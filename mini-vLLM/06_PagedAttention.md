# 06 PagedAttention：注意力层与 KV Cache 读写

> 对标 vLLM：CUDA kernel `paged_attention_v1/v2` → nano-vllm Python + Triton 实现

## 一、Attention 层结构

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])    # 占位，后面由 ModelRunner 填充
```

KV Cache 不在 Attention 内部创建，而是由 ModelRunner 统一分配后注入：
```python
# model_runner.py
module.k_cache = self.kv_cache[0, layer_id]
module.v_cache = self.v_cache = self.kv_cache[1, layer_id]
```

## 二、KV Cache 写入：Triton Kernel

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,            # num_heads * head_dim
):
    idx = tl.program_id(0)      # 第 idx 个 token
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return       # -1 表示不写入
    # 读取 key/value
    key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D))
    value = tl.load(value_ptr + idx * value_stride + tl.arange(0, D))
    # 写入 cache 的对应 slot
    tl.store(k_cache_ptr + slot * D + tl.arange(0, D), key)
    tl.store(v_cache_ptr + slot * D + tl.arange(0, D), value)
```

**slot_mapping** 是核心：
- `slot_mapping[i] = block_id * block_size + offset`
- 把第 i 个 token 的 KV 写入 cache 的第 slot 个位置
- `-1` 表示跳过（warmup 时用）

### 2.1 slot_mapping 构建

**Prefill 时**（model_runner.py:152-161）：
```python
for i in range(start_block, end_block):
    slot_start = seq.block_table[i] * block_size
    if i == start_block:
        slot_start += start % block_size      # prefix cache 命中的部分跳过
    slot_end = seq.block_table[i] * block_size + (block_size 或 end - i*block_size)
    slot_mapping.extend(range(slot_start, slot_end))
```

**Decode 时**（model_runner.py:181）：
```python
slot_mapping.append(seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1)
```

## 三、Attention Forward

```python
def forward(self, q, k, v):
    context = get_context()
    k_cache, v_cache = self.k_cache, self.v_cache

    # 1. 写入 KV Cache
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

    # 2. 计算 attention
    if context.is_prefill:
        if context.block_tables is not None:    # 有 prefix cache
            k, v = k_cache, v_cache             # 从 cache 读
        o = flash_attn_varlen_func(q, k, v,     # 变长 attention
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            causal=True, block_table=context.block_tables)
    else:    # decode
        o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
            cache_seqlens=context.context_lens, block_table=context.block_tables,
            causal=True)
    return o
```

### 3.1 Prefill vs Decode 的 Attention

| | Prefill | Decode |
|--|---------|--------|
| 输入 | q: [total_tokens, heads, dim] | q: [bs, 1, heads, dim] |
| KV 来源 | 当前 k/v + cache（有 prefix 时） | 全部从 cache 读 |
| API | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
| 变长 | cu_seqlens_q/k | cache_seqlens |
| block_table | 可选 | 必须 |

### 3.2 flash_attn_varlen_func

```
输入：
  q = [t0_h0, t0_h1, ..., t1_h0, t1_h1, ...]    # 所有 seq 的 token 拼在一起
  cu_seqlens_q = [0, 3, 7, ...]                   # 每个 seq 的累计长度

效果：
  seq0: q[0:3] @ k[0:3].T / sqrt(dim)
  seq1: q[3:7] @ k[3:7].T / sqrt(dim)
  ...

block_tables: 让 attention 知道 KV Cache 的物理布局
  seq0 的 block_table = [5, 12, 3]
  → k_cache[5], k_cache[12], k_cache[3] 拼接成 seq0 的 K
```

### 3.3 flash_attn_with_kvcache

```
输入：
  q = [bs, 1, heads, dim]           # decode 时每个 seq 只有 1 个 token
  k_cache = [num_blocks, block_size, kv_heads, dim]
  cache_seqlens = [seq0_len, seq1_len, ...]

效果：
  对每个 seq i：
    q[i] @ k_cache[blocks[i][:cache_seqlens[i]]].T / sqrt(dim)
```

## 四、PagedAttention 的本质

```
传统 Attention：
  K = [seq0_k, seq1_k, ...]    # 连续内存
  Q @ K.T

PagedAttention：
  K Cache = [block0, block1, block2, ...]    # 非连续，按 block 分页
  每个 seq 有一个 block_table = [5, 12, 3]   # 指向物理 block
  Q @ K[blocks].T                              # 通过 block_table 间接寻址
```

好处：
- **无碎片**：不需要连续内存，block 可以任意分配
- **共享 prefix**：多个 seq 可以共享同一个 block（ref_count）
- **动态增长**：seq 只在需要时分配新 block

## 五、KV Cache 内存布局

```
kv_cache = [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
           ↑            ↑          ↑          ↑           ↑           ↑
         K/V        层索引     物理block    每block     KV头数      头维度
                    0~31       0~N         256 tokens   8(GQA)     128

一个 block 的大小：
  2 × block_size × num_kv_heads × head_dim × dtype_size
  = 2 × 256 × 8 × 128 × 2 (bf16)
  = 1 MB per block per layer
```

## 六、与 vLLM PagedAttention 对照

| 特性 | nano-vllm | vLLM v1 |
|------|----------|---------|
| 写入 | Triton kernel | CUDA kernel |
| 读取 | flash_attn 内置 | flash_attn / paged_attention kernel |
| KV Cache 布局 | `[2, layers, blocks, ...]` | 分层管理 |
| block_table | 2D tensor | 2D tensor |
| slot_mapping | 1D tensor | 1D tensor |
| GQA | 支持（num_kv_heads < num_heads） | 支持 |

## 七、学习要点

1. **slot_mapping 是 KV Cache 读写的桥梁**：token → slot 的映射
2. **Triton kernel 做写入**：比 Python 循环快，比 CUDA kernel 好写
3. **flash_attn 做读取**：变长 + block_table 支持 PagedAttention
4. **Prefill 用 varlen_func，decode 用 with_kvcache**：两个不同的 API
5. **block_table 是间接寻址**：让 KV Cache 可以非连续存储
