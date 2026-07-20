# 05 ModelRunner：模型执行 + CUDA Graph + SHM IPC

> 对标 vLLM：`GPUModelRunner`（~7000 行）→ nano-vllm `model_runner.py`（257 行）

## 一、初始化流程

```python
class ModelRunner:
    def __init__(self, config, rank, event):
        # 1. 分布式初始化
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size, rank)
        torch.cuda.set_device(rank)

        # 2. 加载模型
        self.model = Qwen3ForCausalLM(hf_config)    # 硬编码 Qwen3
        load_model(self.model, config.model)

        # 3. 采样器
        self.sampler = Sampler()

        # 4. Warmup（测量峰值显存）
        self.warmup_model()

        # 5. 分配 KV Cache
        self.allocate_kv_cache()

        # 6. 捕获 CUDA Graph（可选）
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 7. 多进程 IPC（TP > 1）
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()    # rank 1~N 进入等待循环
```

## 二、KV Cache 分配

```python
def allocate_kv_cache(self):
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

    # 计算每个 block 占多少字节
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * head_dim * dtype.itemsize

    # 可用显存 = 总显存 * 利用率 - 已用 - 峰值 + 当前
    config.num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes

    # 统一 KV Cache tensor：[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    self.kv_cache = torch.empty(2, num_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)

    # 分发给每个 Attention 层
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

关键点：
- **统一 tensor**：所有层的 KV Cache 在一个 tensor 里，按 `[layer, block, ...]` 索引
- **显存计算**：`total * utilization - used - peak + current` 是安全的可用显存
- **block_bytes**：每 block = 2(K+V) × layers × block_size × kv_heads × head_dim × dtype_size

## 三、prepare_prefill vs prepare_decode

### 3.1 Prefill

```python
def prepare_prefill(self, seqs):
    for seq in seqs:
        start = seq.num_cached_tokens      # prefix cache 命中的部分跳过
        end = start + seq.num_scheduled_tokens
        input_ids.extend(seq[start:end])    # 只取需要 prefill 的部分
        positions.extend(range(start, end))
        # cu_seqlens_q/k：变长序列的累计长度（flash_attn 需要）
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        # slot_mapping：token → KV Cache slot 的映射
        for i in range(start_block, end_block):
            slot_mapping.extend(range(slot_start, slot_end))
    # 如果有 prefix cache，需要 block_tables
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
        block_tables = self.prepare_block_tables(seqs)
    set_context(True, cu_seqlens_q, cu_seqlens_k, ..., block_tables)
```

### 3.2 Decode

```python
def prepare_decode(self, seqs):
    for seq in seqs:
        input_ids.append(seq.last_token)        # 只取最后一个 token
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))            # 总长度（attention 需要）
        slot_mapping.append(seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1)
    block_tables = self.prepare_block_tables(seqs)
    set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
```

| | Prefill | Decode |
|--|---------|--------|
| input_ids | 多个 token | 1 个 token |
| positions | 范围 [start:end) | 单个位置 |
| attention | flash_attn_varlen_func | flash_attn_with_kvcache |
| block_tables | 可选（有 prefix cache 时） | 必须 |
| cu_seqlens | 必须 | 不需要 |

## 四、CUDA Graph

```python
def capture_cudagraph(self):
    # 预分配 tensor
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hidden_size)

    # 捕获不同 batch size 的 graph
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        set_context(False, ...)
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
        self.graphs[bs] = graph
```

decode 时使用 graph：

```python
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        bs = input_ids.size(0)
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]  # 找最小的够用的 graph
        graph_vars["input_ids"][:bs] = input_ids     # 写入预分配的 tensor
        graph_vars["positions"][:bs] = positions
        graph.replay()                                # 重放 graph
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

为什么需要 CUDA Graph？
- Decode 的 kernel 很小（bs=1~512，每个 token 1 次 forward）
- CPU launch kernel 的开销（~10μs/kernel）占比大
- CUDA Graph 把整个 forward 录制成一个 graph，一次 launch 完成

## 五、SHM IPC（多进程通信）

```python
# rank 0 → rank 1~N
def write_shm(self, method_name, *args):
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")    # 长度前缀
    self.shm.buf[4:n+4] = data                      # 序列化数据
    for event in self.event:
        event.set()                                  # 通知子进程

# rank 1~N
def read_shm(self):
    self.event.wait()                                # 等待通知
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
    return method_name, args

def loop(self):                                      # 子进程主循环
    while True:
        method_name, args = self.read_shm()
        self.call(method_name, *args)
        if method_name == "exit":
            break
```

- SharedMemory 只有 **1MB**，所以 decode 时只传 `last_token`（不是全部 token_ids）
- `Event` 用于同步：rank 0 写完 → set → rank 1~N 读完 → clear

## 六、与 vLLM GPUModelRunner 对照

| 特性 | nano-vllm | vLLM v1 |
|------|----------|---------|
| 行数 | 257 行 | ~7000 行 |
| 模型 | 硬编码 Qwen3 | 任意 HF 模型 |
| KV Cache | 统一 tensor | 分层管理 |
| CUDA Graph | 有 | 有（更复杂） |
| IPC | SharedMemory + Event | ZMQ |
| TP | 支持 | 支持 |
| PP | 不支持 | 支持 |
| 多模态 | 不支持 | 支持 |

## 七、学习要点

1. **KV Cache 是统一 tensor**：`[2, layers, blocks, block_size, kv_heads, head_dim]`
2. **CUDA Graph 只用于 decode**：prefill 变长，不能用 graph
3. **prepare_prefill/decode 构建 Context**：slot_mapping、cu_seqlens、block_tables
4. **SHM IPC 用 pickle**：简单但有 1MB 限制
5. **257 行 vs 7000 行**：省掉了 PP、多模态、复杂调度
