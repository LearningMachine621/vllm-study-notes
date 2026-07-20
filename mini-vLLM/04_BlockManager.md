# 04 BlockManager：KV Cache Block 分配与 Prefix Cache

> 对标 vLLM：`KVCacheManager`（~400 行）+ `block_pool.py`（~300 行）→ nano-vllm `block_manager.py`（120 行）

## 一、核心数据结构

```python
class Block:
    block_id: int       # 物理 block 编号
    ref_count: int      # 引用计数（prefix cache 共享）
    hash: int           # block 内容的 hash（用于 prefix 匹配）
    token_ids: list     # block 存储的 token ids（用于验证 hash 碰撞）

class BlockManager:
    blocks: list[Block]                    # 所有 block 对象
    hash_to_block_id: dict[int, int]       # hash → block_id（prefix cache 索引）
    free_block_ids: deque[int]             # 空闲 block 池
    used_block_ids: set[int]               # 已用 block 集合
```

## 二、Block 分配流程

```
allocate(seq, num_cached_blocks)

  1. 复用 prefix cache 命中的 block（ref_count++）
     for i in range(num_cached_blocks):
         h = compute_hash(seq.block(i), h)
         block_id = hash_to_block_id[h]
         block.ref_count += 1
         seq.block_table.append(block_id)

  2. 分配新 block
     for i in range(num_cached_blocks, seq.num_blocks):
         block_id = _allocate_block()   # 从 free_block_ids 取
         seq.block_table.append(block_id)

  3. 更新 num_cached_tokens
     seq.num_cached_tokens = num_cached_blocks * block_size
```

## 三、Prefix Cache 机制

### 3.1 Hash 计算

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

- 用 **xxhash**（比 SHA 快 10x）
- **链式 hash**：每个 block 的 hash 包含前一个 block 的 hash
- 这样 `block[0:3]` 的 hash 可以唯一标识前 3 个 block 的内容

### 3.2 Prefix 匹配

```python
def can_allocate(self, seq) -> int:
    h = -1
    num_cached_blocks = 0
    for i in range(seq.num_blocks - 1):          # 最后一个 block 不检查
        token_ids = seq.block(i)
        h = compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break                                  # hash 碰撞或无匹配
        num_cached_blocks += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1                    # 已用的 block 不需要再分配
    if len(self.free_block_ids) < num_new_blocks:
        return -1                                  # block 不够
    return num_cached_blocks
```

关键点：
- **只检查前 N-1 个 block**：最后一个 block 可能不满，hash 不稳定
- **双重验证**：hash 匹配后还要比较 token_ids（防碰撞）
- **ref_count 共享**：已用的 block 可以被多个 seq 共享

### 3.3 Hash 注册（在 postprocess 中）

```python
def hash_blocks(self, seq):
    start = seq.num_cached_tokens // self.block_size
    end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
    if start == end: return                        # 不满一个 block，不 hash
    h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
    for i in range(start, end):
        block = self.blocks[seq.block_table[i]]
        token_ids = seq.block(i)
        h = compute_hash(token_ids, h)
        block.update(h, token_ids)
        self.hash_to_block_id[h] = block.block_id
```

只在 **prefill 完成后** 注册 hash。Decode 阶段不注册（因为只生成 1 个 token，不构成完整 block）。

## 四、Deallocate（释放）

```python
def deallocate(self, seq):
    for block_id in reversed(seq.block_table):     # 反向释放
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)       # ref_count 归零才真正释放
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

- **ref_count > 0**：block 还被其他 seq 使用，只减计数
- **ref_count == 0**：block 无引用，放回 free_block_ids

## 五、Decode 时的 Block 管理

```python
def can_append(self, seq) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq):
    if len(seq) % self.block_size == 1:            # 刚好跨入新 block
        seq.block_table.append(self._allocate_block())
```

- `len(seq) % block_size == 1`：token 数刚好是 block_size 的倍数 + 1
- 比如 block_size=256，seq 有 257 个 token → 需要新 block
- `can_append` 只检查是否有 1 个空闲 block（不是全部）

## 六、与 vLLM KVCacheManager 对照

| 特性 | nano-vllm | vLLM v1 |
|------|----------|---------|
| 数据结构 | `Block` 对象 + `dict` | `KVCacheManager` + `BlockPool` |
| hash 算法 | xxhash（链式） | hash（block 内容） |
| prefix cache | 有（block 级别） | 有（block 级别） |
| ref_count 共享 | 有 | 有 |
| preemption | deallocate + 回 waiting | deallocate + 回 waiting |
| GPU 内存 | ModelRunner 分配 | Worker 分配 |
| block size | 256（固定） | 可配置 |

## 七、Prefix Cache 命中流程

```
seq = Sequence([t0, t1, ..., t511])    # 512 tokens, 2 blocks

Step 1: can_allocate(seq)
  block[0] = [t0..t255] → hash=h0 → hash_to_block_id[h0]=block_5 → 命中!
  block[1] = [t256..t511] → 不检查（最后一个 block）
  num_cached_blocks = 1

Step 2: allocate(seq, 1)
  seq.block_table = [block_5]           # 复用 prefix block
  block_5.ref_count += 1
  分配新 block_7 给 block[1]
  seq.block_table = [block_5, block_7]
  seq.num_cached_tokens = 256           # 只需 prefill 后 256 个 token

Step 3: prefill 只处理 token[256:511]，跳过前 256 个
```

## 八、学习要点

1. **Block 是 KV Cache 的最小单位**：block_size=256 个 token 共享一组 KV
2. **Prefix Cache 靠 hash 链**：每个 block 的 hash 包含前一个 block 的 hash
3. **ref_count 实现共享**：多个 seq 可以共享同一个 prefix block
4. **只 hash 完整 block**：最后一个不完整的 block 不注册到 cache
5. **120 行实现 vLLM 700 行的功能**：核心是 hash_to_block_id 字典
