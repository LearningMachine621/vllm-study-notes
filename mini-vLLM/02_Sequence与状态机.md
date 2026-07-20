# 02 Sequence 与状态机

> 对标 vLLM：`vllm/v1/request.py`（352 行）→ nano-vllm `sequence.py`（83 行）

## 一、Sequence 类设计

```python
class Sequence:
    block_size = 256          # 类变量，由 Config 设置
    counter = count()         # 全局自增 ID

    def __init__(self, token_ids, sampling_params):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)    # prompt token ids
        self.last_token = token_ids[-1]     # decode 时只传最后一个
        self.num_tokens = len(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0          # prefix cache 命中数
        self.num_scheduled_tokens = 0       # 本轮调度的 token 数
        self.is_prefill = True
        self.block_table = []               # KV Cache block 分配表
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```

## 二、状态机

```
              add()
  ┌───────────────────────┐
  │                       ▼
  │                   ┌────────┐
  │                   │ WAITING │ ←── preempt()
  │                   └────┬───┘
  │          schedule()    │
  │          (prefill      │
  │           完成)        ▼
  │                   ┌────────┐
  │                   │ RUNNING │ ←── prefill 未完成时保持 WAITING
  │                   └────┬───┘
  │          postprocess() │
  │          (EOS/max)     ▼
  │                   ┌─────────┐
  │                   │ FINISHED │
  │                   └─────────┘
```

与 vLLM 的差异：
- vLLM 有 5 个状态：`WAITING → RUNNING → FINISHED_OK / FINISHED_LENGTH / PREEMPTED`
- nano-vllm 只有 3 个：`WAITING → RUNNING → FINISHED`
- nano-vllm 的 preemption 直接把状态改回 `WAITING`

## 三、关键属性

### 3.1 num_tokens 动态增长

```python
def append_token(self, token_id):
    self.token_ids.append(token_id)
    self.last_token = token_id
    self.num_tokens += 1
```

每次 decode 生成一个 token，`num_tokens` 加 1。
`num_blocks` 随之变化：`(num_tokens + block_size - 1) // block_size`

### 3.2 block(i) —— 按 block 切分 token

```python
def block(self, i):
    return self.token_ids[i*self.block_size: (i+1)*self.block_size]
```

用于 prefix cache 的 hash 计算。最后一个 block 可能不满。

### 3.3 last_block_num_tokens

```python
@property
def last_block_num_tokens(self):
    return self.num_tokens - (self.num_blocks - 1) * self.block_size
```

用于 decode 时计算 slot_mapping 的偏移。
比如 block_size=256，num_tokens=300 → last_block_num_tokens = 300 - 256 = 44

## 四、pickle 安全（SHM IPC）

```python
def __getstate__(self):
    last_state = self.last_token if not self.is_prefill else self.token_ids
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.num_scheduled_tokens, self.block_table, last_state)

def __setstate__(self, state):
    ...
    if isinstance(last_state, list):    # prefill：传完整 token_ids
        self.token_ids = last_state
        self.last_token = self.token_ids[-1]
    else:                                # decode：只传 last_token
        self.token_ids = []
        self.last_token = last_state
```

为什么这样设计？
- **prefill 时**：需要完整 token_ids 做 attention，传全部
- **decode 时**：只需要 last_token，省带宽（SharedMemory 只有 1MB）
- 这是 nano-vllm 能用 SHM 代替 ZMQ 的关键优化

## 五、与 vLLM Request 对照

| 字段/方法 | nano-vllm Sequence | vLLM Request |
|----------|-------------------|-------------|
| ID | `seq_id` (int) | `request_id` (str) |
| token 存储 | `token_ids` (list) | `_all_token_ids` + `_output_token_ids` (ConstantList) |
| 状态 | `SequenceStatus` (3 态) | `RequestStatus` (5 态) |
| block table | `block_table` (list) | `block_hashes` + `block_table` |
| prefix cache | `num_cached_tokens` | `num_computed_tokens` |
| IPC 序列化 | `__getstate__/__setstate__` | msgspec Struct |
| priority | 无 | `priority: int` |
| workload_type | 无 | 无（vLLM 原版也没有） |

## 六、学习要点

1. **Sequence 是自包含的**：token ids + 状态 + block table + 采样参数全在一个对象里
2. **3 态状态机够用**：WAITING/RUNNING/FINISHED 覆盖了核心生命周期
3. **pickle 优化是亮点**：prefill 传全量、decode 传 last_token，SHM 友好
4. **block_size 是类变量**：所有 Sequence 共享，由 Config 初始化时设置
