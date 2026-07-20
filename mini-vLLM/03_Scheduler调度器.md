# 03 Scheduler 调度器

> 对标 vLLM：`vllm/v1/core/sched/scheduler.py`（~900 行）→ nano-vllm `scheduler.py`（92 行）

## 一、数据结构

```python
class Scheduler:
    def __init__(self, config):
        self.max_num_seqs = config.max_num_seqs           # 最大并发序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 每步最大 token 数
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(num_blocks, block_size)
        self.waiting: deque[Sequence] = deque()    # 等待 prefill
        self.running: deque[Sequence] = deque()    # 正在 decode
```

两个队列，一个 BlockManager，没有优先级队列。

## 二、schedule() 核心逻辑

```
schedule() 返回 (seqs, is_prefill)

阶段 1：Prefill（优先）
  遍历 waiting 队列：
    ├── 检查 prefix cache（can_allocate → num_cached_blocks）
    ├── 计算需要的 token 数 = num_tokens - cached
    ├── 检查 token budget（remaining >= num_tokens）
    ├── 分配 block（allocate）
    ├── 设置 num_scheduled_tokens
    ├── 如果 prefill 完成 → 移到 running
    └── 加入 scheduled_seqs

  如果有 scheduled_seqs → 返回 (seqs, True)

阶段 2：Decode（prefill 为空时才执行）
  遍历 running 队列：
    ├── 检查能否 append（can_append）
    ├── 不能 → preempt 最后一个 running 的 seq
    ├── 能 → 设置 num_scheduled_tokens=1
    └── 加入 scheduled_seqs

  返回 (seqs, False)
```

### 2.1 Chunked Prefill

```python
if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
    break
```

第一个 seq 可以跨多步 prefill（chunked），后面的 seq 必须整块调度。
这是 vLLM 的 ChunkedPrefill 特性的简化版。

### 2.2 Prefill 优先级

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    ...
if scheduled_seqs:
    return scheduled_seqs, True   # prefill 优先
```

只要有 waiting 的 seq，就先做 prefill。Decode 只在 waiting 为空时执行。
这保证了新请求的 TTFT（Time To First Token）。

### 2.3 Preemption 策略

```python
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())  # 抢最后一个
    else:
        self.preempt(seq)  # 抢自己（退回到 waiting）
        break
```

- 抢占 running 队列**最后一个** seq（最不重要的）
- 如果 running 只剩一个，就抢占自己
- 被抢占的 seq 回到 waiting 队列头部（优先重新调度）

## 三、postprocess() 后处理

```python
def postprocess(self, seqs, token_ids, is_prefill):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)        # 1. 计算 block hash（prefix cache）
        seq.num_cached_tokens += seq.num_scheduled_tokens  # 2. 更新 cached 计数
        seq.num_scheduled_tokens = 0               # 3. 重置 scheduled 计数
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue                               # 4. prefill 未完成，跳过
        seq.append_token(token_id)                 # 5. 追加生成的 token
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED    # 6. 结束条件
            self.block_manager.deallocate(seq)      # 7. 释放 block
            self.running.remove(seq)
```

关键点：
- `hash_blocks()` 只在 prefill 后调用，把新 block 注册到 prefix cache
- chunked prefill 时，一次 step 只处理部分 token，不生成 output token
- decode 时，每步生成一个 token，检查 EOS 或 max_tokens

## 四、与 vLLM Scheduler 对照

| 特性 | nano-vllm | vLLM v1 |
|------|----------|---------|
| 队列 | `deque` × 2 | `FCFSQueue` + `PriorityQueue` |
| prefill 优先 | 是 | 是（`schedule_prefills` 先调用） |
| chunked prefill | 简化版 | 完整版（`max_long_partial_prefills`） |
| preemption | 抢最后一个 running | 抢 lowest priority |
| token budget | `max_num_batched_tokens` | `max_num_batched_tokens` |
| seq limit | `max_num_seqs` | `max_num_seqs` |
| async scheduling | 无 | 有（`async_scheduling`） |
| DP/PP | 无 | 有（`connector.py`） |
| 条件编译 | 无 | 有（`@config_compiled_schedule`） |

## 五、调度时序图

```
Step 1 (Prefill):
  waiting: [seq0(prompt=100), seq1(prompt=200)]
  schedule → seq0(100 tokens), seq1(200 tokens), is_prefill=True
  model_runner.run(seqs, prefill=True) → token_ids=[tok0, tok1]
  postprocess: seq0 append tok0, seq1 append tok1
  waiting: [], running: [seq0, seq1]

Step 2 (Decode):
  waiting: [], running: [seq0, seq1]
  schedule → seq0(1 token), seq1(1 token), is_prefill=False
  model_runner.run(seqs, prefill=False) → token_ids=[tok2, tok3]
  postprocess: seq0 append tok2, seq1 append tok3
  ...

Step N (Finished):
  postprocess: token_id == EOS → seq0 FINISHED, deallocate
  running: [seq1]
```

## 六、学习要点

1. **Prefill 优先**：只要 waiting 有 seq，就先做 prefill，保证 TTFT
2. **Token budget 控制**：`max_num_batched_tokens` 限制每步总 token 数
3. **Preemption 是最后手段**：block 不够时抢最后一个 running 的 seq
4. **Postprocess 分两阶段**：prefill 后 hash_blocks，decode 后 append_token
5. **92 行实现核心调度**：vLLM 900 行的 1/10
