# vLLM v1 Python 数据结构速查

> 按 Engine / Scheduler / ModelRunner 三层组织
> 每个数据结构附源码文件 + 行号，可直接跳转

---

## 1. Engine 层：跨线程/跨进程通信

核心职责：Frontend ↔ EngineCore 之间的请求/响应传递，双线程 IO 模型。

### 1.1 queue.Queue — 线程安全消息队列

```python
# vllm/v1/engine/core.py L825-826 (EngineCoreProc)
self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
```

| 特性 | 说明 |
|------|------|
| 线程安全 | 内置锁，`put()`/`get()` 阻塞语义 |
| GIL 释放 | IO 线程等待时释放 GIL，GPU 线程可并行计算 |
| 用途 | Input 线程 → `input_queue` → 主循环 → `output_queue` → Output 线程 |

**面试要点**：vLLM 用双线程 + Queue 实现 IO 与计算并行，避免 ZMQ recv/send 阻塞 GPU 推理。

### 1.2 deque — 双端队列

```python
# vllm/v1/engine/core.py L194 (EngineCore.__init__)
self.batch_queue = deque(maxlen=self.batch_queue_size)

# vllm/v1/engine/core.py L1478 (process_output_sockets)
pending: deque[tuple[zmq.MessageTracker, Any, bytearray]] = deque()
```

| 位置 | 用途 | 操作 |
|------|------|------|
| `batch_queue` (L194) | Pipeline Parallel 批次队列 | `appendleft` 加新批次，`pop` 取最老批次 |
| `pending` (L1478) | ZMQ 零拷贝发送追踪 | 追踪在飞的 MessageTracker |

**面试要点**：`deque(maxlen=N)` 自动丢弃最老元素，适合 PP 的滑动窗口批处理。

### 1.3 threading.Event — 线程间信号通知

```python
# vllm/v1/engine/utils.py L238 (SignalCallback.__init__)
self._event = threading.Event()

# vllm/v1/engine/core.py L889 (EngineCoreProc)
ready_event = threading.Event()
```

| 位置 | 用途 |
|------|------|
| `SignalCallback` (utils.py L233) | 信号处理线程安全：`_event.wait()` 阻塞等待，`_event.set()` 唤醒 |
| `ready_event` (core.py L889) | 三阶段握手中通知输入线程已就绪 |

```python
# 信号处理的安全模式（utils.py L248-257）
def _thread_main(self):
    self._event.wait()       # 阻塞，不占 CPU
    self._callback()         # 被唤醒后执行回调

def _signal_handler(self, signum, frame):
    self._event.set()        # 信号处理函数中只做 set，不执行回调
```

**面试要点**：信号处理函数中不能做复杂操作（可能死锁），所以用 Event 通知专用线程执行。

### 1.4 msgspec.Struct — 高性能序列化

```python
# vllm/v1/engine/__init__.py L80 (EngineCoreRequest)
class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,       # 数组布局，比 dict 快 2-3x
    omit_defaults=True,    # 默认值不序列化，减少数据量
    gc=False,              # 禁用 GC 追踪，减少 GC 压力
):
    request_id: str
    prompt_token_ids: list[int] | None
    ...

# vllm/v1/engine/__init__.py L161 (EngineCoreOutput)
class EngineCoreOutput(msgspec.Struct, ...):
    request_id: str
    new_token_ids: list[int]        # 增量设计！只发新 token
    finish_reason: FinishReason | None = None
    ...

# vllm/v1/engine/__init__.py L206 (EngineCoreOutputs)
class EngineCoreOutputs(msgspec.Struct, ...):
    outputs: list[EngineCoreOutput] = []
    finished_requests: set[str] | None = None
```

| 参数 | 效果 |
|------|------|
| `array_like=True` | 序列化用数组布局，比 dict 快 2-3x |
| `omit_defaults=True` | 默认值字段跳过，减少 IPC 数据量 |
| `gc=False` | 不追踪 GC，减少引用计数开销 |

**面试要点**：msgspec 比 dataclass+pickle 快 20x（~5μs vs ~100μs per request），是 vLLM IPC 的性能基石。

### 1.5 enum.IntEnum / enum.Enum — 枚举

```python
# vllm/v1/engine/__init__.py L42 (FinishReason)
class FinishReason(enum.IntEnum):
    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4

    def __str__(self):
        return FINISH_REASON_STRINGS[self.value]  # 内部整数，外部字符串

# vllm/v1/engine/__init__.py L237 (EngineCoreRequestType)
class EngineCoreRequestType(enum.Enum):
    ADD = b"\x00"               # 1 byte，直接发 ZMQ
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
```

**面试要点**：IntEnum 用整数传输更紧凑（1 byte vs 4+ bytes），`__str__` 对外返回字符串保持可读性。

---

## 2. Scheduler 层：调度排序 + 资源管理

核心职责：决定哪些请求运行、分配多少 token、管理 KV Cache block。

### 2.1 dict — 请求状态索引

```python
# vllm/v1/core/sched/scheduler.py L158 (Scheduler.__init__)
self.requests: dict[str, Request] = {}           # 所有活跃请求

# vllm/v1/core/sched/scheduler.py L369 (schedule())
req_to_new_blocks: dict[str, KVCacheBlocks] = {}  # 本次分配的 block
num_scheduled_tokens: dict[str, int] = {}          # 本次调度的 token 数
scheduled_spec_decode_tokens: dict[str, list[int]] = {}  # 投机解码 token

# vllm/v1/core/sched/scheduler.py L100
finished_req_ids_dict: dict[int, set[str]] | None  # 按 client_index 分组
```

**面试要点**：Scheduler 的 `schedule()` 返回多个 dict，每个 dict 按 request_id 索引——这是 vLLM 的"调度输出协议"。

### 2.2 deque — FCFS 请求队列

```python
# vllm/v1/core/sched/request_queue.py L75
class FCFSRequestQueue(deque[Request], RequestQueue):
    """先到先服务队列"""
    def add(self, request): self.append(request)      # O(1) 尾部插入
    def pop(self): return self.popleft()               # O(1) 头部弹出
    def peek(self): return self[0]                     # O(1) 查看头部
    def prepend(self, request): self.appendleft(request)  # O(1) 头部插入（抢占恢复）
```

| 操作 | 复杂度 | 用途 |
|------|--------|------|
| `append` | O(1) | 新请求入队 |
| `popleft` | O(1) | 取最老请求调度 |
| `appendleft` | O(1) | 抢占恢复的请求放回队头 |

### 2.3 heapq — 优先级请求队列

```python
# vllm/v1/core/sched/request_queue.py L131-184
class PriorityRequestQueue(RequestQueue):
    def __init__(self):
        self._heap: list[Request] = []

    def add(self, request):
        heapq.heappush(self._heap, request)    # O(log n)

    def pop(self):
        return heapq.heappop(self._heap)       # O(log n)

    def peek(self):
        return self._heap[0]                    # O(1)
```

**依赖 `Request.__lt__`**：

```python
# vllm/v1/request.py L296-307
def __lt__(self, other: "Request") -> bool:
    if self.priority != other.priority:
        return self.priority < other.priority       # 优先级低的先调度
    if self.arrival_time != other.arrival_time:
        return self.arrival_time < other.arrival_time  # 先到先服务
    if self.request_id != other.request_id:
        return self.request_id < other.request_id   # 确定性排序
    return id(self) < id(other)                     # 最终 tiebreaker
```

**面试要点**：heapq 不支持随机删除，`remove_requests` 要先标记再 `heapify`（O(n)），这是经典取舍。

### 2.4 侵入式双向链表 — FreeKVCacheBlockQueue

```python
# vllm/v1/core/kv_cache_utils.py L127-128 (KVCacheBlock)
class KVCacheBlock:
    prev_free_block: "KVCacheBlock | None" = None   # 前驱指针
    next_free_block: "KVCacheBlock | None" = None   # 后继指针

# vllm/v1/core/kv_cache_utils.py L162 (FreeKVCacheBlockQueue)
class FreeKVCacheBlockQueue:
    """侵入式双向链表，不额外分配节点对象"""
```

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| `popleft` | O(1) | 从头部取空闲 block |
| `append` | O(1) | 释放 block 放回尾部 |
| 任意位置摘除 | O(1) | 驱逐缓存时直接修改 `prev/next` 指针 |

**面试要点**：为什么不用 `collections.deque`？因为 deque 不支持 O(1) 从中间删除。侵入式链表把指针存在 block 对象本身上，实现 O(1) 任意位置操作。

### 2.5 set — 去重与快速查找

```python
# vllm/v1/core/sched/scheduler.py L176
self.finished_req_ids: set[str] = set()   # 本 step 完成的请求

# vllm/v1/core/sched/request_queue.py L111, 182
requests_to_remove = set(requests)        # O(1) 查找加速批量删除
```

### 2.6 BlockHashToBlockMap — 前缀缓存查找

```python
# vllm/v1/core/block_pool.py L171
self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

# 内部实现（vllm/v1/core/kv_cache_utils.py L58-59）
# dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]
```

**面试要点**：前缀缓存的核心——如果两个请求共享相同 prompt 前缀，可以复用已计算的 KV Cache block，避免重复计算。

### 2.7 list — 运行中请求列表

```python
# vllm/v1/core/sched/scheduler.py L170
self.running: list[Request] = []
```

**为什么 running 用 list 不用 deque？**
- 需要遍历 + 按索引删除（抢占时 `running.remove(req)` 或 `running.pop(i)`）
- list 的 `remove` 是 O(n)，但 running 列表通常很短（受 `max_num_running_reqs` 限制）

### 2.8 RequestStatus — 状态机枚举

```python
# vllm/v1/request.py L310-326
class RequestStatus(enum.IntEnum):
    WAITING = enum.auto()
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # --- 以下都是终态 ---
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED  # O(1) 数值比较
```

**面试要点**：`is_finished` 用 `>` 而不是枚举所有终态——利用 IntEnum 的数值顺序，新增终态时不需要改判断逻辑。

---

## 3. ModelRunner 层：GPU 计算 + Buffer 管理

核心职责：管理 persistent batch 的 GPU/CPU buffer，执行模型推理。

### 3.1 dict — Persistent Batch 状态索引

```python
# vllm/v1/worker/gpu_input_batch.py L91, L127 (InputBatch)
class InputBatch:
    def __init__(self, ...):
        self.req_id_to_index: dict[str, int] = {}    # request_id → slot 索引
        self.req_ids: list[str] = []                   # slot → request_id（反向映射）
```

**Persistent Batch 的核心设计**：

```
req_id_to_index = {"req-001": 0, "req-003": 1, "req-007": 2}
                   ↓
GPU buffers 按 slot 索引：
  input_ids[0] = req-001 的 input_ids
  input_ids[1] = req-003 的 input_ids
  input_ids[2] = req-007 的 input_ids
```

**面试要点**：Persistent Batch 用固定大小的 GPU buffer（不随 batch size 变化），配合 CUDA Graph 实现零 CPU 开销推理。dict 提供 O(1) 的 request_id → slot 映射。

### 3.2 torch.Tensor — GPU 固定 Buffer

```python
# vllm/v1/worker/gpu_input_batch.py (InputBatch 的 GPU buffers)
self.input_ids: torch.Tensor          # [max_num_reqs] int32
self.positions: torch.Tensor          # [max_num_reqs] int32
self.seq_lens: torch.Tensor           # [max_num_reqs] int32
self.block_tables: torch.Tensor       # [max_num_reqs, max_num_blocks] int32

# vllm/v1/worker/gpu_model_runner.py L397-418 (GPUModelRunner.__init__)
# 持久化 CUDA buffer，配合 CUDA Graph
```

**为什么用固定大小 Tensor？**
- CUDA Graph 要求每次推理的 kernel 签名完全一致（tensor shape、stride）
- 固定 buffer + padding = 同一个 Graph 可以复用
- 变长请求通过 `num_scheduled_tokens` 和 `seq_lens` 标记有效范围

### 3.3 numpy.ndarray — CPU Staging Buffer

```python
# vllm/v1/worker/gpu_input_batch.py L162-168
self.num_computed_tokens_cpu_tensor = torch.zeros(max_num_reqs, dtype=torch.int32)
self.num_computed_tokens_cpu = self.num_computed_tokens_cpu_tensor.numpy()  # 零拷贝视图
```

**CPU → GPU 数据流**：

```
Python dict/list  →  numpy 操作  →  torch.Tensor (CPU pinned)  →  .copy_(non_blocking)  →  GPU
     (慢)              (批量快)         (零拷贝视图)                    (异步 DMA)
```

**面试要点**：numpy 用于 CPU 端的批量操作（如 `num_computed_tokens_cpu[:] = values`），比逐个 Python 赋值快得多。

### 3.4 NamedTuple — 轻量不可变返回值

```python
# vllm/v1/worker/gpu_model_runner.py L378 (ExecuteModelState)
class ExecuteModelState(NamedTuple):
    hidden_states: torch.Tensor | None
    logits: torch.Tensor | None
    speculative_decode_metadata: Any | None
    scheduler_output: Any | None
```

**为什么用 NamedTuple 不用 dataclass？**
- 不可变（immutable），线程安全
- 不需要 `__init__` 生成代码，更轻量
- 支持解包：`hidden, logits, *_ = state`

### 3.5 Callable / 闭包 — 延迟执行

```python
# vllm/v1/worker/gpu_model_runner.py L81 (AsyncIntermediateTensors)
class AsyncIntermediateTensors(IntermediateTensors):
    comm_postprocess: list[Callable[[], None]]  # PP 通信后处理回调

# vllm/v1/request.py L159 (_block_hasher)
self._block_hasher: Callable[[Request], list[BlockHash]] | None
```

| 用途 | 说明 |
|------|------|
| `comm_postprocess` | PP 通信完成后执行的回调，延迟到实际需要时才调用 |
| `_block_hasher` | 避免引用循环（Request → partial → Request），通过回调解耦 |

### 3.6 AsyncIntermediateTensors — 懒同步代理

```python
# vllm/v1/worker/gpu_worker.py L74
class AsyncIntermediateTensors(IntermediateTensors):
    def __getattribute__(self, name):
        if name == "tensors":
            self.wait_for_comm()  # 第一次访问 .tensors 时才等待通信完成
        return super().__getattribute__(name)
```

**面试要点**：Pipeline Parallel 的关键优化——PP 前一层的通信和当前层的计算可以重叠。`__getattribute__` 拦截实现了"用到才等"的懒同步。

### 3.7 torch.cuda.Stream / Event — 异步拷贝

```python
# vllm/v1/worker/gpu_model_runner.py L227 (AsyncGPUModelRunnerOutput)
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(self, ...):
        self.sampled_token_ids_cpu = sampled_token_ids.to("cpu", non_blocking=True)
        # 用单独的 CUDA stream 做 D2H 拷贝，不阻塞主 stream
```

**异步 D2H 拷贝流程**：

```
主 Stream:  [compute] → [record event] → [继续 compute]
副 Stream:  [D2H copy sampled_token_ids] → [event ready]
Output 线程: [wait event] → [send via ZMQ]
```

---

## 4. 万能回答框架

当面试官问"vLLM 用了什么数据结构"时：

> "分三层看：
> - **Engine 层**是跨线程/进程通信，用 `Queue` 和 `deque` 做消息传递，`msgspec.Struct` 做高性能序列化
> - **Scheduler 层**是调度排序 + 资源管理，用 `heapq` 做优先级调度，用侵入式双向链表管理 KV Cache block 实现 O(1) 任意位置操作
> - **Runner 层**是 GPU 计算，用固定 shape 的 `torch.Tensor` buffer 配合 CUDA Graph，用 `dict` 按 slot 索引管理 persistent batch 状态"

---

## 5. 面试优先级

| 优先级 | 数据结构 | 被问概率 | 典型问题 |
|--------|---------|---------|---------|
| **必须熟练** | `heapq` | >70% | "vLLM 怎么做优先级调度？" |
| **必须熟练** | `deque` | >70% | "FCFS 队列为什么用 deque？" |
| **必须熟练** | `queue.Queue` | >60% | "双线程 IO 怎么实现的？" |
| **必须熟练** | `dict` + 固定 buffer | >60% | "Persistent Batch 是什么？" |
| **值得了解** | 侵入式双向链表 | 40% | "KV Cache block 怎么管理？" |
| **值得了解** | `msgspec.Struct` | 30% | "IPC 怎么优化？" |
| **值得了解** | `threading.Event` | 20% | "信号处理怎么做？" |
| **知道就行** | `NamedTuple` | 10% | 返回值，不会单独问 |
| **知道就行** | numpy CPU buffer | 10% | 异步拷贝 staging 区 |

---

## 6. 源码文件索引

| 文件                                    | 核心数据结构                                | 行数   |
| ------------------------------------- | ------------------------------------- | ---- |
| `vllm/v1/engine/core.py`              | queue.Queue, deque, threading         | 2146 |
| `vllm/v1/engine/utils.py`             | threading.Event, weakref              | 1277 |
| `vllm/v1/engine/__init__.py`          | msgspec.Struct, enum                  | 270  |
| `vllm/v1/request.py`                  | `ConstantList, __lt__, RequestStatus` | 352  |
| `vllm/v1/core/sched/scheduler.py`     | dict, list, set                       | ~900 |
| `vllm/v1/core/sched/request_queue.py` | deque, heapq                          | ~200 |
| `vllm/v1/core/kv_cache_utils.py`      | 侵入式双向链表                               | ~500 |
| `vllm/v1/core/block_pool.py`          | dict (BlockHashToBlockMap)            | ~300 |
| `vllm/v1/worker/gpu_model_runner.py`  | torch.Tensor, NamedTuple, Callable    | 7123 |
| `vllm/v1/worker/gpu_input_batch.py`   | dict, numpy, torch.Tensor             | ~600 |
| `vllm/v1/worker/gpu_worker.py`        | AsyncIntermediateTensors              | 1087 |
## 数据结构速查

|优先级|数据结构|在 vLLM 里对应什么|你要理解到什么程度|
|---|---|---|---|
|P0|`dict[str, Request]`|`Scheduler.requests`|request 全局索引表，所有状态流转都绕不开|
|P0|`list[Request]`|`Scheduler.running`|当前正在被调度/已占 KV 的请求集合|
|P0|`deque[Request]`|`FCFSRequestQueue`|FCFS 等待队列|
|P0|`heapq + Request.__lt__`|`PriorityRequestQueue`|priority scheduling 的本质|
|P0|`dict[str, int]` / `dict[str, list]`|`num_scheduled_tokens`、`scheduled_spec_decode_tokens` 等|一轮 schedule 的结构化结果|
|P0|固定数组 `list[KVCacheBlock]`|`BlockPool.blocks`|block_id 到 block metadata 的稳定映射|
|P0|侵入式双向链表|`FreeKVCacheBlockQueue`|free block / eviction candidate 的 O(1) 管理|
|P0|hashmap|`BlockHashToBlockMap`|prefix cache 的 hash → block 查询|
|P0|ref count|`KVCacheBlock.ref_cnt`|block 是否可释放、可复用、可驱逐|
|P1|`msgspec.Struct`|`EngineCoreRequest` / `EngineCoreOutput` / `EngineCoreOutputs`|进程间通信 DTO，追求低序列化开销|
|P1|dataclass DTO|`SchedulerOutput` / `NewRequestData` / `CachedRequestData` / `ModelRunnerOutput`|Scheduler 和 ModelExecutor 之间的接口|
|P1|`deque(maxlen=...)`|`EngineCore.batch_queue`|pipeline/concurrent batch 的执行流水|
|P1|`set[str]`|`finished_req_ids`、`preempted_req_ids`|批量状态更新和释放|
|P2|`queue.Queue`|`aborts_queue`、input queue|线程/进程入口层，不是调度核心|
|P2|`defaultdict(list)`|`req_to_blocks`|每个 request 当前持有哪些 KV blocks|