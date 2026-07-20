# EngineCore 关键函数

> 文件: `vllm/v1/engine/core.py`
> 聚焦: `EngineCore.__init__()` / `add_request()` / `step()`
> 职责: vLLM v1 的"心脏"——组装 Scheduler + Executor，驱动推理主循环

---

## 0. 类层次（只看 EngineCore 基类）

```
EngineCore (L91)              ← 本文件聚焦的类
  └─ EngineCoreProc (L806)    ← + ZMQ IPC、IO 线程、busy loop
       └─ DPEngineCoreProc    ← + Data Parallel
```

EngineCore 只负责**逻辑**：初始化、添加请求、执行一步。通信和进程管理交给子类。

---

## 1. `__init__()` (L94-229)

**一句话**：按顺序创建 Executor → Profile 显存 → 初始化 KV Cache → 创建 Scheduler → 冻结 GC。

### 初始化顺序

```
__init__(vllm_config, executor_class, log_stats)
  │
  ├─ 1. load_general_plugins()              ← 插件系统（L103-105）
  │
  ├─ 2. self.model_executor = executor_class(vllm_config)   ← 创建 Executor（L118）
  │
  ├─ 3. kv_cache_config = _initialize_kv_caches(vllm_config) ← Profile 显存 + 分配 KV Cache（L128）
  │     │
  │     ├─ executor.get_kv_cache_specs()          ← 模型需要什么规格的 KV Cache
  │     ├─ executor.determine_available_memory()   ← Profile GPU 显存剩余
  │     ├─ get_kv_cache_configs()                  ← 计算 block 数量和大小
  │     └─ executor.initialize_from_config()       ← 实际分配 + warmup
  │
  ├─ 4. self.scheduler = Scheduler(...)      ← 创建调度器（L145-153）
  │
  ├─ 5. self.request_block_hasher = ...      ← 前缀缓存的 hash 函数（L202-211）
  │
  ├─ 6. self.step_fn = step 或 step_with_batch_queue  ← 根据 PP 配置选择（L213-214）
  │
  └─ 7. freeze_gc_heap()                    ← 冻结 GC，减少暂停（L224）
```

### 关键代码

```python
# L118: 创建 Executor（管理 GPU Worker）
self.model_executor = executor_class(vllm_config)

# L128: Profile 显存 → 分配 KV Cache → warmup 模型
kv_cache_config = self._initialize_kv_caches(vllm_config)

# L145-153: 创建 Scheduler，注入 KV Cache 配置
self.scheduler: SchedulerInterface = Scheduler(
    vllm_config=vllm_config,
    kv_cache_config=kv_cache_config,
    structured_output_manager=self.structured_output_manager,
    ...
)

# L213-214: 普通模式 vs Pipeline Parallel 模式
self.step_fn = (
    self.step if self.batch_queue is None else self.step_with_batch_queue
)

# L194: Pipeline Parallel 时启用 batch_queue
if self.batch_queue_size > 1:
    self.batch_queue = deque(maxlen=self.batch_queue_size)

# L224: 冻结 GC——启动时分配的大量对象（模型权重、KV Cache）生命周期=进程生命周期
freeze_gc_heap()
```

### `_initialize_kv_caches()` 内部 (L232-310)

```python
def _initialize_kv_caches(self, vllm_config):
    # 1. 模型需要什么 KV Cache 规格
    kv_cache_specs = self.model_executor.get_kv_cache_specs()

    # 2. Profile GPU 显存，确定可用空间
    available_gpu_memory = self.model_executor.determine_available_memory()

    # 3. 计算 KV Cache 配置（block 数量、大小）
    kv_cache_configs = get_kv_cache_configs(vllm_config, kv_cache_specs, available_gpu_memory)

    # 4. 实际分配 KV Cache + warmup 模型
    self.model_executor.initialize_from_config(kv_cache_configs)
```

**面试高频**："vLLM 启动时做了什么？" → Profile 显存 → 计算 block 数量 → 分配 KV Cache → warmup。

### __init__ 后 EngineCore 持有的对象

```
EngineCore
├── model_executor: Executor         ← GPU 计算
├── scheduler: SchedulerInterface    ← 调度决策
│   └── kv_cache_manager             ← KV Cache 分配（Scheduler 的子组件）
├── request_block_hasher: Callable   ← 前缀缓存 hash
├── step_fn: Callable                ← step 或 step_with_batch_queue
├── batch_queue: deque | None        ← PP 模式的批次队列
└── aborts_queue: queue.Queue        ← 中途 abort 的请求
```

### `__init__()` 总结

```
代码位置
  vllm/v1/engine/core.py L94-229
  _initialize_kv_caches: L232-310
    ↓
做了什么
  按顺序创建 Executor → Profile 显存 → 初始化 KV Cache → 创建 Scheduler → 冻结 GC
  _initialize_kv_caches 内部：get_kv_cache_specs → determine_available_memory → get_kv_cache_configs → initialize_from_config
    ↓
持有哪些对象
  model_executor: Executor              ← GPU 计算（拥有 Worker 进程）
  scheduler: SchedulerInterface         ← 调度决策（内部持有 KVCacheManager）
  request_block_hasher: Callable | None ← 前缀缓存 hash 函数
  step_fn: Callable                     ← step() 或 step_with_batch_queue()
  batch_queue: deque | None             ← PP 模式的批次队列（maxlen=batch_queue_size）
  aborts_queue: queue.Queue             ← 中途 abort 的请求 ID 列表
  structured_output_manager             ← 结构化输出（JSON Schema 等）
    ↓
调用了谁
  load_general_plugins()                        ← 插件系统
  executor_class(vllm_config)                   ← 创建 Executor
  _initialize_kv_caches(vllm_config)            ← Profile + 分配 KV Cache
    ├─ executor.get_kv_cache_specs()            ← 模型 KV Cache 规格
    ├─ executor.determine_available_memory()    ← GPU 显存 Profile
    ├─ get_kv_cache_configs()                   ← 计算 block 数量
    └─ executor.initialize_from_config()        ← 实际分配 + warmup
  Scheduler(vllm_config, kv_cache_config, ...)  ← 创建调度器
  resolve_kv_cache_block_sizes()                ← 确定 block 大小
  freeze_gc_heap()                              ← 冻结 GC
    ↓
改变了什么状态
  EngineCore 从"未初始化"变为"就绪"
  GPU 显存已分配（模型权重 + KV Cache blocks）
  Scheduler 的 RequestQueue 已就绪，可以接收请求
  GC 堆被冻结，减少推理时的 GC 暂停
    ↓
为下一步传了什么数据结构
  无直接传递。__init__ 建立了 add_request() 和 step() 所需的全部基础设施：
    - scheduler 接收 Request
    - step_fn 驱动 schedule → execute → update 循环
```

---

## 2. `add_request()` (L315-346)

**一句话**：校验请求参数，然后转发给 Scheduler。

### 代码

```python
def add_request(self, request: Request, request_wave: int = 0):
    # 1. 校验 request_id 类型
    if not isinstance(request.request_id, str):
        raise TypeError(...)

    # 2. 校验 pooling 任务是否支持
    if pooling_params := request.pooling_params:
        supported_pooling_tasks = [...]
        if pooling_params.task not in supported_pooling_tasks:
            raise ValueError(...)

    # 3. 警告：有 kv_transfer_params 但没有 KVConnector
    if request.kv_transfer_params is not None and not self.scheduler.get_kv_connector():
        logger.warning(...)

    # 4. 核心：转交给 Scheduler
    self.scheduler.add_request(request)
```

### 调用链

```
Input Thread (ZMQ recv)
  → _handle_client_request()
    → EngineCore.add_request(request)
      → Scheduler.add_request(request)
        → RequestQueue.add(request)   ← FCFS 或 Priority 队列
```

**关键理解**：EngineCore 不做任何调度决策，它只是个"门卫"——校验参数后直接转交给 Scheduler。

### `request_wave` 参数

```python
def add_request(self, request: Request, request_wave: int = 0):
```

- 普通模式：`request_wave = 0`，忽略
- Data Parallel 模式：标识请求属于哪个"波次"，用于 DP 同步

### `add_request()` 总结

```
代码位置
  vllm/v1/engine/core.py L315-346
    ↓
做了什么
  校验请求参数（request_id 类型、pooling 任务支持、KVConnector 一致性）
  然后直接转发给 Scheduler
    ↓
持有哪些对象
  不持有新对象。读取 self.scheduler（由 __init__ 创建）
    ↓
调用了谁
  request.request_id 类型检查                    ← isinstance(request_id, str)
  request.pooling_params 任务校验                ← 检查 supported_pooling_tasks
  self.scheduler.get_kv_connector()              ← 检查 KVConnector 存在性
  self.scheduler.add_request(request)            ← 核心：转交 Scheduler
    ↓
改变了什么状态
  Scheduler 的 RequestQueue 中新增一个 WAITING 状态的 Request
  Request 对象由 InputProcessor 创建，此处不做修改
    ↓
为下一步传了什么数据结构
  request: Request                          ← 完整的请求对象（含 prompt_token_ids、
                                               sampling_params、arrival_time 等）
                                               传入 Scheduler.add_request() → RequestQueue
                                               下一步由 step() 中 scheduler.schedule() 取出
```

---

## 3. `step()` (L402-431)

**一句话**：一次完整的推理循环——schedule → execute → update。

### 代码

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    # 1. 没有请求就直接返回
    if not self.scheduler.has_requests():
        return {}, False

    # 2. 调度：Scheduler 选择哪些请求参与本轮，分配 KV Cache block
    scheduler_output = self.scheduler.schedule()

    # 3. 异步执行：把 scheduler_output 交给 GPU（non_block=True）
    future = self.model_executor.execute_model(scheduler_output, non_block=True)

    # 4. GPU 计算的同时，准备结构化输出的 grammar bitmask
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)

    # 5. 等待 GPU 完成
    with self.log_error_detail(scheduler_output):
        model_output = future.result()
        if model_output is None:
            # 需要 grammar bitmask 才能采样（JSON Schema 等约束）
            model_output = self.model_executor.sample_tokens(grammar_output)

    # 6. 处理中途 abort 的请求
    self._process_aborts_queue()

    # 7. 从输出更新 Scheduler 状态（标记完成的请求、释放 block 等）
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )

    # 返回：本轮的输出 + 是否实际执行了模型
    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

### 数据流图

```
step() 一轮的数据流：

Scheduler                    Executor                     Scheduler
   │                            │                            │
   │  schedule()                │                            │
   │ ──→ SchedulerOutput ──────→│                            │
   │     (哪些请求、多少 token、  │  execute_model()           │
   │      block_tables 等)       │ ──→ GPU 计算 ──→ ModelRunnerOutput
   │                            │                            │
   │  get_grammar_bitmask()     │                            │
   │ ──→ grammar bits           │                            │
   │                            │  sample_tokens(grammar)    │
   │                            │ ──→ sampled_token_ids      │
   │                            │                            │
   │                            │                            │  update_from_output()
   │                            │                            │ ──→ EngineCoreOutputs
   │                            │                            │     (新 token、完成状态)
```

### 三个关键调用

| 调用 | 位置 | 做什么 |
|------|------|--------|
| `scheduler.schedule()` | L413 | 决定本轮处理哪些请求、每个请求多少 token、分配哪些 block |
| `executor.execute_model()` | L414 | 把 SchedulerOutput 送给 GPU，执行模型前向推理 |
| `scheduler.update_from_output()` | L427-429 | 根据采样结果更新请求状态（新 token、完成原因、释放 block） |

### 为什么 `execute_model(non_block=True)`？

```python
future = self.model_executor.execute_model(scheduler_output, non_block=True)
# GPU 开始计算...
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)  # CPU 同时准备 grammar
model_output = future.result()  # 等 GPU 完成
```

**GPU 和 CPU 并行**：GPU 在做矩阵乘法的同时，CPU 在准备结构化输出的 grammar bitmask。如果 `non_block=False`，CPU 会白白等 GPU。

### `model_output is None` 的情况

```python
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
```

**什么时候 `None`？** 当请求有结构化输出约束（JSON Schema、正则）时，`execute_model` 返回 None——它需要先拿到 grammar bitmask 才能采样（确保生成的 token 符合 grammar 约束）。

### `step()` 总结

```
代码位置
  vllm/v1/engine/core.py L402-431
    ↓
做了什么
  一次完整的推理循环：
    scheduler.schedule()           → 选请求、分 block、生成 SchedulerOutput
    executor.execute_model()       → GPU 异步执行模型前向推理
    scheduler.get_grammar_bitmask()→ CPU 同时准备结构化输出 grammar
    future.result()                → 等 GPU 完成，拿到 ModelRunnerOutput
    scheduler.update_from_output() → 更新请求状态、输出新 token
    ↓
持有哪些对象
  读取 self.scheduler                  ← 调度决策
  读取 self.model_executor             ← GPU 执行
  持有 self.aborts_queue               ← 中途 abort 的请求
    ↓
调用了谁
  self.scheduler.has_requests()                    ← 检查是否有请求
  self.scheduler.schedule()                        ← 生成 SchedulerOutput
  self.model_executor.execute_model(sched, True)   ← GPU 异步推理，返回 Future
  self.scheduler.get_grammar_bitmask(sched)        ← 准备 grammar bitmask
  future.result()                                  ← 等待 GPU 完成
  self.model_executor.sample_tokens(grammar)       ← 延迟采样（结构化输出时）
  self._process_aborts_queue()                     ← 处理中途 abort
  self.scheduler.update_from_output(sched, output) ← 更新请求状态
    ↓
改变了什么状态
  Scheduler 内部：
    - WAITING 请求变为 RUNNING（被 schedule 选中）
    - RUNNING 请求收到新 token（num_computed_tokens 增加）
    - 完成的请求变为 FINISHED_*，释放 KV Cache block
    - 被抢占的请求变为 PREEMPTED
  EngineCore 内部：
    - aborts_queue 被清空
    ↓
为下一步传了什么数据结构
  scheduler.schedule() 输出：
    scheduler_output: SchedulerOutput          ← 包含 scheduled_new_reqs、
                                                  scheduled_resumed_reqs、
                                                  num_scheduled_tokens、
                                                  block_tables 等
                                                  → 传给 executor.execute_model()

  executor.execute_model() 输出：
    model_output: ModelRunnerOutput            ← 包含 sampled_token_ids（GPU tensor）、
                                                  logprobs、prompt_logprobs 等
                                                  → 传给 scheduler.update_from_output()

  scheduler.update_from_output() 输出：
    engine_core_outputs: dict[int, EngineCoreOutputs]
                                                 ← 每个 client_index 对应一组输出
                                                  EngineCoreOutputs 包含：
                                                    outputs: list[EngineCoreOutput]
                                                      → new_token_ids, finish_reason, events
                                                    finished_requests: set[str]
                                                    scheduler_stats
                                                  → 传给 Output Thread → ZMQ → Frontend
```

---

---

## 4. 三函数协作全景

```
__init__()
  创建 Executor + Scheduler + KV Cache
  ↓
  系统就绪，等待请求
  ↓
add_request() × N
  校验 → Scheduler.add_request() → RequestQueue
  ↓
  请求进入等待队列
  ↓
step() 循环（每步一次）
  ├─ scheduler.schedule()      → 选请求、分 block
  ├─ executor.execute_model()  → GPU 推理
  └─ scheduler.update_from_output()  → 更新状态、输出 token
  ↓
  请求完成 → EngineCoreOutputs 返回给 Frontend
```

---

## 5. 面试问答

### Q1: `__init__` 时 KV Cache 是怎么初始化的？

```
1. executor.get_kv_cache_specs()
   → 模型告诉系统：我需要什么规格的 KV Cache（层数、head 数、head_dim）

2. executor.determine_available_memory()
   → 实际加载模型到 GPU，Profile 峰值显存，算出剩余空间

3. get_kv_cache_configs()
   → 剩余空间 ÷ 单个 block 大小 = 可分配的 block 数量

4. executor.initialize_from_config()
   → 实际分配 GPU 内存 + warmup（跑一次空推理，触发 CUDA Graph capture）
```

### Q2: `step()` 中 Scheduler 和 Executor 的分工？

```
Scheduler（CPU）：
  - 决定哪些请求进 batch
  - 决定每个请求处理多少 token（chunked prefill）
  - 分配/释放 KV Cache block
  - 处理抢占

Executor（GPU）：
  - 把 SchedulerOutput 写入 GPU buffer
  - 执行模型前向推理
  - 采样 token（或等 grammar bitmask 后采样）
  - 返回采样结果

step() 是两者之间的"胶水"——调度→执行→更新，每步循环一次。
```

### Q3: `add_request` 为什么这么简单？

```
因为 EngineCore 是"协调者"，不是"决策者"。

add_request 的职责：
  ✓ 校验参数（request_id 类型、pooling 任务支持）
  ✓ 转发给 Scheduler

add_request 不做的事：
  ✗ 决定何时调度（Scheduler 决定）
  ✗ 分配 KV Cache（Scheduler 委托 KVCacheManager）
  ✗ 启动 GPU 计算（step() 中 Executor 做）
```

---

## 6. 源码索引

| 函数 | 行号 | 行数 | 复杂度 |
|------|------|------|--------|
| `EngineCore.__init__` | L94-229 | ~135 | 中（初始化链长但逻辑线性） |
| `_initialize_kv_caches` | L232-310 | ~78 | 中（Profile + 分配 + warmup） |
| `add_request` | L315-346 | ~31 | 低（校验 + 转发） |
| `step` | L402-431 | ~29 | 低（但理解每步做什么需要上下文） |
