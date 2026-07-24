# A2 · AsyncLLM 前端初始化

> **版本**：vLLM 0.24.0，V1 在线服务主路径
>
> **阶段**：M2 → M3，构造中同步穿越 M4–M11，最后到 M12
>
> **进程**：API Server 进程（EngineCore 在子进程）
>
> **主要源码**：`vllm/v1/engine/async_llm.py`、`vllm/transformers_utils/config.py`、`vllm/tracing/`

> **术语规则**：本章第一次引入关键英文词时先给出中文含义；源码标识符保持原名，避免搜索源码时失去对应关系。

---

## 1. 一句话定位

`AsyncLLM` 是 API Server 侧实现 `EngineClient` 协议的异步门面：它持有输入/输出处理器和 EngineCore client，把用户请求转换为核心请求，再把核心输出还原为流式结果。

```text
FastAPI / OpenAIServing / 编程用户
                  │ EngineClient 协议
                  ▼
              AsyncLLM
              ├─ BaseRenderer/tokenizer
              ├─ InputProcessor
              ├─ OutputProcessor
              ├─ StatLoggerManager（M12，可选）
              └─ EngineCoreClient
                     │ IPC / ZMQ runtime channels
                     ▼
                EngineCoreProc
```

它不做 Scheduler 的资源决策、模型 forward、权重加载或 GPU KV tensor 分配。

---

## 2. 工厂入口与实例初始化边界

在线主路径从 A1 的 M2 进入：

```text
build_async_engine_client_from_engine_args(...)
└─ vllm_config = engine_args.create_engine_config(...)        🚩 M2
   └─ AsyncLLM.from_vllm_config(vllm_config, ...)
      ├─ executor_class = Executor.get_class(vllm_config)     # 只选类
      └─ return AsyncLLM(vllm_config, executor_class, ...)
         └─ AsyncLLM.__init__()                               # 本章主体
```

| 符号 | 性质 | 作用 |
|---|---|---|
| `from_vllm_config()` | classmethod 工厂 | 选 Executor 类、翻译日志参数、调用 `cls(...)` |
| `AsyncLLM(...)` | 类实例化表达式 | 分配对象并触发 `__init__()` |
| `AsyncLLM.__init__()` | 初始化方法 | 创建前端组件，阻塞等待 EngineCore ready，再完成收尾 |
| `Executor.get_class()` | 类选择 | 不创建 Executor/Worker/CUDA |

真正的 `executor_class(vllm_config)` 实例化发生在 EngineCore 子进程，而不是 `from_vllm_config()` 中。

---

## 3. M3 的三个模块：先看目的、源码动作和产物

M3 不是三个彼此平级、完全独立的“对象”。它是 `AsyncLLM.__init__()` 在启动 EngineCore 前必须完成的三组准备工作：

| 模块 | 为什么必须在 EngineCore 前完成 | 源码中的直接动作 | M3 产物 | 运行期谁消费 |
|---|---|---|---|---|
| M3-A 配置与跨进程准备 | 后面要 spawn 子进程；前端组件也要反复读取配置 | 尝试注册序列化规则；保存 config 引用 | 进程级序列化规则、`self.vllm_config` 等引用 | multiprocessing/Ray、renderer、processor、EngineCoreClient |
| M3-B 运行证据策略 | OutputProcessor 和 EngineCoreClient 创建时就要知道是否收集 stats/trace | 初始化可选 tracer；保存 request-log 开关；收集 logger factories；计算 `self.log_stats` | 三条运行期证据链的开关和基础设施 | OutputProcessor、output handler、StatLoggerManager |
| M3-C 请求数据转换 | EngineCore 只消费内部请求，不应理解 HTTP/文本协议 | 创建 renderer、InputProcessor、OutputProcessor | 前端输入/输出转换对象 | 每个在线请求 |

这里的“运行证据”是一个总称：系统运行后，为了知道“发生了什么、性能如何、某个请求走过哪些阶段”而产生的数据。

```text
AsyncLLM.__init__()
│
├─ M3-A 配置与跨进程准备
│  ├─ maybe_register_config_serialize_by_value()
│  ├─ self.vllm_config = vllm_config
│  ├─ self.model_config = vllm_config.model_config
│  └─ self.observability_config = vllm_config.observability_config
│
├─ M3-B 运行证据策略
│  ├─ tracing endpoint → 可选 init_tracer()
│  ├─ self.log_requests
│  ├─ custom StatLoggerFactory 列表
│  └─ self.log_stats
│
├─ M3-C 请求数据转换
│  ├─ BaseRenderer/tokenizer
│  ├─ InputProcessor
│  └─ OutputProcessor
│                                                        🚩 M3
│
├─ EngineCoreClient.make_async_mp_client(...)            🚩 M4–M11
│
├─ StatLoggerManager / output handler / frontend profiler
└─ AsyncLLM.__init__() return                            🚩 M12
```

M3-A 解决“配置怎样被当前前端使用、以后怎样安全跨进程”；M3-B 解决“运行后要不要留下三类证据，以及这些证据送到哪里”；M3-C 才解决“请求怎样进、结果怎样出”。

---

## 4. M3-A：让配置可引用、可跨进程

### 4.1 源码定位：M3-A 实际只有两类动作

`vllm/v1/engine/async_llm.py::AsyncLLM.__init__()` 的相关骨架：

```python
# 1. 注册将来跨进程时使用的序列化规则
maybe_register_config_serialize_by_value()

# 2. 保存完整配置和两个常用子配置的引用
self.vllm_config = vllm_config
self.model_config = vllm_config.model_config
self.observability_config = vllm_config.observability_config
```

这几行不会创建一个名为“配置管理器”的对象。M3-A 的正常产物是：

1. Python 进程中已尝试注册的序列化规则；
2. `AsyncLLM` 上指向 M2 配置对象图的引用。

### 4.2 动作一：为什么要注册“按值序列化”

先定义四个词：

| 词 | 本文含义 |
|---|---|
| 序列化（serialization） | 把 Python 对象转换成可跨进程/网络传递的字节表示 |
| 反序列化（deserialization） | 在接收端从字节恢复对象 |
| 按引用（by reference） | 只记录“模块路径 + 类名”；接收端必须能 import 同一个类 |
| 按值（by value） | 把动态类定义连同实例数据一起带过去；接收端不必能 import 原模块 |
| pickle | Python 标准序列化机制/模块 |
| cloudpickle | 对动态函数、动态类等支持更强的序列化库 |
| reducer | 告诉 pickle“把此对象还原为哪个函数及其参数”的规则 |
| spawn | 创建全新 Python 子进程的一种启动方式 |
| Ray | 可管理远程 actor/进程的分布式执行框架 |

普通 Transformers 内置配置类可以按引用恢复：

```text
父进程对象：transformers.models.llama.LlamaConfig(...)
→ 记录模块路径和类名
→ 子进程 import transformers.models.llama.LlamaConfig
→ 恢复实例
```

`--trust-remote-code` 可能从模型仓库动态加载自定义 config 类，例如它的模块位于运行时生成的 `transformers_modules...` 中。父进程能 import，不保证 spawn 子进程或另一节点也有同一动态模块：

```text
父进程：存在动态类定义
→ 若只按引用记录模块路径
→ 子进程无法 import
→ VllmConfig 反序列化失败
```

因此 `vllm/transformers_utils/config.py` 中的函数做两层注册，等价骨架为：

```python
def _reduce_config(config: VllmConfig):
    payload = cloudpickle.dumps(config)
    return pickle.loads, (payload,)

multiprocessing.reducer.register(VllmConfig, _reduce_config)

if transformers_modules_available:
    cloudpickle.register_pickle_by_value(transformers_modules)
    if ray:
        ray.cloudpickle.register_pickle_by_value(transformers_modules)
```

逐行理解：

1. `multiprocessing.reducer.register(...)` 告诉 multiprocessing：以后遇到 `VllmConfig`，不要只依赖默认 pickle；
2. `cloudpickle.dumps(config)` 可以处理更多动态 Python 对象；
3. `register_pickle_by_value(transformers_modules)` 要求动态模块中的类定义按值携带；
4. Ray 自带一份 cloudpickle，因此 Ray 路径也要单独注册。

> 这里的“注册”类似先安装一条打包规则。它既没有立即调用 `cloudpickle.dumps(vllm_config)`，也没有在这里启动子进程。

这个函数属于防御性注册：加载 `trust_remote_code` config 时可能已经调用过，`AsyncLLM.__init__()` 再调用一次，确保进入进程启动边界前规则存在。函数内部会捕获注册异常并记录 warning，不一定在 M3 当场抛错；如果确实依赖动态类，失败可能延迟到 M4 的进程参数序列化/反序列化阶段暴露。

### 4.3 动作一何时真正生效：在后续 spawn，而不是此刻走 ZMQ

把主语写完整后，时序如下：

```text
调用 maybe_register_config_serialize_by_value()
│
├─ 当前动作：把规则登记到 multiprocessing/cloudpickle
└─ 当前没有：发送 VllmConfig、创建 socket、spawn 进程

随后 EngineCoreClient.make_async_mp_client()
└─ launch_core_engines()
   └─ multiprocessing.Process(..., kwargs={"vllm_config": vllm_config})
      └─ multiprocessing 需要传递 Process 参数
         └─ 此时才触发已注册的 reducer/cloudpickle 规则
            └─ EngineCore 子进程得到独立的 VllmConfig 副本
```

ZMQ（ZeroMQ）在这里是 vLLM 的运行期 IPC 通道之一。IPC 是 inter-process communication，即进程间通信。

```text
进程启动参数通道：
VllmConfig → multiprocessing/Ray serialization → 子进程

运行期 ZMQ 通道：
EngineCoreRequest / EngineCoreOutputs / control RPC
```

所以 4.3 的结论只是否定一个常见误解：

> `maybe_register_config_serialize_by_value()` 负责“以后怎样序列化”，不负责“现在通过 ZMQ 把 VllmConfig 发出去”。

### 4.4 动作二：保存 config 引用有什么用

引用（reference）可以理解为“另一个变量名指向同一个 Python 对象”，不是复制：

```text
vllm_config.model_config
          ▲
          │ 同一个对象
self.model_config
```

源码保存三层入口：

| 引用 | 为什么保存 | 后续直接消费者 |
|---|---|---|
| `self.vllm_config` | 保留配置聚合根；很多组件需要多个子配置 | renderer、InputProcessor、EngineCoreClient、StatLoggerManager |
| `self.model_config` | 请求处理高频读取模型能力、长度、任务等 | AsyncLLM 请求校验/辅助方法 |
| `self.observability_config` | tracing/metrics 策略集中在该子配置 | M3-B、`is_tracing_enabled()` |

其他子配置不一定保存成 `self.xxx_config`，需要时可以继续从聚合根读取：

```python
self.vllm_config.scheduler_config.stream_interval
self.vllm_config.profiler_config
self.vllm_config.parallel_config
```

因此 `save config refs` 的功能不是重新解析配置，而是：

- 确保 AsyncLLM 生命周期内持有 M2 配置图；
- 给高频使用的子配置建立简短入口；
- 把同一份配置引用注入后续前端组件；
- 把完整配置交给 EngineCoreClient，供下一阶段跨进程传递。

### 4.5 M3-A 的完整输入、产物和消费链

```text
输入：
M2 返回的 VllmConfig
  + Python 中可能存在的 transformers_modules 动态类

M3-A 动作：
尝试注册 reducer / by-value policy
  + 保存 full/model/observability config refs

M3-A 产物：
① 序列化规则（进程级注册状态）
② AsyncLLM 上的配置引用

消费者：
① M4 的 multiprocessing/Ray spawn
② M3-B tracing/stats 策略
③ M3-C renderer/processors
④ M4–M12 EngineCoreClient、StatLoggerManager、profiler
```

所有主要配置从 CLI 到最终消费者的索引集中在 [AX1-配置传递与最终消费链路.md](AX1-配置传递与最终消费链路.md)。

---

## 5. M3-B：决定运行期留下哪些证据

### 5.1 M3-B 的主要作用不是“打印日志”

M3-B 是 **运行证据策略初始化**。它在请求到来前确定三条互相独立的数据链：

| 证据链 | 回答的问题 | 数据粒度 | 典型去向 |
|---|---|---|---|
| tracing | 某一个请求经过了哪些阶段，各阶段用了多久？ | 单请求、跨阶段 | OpenTelemetry Collector / trace backend |
| stats/metrics | 整个 Engine 最近吞吐、队列、cache、时延分布如何？ | 聚合统计 | Prometheus、周期日志、自定义 logger |
| request log | 哪个请求被加入、取消或失败？ | 单个离散事件的文本 | Python logging/stdout |

所以 M3-B 不能简单概括为“日志初始化”：

- tracing 是带父子关系的时间区间；
- metrics 是可以聚合、计数和画图的数值；
- request log 是便于人阅读的事件文本。

### 5.2 第一次出现的观测术语

| 英文 | 中文理解 | 例子 |
|---|---|---|
| observability | 可观测性：从外部数据理解系统内部状态的能力 | traces、metrics、logs |
| telemetry | 遥测数据：程序运行时自动采集并送出的观测数据总称 | 一次请求的 span、吞吐计数 |
| trace | 一条追踪链：描述一个请求从开始到结束的完整路径 | HTTP 接收 → 排队 → prefill → decode |
| span | trace 中一个有开始/结束时间的工作区间 | “该请求排队 8 ms” |
| span attribute | 附着在 span 上的键值信息 | request ID、prompt tokens、TTFT |
| trace context | 用于把上下游 span 串成同一条 trace 的身份信息 | trace ID、parent span ID |
| endpoint | 数据要发送到的目标地址 | `http://otel-collector:4317` |
| exporter | 把已完成的 telemetry 编码并发送出去的组件 | OTLP span exporter |
| provider | 创建 tracer 并管理处理管线的 SDK 根对象 | `TracerProvider` |
| processor | 位于 span 与 exporter 之间，负责批处理等工作 | `BatchSpanProcessor` |
| collector | 接收、加工、转发 telemetry 的独立服务 | OpenTelemetry Collector |
| backend | 最终查询/展示数据的系统 | Jaeger、Tempo 等 trace backend |

**endpoint** 不是“端点对象”，而是目的地址。它告诉 exporter 把 trace 发到哪里；它不是 vLLM 用来收请求的 FastAPI 地址，也不是 EngineCore 的 ZMQ 地址。

**OTLP** 是 OpenTelemetry Protocol，即 OpenTelemetry 传输遥测数据的协议。这里用它发送 trace spans。

### 5.3 M3-B 在 AsyncLLM 源码中的完整位置

下面是 `AsyncLLM.__init__()` 相关源码的等价摘录：

```python
self.observability_config = vllm_config.observability_config
tracing_endpoint = self.observability_config.otlp_traces_endpoint

if tracing_endpoint is not None:
    init_tracer("vllm.llm_engine", tracing_endpoint)

self.log_requests = log_requests

custom_stat_loggers = list(stat_loggers or [])
custom_stat_loggers.extend(load_stat_logger_plugin_factories())
has_custom_loggers = bool(custom_stat_loggers)
self.log_stats = log_stats or has_custom_loggers
```

随后创建 OutputProcessor：

```python
self.output_processor = OutputProcessor(
    renderer.tokenizer,
    log_stats=self.log_stats,
    stream_interval=vllm_config.scheduler_config.stream_interval,
    tracing_enabled=tracing_endpoint is not None,
)
```

这解释了 M3-B 为什么必须早于 M3-C 的 OutputProcessor：

- `log_stats` 决定是否为每个请求创建统计状态；
- `tracing_enabled` 决定请求完成时是否创建 trace span；
- 两个布尔值都必须在 OutputProcessor 构造前确定。

### 5.4 tracing 分支：从 CLI endpoint 到一个请求 span

#### 5.4.1 endpoint 从哪里来

```text
CLI --otlp-traces-endpoint
→ AsyncEngineArgs/EngineArgs.otlp_traces_endpoint
→ create_engine_config()
→ ObservabilityConfig.otlp_traces_endpoint
→ VllmConfig.observability_config
→ AsyncLLM.__init__()
→ tracing_endpoint
```

若该字段为 `None`：

- 不调用 `init_tracer()`；
- `OutputProcessor(tracing_enabled=False)`；
- 不为完成请求执行 `do_tracing()`。

若字段非空，配置校验还会确认 OpenTelemetry 依赖可用。`collect_detailed_traces` 必须配合 endpoint 使用，并可能带来明显开销。

#### 5.4.2 `init_tracer()` 创建了什么

贴近 v0.24.0 实现的概念骨架：

```python
provider = TracerProvider(...)
exporter = get_span_exporter(endpoint)
processor = BatchSpanProcessor(exporter)

provider.add_span_processor(processor)
set_tracer_provider(provider)
tracer = provider.get_tracer("vllm.llm_engine")
```

逐步解释：

```text
TracerProvider
│  负责创建 tracer，并持有处理管线
│
└─ BatchSpanProcessor
   │  span 完成后先进入这里；批量发送以减少每次请求直接网络发送的开销
   │
   └─ OTLP SpanExporter
      │  把 span 编码成 OTLP
      │
      └─ endpoint
         └─ Collector / tracing backend
```

`AsyncLLM.__init__()` 调用 `init_tracer()` 时没有把返回值保存到 `self.tracer`。初始化函数把 provider 注册进 OpenTelemetry 的全局 tracing 基础设施；运行期的 `instrument_manual()` 再从这套基础设施创建/提交 span。

M3 创建的是“管线”，不是请求 trace。此时还没有用户请求，所以没有 request span 可记录。

#### 5.4.3 运行期到底创建什么 span

请求完成时，`OutputProcessor.process_outputs()` 在 tracing 开启时调用 `do_tracing()`。v0.24.0 生成名为 `llm_request` 的 server span，并写入：

- time to first token（TTFT，首 token 延迟）；
- end-to-end latency（E2E，请求端到端延迟）；
- queue time（排队时间）；
- prefill、decode、inference 时间；
- prompt/completion token 数；
- request ID；
- 可选的 top-p、temperature、max tokens、`n`。

```text
HTTP trace headers
→ InputProcessor 写入 EngineCoreRequest.trace_headers
→ EngineCore output 带回 trace_headers
→ OutputProcessor.extract_trace_context(...)
→ instrument_manual(span_name="llm_request", ...)
→ provider → processor → exporter → endpoint
```

这里的 **trace headers** 是 HTTP header 中传播 trace context 的字段，用于让 vLLM 的 `llm_request` span 接到调用方已有 trace 上；它不是普通 request log。

### 5.5 stats/metrics 分支：为什么有 factory、gate、manager 三层

先定义：

| 词 | 含义 |
|---|---|
| stat | statistic，统计数据；例如本轮生成 token 数 |
| metric | 可长期聚合/查询的数值指标；例如吞吐 histogram/counter/gauge |
| logger | 消费统计数据并输出到某个后端的对象 |
| factory | 创建 logger 的类或可调用配方 |
| manager | 按 engine rank 管理多个 logger，并把数据路由给它们 |
| gate | 总开关；决定是否付出采集和维护统计状态的成本 |

#### 5.5.1 M3 先收集 factory

```python
custom_stat_loggers = list(stat_loggers or [])
custom_stat_loggers.extend(load_stat_logger_plugin_factories())
```

来源有两类：

1. 调用方显式传入的 `stat_loggers`；
2. vLLM 插件组加载到的自定义 logger 类。

此时只收集“怎样创建 logger”的配方，不实例化每个 engine logger，因此不需要 `engine_ranks_managed`。

#### 5.5.2 `self.log_stats` 是采集 gate

```python
has_custom_loggers = bool(custom_stat_loggers)
self.log_stats = log_stats or has_custom_loggers
```

其中：

- 参数 `log_stats`：是否启用 vLLM 默认 stats logger；
- `has_custom_loggers`：是否存在自定义消费者；
- `self.log_stats`：只要任一消费者需要数据，就必须启用统计采集。

```text
默认 logger 开启  custom logger 存在  self.log_stats
      False              False             False
      True               False             True
      False              True              True
      True               True              True
```

为什么 OutputProcessor 需要这个 gate？源码中每个 `RequestState` 只有在 `log_stats=True` 时才创建 `RequestStateStats`；output handler 也只在需要时创建 `IterationStats`。关闭 gate 可以避免无消费者时仍维护统计状态。

#### 5.5.3 M12 才实例化 manager

EngineCoreClient READY 后：

```python
self.logger_manager = StatLoggerManager(
    vllm_config=vllm_config,
    engine_idxs=self.engine_core.engine_ranks_managed,
    custom_stat_loggers=custom_stat_loggers,
    enable_default_loggers=log_stats,
    ...)
```

此时才知道当前 client 实际管理哪些 engine ranks，因此 manager 必须晚于 EngineCoreClient。

注意两个开关没有重复：

```text
self.log_stats
└─ 是否收集统计数据（默认或自定义消费者任一存在）

enable_default_loggers=log_stats
└─ 是否创建 vLLM 默认 logger
```

用户关闭默认 logger、但安装了 custom logger 时：

- 统计数据仍会采集；
- 默认 logger 不创建；
- manager 把数据交给 custom logger。

#### 5.5.4 统计数据运行期怎样流动

```text
EngineCoreOutputs
├─ scheduler_stats
└─ 每请求时间戳/输出
        │
        ▼
AsyncLLM output handler
├─ 创建 IterationStats（self.log_stats=True 时）
├─ OutputProcessor 更新 RequestStateStats
└─ logger_manager.record(
     engine_idx,
     scheduler_stats,
     iteration_stats,
     mm_cache_stats)
        │
        ├─ LoggingStatLogger：周期性文本统计
        ├─ PrometheusStatLogger：Counter/Gauge/Histogram
        └─ custom StatLogger：用户后端
```

因此“stats log”和“指标”有重叠但不完全相同：同一批 stats 可以被格式化为周期日志，也可以转成 Prometheus metrics。

### 5.6 request log 分支：最普通，也最独立

```python
self.log_requests = log_requests
```

它只保存一个普通日志开关。运行期在请求加入、取消、校验失败、Engine 死亡等位置执行类似：

```python
if self.log_requests:
    logger.info("Added request %s.", request.request_id)
```

它不创建 `RequestStateStats`，不进入 `StatLoggerManager`，也不产生 span。

```text
self.log_requests
→ Python logger.info(...)
→ stdout / 日志采集系统
```

### 5.7 M3-B 的最终产物

| M3-B 产物 | 现在完成什么 | 以后在哪里生效 |
|---|---|---|
| tracer/provider/exporter 管线（可选） | trace 已有发送能力 | OutputProcessor 完成请求时创建 span |
| `self.log_requests` | 请求事件日志策略确定 | `add_request()`、`abort()`、异常处理 |
| `custom_stat_loggers` | logger 创建配方收集完成 | M12 构造 StatLoggerManager |
| `self.log_stats` | stats 采集总 gate 确定 | OutputProcessor、EngineCoreClient、output handler |

所以 M3-B 的 milestone 含义是：

> 三条运行证据链的策略与前置基础设施已经确定；此时尚未产生用户请求日志、请求 span 或运行期统计样本，`StatLoggerManager` 也要到 M12 才实例化。

---

## 6. M3-C：Renderer、InputProcessor、OutputProcessor

### 6.1 BaseRenderer/tokenizer

```text
self.renderer = renderer_from_config(vllm_config)
```

它统一提供 tokenizer、prompt/template、文本/token 输入规范化、多模态渲染接口以及输出 detokenization 所需能力。默认 Engine 在线路径中，它由 AsyncLLM 创建；M13 的 `OpenAIServingRender` 再复用 `engine_client.renderer`，并非重新创建同级 tokenizer。

显式跳过 tokenizer 初始化等特殊配置下，`renderer.tokenizer` 可为 `None`；不要把“一定有 tokenizer”写成无条件结论。

### 6.2 InputProcessor

```text
用户/Serving 输入
→ Renderer 产出的 EngineInput / PromptType
→ InputProcessor(vllm_config, renderer)
→ EngineCoreRequest
```

初始化期只创建 processor 及其依赖；逐请求校验、token/多模态处理、LoRA/长度/参数检查属于运行期。

### 6.3 OutputProcessor

```text
EngineCoreOutputs
→ OutputProcessor(renderer.tokenizer, ...)
→ 更新 request state / detokenize / stop
→ RequestOutput / Streaming response
```

其构造依赖：

- `renderer.tokenizer`；
- M3-B 已确定的 `self.log_stats`；
- `scheduler_config.stream_interval`；
- tracing endpoint 是否存在。

这就是 M3 内部不能任意交换所有步骤的原因。

### 6.4 M3 内部依赖图

```text
A. register serialization policy
   └─ 必须早于后面的 EngineCore spawn

A. save config refs
   ├──────────────→ B. 运行证据策略
   └──────────────→ C. renderer
                         ├─→ InputProcessor
                         └─→ OutputProcessor
                              ▲       ▲
                              │       └─ scheduler stream_interval
                              └─ B.self.log_stats / tracing_enabled
```

关系总结：

- serialization registration 与 renderer 没有直接数据依赖，但必须发生在 spawn 前；
- tracing/logging policy 与 renderer 大体可独立准备；
- renderer 必须先于两个 processor；
- OutputProcessor 是 B、C 两条分支的汇合点；
- 三个模块全部完成才达到 M3。

> 🚩 **M3 · AsyncLLM 前端基础就绪**
>
> 配置/跨进程准备、可观测性/日志策略、renderer/input/output 数据转换组件均已建立。EngineCoreClient、StatLoggerManager 实例和 output handler 尚未完成。

---

## 7. EngineCoreClient 是 M3 与 M12 之间的同步屏障

```python
self.engine_core = EngineCoreClient.make_async_mp_client(
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=self.log_stats,
    client_addresses=client_addresses,
    client_count=client_count,
    client_index=client_index,
)
```

典型选择：

| 条件 | client 形态 |
|---|---|
| DP=1 | `AsyncMPClient` |
| DP>1、external LB | `DPAsyncMPClient` |
| DP>1、内部/混合 LB | `DPLBAsyncMPClient` |

工厂内部同步完成 M4–M11：ZMQ/socket、spawn、EngineCoreProc、Worker/model、KV、Scheduler、I/O threads、READY handshake。只有 client 收齐 READY 并返回，`AsyncLLM.__init__()` 才继续。

因此模型加载/KV 初始化失败时可能出现：M3 已完成，但 M12 永远到不了；这时 `StatLoggerManager`、output handler 和前端 profiler 尚未创建。

---

## 8. M12：Engine READY 后的前端收尾

严格顺序：

```text
1. 若 self.log_stats：
   StatLoggerManager(engine_idxs=engine_core.engine_ranks_managed, ...)
   └─ logger_manager.log_engine_initialized()
2. self._client_count = client_count
3. self.output_handler = None
4. 若当前线程已有 running asyncio loop：_run_output_handler()
5. 按 ProfilerConfig 创建可选前端 torch CPU profiler
6. AsyncLLM.__init__() 返回                              🚩 M12
```

### 8.1 output handler 的条件性

默认 `vllm serve` 路径在运行中的 uvloop/asyncio loop 内构造，通常在 M12 立即创建 task：

```text
engine_core.get_output_async()
→ OutputProcessor.process_outputs()
→ 更新 scheduler stats / abort stop-string requests
→ StatLoggerManager.record(...)
→ RequestOutputCollector 被唤醒
```

通用编程入口可在没有 running loop 的同步上下文构造。此时初始化捕获 `RuntimeError`，第一次 `add_request()` 再调用 `_run_output_handler()`。因此：

- 在线主路径：M12 通常已启动 handler task；
- 跨入口语义：M12 至少已完成 handler 管理状态和可启动条件。

### 8.2 前端 profiler 不是模型 compilation

当 `profiler_config.profiler == "torch"` 且未忽略 frontend 时，M12 创建 AsyncLLM 侧 CPU profiler。它与 M9 的模型 `torch.compile`、CUDA Graph capture/warmup 是不同机制。

> 🚩 **M12 · AsyncLLM 完整初始化完成**
>
> EngineCoreClient 已收齐 READY；统计管理器（若需要）、engine-initialized 记录、output-handler 管理/在线 task、可选前端 profiler 已完成，`AsyncLLM.__init__()` 返回。

---

## 9. 运行期预览：初始化对象怎样协作

```text
OpenAIServing / API route
→ AsyncLLM.generate()/add_request()
→ InputProcessor.process_inputs()
→ OutputProcessor.add_request()          # 先登记前端状态
→ EngineCoreClient.add_request_async()
→ ZMQ → EngineCore/Scheduler/Worker
→ EngineCoreOutputs → ZMQ
→ output handler
→ OutputProcessor.process_outputs()
→ RequestOutputCollector / async generator
→ HTTP SSE/response
```

`abort()` 同时清前端 request state 并通知 EngineCore；`shutdown()` 逆序清 Prometheus、renderer、engine client/子进程和 output-handler task。这些是运行控制/生命周期，不属于 M3 的轻组件构造。

---

## 10. 所有权与 FastAPI 的关系

默认在线路径：

- `run_server_worker()` 的 Engine async context 持有 `AsyncLLM` 生命周期；
- M13 时 FastAPI `app.state.engine_client` 引用它；
- 多个 `OpenAIServing*` 对象共享同一个 EngineClient/renderer；
- `AsyncLLM` 持有 EngineCoreClient，后者管理 IPC 和 EngineCore 子进程。

这不是“FastAPI 最后才初始化 AsyncLLM”，而是 **Engine 先完成 M12，FastAPI/Serving 再把它注入 app state，最后 uvicorn 到 M14**。

DP、多 API server worker、外部 Engine 管理或 CPU-only render server 会改变实例数量，不能概括为“一台机器永远只有一个 AsyncLLM”。

---

## 11. 失败定位与常见误区

| 现象 | 优先阶段 |
|---|---|
| 动态 Transformers config 在子进程反序列化失败 | M3-A → M4 |
| OTLP endpoint/SDK 初始化错误 | M3-B |
| tokenizer/renderer 创建失败 | M3-C |
| custom logger 存在但没有统计数据 | 检查 M3-B `self.log_stats` 与 M12 manager |
| 长时间卡在 client 构造 | M4–M11，通常不是 M3 轻组件 |
| 收不到 READY | M5–M11 任一 proc 端失败 |
| Engine ready 但 AsyncLLM 未返回 | M12 收尾/output handler/profiler |

常见误区：

- **“按值序列化就是经 ZMQ 发 config”**：错；它先注册 pickle/cloudpickle 策略。
- **“OTLP endpoint 是 ZMQ endpoint”**：错；它是 telemetry collector/backend 地址。
- **“self.log_stats 就是 StatLoggerManager”**：错；前者是数据收集 gate，后者是 M12 的管理器实例。
- **“关闭默认 stats 等于自定义 logger 也停掉”**：错；有 custom factory 时 `self.log_stats` 仍为真。
- **“M3 只有 renderer、InputProcessor、OutputProcessor”**：不完整；还包括配置跨进程准备和观测/日志策略。

---

## 12. 本章源码定位索引

| 主题 | v0.24.0 源码位置 |
|---|---|
| M3-A/M3-B 总顺序 | `vllm/v1/engine/async_llm.py::AsyncLLM.__init__` |
| 自定义 config 按值序列化 | `vllm/transformers_utils/config.py::maybe_register_config_serialize_by_value` |
| endpoint 和 detailed traces 配置 | `vllm/config/observability.py::ObservabilityConfig` |
| provider/processor/exporter 初始化 | `vllm/tracing/` 中的 OTel tracer 初始化 |
| 请求 span 属性与创建时机 | `vllm/v1/engine/output_processor.py::OutputProcessor.do_tracing` |
| stats factory 和 logger 接口 | `vllm/v1/metrics/loggers.py` |
| stats 运行期路由 | `vllm/v1/engine/async_llm.py::AsyncLLM._run_output_handler` |
| request event log | `AsyncLLM._add_request`、`generate`、`abort` |

下一阶段：[A3-EngineCoreClient与ZMQ.md](A3-EngineCoreClient与ZMQ.md)。
