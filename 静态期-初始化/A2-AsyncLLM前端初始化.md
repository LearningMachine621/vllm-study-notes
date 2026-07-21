# A2 · AsyncLLM 前端初始化

> **版本**：vLLM 0.24.0，V1 在线服务主路径
>
> **阶段**：M2 → M3，构造中同步穿越 M4–M11，最后到 M12
>
> **进程**：API Server 进程（EngineCore 在子进程）
> **主要源码**：`vllm/v1/engine/async_llm.py`、`vllm/transformers_utils/config.py`、`vllm/tracing.py`

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

## 3. 唯一主骨架：M3 三模块 → EngineCore 屏障 → M12 收尾

```text
AsyncLLM.__init__()
│
├─ A. 配置与跨进程准备
│   ├─ register serialization-by-value policy
│   └─ save config refs
│
├─ B. 可观测性与日志策略
│   ├─ optional tracing pipeline
│   ├─ request logging switch
│   ├─ custom StatLoggerFactory list
│   └─ self.log_stats collection gate
│
├─ C. 请求数据转换组件
│   ├─ BaseRenderer/tokenizer
│   ├─ InputProcessor
│   └─ OutputProcessor
│                                                        🚩 M3
│
├─ EngineCoreClient.make_async_mp_client(...)
│   └─ IPC → proc → model → KV → Scheduler → READY      🚩 M4–M11
│
├─ StatLoggerManager（若 self.log_stats）
├─ output handler（在线路径通常立即启动）
├─ optional frontend torch profiler
└─ AsyncLLM.__init__() return                            🚩 M12
```

把 M3 概括成三个模块是合理的，但第一类不宜只叫“辅助配置”。更准确的命名是：

1. **配置与跨进程准备**；
2. **可观测性与日志策略**；
3. **请求数据转换组件（renderer + processors）**。

前两类属于控制面准备，第三类属于前端数据面。这样既简洁，也不会把 tracing/stat policy 隐藏在含义过宽的“辅助配置”中。

---

## 4. M3-A：配置与跨进程准备

### 4.1 `maybe_register_config_serialize_by_value()` 做了什么

`AsyncLLM.__init__()` 首先调用：

```python
maybe_register_config_serialize_by_value()
```

它解决 `trust_remote_code` 自定义 Transformers config 的类定义可达性问题。动态类常位于运行时生成的 `transformers_modules...` 模块中：父进程能 import，不代表 spawn 的子进程或另一节点也能用相同模块路径重新 import。

v0.24.0 的核心策略是：

```text
VllmConfig 的 multiprocessing reducer
└─ cloudpickle.dumps(config)

动态 transformers_modules
├─ cloudpickle.register_pickle_by_value(...)
└─ Ray 路径也注册 ray.cloudpickle 按值行为
```

“按值”强调把动态类定义连同实例状态一起序列化；“按引用”通常只记录模块名和类名，接收端必须能重新 import。

### 4.2 它不等于“立即通过 ZMQ 发 VllmConfig”

必须分开三个时刻：

```text
M3：注册序列化规则
     │ 只改变以后 pickle/cloudpickle 怎样处理对象
     ▼
M4：spawn EngineCore process
     │ VllmConfig 作为 Process kwargs 被序列化到子进程
     ▼
运行期：ZMQ request/output/control channels
     │ 传 EngineCoreRequest、EngineCoreOutputs、控制 RPC 等
```

因此：

- 该函数调用本身不发送数据；
- 默认本地 MP 路径的初始 VllmConfig 主要随 `multiprocessing` 进程参数传递；
- Ray 路径使用 Ray/cloudpickle 机制；
- 运行期 ZMQ 是另一条 IPC 数据面，不能说“这里把 VllmConfig 打包后经 ZMQ 发走”。

### 4.3 配置引用

随后保存：

```text
self.vllm_config
self.model_config = vllm_config.model_config
self.observability_config = vllm_config.observability_config
```

它们主要是 M2 对象图的引用，不是重新构造的独立 config。M3-B/C 都依赖这些引用：

```text
config refs
├─ observability_config → tracing endpoint / detailed trace policy
├─ scheduler_config → stream_interval
├─ model/config fields → renderer / processors
└─ full vllm_config → EngineCoreClient
```

---

## 5. M3-B：OTLP tracing、请求日志与统计策略

### 5.1 OTLP、endpoint、provider、processor、exporter

**OTLP** 是 OpenTelemetry Protocol：用于把 telemetry 数据传到 OpenTelemetry Collector 或兼容后端的协议。这里关注的是 trace spans。

endpoint 的来源链是：

```text
CLI --otlp-traces-endpoint
→ AsyncEngineArgs / EngineArgs
→ ObservabilityConfig.otlp_traces_endpoint
→ VllmConfig.observability_config
→ AsyncLLM.__init__()
```

当 endpoint 非空时：

```python
init_tracer("vllm.llm_engine", tracing_endpoint)
```

概念管线：

```text
业务代码创建/结束 Span
          │
          ▼
Tracer（由 Provider 提供）
          │
          ▼
TracerProvider
  └─ SpanProcessor（通常批量缓冲/调度导出）
       └─ OTLP SpanExporter
            └─ endpoint → Collector / tracing backend
```

| 概念 | 职责 |
|---|---|
| `Tracer` | 业务代码用它创建 span |
| `TracerProvider` | tracer 的 SDK 根对象，管理 processor/resource/sampling 等 |
| `SpanProcessor` | 接收 span 生命周期事件，通常批量化后交给 exporter |
| `OTLPSpanExporter` | 把已完成 span 编码为 OTLP 并发送到 endpoint |
| endpoint | Collector/后端的目标地址，不是 vLLM 内部 ZMQ 地址 |

所以 provider 和 exporter 不重复：provider 管理“如何产生/处理 trace”，exporter 负责“怎样送出完成的 span”。

### 5.2 为什么它属于 tracing，主要 trace 什么

M3 只装好管线，不是在启动期凭空生成请求 trace。运行期由请求携带/传播的 trace context 和 OutputProcessor 的 request state 形成 spans/attributes，典型关注：

- 请求端到端延迟；
- time to first token；
- queue、prefill、decode/inference 等阶段耗时；
- prompt/completion token 数；
- request ID、模型、采样参数等请求属性；
- 配置开启的 model/worker detailed traces（更细、更有开销）。

`collect_detailed_traces` 与 endpoint 相关：没有 trace exporter 目标却请求 detailed trace 没有完整意义；详细追踪也可能增加性能开销。普通 Prometheus 指标不是 trace span，两者都属于 observability，但数据模型不同。

### 5.3 三种容易混淆的“log”

| 名称 | 初始化字段/对象 | 运行期内容 |
|---|---|---|
| 普通请求日志 | `self.log_requests` | add/finish/abort/error 等文本事件 |
| stats/metrics | `self.log_stats` + `StatLoggerManager` | 吞吐、时延、队列、KV/cache 等聚合统计/指标 |
| tracing | tracer/provider/exporter | 单个请求的跨阶段 span 与上下文 |

`self.log_requests` 和 `self.log_stats` 是两条独立控制链；请求文本日志不会流进 `StatLoggerManager`。

### 5.4 为什么 factory、`self.log_stats`、manager 要分三步

源码顺序可抽象为：

```python
custom_stat_loggers = list(stat_loggers or [])
custom_stat_loggers.extend(load_stat_logger_plugin_factories())
has_custom_loggers = bool(custom_stat_loggers)

self.log_stats = log_stats or has_custom_loggers
```

EngineCore READY 后才做：

```python
StatLoggerManager(
    engine_idxs=self.engine_core.engine_ranks_managed,
    custom_stat_loggers=custom_stat_loggers,
    enable_default_loggers=log_stats,
    ...)
```

三者并不重复：

| 对象/值 | M3 时是什么 | 为什么此时可确定 |
|---|---|---|
| `custom_stat_loggers` | factory 列表/创建配方 | 插件和用户参数已知，不需要 engine ranks |
| `self.log_stats` | 是否收集统计的总开关 | 默认 logger 或任一 custom logger 需要数据即可为真 |
| `StatLoggerManager` | 实际 logger 管理/路由实例 | 需要 M11 后的 `engine_ranks_managed`，所以到 M12 才创建 |

特别重要的布尔关系：

```text
self.log_stats = default_stats_enabled OR custom_logger_exists
enable_default_loggers = default_stats_enabled
```

所以即使用户关闭默认 stats 输出，只要发现自定义 logger：

- `self.log_stats` 仍为真，OutputProcessor/output handler 继续收集统计；
- `enable_default_loggers` 仍为假，不会偷偷重新开启默认 logger；
- M12 的 manager 只把数据送到自定义后端。

`self.log_stats` 不是 logger 实例，也不是“是否打印一行日志”的开关，而是前端统计数据路径的总 gate。它必须在创建 `OutputProcessor` 和 EngineCoreClient 前确定。

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
   ├──────────────→ B. tracing / logging policy
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

下一阶段：[A3-EngineCoreClient与ZMQ.md](A3-EngineCoreClient与ZMQ.md)。
