# A2 · AsyncLLM 前端初始化

> **阶段**：M2 → M3，随后同步进入 M4–M12  
> **进程**：API Server 进程  
> **主要源码**：`vllm/v1/engine/async_llm.py`

---

## 1. 定位

`AsyncLLM` 是前端异步门面，实现对外的 `EngineClient` 协议。它负责：

- 将用户输入转换为 EngineCore 请求；
- 将 EngineCore 输出组装为用户可见的流式结果；
- 管理每个前端请求的状态和输出队列；
- 通过 `EngineCoreClient` 向 proc 端发送请求、接收输出；
- tracing、统计、abort 和 shutdown 的前端编排。

它不负责：

- Scheduler 的资源决策；
- 模型 forward；
- GPU KV tensor；
- 模型权重加载。

```text
OpenAIServing / 编程用户
            │ EngineClient 协议
            ▼
AsyncLLM
├── InputProcessor
├── OutputProcessor
├── EngineCoreClient
└── StatLoggerManager
            │ ZMQ
            ▼
EngineCoreProc
```

---

## 2. 构造入口

v0.24.0 在线主路径使用：

```text
build_async_engine_client_from_engine_args()
├── engine_args.create_engine_config()       # A1, M2
└── AsyncLLM.from_vllm_config(vllm_config)
    ├── Executor.get_class(vllm_config)
    └── AsyncLLM(vllm_config, executor_class, ...)
```

`from_vllm_config()` 的两个关键动作：

1. 根据 `parallel_config.distributed_executor_backend` 等确定 **Executor 类**；
2. 把完整 `VllmConfig` 和 Executor 类传进 `AsyncLLM.__init__()`。

这里得到的只是类选择，Executor 实例要到 EngineCore 子进程中的 A5/A6 才创建。

必须区分三个符号：

| 符号 | 性质 | 做什么 |
|---|---|---|
| `AsyncLLM.from_vllm_config(...)` | `@classmethod` 工厂入口 | 选择 Executor 类并调用 `cls(...)` |
| `AsyncLLM(...)` | 类实例化表达式 | Python 分配实例并触发 `__init__()` |
| `AsyncLLM.__init__(...)` | 实例初始化方法 | 建立 renderer、processor、EngineCoreClient、logger、output handler |

因此说“调用 `from_vllm_config()` 拉起 AsyncLLM”在入口层面没错；下钻源码时应写成：

```text
from_vllm_config()
→ executor_class = Executor.get_class(vllm_config)   # 只选类
→ return cls(vllm_config, executor_class, ...)
  → AsyncLLM.__init__()                              # 实例初始化
```

`Executor.get_class()` 不创建 Worker、CUDA context 或模型。真正的：

```python
model_executor = executor_class(vllm_config)
```

发生在 EngineCore 子进程。

---

## 3. AsyncLLM.__init__ 的精确顺序

主干顺序可概括为：

```text
保存 VllmConfig 引用
→ 初始化 tracing
→ 配置请求日志
→ renderer_from_config()
→ InputProcessor(...)
→ OutputProcessor(...)
                                      🚩 M3
→ EngineCoreClient.make_async_mp_client(...)
                                      # 同步进入 A3–A9
  → spawn + 模型加载 + KV + scheduler + READY
                                      🚩 M12 后返回
→ StatLoggerManager(...)
→ 准备/启动 output handler
→ AsyncLLM.__init__ 返回
```

因此 M3 只是“前端轻组件就绪”，不是 AsyncLLM 构造完成。最重的子进程启动嵌在 `EngineCoreClient.make_async_mp_client()` 里面。

---

## 4. 配置引用

`AsyncLLM` 保存整个 `VllmConfig`，并抽取常用引用，例如：

- `model_config`；
- `observability_config`；
- scheduler 的 stream/output 相关配置；
- renderer/tokenizer 所需配置。

这些不是重新构造的独立 config，而是 M2 对象图中的引用。

---

## 5. Renderer

`renderer_from_config(vllm_config)` 统一处理输入渲染相关能力：

- tokenizer（正常文本路径；显式跳过 tokenizer 初始化等场景可为 `None`）；
- prompt/template；
- text/token 输入规范化；
- 多模态输入渲染接口；
- 后续 detokenization 需要的 tokenizer 能力。

它属于前端，因为用户协议中的输入输出表现不应由 GPU EngineCore 负责。

---

## 6. InputProcessor

它处在两种请求模型之间：

```text
用户/Serving 输入
→ EngineInput / TokensPrompt / multimodal input
→ InputProcessor
→ EngineCoreRequest
```

启动时创建 processor 及其依赖；真正针对某个请求的处理属于运行期。

典型职责：

- 校验 prompt/token IDs；
- tokenization 或接收预分词 token；
- 合并多模态特征和占位信息；
- 构建采样/池化相关参数；
- 生成 EngineCore 可序列化的请求对象；
- 根据模型长度、缓存、LoRA 等配置做前端校验。

它必须在 API Server 进程，因为 tokenizer 和 HTTP 输入都在前端，且不应让 EngineCore 处理用户协议。

---

## 7. OutputProcessor

数据方向与 InputProcessor 相反：

```text
EngineCoreOutputs
→ OutputProcessor
→ 更新前端请求状态
→ RequestOutput / Streaming response
```

初始化时主要保存：

- tokenizer/renderer；
- request state 容器；
- detokenization 和 stop 条件所需结构；
- scheduler stream interval 等输出配置；
- completion callback / output queue 协作信息。

真正消费 token、累积文本、判断 finished 属于运行期。

---

## 8. API Server 进程中的 observability、日志和控制协议

这里容易把五件事混成“观测能力”，实际应分开：

| 能力 | 目的 | 类型 |
|---|---|---|
| tracing | 追踪一个请求跨层耗时和上下文 | observability |
| runtime stats | 聚合 Scheduler/iteration/cache 性能数据 | observability |
| request logging | 记录请求 ID、参数、输入或状态事件 | 普通日志 |
| abort | 同步终止前端与 EngineCore 中的请求 | 运行时控制 |
| shutdown | 逆序释放前端、IPC、子进程资源 | 生命周期控制 |

### 8.1 tracing 初始化的意义

当：

```python
observability_config.otlp_traces_endpoint is not None
```

时，AsyncLLM 调用：

```python
init_tracer("vllm.llm_engine", tracing_endpoint)
```

其目的不是“启动时生成一条请求 trace”，而是预先建立 OpenTelemetry 基础设施：

- `TracerProvider`；
- OTLP span exporter；
- batch span processor；
- 模块 tracer；
- 后续从请求 trace headers 继续父子 span 的能力。

随后 `OutputProcessor` 收到：

```python
tracing_enabled=True
```

在请求完成时可结合 request state、EngineCore output 和 iteration stats 记录：

- time to first token；
- end-to-end latency；
- queue/prefill/decode/inference time；
- prompt/completion token 数；
- request ID 和采样参数等 span attributes。

所以初始化 tracer 是“先装好追踪管线”，真正的 span/attribute 产生在运行期。

### 8.2 请求日志

`enable_log_requests` 最终控制：

```python
self.log_requests
```

它用于普通事件日志，例如：

- request ID；
- SamplingParams/LoRA；
- DEBUG 级别下的 prompt/token IDs；
- 请求完成、abort、bad request、engine dead。

它关注“这个请求发生了什么”，不是聚合性能指标。

### 8.3 StatLoggerManager

统计开关与插件逻辑：

```text
disable_log_stats / log_stats
→ 加载自定义 StatLoggerFactory
→ 有自定义 logger 时，即使关闭默认 logger 也要保留统计
```

`StatLoggerManager` 在 EngineCoreClient READY 后创建，原因包括：

- 需要 `engine_core.engine_ranks_managed`；
- DP 下可能一个 AsyncLLM 对应多个 EngineCore；
- proc 已返回真实配置和 Engine identity；
- manager 要为各 EngineCore 建立/聚合 logger。

运行期 output handler 从 EngineCoreOutputs 获得 SchedulerStats、IterationStats、多模态 cache stats 等，交给 OutputProcessor 和 StatLoggerManager 记录/输出。

### 8.4 请求日志与 StatLoggerManager 的关系

它们是两条独立链：

```text
enable_log_requests
→ self.log_requests
→ 每请求事件文本日志

log_stats / custom stat loggers
→ StatLoggerManager
→ 吞吐、时延、队列、cache 等统计/指标
```

两者都使用配置和 logging 基础设施，但 request logging 不会“流入” StatLoggerManager。

### 8.5 abort

`AsyncLLM.abort()` 同时清理两侧：

```text
request IDs
→ OutputProcessor.abort_requests()
  └─ 清前端 request state/collector/parent-child 映射
→ EngineCoreClient.abort_requests_async()
  └─ 经 IPC 让 Scheduler 结束请求并释放逻辑 KV blocks
```

常见触发：

- HTTP client 断开导致 generate coroutine 取消；
- 请求处理异常；
- 前端 stop string 已满足，但 EngineCore 尚未标 finished；
- 用户显式 abort。

### 8.6 shutdown

v0.24.0 `AsyncLLM.shutdown()` 的前端逆序清理包括：

```text
shutdown_prometheus()
→ renderer.shutdown()
→ engine_core.shutdown(timeout)
→ cancel output_handler task
```

EngineCoreClient 再负责 IPC、monitor、EngineCore/Worker 进程的清理。在线服务的 async context manager 在退出时调用它。

“logger 在 M3 就绪”并不精确；M3 只覆盖 tracer 基础、renderer/input/output 等轻组件，StatLoggerManager 和完整 AsyncLLM 要等 M12。

---

## 9. 创建 EngineCoreClient

v0.24.0 中的正确调用关系是 `EngineCoreClient` 类方法：

```python
self.engine_core = EngineCoreClient.make_async_mp_client(
    vllm_config=vllm_config,
    executor_class=executor_class,
    ...
)
```

该方法根据并行配置选择并构造具体 client；不要把其他版本或设计草稿中的模块级工厂写法套到 v0.24.0。

工厂根据 DP/LB 配置返回：

| 条件 | 典型实现 |
|---|---|
| DP=1 | `AsyncMPClient` |
| DP>1，external LB | `DPAsyncMPClient` |
| DP>1，内部/混合 LB | `DPLBAsyncMPClient` |

这个调用同步完成 M4–M12：建 socket、spawn、等待模型/KV/Scheduler、接收 READY。它是 `AsyncLLM.__init__()` 的启动屏障。

---

## 10. Output handler：静态对象与动态执行

AsyncLLM 需要一个后台 asyncio task 持续：

```text
await engine_core.get_output_async()
→ OutputProcessor.process_outputs()
→ 唤醒相应请求的 AsyncStream
```

需要区分：

- output handler 的管理结构在启动时准备；
- 如果构造时已有 event loop，可立即创建 task；
- 某些编程/测试上下文没有运行中的 loop，会延迟到第一次请求；
- task 每次处理输出属于运行期。

所以“output handler 一定在服务启动时已经跑起来”不是跨入口成立的绝对事实；在线 API Server 路径通常有事件循环。

---

## 11. 请求流预览

以下只说明静态组件的用途：

```text
AsyncLLM.generate()
→ add_request()
→ InputProcessor
→ EngineCoreClient.add_request_async()
→ ZMQ → EngineCore
→ Scheduler / Worker
→ ZMQ → EngineCoreClient.get_output_async()
→ OutputProcessor
→ AsyncGenerator yield
```

`generate()`、`add_request()`、abort 和逐 token 输出不属于本阶段。

---

## 12. 生命周期与所有权

在线服务中通常：

- 一个 server worker 持有一个 `AsyncLLM`；
- FastAPI `app.state.engine_client` 保存它；
- 多个 `OpenAIServing*` 对象共享同一引用；
- `AsyncLLM` 独占其 client/输出处理生命周期；
- client 再管理 EngineCore 子进程。

DP、多个 server worker 或外部 Engine 管理模式会改变数量关系，不能把“一台机器永远一个 AsyncLLM”写成普遍定律。

---

## 13. 本阶段失败定位

| 现象 | 更可能的阶段 |
|---|---|
| tokenizer/renderer 创建失败 | M3 |
| OTLP/tracer 配置错误 | M3 |
| 日志停在 EngineCore 启动前 | M3→M4 |
| 长时间卡在 client 构造 | M4→M12，通常不是 AsyncLLM 轻组件 |
| EngineCore 子进程 OOM | M7–M9 |
| 收不到 READY | M5–M11 任一 proc 端失败 |

---

## 14. Milestone

> 🚩 **M3 · AsyncLLM 前端轻组件就绪**
>
> renderer、InputProcessor、OutputProcessor 和前端 observability 基础已经建立。此时 EngineCoreClient 仍可能正在同步构造，AsyncLLM 尚不能对外服务。

完整 AsyncLLM 的完成标志是 A3 中的 M12。

下一阶段：[A3-EngineCoreClient与ZMQ.md](A3-EngineCoreClient与ZMQ.md)。
