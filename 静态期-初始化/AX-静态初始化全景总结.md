# AX · vLLM V1 静态初始化全景总结

> **基准**：vLLM 0.24.0  
> **主线**：`vllm serve` → server 编排 → EngineArgs/Config → AsyncLLM 工厂与构造 → AsyncMPClient → EngineCoreProc → Executor/Worker → KV/Scheduler → FastAPI/uvicorn

---

## 1. 一句话

vLLM 静态初始化把用户配置编译成 `VllmConfig`，在前端创建异步协议门面和 ZMQ client，在子进程/Worker 中建立设备并加载模型，依据真实峰值显存创建 KV Cache，建立 Scheduler 的逻辑 block 账本，最后通过 READY handshake 把 Engine 汇聚回 FastAPI 并开始监听。

---

## 2. 完整启动链：控制流所有权 + 资源就绪时序

```text
$ vllm serve <model>
│
▼
CLI main / ServeSubcommand
│
▼
run_server()
├─ setup_server()：提前 bind socket，尚未开始 HTTP 服务
└─ run_server_worker()
   │
   ▼
async with build_async_engine_client(args, client_config=...)
├─ AsyncEngineArgs.from_cli_args(args)
│  └─ AsyncEngineArgs：CLI 输入边界收口                      M1
│
└─ async with build_async_engine_client_from_engine_args(
     engine_args, usage_context=OPENAI_API_SERVER, ...)
   ├─ engine_args.create_engine_config(usage_context)
   │  └─ VllmConfig                                         M2
   │
   ├─ AsyncLLM.from_vllm_config(vllm_config, ...)
   │  ├─ Executor.get_class(vllm_config)：只选择 Executor 类
   │  └─ AsyncLLM(vllm_config, executor_class, ...)
   │     └─ AsyncLLM.__init__()
   │        ├─ A. config serialization policy + config refs
   │        ├─ B. tracing + request/stat logging policy
   │        ├─ C. BaseRenderer + InputProcessor + OutputProcessor M3
   │        ├─ EngineCoreClient.make_async_mp_client()
   │        │  ├─ ZMQ ROUTER/PULL bind
   │        │  ├─ launch_core_engines()                     M4
   │        │  │
   │        │  │   EngineCore process
   │        │  │   └─ EngineCoreProc
   │        │  │       ├─ handshake / identity / DP env     M5
   │        │  │       └─ EngineCore.__init__()
   │        │  │           ├─ Executor
   │        │  │           │   └─ Worker.init_device()
   │        │  │           │       └─ GPUModelRunner        M6
   │        │  │           ├─ Worker.load_model()           M7
   │        │  │           ├─ KV specs/profile/config       M8
   │        │  │           ├─ KV tensors + compile/warmup   M9
   │        │  │           ├─ StructuredOutputManager
   │        │  │           └─ Scheduler
   │        │  │               └─ KVCacheManager
   │        │  │                   └─ Coordinator/BlockPool M10
   │        │  │
   │        │  ├─ proc I/O threads + READY；进入 busy loop  M11
   │        │  └─ client 收齐 READY / 应用派生配置
   │        ├─ StatLoggerManager（若启用）
   │        ├─ output handler（在线主路径立即启动）
   │        ├─ optional frontend torch profiler
   │        └─ AsyncLLM.__init__() return                   M12
   ├─ await engine_client.reset_mm_cache()
   ├─ yield engine_client
   │  └─ build_and_serve(..., engine_client, ...)
   │     ├─ FastAPI app + app.state + OpenAIServing*        M13
   │     └─ uvicorn listen                                  M14
   └─ finally: engine_client.shutdown(...)
```

> 缩进表示调用/生命周期所有权；同一缩进内的纵向顺序表示先后。`build_async_engine_client()` 是外层 async context manager，不是 AsyncLLM 方法。它在 `yield` 前完成 M1–M12，HTTP 服务在 yield 期间运行，退出后逆序 shutdown。`Executor.get_class()` 只返回类，真正的 Executor 实例到 EngineCore 子进程中才创建。

M3 的三个块存在依赖，而非任意并列：config refs 供 B/C 使用；renderer 先于两个 processor；`OutputProcessor` 同时依赖 renderer.tokenizer、`self.log_stats`、stream interval 和 tracing 开关；序列化策略只需保证在 EngineCore spawn 前注册。

---

## 3. Milestone 速查

| 标志 | 谁完成 | 判断标准 | 尚未完成 |
|---|---|---|---|
| M0 | 知识边界 | 版本/主路径锁定 | 任何对象 |
| M1 | CLI/Engine 参数边界 | `AsyncEngineArgs` 返回；下游不再依赖原始 Namespace | config 构造与跨配置校验 |
| M2 | Config | `VllmConfig` 返回 | GPU 实测派生值 |
| M3 | AsyncLLM 前端 | A 配置/跨进程准备 + B 可观测性/日志策略 + C renderer/processors 完成 | Engine client 与后置收尾 |
| M4 | MPClient | socket bind + process start | Engine ready |
| M5 | Proc | 地址/identity/DP 环境 | model/KV |
| M6 | Worker | device/distributed/runner | weights |
| M7 | Executor | 所有模型 shard loaded | KV/Scheduler |
| M8 | EngineCore | KV specs/capacity known | tensor/warmup |
| M9 | Worker | KV physical + warmup ready | logical pool |
| M10 | Scheduler | queues/managers/BlockPool | proc IPC threads |
| M11 | Proc | I/O threads ready，发送 READY，随后进 busy loop | client 汇聚 |
| M12 | AsyncLLM | client READY + logger/output handler/profiler 收尾，`__init__` 返回 | HTTP app |
| M13 | FastAPI | state/Serving/routes ready | listen |
| M14 | uvicorn | 对外监听 | — |

---

## 4. 两个进程中的持久对象

### API Server

```text
FastAPI
└── AsyncLLM
    ├── tracer pipeline（若配置 OTLP endpoint）
    ├── renderer
    ├── InputProcessor
    ├── OutputProcessor
    ├── StatLoggerManager
    └── AsyncMPClient
        ├── ROUTER/PULL
        ├── encoder/decoder
        ├── output task
        └── process manager/monitor
```

### EngineCore

```text
EngineCoreProc
├── queues + I/O threads
└── EngineCore
    ├── Executor
    │   └── Worker(s)
    │       └── GPUModelRunner
    │           ├── model weights
    │           └── physical KV tensors
    ├── Scheduler
    │   └── KVCacheManager
    │       └── logical BlockPool
    └── auxiliary managers
```

TP/PP/Ray 会把 Worker 扩展到更多进程/actor，但职责分层不变。

---

## 5. 五种“ready”

| ready | 含义 |
|---|---|
| config ready | 参数组合已经收口 |
| model ready | 权重已加载 |
| memory ready | KV 容量、物理 tensor、warmup 已完成 |
| Engine ready | Scheduler + proc I/O + handshake 完成 |
| service ready | HTTP app 已监听 |

看到日志“Loading model weights took …”只能说明 M7，不能推断服务马上可用。

---

## 6. 三类数据

### 配置数据

```text
CLI → EngineArgs → VllmConfig
→ M3 注册按值序列化规则
→ multiprocessing/Ray 序列化到 proc/Worker（不是运行期 ZMQ request）
→ profiling 派生值通过 READY 回前端
```

### 请求数据

```text
HTTP → AsyncLLM/InputProcessor
→ ROUTER/DEALER → EngineCore
→ Scheduler/Worker
→ PUSH/PULL → OutputProcessor
→ SSE/response
```

### KV 数据

```text
ModelRunner 在 GPU tensor 中读写真实 K/V
Scheduler 用 block IDs 管理逻辑所有权
BlockPool 用 hash/refcount 管理复用和回收
```

---

## 7. 最关键的依赖关系

```text
VllmConfig
    ↓
Executor/Worker/device
    ↓
model weights
    ↓
真实执行 profiling
    ↓
KV capacity
    ↓
physical KV + warmup
    ↓
Scheduler logical BlockPool
    ↓
proc READY
    ↓
client/HTTP ready
```

任何箭头上游失败，都不应把下游对象写成“已初始化”。

---

## 8. 静态与运行期对照

| 静态创建一次 | 运行期重复使用 |
|---|---|
| InputProcessor | 处理每个输入 |
| OutputProcessor | 处理每批输出 |
| ZMQ sockets | 发送/接收消息 |
| model weights | 每 step forward |
| physical KV tensors | 写入/读取 token KV |
| Scheduler | 每 step schedule |
| BlockPool | allocate/free/cache blocks |
| structured output manager | 为请求创建/推进 grammar |
| batch queue | 流水化 batch |

每 step 的 token budget、临时结果桶、block table 更新不是静态初始化。

---

## 9. 常见错误结论

### 错：AsyncLLM 创建完成后才启动 EngineCore

正确：EngineCore client 的同步构造嵌在 `AsyncLLM.__init__()` 中；M12 前 AsyncLLM 尚未返回。

### 错：EngineCoreProc 先启动 I/O threads，再加载模型

正确：v0.24.0 先完成 `EngineCore.__init__()`，随后启动 input/output threads。

### 错：EngineCoreClient 是 AsyncLLM 的父类

正确：AsyncLLM 实现 EngineClient 协议，并持有 EngineCoreClient。

### 错：BlockPool 分配 GPU KV tensor

正确：Worker/ModelRunner 在 M9 创建物理 tensor；BlockPool 在 M10 管理逻辑 IDs。

### 错：profiling 完成就等于 KV 完全 ready

正确：M8 只保证容量已知；M9 才保证物理 KV 和 warmup 成功。

### 错：模型 loaded 就可接 HTTP

正确：M7 后还要 M8–M14。

---

## 10. 启动耗时地图

通常：

- M1–M3：轻，受 tokenizer/config/插件影响；
- M4–M5：进程与 IPC；
- M6：CUDA/NCCL/Worker；
- M7：权重 I/O、反序列化、量化；
- M8：profiling forward；
- M9：KV allocation、compile、CUDA Graph warmup；
- M10–M13：大多较轻，但 connector/grammar/多模态分支可能增加工作；
- M14：server startup。

不要写死秒数。模型、GPU、磁盘、并行、compile 和 feature 组合会让耗时相差数个数量级。

---

## 11. 排障决策表

| 最后日志/现象 | 优先看 |
|---|---|
| 参数/config error | A1，M1–M2 |
| tokenizer/renderer error | A2，M3 |
| spawn/handshake timeout | A3/A4，M4–M5 |
| NCCL/rank/device error | A6，M6 |
| checkpoint/quantization error | A6，M7 |
| profile/no memory/max len | A7，M8 |
| KV allocate/compile/capture OOM | A7，M9 |
| block/group/connector error | A8/A9，M10 |
| proc ready 未返回 | A4，M11 |
| AsyncLLM 构造不返回 | A3，M12 汇聚 |
| engine ready 但端口不可用 | A10，M13–M14 |

---

## 12. 分支如何挂到主线

```text
多模态       M2/M3/M5/M6/M8/M10
投机解码     M2/M6/M8/M9/M10
DP           M2/M4/M5/M11/M12
TP/PP        M2/M6/M7/M8/M9/M10
Ray/external M2/M4/M6/M7
P/D          M2/M6/M9/M10/M11
LoRA         M2/M3/M6，adapter 可运行期加载
```

这些分支不会推翻主 milestone，只会让某节点包含更多参与者和完成条件。

---

## 13. 面向源码的检查清单

切换 vLLM 版本时优先重新检查：

1. API Server 是否仍显式使用 `from_vllm_config()`；
2. client 工厂和 DP 类继承是否变化；
3. ZMQ socket 类型和 bind/connect；
4. EngineCoreProc 中 `super().__init__` 与线程启动顺序；
5. Executor 是否在构造期完成 load_model；
6. `_initialize_kv_caches()` 是否仍包含 warmup；
7. KV groups/coordinator 类拆分；
8. Scheduler 构造参数；
9. READY response 字段；
10. FastAPI app state 初始化顺序。

只更新行号而不检查这些语义，没有意义。

---

## 14. 文件索引

| 文件 | 核心问题 |
|---|---|
| [A0](A0-阅读地图与版本边界.md) | 边界、进程图、全局 milestone |
| [A1](A1-CLI-EngineArgs-VllmConfig.md) | 配置如何诞生和流转 |
| [A2](A2-AsyncLLM前端初始化.md) | 前端门面创建了什么 |
| [A3](A3-EngineCoreClient与ZMQ.md) | IPC、spawn、READY |
| [A4](A4-EngineCoreProc与跨进程握手.md) | proc 顺序、线程、busy loop |
| [A5](A5-EngineCore编排初始化.md) | 核心组件为什么按此顺序 |
| [A6](A6-Executor-Worker-GPUModelRunner.md) | 设备、分布式、模型加载 |
| [A7](A7-KVCache-Profiling-分配-Warmup.md) | M8 与 M9 的边界 |
| [A8](A8-Scheduler-KVCacheManager-BlockPool.md) | 逻辑 KV 账本 |
| [A9](A9-辅助组件与可选分支.md) | 多模态、DP/TP/PP、P/D 等 |
| [A10](A10-FastAPI-Serving与服务就绪.md) | Engine ready 到 HTTP ready |
| [A11](A11-FastAPI-OpenAIServing-AsyncLLM关系与时序.md) | FastAPI、Serving、AsyncLLM 的对象关系与严格时序 |

---

## 15. 最终完成标志

> 🚩 **M14 · 静态初始化完成**
>
> 从 CLI、配置、AsyncLLM、IPC、EngineCore、Worker/model、KV、Scheduler 到 FastAPI 的整条依赖链已完成，uvicorn 正在监听。M14 之后才进入“请求处理期”。
