# A11 · FastAPI、OpenAIServing 与 AsyncLLM 的关系和初始化时序

> **版本基准**：vLLM `0.24.0`，V1 Engine  
> **范围**：默认 `vllm serve <model>`、单 API Server worker、非 CPU-only render server  
> **主要源码**：`vllm/entrypoints/openai/api_server.py`、`vllm/v1/engine/async_llm.py`、各 task 的 `api_router.py`/`serving.py`

---

## 1. 先给结论

### 问题一：FastAPI app、OpenAIServing、AsyncLLM 是什么关系？

它们是三层，不是继承关系：

```text
FastAPI app                         HTTP/ASGI 容器、路由、middleware、共享 state
└── app.state
    ├── engine_client ──────────────┐
    │                               │
    ├── openai_serving_chat ────────┼── 持有同一个 EngineClient
    ├── openai_serving_completion ──┤
    ├── 其他 Serving* ──────────────┤
    └── models/render/tokenization  │
                                    ▼
                              AsyncLLM
                              （EngineClient 的具体实现）
                                    │
                                    ▼
                              EngineCoreClient
                                    │ ZMQ
                                    ▼
                              EngineCoreProc
```

- FastAPI app 是容器。
- OpenAIServing* 是 HTTP/OpenAI 协议业务层。
- AsyncLLM 是 Engine 前端门面。
- OpenAIServing* 通过 `EngineClient` 接口调用 AsyncLLM。
- FastAPI 与 OpenAIServing* 都不会直接操作 Scheduler、Worker 或 GPU。

### 问题二：静态初始化是从 `vllm serve` 直接拉起 AsyncLLM 吗？

概念上是，但源码上不是一条直接调用：

```text
vllm serve
→ run_server()
→ run_server_worker()
→ build_async_engine_client()
→ build_async_engine_client_from_engine_args()
→ AsyncLLM.from_vllm_config()
```

`build_async_engine_client*` 是 AsyncLLM 的生命周期工厂和 async context manager。

### 问题三：FastAPI/uvicorn 在最后初始化吗？

要拆成三件事回答：

1. **监听 socket 最先 bind**：`setup_server()` 在 Engine 启动前占住端口。
2. **FastAPI app 在 AsyncLLM/Engine READY 后构造**：是靠后的。
3. **uvicorn 真正启动服务在最后**：app/state/Serving 都完成后才调用 `serve_http()`。

所以：

> “端口先绑定”和“FastAPI/uvicorn 最后开始服务”同时成立。

---

## 2. 三层分别负责什么

## 2.1 FastAPI app：HTTP 容器

`build_app()` 创建 FastAPI 对象并注册：

- OpenAI-compatible routes；
- `/health`、models、metrics 等适用路由；
- CORS、authentication、request ID；
- exception handlers；
- scaling middleware；
- lifespan；
- 按 supported tasks 加载的 task router。

FastAPI app 不执行模型推理。它解决：

```text
网络请求
→ 路由匹配
→ HTTP 参数与异常
→ 找到 app.state 中的 Serving 对象
```

### app.state 的作用

`app.state` 是整个 API Server worker 内跨请求共享的对象容器。

`init_app_state()` 至少会保存：

```python
state.engine_client = engine_client
state.vllm_config = engine_client.vllm_config
state.args = args
```

并创建 models、render、tokenization 以及具体 task 对应的 Serving 对象。

---

## 2.2 OpenAIServing*：协议业务层

“OpenAIServing”不是单指一个全局单例，而是一族对象，例如：

- `OpenAIServingChat`；
- completion/generation Serving；
- pooling/embedding/score Serving；
- transcription/realtime 等适用 Serving；
- `OpenAIServingModels`；
- `OpenAIServingRender`。

具体创建哪些对象取决于：

- `await engine_client.get_supported_tasks()`；
- 模型 runner/task；
- server CLI 配置；
- 是否启用 speech、pooling、render 等能力。

### 它们的职责

以 chat/completion 为例：

```text
OpenAI HTTP request
→ 校验 model/LoRA
→ chat template / tool parser / reasoning parser
→ 构造 Engine 输入和 SamplingParams
→ 调 engine_client.generate(...)
→ 将 RequestOutput 组装为 OpenAI response/SSE
```

它们负责“OpenAI 协议语义”，不负责：

- EngineCore 调度；
- block 分配；
- CUDA forward；
- 权重/KV Cache。

### 与 AsyncLLM 的关系

Serving 构造器接收的通常是：

```python
engine_client: EngineClient
```

在 v0.24.0 默认 V1 在线路径中，运行时对象实际是：

```python
engine_client is AsyncLLM instance
```

因此关系是：

```text
OpenAIServing* --持有/引用--> EngineClient
                                  ▲
                                  │ 实际实现
                               AsyncLLM
```

它是依赖接口的组合关系，不是继承关系。

---

## 2.3 AsyncLLM：Engine 前端门面

`AsyncLLM(EngineClient)` 负责把协议层与 EngineCore 隔开。

启动时持有：

- `renderer`；
- `InputProcessor`；
- `OutputProcessor`；
- `EngineCoreClient`；
- `StatLoggerManager`；
- output handler task。

运行时提供：

- `generate()`；
- `encode()`/pooling 相关接口；
- `abort()`；
- `get_supported_tasks()`；
- LoRA、health、utility；
- shutdown。

它不理解 HTTP Request/Response，也不知道 FastAPI route。它可以被其他 Python 调用方复用。

---

## 2.4 BaseRenderer、OpenAIServingRender 与 Tokenizer

三者不是同一个对象，也不是各自独立创建一套 tokenizer。

### Tokenizer

负责较底层的：

```text
text ↔ token IDs
decode token IDs → text
tokenizer metadata/special tokens
```

通常由 BaseRenderer 暴露为 `renderer.tokenizer`；启用 `skip_tokenizer_init`、prompt-embeds/tokenizerless 等特殊配置时它可以是 `None`，这也是 `OutputProcessor` 接受可选 tokenizer 的原因。

### BaseRenderer

`renderer_from_config(vllm_config)` 创建的模型输入渲染器。它**持有或暴露可选 tokenizer**，并提供比 tokenizer 更高层的模型输入能力：

- prompt/chat 输入规范化；
- chat template 渲染；
- text/token prompt 转换；
- 多模态输入与占位信息；
- 生成 `EngineInput`。

默认在线路径中它在 `AsyncLLM.__init__()` 里创建：

```python
self.renderer = renderer_from_config(self.vllm_config)
self.input_processor = InputProcessor(self.vllm_config, self.renderer)
self.output_processor = OutputProcessor(self.renderer.tokenizer, ...)
```

因此：

- AsyncLLM 持有 `BaseRenderer`；
- InputProcessor 复用它；
- OutputProcessor 直接引用它的 tokenizer。

### OpenAIServingRender

这是 OpenAI/HTTP 协议层的包装器，不是第二个 `BaseRenderer`。`init_app_state()` 在 AsyncLLM READY 后：

```python
state.openai_serving_render = OpenAIServingRender(
    model_config=engine_client.model_config,
    renderer=engine_client.renderer,
    ...
)
```

即：

```text
AsyncLLM.renderer ───────────────┐
                                ▼
                          同一个 BaseRenderer
                                ▲
OpenAIServingRender.renderer ────┘
```

`OpenAIServingRender` 额外负责：

- OpenAI chat/completion request 预处理；
- model/LoRA registry；
- chat template 选择和信任策略；
- tool/reasoning parser；
- SamplingParams/GenerateRequest；
- OpenAI response 的 render/derender。

`OpenAIServingChat`、`OpenAIServingCompletion` 等再委托给这个包装器。

### 谁创建谁

正确顺序：

```text
AsyncLLM.__init__()
→ 创建 BaseRenderer + tokenizer
→ AsyncLLM/Engine READY
→ init_app_state()
→ 创建 OpenAIServingRender
  └─ 注入 engine_client.renderer（复用，不重新创建）
→ 创建 OpenAIServingChat/Completion/ServingTokenization
  └─ 继续复用 OpenAIServingRender/BaseRenderer/tokenizer
```

不是 AsyncLLM 实例化 OpenAIServingRender；是 API Server 在后续 `init_app_state()` 中用 AsyncLLM 暴露的 renderer 创建协议层包装器。

CPU-only render server 是例外：没有 AsyncLLM，因此 `init_render_app_state()` 自己调用 `renderer_from_config()`。

---

## 3. 谁创建谁

精确创建关系：

```text
run_server_worker()
│
├── build_async_engine_client(args)
│   ├── AsyncEngineArgs.from_cli_args(args)
│   └── build_async_engine_client_from_engine_args(...)
│       ├── engine_args.create_engine_config()
│       ├── AsyncLLM.from_vllm_config(...)
│       │   └── AsyncLLM.__init__()
│       │       └── EngineCoreClient.make_async_mp_client(...)
│       │           └── EngineCore 子进程完整启动
│       ├── await async_llm.reset_mm_cache()
│       └── yield async_llm
│
└── build_and_serve(engine_client=async_llm, ...)
    ├── await engine_client.get_supported_tasks()
    ├── app = build_app(...)
    ├── await init_app_state(engine_client, app.state, ...)
    │   ├── state.engine_client = engine_client
    │   ├── OpenAIServingModels(engine_client=...)
    │   ├── OpenAIServingRender(...)
    │   ├── ServingTokenization(...)
    │   └── task-specific init_*_state()
    │       └── 创建 OpenAIServingChat/Completion/... 
    └── serve_http(app, sock=...)
        └── uvicorn server
```

因此：

- `vllm serve` 的 API Server 路径创建 AsyncLLM；
- AsyncLLM 不是由 FastAPI app 创建；
- FastAPI app 和 Serving 对象反而使用已经 READY 的 AsyncLLM；
- AsyncLLM 的生命周期由 `build_async_engine_client*` async context manager 管理。

---

## 4. 严格初始化时间线

## T0：CLI 入口

```text
vllm serve <model>
→ ServeSubcommand
→ uvloop.run(run_server(args))
```

此时只是从同步 CLI 进入异步 server 主协程。

---

## T1：setup_server 先绑定 socket

`run_server()` 第一阶段：

```python
listen_address, sock = setup_server(args)
```

`setup_server()`：

1. 记录版本和参数；
2. 加载 tool/reasoning parser plugin；
3. 校验 API server 参数；
4. 创建 TCP 或 Unix socket；
5. `sock.bind(...)`；
6. 调整文件描述符上限；
7. 返回 `listen_address, sock`。

源码注释明确说明：在 Engine setup 前先 bind port，用来规避 Ray 等场景的端口竞态。

### 这时能接 HTTP 吗？

不能据此认为服务已经可用。

此时：

- socket 地址已被占用；
- FastAPI app 尚未创建；
- uvicorn 尚未接管 socket；
- AsyncLLM 尚未创建。

它只是“先把门牌号占住”，还没有营业。

---

## T2：进入 run_server_worker

```python
await run_server_worker(listen_address, sock, args)
```

核心结构：

```python
async with build_async_engine_client(args) as engine_client:
    shutdown_task = await build_and_serve(
        engine_client, listen_address, sock, args
    )
    await shutdown_task
```

这个代码块同时确定：

- Engine 必须先构造成功；
- app/uvicorn 在 Engine context 内运行；
- server 退出后 context manager 负责 shutdown Engine。

---

## T3：构造 AsyncEngineArgs 与 VllmConfig

```text
AsyncEngineArgs.from_cli_args(args)
→ engine_args.create_engine_config()
→ VllmConfig
```

对应知识库 M1、M2。

---

## T4：直接构造 AsyncLLM，而不是先构造 FastAPI

`build_async_engine_client_from_engine_args()`：

```python
from vllm.v1.engine.async_llm import AsyncLLM

async_llm = AsyncLLM.from_vllm_config(
    vllm_config=vllm_config,
    ...
)
```

注意它是普通同步构造调用，虽然外层函数是 async context manager。

`AsyncLLM.__init__()` 内部：

```text
renderer / InputProcessor / OutputProcessor
→ EngineCoreClient.make_async_mp_client()
→ spawn EngineCore
→ Worker + model
→ KV Cache
→ Scheduler
→ READY handshake
→ logger/output handler
```

所以 `AsyncLLM.from_vllm_config()` 返回时，EngineCore 重型初始化已经完成。

---

## T5：AsyncLLM yield 前的小收尾

源码在 yield 前执行：

```python
await async_llm.reset_mm_cache()
yield async_llm
```

这意味着 `run_server_worker()` 拿到的 `engine_client` 已是可用 EngineClient，而不是一个“后台仍在加载模型”的 future。

对应 M12。

---

## T6：查询支持的 tasks

`build_and_serve()` 首先：

```python
supported_tasks = await engine_client.get_supported_tasks()
model_config = engine_client.model_config
```

为什么在 build_app 前：

- route 集合依赖模型支持 generate、pooling、speech 等哪些 tasks；
- app 和 Serving 对象只应创建适用能力；
- task 信息来自已经 READY 的 EngineCore。

---

## T7：构造 FastAPI app

```python
app = build_app(args, supported_tasks, model_config)
```

此时完成：

- FastAPI 对象；
- routes；
- middleware；
- exception handlers；
- lifespan 配置。

但 `app.state.engine_client` 和 Serving 实例还没完全挂入。

---

## T8：init_app_state 创建 Serving 对象

```python
await init_app_state(
    engine_client,
    app.state,
    args,
    supported_tasks,
)
```

关键操作：

```python
state.engine_client = engine_client
state.vllm_config = engine_client.vllm_config
state.openai_serving_models = OpenAIServingModels(
    engine_client=engine_client,
    ...
)
```

随后创建 render/tokenization，并按 task 调：

- `init_generate_state(...)`；
- `init_pooling_state(...)`；
- `init_speech_to_text_state(...)`；
- 其他相应初始化器。

这些 task-specific initializer 再创建对应 OpenAIServing*，并保存进 `app.state`。

对应 M13。

---

## T9：最后启动 uvicorn

只有 app 和 state 都完成后才：

```python
await serve_http(app, sock=sock, ...)
```

`serve_http()`：

- 使用已经 bind 的 socket；
- 建立 uvicorn server；
- 执行 ASGI lifespan startup；
- 开始 accept/处理请求；
- 返回 shutdown task。

对应 M14。

---

## 5. 时间线总图

```text
时间 ────────────────────────────────────────────────────────────────→

CLI/API Server 主协程

T0  vllm serve
 │
T1  setup_server()
 │   └─ TCP/Unix socket.bind()                 端口已占用，但不能服务
 │
T2  run_server_worker()
 │
T3  AsyncEngineArgs → VllmConfig               M1/M2
 │
T4  AsyncLLM.from_vllm_config()
 │   ├─ 前端组件                               M3
 │   └─ EngineCoreClient.make_async_mp_client()
 │       └─ EngineCore/Worker/model/KV/Scheduler
 │                                              M4–M11
 │
T5  AsyncLLM READY，被 context manager yield    M12
 │
T6  get_supported_tasks()
 │
T7  build_app()                                 FastAPI 对象/路由
 │
T8  init_app_state()
 │   ├─ state.engine_client = AsyncLLM
 │   └─ OpenAIServing* 持有同一 EngineClient    M13
 │
T9  serve_http(app, pre-bound sock)
     └─ uvicorn lifespan + accept               M14
```

---

## 6. 对象关系图

```text
run_server_worker 的 async context
│
├── 生命周期拥有：AsyncLLM
│   ├── renderer
│   ├── input_processor
│   ├── output_processor
│   └── engine_core: EngineCoreClient
│
└── 生命周期内运行：FastAPI/uvicorn
    │
    └── app.state
        ├── engine_client ─────────────────────────┐
        │                                          │
        ├── openai_serving_models                  │
        │   └── engine_client ─────────────────────┤
        │                                          │
        ├── openai_serving_chat                    │
        │   └── engine_client ─────────────────────┤
        │                                          │
        ├── openai_serving_completion              │
        │   └── engine_client ─────────────────────┤
        │                                          │
        └── other task Serving*                    │
            └── engine_client ─────────────────────┤
                                                   ▼
                                           同一个 AsyncLLM 实例
```

关键点：

- `app.state.engine_client` 和 Serving 中的 `self.engine_client` 指向同一个对象；
- 多个引用不表示创建了多个 AsyncLLM；
- context manager 才是 Engine 生命周期的外层拥有者；
- app.state 让路由能找到共享对象；
- Serving 通过接口调用它。

---

## 7. 一个请求如何穿过三层

静态初始化结束后：

```text
HTTP POST /v1/chat/completions
        │
        ▼
FastAPI route
        │ 从 request.app.state 取得
        ▼
OpenAIServingChat.create_chat_completion()
        │
        ├─ model/LoRA 校验
        ├─ chat template / render
        ├─ SamplingParams
        └─ self.engine_client.generate(...)
                            │
                            ▼
                         AsyncLLM
                            ├─ InputProcessor
                            ├─ EngineCoreClient → ZMQ → EngineCore
                            └─ OutputProcessor
                            │
                            ▼
OpenAIServingChat 组装 OpenAI response/SSE
        │
        ▼
FastAPI/uvicorn 写回 HTTP
```

因此三者关系可以概括为：

```text
FastAPI 负责“请求到了谁”
OpenAIServing 负责“OpenAI 协议怎么解释”
AsyncLLM 负责“怎样异步驱动 vLLM Engine”
```

---

## 8. 为什么要 Engine 先于 FastAPI app

### 8.1 routes 依赖 supported tasks

app 要知道模型支持：

- generate；
- pooling/embedding；
- score/classify；
- transcription/realtime；
- render。

这些能力由 `engine_client.get_supported_tasks()` 给出。

### 8.2 Serving 构造依赖 Engine 配置

Serving 需要：

- `model_config`；
- renderer/tokenizer；
- served model names；
- LoRA modules；
- tool/reasoning parser；
- chat template；
- Engine health/generate 接口。

### 8.3 避免“HTTP 已监听但 Engine 还没 ready”

如果先启动 uvicorn 再后台加载模型，请求会遇到复杂的：

- 503/readiness；
- 排队和超时；
- 启动失败后的连接语义；
- half-initialized state。

v0.24.0 主路径用 Engine context 作为 server startup 屏障，只有 M12 后才走到 M13/M14。

---

## 9. 为什么 socket 又要最先 bind

这不是为了提前服务，而是为了：

- 提前确定端口；
- 阻止另一个进程抢占；
- 避免 Ray/多进程启动中的端口竞态；
- 将同一个 socket 最后交给 uvicorn。

类比：

```text
bind socket       先取得店铺地址
load Engine       搬入设备和库存
build app/state   布置柜台和业务员
uvicorn accept    正式开门营业
```

---

## 10. 生命周期与 shutdown

`build_async_engine_client_from_engine_args()` 结构：

```python
try:
    async_llm = AsyncLLM.from_vllm_config(...)
    yield async_llm
finally:
    if async_llm:
        async_llm.shutdown(...)
```

而 `run_server_worker()` 在 context 内等待 uvicorn shutdown task。

正常逆序：

```text
uvicorn 停止
→ socket close
→ 离开 async engine context
→ AsyncLLM.shutdown()
→ output handler / renderer
→ EngineCoreClient shutdown
→ EngineCore/Worker 退出
```

FastAPI app 并不是唯一的 Engine owner；真正保证 Engine 被清理的是 async context manager。

---

## 11. 特殊路径不能套用默认结论

### CPU-only render server

`init_render_app_state()` 可以没有 EngineClient：

```python
state.engine_client = None
```

它直接从 `VllmConfig` 建 renderer/input pipeline。此时“Serving 都持有 AsyncLLM”不成立。

### 多 API worker / DP / external Engine

- 可能有多个 AsyncLLM/client 实例；
- client 可连接外部管理的 Engine；
- 不一定由该 API worker 本地 spawn 所有 EngineCore；
- app.state 与其所在 worker 的 EngineClient 对应。

本文结论针对默认单 worker 在线主路径。

---

## 12. 最容易说错的五句话

### 错误一：“FastAPI 创建 AsyncLLM”

正确：

> `run_server_worker()` 的 Engine context 先创建 AsyncLLM；随后 `build_and_serve()` 创建 FastAPI，并把 AsyncLLM 挂到 app.state。

### 错误二：“OpenAIServing 就是 AsyncLLM”

正确：

> OpenAIServing 是协议业务层，持有类型为 EngineClient 的引用；默认 V1 路径中该引用实际指向 AsyncLLM。

### 错误三：“vllm serve 一启动就先 new FastAPI”

正确：

> 它先 bind server socket，再完整创建 AsyncLLM/Engine，然后才 `build_app()`。

### 错误四：“端口 bind 后服务已经 ready”

正确：

> bind 只占住地址；`serve_http()` 启动 uvicorn 后才真正 accept 请求。

### 错误五：“AsyncLLM 在后台异步加载模型，FastAPI 同时启动”

正确：

> `AsyncLLM.from_vllm_config()` 构造过程同步等待 EngineCore READY；返回并 yield 后才构造 app/Serving。

---

## 13. 最终记忆模型

记住四个动词：

```text
setup_server()              bind
build_async_engine_client() build Engine
init_app_state()            wire objects
serve_http()                serve
```

以及一句总纲：

> vLLM 0.24.0 默认 V1 在线路径先占端口，再同步拉起并等待 AsyncLLM/Engine 完全就绪；然后构造 FastAPI 和 OpenAIServing，把同一个 AsyncLLM 作为 EngineClient 注入 app.state 与各 Serving；最后 uvicorn 才开始处理 HTTP 请求。

---

## 14. 对应 milestone

| 时序点 | Milestone |
|---|---|
| AsyncLLM 前端轻组件 | M3 |
| EngineCore 进程启动到 READY | M4–M11 |
| AsyncLLM 被 context yield | M12 |
| FastAPI + app.state + Serving | M13 |
| uvicorn 开始服务 | M14 |

关联阅读：

- [A2-AsyncLLM前端初始化.md](A2-AsyncLLM前端初始化.md)
- [A3-EngineCoreClient与ZMQ.md](A3-EngineCoreClient与ZMQ.md)
- [A10-FastAPI-Serving与服务就绪.md](A10-FastAPI-Serving与服务就绪.md)
- [AX-静态初始化全景总结.md](AX-静态初始化全景总结.md)

官方源码：

- [v0.24.0 `api_server.py`](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/entrypoints/openai/api_server.py)
- [v0.24.0 `async_llm.py`](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/v1/engine/async_llm.py)
