# A10 · FastAPI、OpenAI Serving 与服务就绪

> **阶段**：M12 → M13 → M14  
> **进程**：API Server 进程  
> **主要源码**：`vllm/entrypoints/openai/api_server.py`、`vllm/entrypoints/openai/serving_engine.py` 等

---

## 1. 为什么 HTTP 层属于静态初始化

Engine M12 ready 只表示 Python 内已有可用 `AsyncLLM`，不表示客户端能连接。

完整 `vllm serve` 启动还要：

```text
build_app()
→ attach routes/middleware
→ init_app_state(engine_client)
→ 构造 OpenAIServing*
→ lifespan/startup hooks
→ serve_http()
→ uvicorn 开始监听
```

所以服务级最终 milestone 是 M14，不是模型加载 M7，也不是 Engine READY M12。

---

## 2. server socket 与监听的区别

在线主路径可能先执行 `setup_server()` 绑定/准备 socket，再启动 Engine，以避免端口和分布式启动竞态。

必须区分：

- socket 已创建/绑定；
- uvicorn 已开始 accept；
- application startup 已完成。

只有后两者满足时，外部请求才真正可服务。知识库把最终完成点定义为 M14。

---

## 3. build_app

典型静态产物：

- FastAPI app；
- OpenAI-compatible routes；
- health/version/metrics routes；
- middleware；
- CORS/auth；
- exception handlers；
- lifespan hooks。

路由函数此时只是注册；并未处理用户请求。

---

## 4. app.state

`init_app_state` 将跨请求共享对象放入 app：

```text
app.state
├── engine_client: AsyncLLM
├── openai_serving_chat
├── openai_serving_completion
├── openai_serving_embedding / pooling / score ...
├── model/path/tool/reasoning 配置
└── metrics/其他共享服务
```

具体 Serving 集合取决于模型支持的 tasks 和 CLI 配置。

---

## 5. OpenAIServing* 的角色

这些对象是协议业务层：

```text
HTTP JSON
→ 参数校验 / chat template / tool/reasoning
→ self.engine_client.generate()/encode()
→ AsyncLLM
→ RequestOutput
→ OpenAI-compatible response/SSE
```

它们共享同一个 `AsyncLLM` 引用，不直接访问：

- Scheduler；
- BlockPool；
- Worker；
- GPUModelRunner；
- ZMQ socket。

这保持了 HTTP 协议层与 Engine 实现的边界。

---

## 6. M13 的完成条件

> 🚩 **M13 · HTTP 业务对象就绪**
>
> FastAPI app、routes、`app.state.engine_client` 和适用的 OpenAIServing 对象已经构造。进程内 HTTP 调用链完整，但 uvicorn 可能尚未正式监听。

---

## 7. uvicorn serve

`serve_http(app, sock, ...)` 把 app、预先准备的 socket 和 server 配置交给 uvicorn。

当：

- application lifespan startup 成功；
- server 开始监听/accept；
- shutdown task 进入等待

时，外部客户端才可连接。

> 🚩 **M14 · vLLM 服务完全就绪**
>
> uvicorn 正在监听，HTTP 请求可以进入 FastAPI，并通过共享 AsyncLLM 到达已经 READY 的 EngineCore。

---

## 8. 为什么启动期间连接失败

主路径时序：

```text
准备 socket
→ M2 config
→ M3 frontend
→ M4 spawn
→ M5–M11 proc 重型初始化
→ M12 client/AsyncLLM
→ M13 app state/Serving
→ M14 listen
```

M4–M12 期间 client 在等 READY；HTTP server 尚未对外工作。模型越大，M7 权重加载和 M8/M9 KV/warmup 越久。

---

## 9. readiness 与 liveness

- **process alive**：进程存在，不代表 Engine ready；
- **Engine READY**：M12，内部可服务；
- **HTTP liveness**：server 能响应基础端点；
- **HTTP readiness**：路由背后的 Engine 可接请求；
- **首请求已 warm**：M9 已尽量搬走一次性开销，但某些 feature 仍可能按需初始化。

部署探针应明确自己测的是哪一种语义。

---

## 10. Serving 数量不是固定常量

模型 runner/task 会影响：

- chat/completion；
- embeddings/pooling；
- classify/score；
- rerank；
- tokenize；
- transcriptions 或其他入口。

因此“一定创建 Chat 和 Completion 两个对象”不是跨模型事实。稳定事实是：

> app.state 根据模型能力和 server 配置创建适用 Serving 对象，并让它们共享 EngineClient。

---

## 11. shutdown 边界

虽然不是本知识库主线，所有权决定了逆序清理：

```text
停止接受 HTTP
→ shutdown Serving/background tasks
→ AsyncLLM shutdown
→ EngineCoreClient sockets/tasks/monitor
→ EngineCore processes
→ Executor/Workers
→ model/KV/distributed resources
```

正常 shutdown 应让 A0–A10 创建的长期资源可回收，尤其是 EngineCore 中的 GC unfreeze 和分布式清理。

---

## 12. Milestone

> 🚩 **M13 · FastAPI/OpenAIServing 就绪**
>
> EngineClient 已挂入 app.state，适用协议层对象与路由完成。

> 🚩 **M14 · 对外服务就绪**
>
> uvicorn 已完成 startup 并开始监听。这是 `vllm serve` 静态初始化的最终完成点。

总结见 [AX-静态初始化全景总结.md](AX-静态初始化全景总结.md)。
