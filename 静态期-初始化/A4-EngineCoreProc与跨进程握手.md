# A4 · EngineCoreProc 与跨进程握手

> **阶段**：M4 → M5 → M11  
> **进程**：EngineCore 子进程  
> **主要源码**：`vllm/v1/engine/core.py` 中 `EngineCoreProc`

---

## 1. 定位

`EngineCoreProc` 是 `EngineCore` 的 ZMQ/进程外壳：

```text
EngineCore                         纯 Engine 业务逻辑
└── EngineCoreProc                 增加：
    ├── 启动 handshake
    ├── ZMQ socket
    ├── input/output queue
    ├── I/O threads
    ├── busy loop
    └── READY / shutdown 协议
```

EngineCore 不必知道前端用 HTTP、asyncio 还是 ZMQ；EngineCoreProc 把传输层请求转换为 EngineCore 方法调用。

---

## 2. 子进程入口

前端进程管理器以 `EngineCoreProc.run_engine_core` 为 target。子进程入口主要做：

- 设置进程名、日志和信号处理；
- 恢复传入的 `VllmConfig`；
- 根据 DP 情况选择 `EngineCoreProc`/DP 扩展；
- 构造 proc 对象；
- 构造成功后进入 busy loop；
- 异常时记录根因并向父端暴露启动失败。

不要把 `Process.start()` 与 `EngineCoreProc.__init__()` 完成画成同一个节点；M4 只是子进程开始执行。

---

## 3. __init__ 的准确顺序

v0.24.0 的关键顺序是：

```text
创建 input_queue / output_queue
→ 设置 engine_index / identity / shutdown state
→ 可选 TensorIpcReceiver
→ 进入 _perform_handshakes(...) 上下文
  → 获得 EngineZmqAddresses
  → 设置 coordinator / stats 信息
  → 初始化 DP 环境
                                      🚩 M5
  → super().__init__(...)             # EngineCore，完成 M6–M10
  → 创建并启动 input thread
  → 创建并启动 output thread
  → 等待 DP coordinator READY（若适用）
→ 握手上下文完成，向前端返回 EngineCoreReadyResponse
                                      🚩 M11
→ run_busy_loop()
```

必须纠正的错误说法：

> 不是“先启动 I/O 守护线程，再调用 `EngineCore.__init__()`”。v0.24.0 中重型 EngineCore 初始化在前，I/O threads 在后。

---

## 4. 队列为什么先创建

```text
input thread → input_queue → busy loop
busy loop → output_queue → output thread
```

队列在重型 EngineCore 初始化前创建，因为：

- Executor failure callback 需要一个可用通道把失败事件投给主循环；
- proc 对象从一开始就需要稳定的内部通信端点；
- I/O 线程虽然稍后启动，但其目标队列必须先存在。

这里使用 `queue.Queue`，因为生产者/消费者是同一进程内的同步线程。`asyncio.Queue` 依赖 event loop，并不适合 proc 主线程的同步 busy loop。

---

## 5. identity

每个 EngineCore 有 engine index/DP rank，并编码成 ZMQ identity。ROUTER 可据此：

- 区分不同 DP Engine；
- 发送定向 utility/request；
- 汇聚 READY；
- 在内部 LB 中完成路由。

identity 是传输身份，不等同于 TP rank 或 PP rank。一个 DP Engine 内部还可能有多个 TP/PP Worker。

---

## 6. _perform_handshakes

握手上下文负责在重型初始化前后形成一个启动事务：

### 进入上下文

- 连接 handshake endpoint；
- 向对应前端/协调器声明 identity；
- 获得数据通道地址；
- DP external/hybrid 模式下可能进行两组握手；
- 获取 coordinator 地址或进程组信息。

### 上下文内部

- 初始化 DP 环境；
- 构造 EngineCore；
- 启动数据通道线程；
- 等待 coordinator ready 条件。

### 正常退出上下文

- 形成 `EngineCoreReadyResponse`；
- 把 max model length、KV block 容量、支持 task 等元数据发回；
- 前端 M12 的等待条件得到满足。

若上下文内部任一步异常，不能发送成功 READY；前端最终由进程监控、错误通道或 timeout 感知失败。

---

## 7. M5：进程外壳已就绪，不等于模型就绪

M5 的产物：

- engine identity；
- input/output/coordinator 地址；
- DP 环境的进程级信息；
- tensor IPC receiver（若使用）；
- 内部 queues 和 failure callback。

M5 尚无：

- 已加载模型；
- KV Cache；
- Scheduler；
- 运行中的 I/O threads。

> 🚩 **M5 · EngineCoreProc 握手与进程环境就绪**
>
> proc 已知道自己是谁、连到哪里、属于哪个 DP 拓扑，可以开始构造 EngineCore。

---

## 8. 调用 EngineCore.__init__

`super().__init__()` 是 EngineCoreProc 启动中最重的同步屏障，内部依次：

```text
Executor / Worker / device            M6
model load                            M7
KV profile + capacity                 M8
KV allocation + compile/warmup        M9
Scheduler / logical KV managers       M10
```

详细见 A5–A9。

EngineCoreProc 只有在 `super().__init__()` 成功返回后，才有可供 busy loop 使用的 scheduler、executor 和 step function。

---

## 9. input thread

`process_input_sockets()` 负责：

```text
DEALER/coordinator socket recv
→ decode message type + payload
→ 必要的 request 预处理
→ input_queue.put(...)
```

它还承担启动/协调器 READY 相关的 socket 行为。

运行期主线程从 `input_queue` drain：

- ADD request；
- ABORT；
- utility call；
- profile/sleep/reconfigure；
- shutdown/failure signal。

把 ZMQ I/O 放在线程中，可以让主线程专注 schedule/execute，同时 ZMQ 等待和部分序列化工作与 GPU 执行重叠。

---

## 10. output thread

`process_output_sockets()` 负责：

```text
output_queue.get()
→ encode EngineCoreOutputs
→ PUSH/coordinator socket send
```

主线程不直接执行网络发送，避免序列化或 socket 背压阻塞 Engine step。

“I/O 能释放 GIL”只表示阻塞 I/O 可与其他 Python 工作并发，不表示所有 msgpack 序列化或 Python 预处理都完全无 GIL。

---

## 11. ready_event

在线 DP=1 与 DP coordinator 模式的 ready 条件不同：

- 无 coordinator 时，input socket thread 完成基础连接即可；
- 有 coordinator 时，必须等 coordinator READY；
- 主线程周期性检查 input thread 是否仍存活，避免无限等一个已经死亡的线程。

只有这些条件满足，启动握手才会成功结束。

---

## 12. busy loop

构造成功后，子进程主线程进入 `run_busy_loop()`：

```text
while running:
  drain input_queue
  apply add/abort/utility
  if scheduler has work:
      step_fn()
      output_queue.put(outputs)
  handle pause/shutdown/failures
```

busy loop 的创建/进入属于 M11；每次 `step_fn()` 属于运行期。

---

## 13. 三线程模型

```text
EngineCore process

input socket thread
   │ decode
   ▼
input_queue
   │
   ▼
main thread: busy loop
   │ schedule + execute
   ▼
output_queue
   │ encode
   ▼
output socket thread
```

队列提供线程安全和背压边界；EngineCore 与 ZMQ 解耦。

---

## 14. DP/PP/TP 的位置

- DP rank 通常对应独立 EngineCoreProc；
- TP/PP Worker 属于某个 EngineCore 的 Executor；
- `engine_index`/ZMQ identity 主要区分 DP Engine；
- PP 的异步 batch queue 在 EngineCore 中；
- TP collective 在 Worker/Executor 分布式环境中。

所以不要把“EngineCoreProc 数 = GPU 数”当成固定关系。

---

## 15. 启动失败语义

| 失败位置 | 前端表现 |
|---|---|
| 地址握手失败 | M5 未到，client 等待/超时 |
| DP init 失败 | proc 退出，monitor 报 Engine 启动失败 |
| Executor/model 失败 | M6/M7 未到 |
| KV profiling/OOM | M8/M9 未到 |
| Scheduler 构造失败 | M10 未到 |
| I/O thread 立即死亡 | M11 未到，不发成功 READY |

日志中的最后一个 milestone 比泛化的“Engine core initialization failed”更有定位价值。

---

## 16. Milestone

> 🚩 **M5 · EngineCoreProc 启动外壳完成**
>
> identity、地址、DP 进程环境和内部队列就绪，开始进入重型 EngineCore 初始化。

> 🚩 **M11 · proc 端可服务**
>
> EngineCore、I/O threads 和必要 coordinator ready 条件完成；READY response 已发送，主线程可进入 busy loop。

下一阶段：[A5-EngineCore编排初始化.md](A5-EngineCore编排初始化.md)。
