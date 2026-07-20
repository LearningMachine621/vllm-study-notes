# A3 · EngineCoreClient 与 ZMQ IPC

> **阶段**：M3 → M4，随后等待 M11 → M12  
> **进程**：API Server 进程  
> **主要源码**：`vllm/v1/engine/core_client.py`、`vllm/v1/engine/utils.py`、`vllm/utils/network_utils.py`

---

## 1. 定位

`EngineCoreClient` 是前端对 EngineCore 的传输抽象：

```text
AsyncLLM
└── engine_core: EngineCoreClient
    ├── add_request_async()
    ├── abort_requests_async()
    ├── get_output_async()
    ├── utility RPC
    └── shutdown()
```

在线默认 `AsyncMPClient` 通过 ZMQ 与 EngineCore 子进程通信。它不调度、不 forward，只负责：

- client 类型选择；
- ZMQ context/socket；
- 序列化；
- EngineCore 进程启动；
- 启动握手与 READY 汇聚；
- 输出接收和存活监控；
- DP 请求路由。

---

## 2. 类族

```text
EngineCoreClient
├── InprocClient
└── MPClient
    ├── SyncMPClient
    └── AsyncMPClient
        └── DPAsyncMPClient
            └── DPLBAsyncMPClient
```

- `InprocClient`：直接构造 EngineCore，适合特定离线/测试路径；没有 ZMQ 进程边界。
- `SyncMPClient`：同步消费者使用。
- `AsyncMPClient`：AsyncLLM 主干。
- `DPAsyncMPClient`：多个 DP Engine，外部 LB。
- `DPLBAsyncMPClient`：前端/Coordinator 参与内部或混合负载均衡。

---

## 3. 客户端工厂

v0.24.0 的类方法 `EngineCoreClient.make_async_mp_client()` 读取：

- `parallel_config.data_parallel_size`；
- external/internal/hybrid load-balancing；
- client address / external engine 管理信息；
- client index/count。

它只选择并调用具体构造器。真正的 socket、spawn 和 handshake 在 `MPClient.__init__()` 及子类扩展中发生。

---

## 4. 两条数据通道

最简 DP=1：

| 方向 | 前端 socket | proc socket | 内容 |
|---|---|---|---|
| client → core | `ROUTER`, bind | `DEALER`, connect | 请求、abort、utility RPC、READY/路由消息 |
| core → client | `PULL`, bind | `PUSH`, connect | `EngineCoreOutputs` |

为什么不用一个 REQ/REP：

- 输入和输出是异步、多路、非严格一问一答；
- 一个请求会产生多次流式输出；
- utility RPC、abort 和普通请求可能交错；
- ROUTER/DEALER 可使用 identity 区分 DP engine；
- PUSH/PULL 让输出路径保持简单。

READY/handshake 的控制消息与正常运行期消息共享或配合启动握手通道，具体细节应以 v0.24.0 符号为准，不要把“ROUTER 只发送请求”理解成绝对限制。

---

## 5. endpoint 如何选择

`get_engine_zmq_addresses(vllm_config)` 根据部署拓扑选择：

- 同机、本地 client：优先 `ipc://...`；
- 跨节点/外部管理：使用 `tcp://host:port`；
- `tcp://host:0` 允许 OS 选择端口，再从 `LAST_ENDPOINT` 读取实际地址。

这些地址不是简单来自某一个 CLI 字段，而是运行时结合：

- DP master IP；
- client 是否与 Engine 同机；
- Engine 是否由本 client 管理；
- external/hybrid LB；
- 本地 IPC 根目录。

---

## 6. MPClient 初始化主干

```text
创建 zmq Context
→ 创建 BackgroundResources
→ 读取 ParallelConfig
→ 计算/接受 EngineZmqAddresses
→ 建 ROUTER + PULL sockets
→ 解析动态端口
→ launch_core_engines(...)
                                      🚩 M4
→ 创建 Msgpack encoder/decoder
→ 计算所管理的 engine ranks/identities
→ 用同步 shadow socket 等待 READY
→ _apply_ready_response(...)
→ 启动 EngineCore monitor
                                      🚩 M12
```

`BackgroundResources` 负责在正常 shutdown 或异常构造失败时回收：

- sockets/context；
- EngineCore process manager；
- 后台 task/thread；
- 其他清理 callback。

这是启动异常不泄漏孤儿进程的关键结构。

---

## 7. launch_core_engines

前端把以下信息交给进程管理器：

- `VllmConfig`；
- Executor 类；
- usage/log stats；
- handshake 地址；
- engine index / DP rank；
- tensor IPC queue 等可选资源。

进程创建通常采用 multiprocessing context，并以：

```text
EngineCoreProc.run_engine_core
```

为 target。`VllmConfig` 被序列化进入子进程，不是共享引用。

> 🚩 **M4 · IPC endpoint + EngineCore 进程启动**
>
> 完成条件：前端 socket 已绑定，实际地址已确定，受管理 EngineCore 进程已经 start。此时子进程通常还在 M5–M10，前端不能发送业务请求。

---

## 8. 启动 handshake 的两个层次

容易混淆的地方是“交换地址”和“Engine READY”不是同一语义：

1. **启动地址/元数据握手**：proc 获得 input/output/coordinator 地址，并交换必要元数据；
2. **READY 汇聚**：proc 完成 EngineCore、Worker、KV、Scheduler、线程后，向 client 表示可以工作。

前端在 M4 后阻塞等待第二类结果。大型模型的主要启动耗时发生在这个等待窗口内，但耗时工作实际上在 proc/worker。

---

## 9. READY response 为什么有数据

proc 端真实执行后才知道：

- 实际 `num_gpu_blocks`；
- auto-fit 后的 `max_model_len`；
- 支持的 task；
- Engine/DP 相关元数据；
- 可能的启动诊断信息。

`_apply_ready_response()` 把关键派生值写回前端那份 `VllmConfig`。这是必要的，因为：

```text
parent VllmConfig ──serialize──> child VllmConfig
```

二者不共享普通 Python 对象内存。

DP>1 时 client 要收齐其管理的所有 engine rank 的 READY，并对容量等字段按相应语义汇聚，不能看到一个 rank ready 就认为整个 client ready。

---

## 10. 序列化

控制/请求输出通常使用 msgpack 族编码器：

- encoder 把 `EngineCoreRequest`、utility call 等变为 bytes；
- decoder 恢复 `EngineCoreOutputs`；
- identity/routing frame 与 payload 分开；
- 多模态大 tensor 可使用单独的 tensor IPC 机制，避免所有数据都塞进 msgpack。

“零拷贝”应谨慎表述：ZMQ、msgpack 和 tensor IPC 各有不同的复制边界；不能笼统宣称整条链完全零拷贝。

---

## 11. 运行期异步接收

`AsyncMPClient` 在 MPClient 基础上增加：

- asyncio 兼容的 socket/queue；
- output queue task；
- `get_output_async()`；
- 将 Engine 死亡状态传播给等待中的协程。

输出拉取 task 的静态管理在启动期完成，真正 recv/decode/dispatch 是运行期。

---

## 12. 存活监控

EngineCore 可能因为：

- CUDA OOM；
- NCCL/进程组错误；
- 模型加载异常；
- native kernel 崩溃；
- handshake 超时；
- Worker 意外退出

而死亡。client monitor 的责任是：

- 观察受管子进程；
- 设置 engine-dead 状态；
- 让输出等待者得到异常而不是永久挂起；
- 触发资源清理。

“ZMQ 没有消息”本身不能区分模型仍在加载还是 proc 已死，因此进程监控与超时都需要。

---

## 13. DP 分支

### external LB

每个前端或上游负载均衡器知道如何选择 DP rank，`DPAsyncMPClient` 更接近“指定 rank 转发”。

### internal/hybrid LB

`DPLBAsyncMPClient` 与 `DPCoordinator` 协作：

- 汇聚各 rank 队列/负载；
- 路由请求；
- 处理 coordinator socket；
- 等待所有相关 rank 就绪。

DP 分支改变 socket 数量、identity 和 READY 汇聚，但不改变核心原则：

> 前端 client 只在所有受管 Engine 达到可服务状态后完成构造。

---

## 14. 常见误区

### “EngineCoreClient 是一个实际进程”

不是。它是 API Server 进程中的客户端对象；它管理/连接 EngineCore 进程。

### “M4 表示 Engine 已经 ready”

不是。M4 只表示进程已 start、IPC endpoint 已建立。

### “ROUTER/PULL 都是 connect”

主干前端通常 bind，proc 的 DEALER/PUSH connect。跨节点外部管理分支需按实际地址配置核对。

### “handshake 只发一个 READY 空包”

不完整。启动有地址/元数据交换，最终 ready response 还携带需要同步回前端的派生信息。

---

## 15. Milestone

> 🚩 **M4 · ZMQ 与 EngineCore 进程已启动**
>
> 前端 IPC endpoint 已绑定、子进程已经 start；重型初始化尚未完成。

> 🚩 **M12 · EngineCoreClient 与 AsyncLLM 完成**
>
> client 收齐所有受管 Engine READY，应用派生配置，启动存活监控；`EngineCoreClient.make_async_mp_client()` 返回，AsyncLLM 完成 logger/output handler 等收尾并返回。

M4 与 M12 之间的 proc 端过程见 [A4-EngineCoreProc与跨进程握手.md](A4-EngineCoreProc与跨进程握手.md)。
