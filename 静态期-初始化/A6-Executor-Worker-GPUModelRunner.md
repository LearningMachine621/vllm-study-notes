# A6 · Executor → Worker → GPUModelRunner

> **阶段**：M5 → M6 → M7  
> **进程**：EngineCore 进程和/或 Worker 进程  
> **主要源码**：`vllm/v1/executor/`、`vllm/v1/worker/`

---

## 1. 三层职责

```text
Executor
└── 决定 Worker 数量、位置、RPC/collective 方式
    └── Worker
        └── 管理 rank、device、distributed env、内存快照
            └── GPUModelRunner
                └── 模型、输入 batch、采样、compile/CUDA Graph、KV 接口
```

### Executor

回答“在哪些进程/设备执行同一个方法”。

### Worker

回答“本 rank 的设备环境如何初始化并执行 Worker 级 RPC”。

### GPUModelRunner

回答“如何把 SchedulerOutput 转换为 GPU forward/sample，以及模型和 KV tensor 如何组织”。

---

## 2. Executor 类选择

Executor 类在 A2 的 `AsyncLLM.from_vllm_config()` 附近通过配置选出，实例在 EngineCore 中创建。

| 场景 | 典型实现 | 进程特征 |
|---|---|---|
| 单 rank 本地 | `UniProcExecutor` | Worker 与 EngineCore 同进程 |
| 本地多 rank | `MultiprocExecutor` | EngineCore 管理多个 Worker 进程 |
| Ray | `RayDistributedExecutor` | Worker/actor 由 Ray 管理 |
| external launcher | 对应 external executor | world 由 torchrun 等建立 |

DP 通常是多个 EngineCore；TP/PP 通常是单个 EngineCore 内部的多个 Worker。两者不能混为“多 GPU 都是 MultiprocExecutor”。

---

## 3. Executor 的统一接口

EngineCore 依赖的典型能力：

- `get_kv_cache_specs()`；
- `determine_available_memory()`；
- `initialize_from_config()`；
- `execute_model()`；
- `sample_tokens()`；
- `collective_rpc()`；
- connector metadata；
- LoRA、profile、sleep/wake、shutdown 等 utility。

不同后端把同一调用分发到本地对象、多进程 Worker 或 Ray actor，使 EngineCore 不依赖具体通信机制。

---

## 4. UniProcExecutor 主干

示意：

```text
Executor.__init__(vllm_config)
→ UniProcExecutor._init_executor()
  → WorkerWrapperBase(rpc_rank=0)
  → init_worker(kwargs)
      → Worker(...)
  → init_device()
      → CUDA/distributed/GPUModelRunner
                                      🚩 M6
  → load_model()
      → GPUModelRunner.load_model()
                                      🚩 M7
```

`WorkerWrapperBase` 先作为稳定 RPC 外壳存在，再动态构造平台对应 Worker。

---

## 5. Worker 初始化分三层

### 5.1 `Worker.__init__`

主要保存配置和 rank 信息：

- `VllmConfig` 及子 config；
- local/global/rpc rank；
- distributed init method；
- driver worker 标志；
- device/model runner 占位。

通常还没有完整 CUDA 副作用。

### 5.2 `init_device()`

负责真实设备环境：

1. 选择 `cuda:{local_rank}`；
2. 设置当前 CUDA device；
3. 检查 dtype/设备能力；
4. 初始化 distributed environment；
5. 初始化 model parallel groups；
6. 设置随机种子；
7. 清理/记录初始显存；
8. 创建 GPUModelRunner。

### 5.3 `load_model()`

调用 ModelRunner 的加载器把权重放到目标设备，并记录权重占用等信息。

把这三层区分开，可避免把“Worker Python 对象已存在”误当成“GPU 和模型已 ready”。

---

## 6. 为什么先初始化 distributed 再创建 ModelRunner

GPUModelRunner 的构造和模型加载可能依赖：

- TP/PP group；
- rank-aware model partition；
- custom all-reduce；
- tensor-parallel layer 的 world size；
- communicator 或 platform backend；
- seed 在各 rank 上的一致/差异策略。

因此正确依赖是：

```text
device
→ distributed process groups
→ model-parallel state
→ GPUModelRunner
→ model shard load
```

否则模型层在构造时无法知道自己加载哪一 shard、与谁 collective。

---

## 7. M6 的精确含义

M6 不是“模型已加载”，而是执行环境可承载模型：

- Worker 对象存在；
- CUDA device 已选择；
- distributed/model-parallel groups 已建立；
- random seed 和平台状态已设置；
- GPUModelRunner 已创建；
- 预分配的输入/中间管理结构已准备。

> 🚩 **M6 · Worker/设备/分布式环境就绪**
>
> 此时可以开始加载模型，但 GPU 上不一定已有完整模型权重。

---

## 8. GPUModelRunner 启动产物

具体字段随模型/backend 变化，但概念上包括：

- model placeholder，随后由 `load_model()` 填充；
- input batch / request-to-token 映射；
- GPU 输入 buffer；
- sampling 组件；
- attention/KV cache 接口；
- multimodal encoder cache/输入结构；
- speculative drafter 或 rejection sampler（可选）；
- compile/cudagraph dispatcher；
- CUDA stream/event 或 platform 相关执行资源。

并非每个 buffer 都在 `__init__` 一次性完成；有些在 KV 初始化和 warmup 阶段按最终容量注册。

---

## 9. 模型加载

`load_model()` 的关键阶段：

```text
选择 loader / architecture
→ 构造模型模块
→ 按 load format 读取 checkpoint
→ 按 TP/PP rank 选择/切分权重
→ dtype/quantization 转换
→ 放到 device
→ 处理 tied weights / post-load
→ 记录模型内存占用
```

加载格式可能是 safetensors、sharded state、dummy、bitsandbytes 等；量化模型还可能创建 scale、workspace 或后处理结构。

“模型加载完成”应以所有受管 Worker 的 `load_model()` 成功返回为准。

---

## 10. M7 与 Executor 构造

在 v0.24.0 主干中，Executor 初始化流程会执行 Worker 的 device init 和 model load。因此：

```text
self.model_executor = executor_class(vllm_config)
```

这一行返回时，受管 Worker 的模型已加载。

> 🚩 **M7 · 模型权重加载完成**
>
> 所有相关 Worker 的模型 shard 已在目标设备上，Executor 可开始 KV spec/profiling。KV Cache 和执行 warmup 尚未完成。

---

## 11. MultiprocExecutor

典型拓扑：

```text
EngineCoreProc
└── MultiprocExecutor
    ├── Worker process rank 0
    ├── Worker process rank 1
    └── ...
```

启动额外包含：

- 选择 multiprocessing context；
- 为每个 rank 构造 WorkerProc；
- 建立 RPC queue/socket；
- 等待 Worker ready；
- 广播 `init_worker`、`init_device`、`load_model`；
- 汇聚异常和返回值；
- 建立 Worker 存活监控。

某些 rank 可与 driver/EngineCore colocate，具体以配置和实现为准。

M6/M7 的完成条件仍是“所有参与 rank 完成”，不是 rank 0 单独完成。

---

## 12. collective_rpc

Executor 把 EngineCore 的一个逻辑调用变成多 Worker 调用：

```text
EngineCore
→ model_executor.collective_rpc("method", ...)
  ├── worker rank 0.method(...)
  ├── worker rank 1.method(...)
  └── ...
→ 汇聚 list[result]
```

启动期常用于：

- 获取 KV specs；
- 测可用显存；
- 初始化 KV Cache；
- 同步 auto-fit 后的 max model length；
- 初始化 connector metadata。

它不是 NCCL collective 的同义词；它是 Executor 控制面的“对所有 Worker 调方法”，方法内部才可能执行 NCCL collective。

---

## 13. MemorySnapshot

Worker 在合适阶段记录显存状态，用于之后区分：

- 总显存；
- 启动/模型权重后的已用与空闲；
- profiling 前后 peak/non-torch memory；
- 用户指定 `gpu_memory_utilization` 的目标预算。

不能用简单公式“总显存 - 权重文件大小”推导 KV 容量，因为还存在：

- CUDA context；
- allocator 保留；
- activation peak；
- attention/kernel workspace；
- NCCL buffer；
- compile/CUDA Graph 相关资源；
- non-torch allocations。

---

## 14. 多模态分支

GPUModelRunner 可能额外创建：

- encoder model/执行路径；
- encoder input/cache；
- multimodal token/feature 映射；
- tensor IPC 接收后的设备拷贝结构；
- encoder profiling input。

这会影响 M8 的峰值测量和 KV 预算，不能把纯文本 profiling 数值直接套给多模态模型。

---

## 15. 投机解码分支

可能增加：

- draft model 或 n-gram drafter；
- spec token buffer；
- rejection sampling/验证；
- target/draft 间状态；
- 更复杂的 compile/capture 路径。

它改变 GPUModelRunner 和运行期 step，但 Scheduler/EngineCore 的主干持有关系不变。

---

## 16. Ray/external launcher

变化的是：

- 谁创建 Worker；
- RPC 如何实现；
- rank/world 如何获得；
- 生命周期和 failure propagation；
- 配置如何到各进程。

不变的是完成条件：

```text
所有 Worker device/distributed ready → M6
所有模型 shard load ready → M7
```

---

## 17. 本阶段尚未完成的内容

M7 后仍没有：

- 最终 KV Cache 容量；
- 物理 KV tensors；
- compile/CUDA Graph warmup 完成保证；
- Scheduler/BlockPool；
- proc I/O threads；
- 前端 READY。

这就是为什么日志显示“模型加载完成”后仍可能等待较长时间。

---

## 18. Milestone

> 🚩 **M6 · 执行环境完成**
>
> Worker、CUDA device、分布式/model-parallel 环境和 GPUModelRunner 已就绪。

> 🚩 **M7 · 模型完成**
>
> 所有相关 Worker 的模型权重加载成功，Executor 构造返回；下一步可基于真实模型执行 profiling。

下一阶段：[A7-KVCache-Profiling-分配-Warmup.md](A7-KVCache-Profiling-分配-Warmup.md)。
