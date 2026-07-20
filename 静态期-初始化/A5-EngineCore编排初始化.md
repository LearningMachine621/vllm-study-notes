# A5 · EngineCore 编排初始化

> **阶段**：M5 → M10  
> **进程**：EngineCore 子进程  
> **主要源码**：`vllm/v1/engine/core.py` 中 `EngineCore`

---

## 1. EngineCore 是什么

`EngineCore` 是 V1 的内循环容器：

```text
EngineCore
├── model_executor
├── Scheduler
├── StructuredOutputManager
├── multimodal receiver cache
├── request block hasher
├── batch queue
├── abort queue
└── step_fn
```

它同时持有“执行”和“调度”两侧：

- Executor 回答“如何在设备/Worker 上执行”；
- Scheduler 回答“本 step 执行哪些请求、占用哪些逻辑 KV blocks”；
- EngineCore 把二者串成 `schedule → execute → update`。

EngineCoreProc 在它外面增加 ZMQ 和进程循环。

---

## 2. 初始化顺序

v0.24.0 主干：

```text
load_general_plugins()
→ 保存 VllmConfig / log_stats
→ model_executor = executor_class(vllm_config)
                                      M6 + M7
→ _initialize_kv_caches(vllm_config)
                                      M8 + M9
→ StructuredOutputManager(vllm_config)
→ Scheduler(vllm_config, kv_cache_config, ...)
                                      M10
→ 初始化 connector/mm/hash/batch/abort 等辅助状态
→ freeze_gc_heap()
→ enable_envs_cache()
```

顺序不是风格偏好，而是依赖关系。

---

## 3. 为什么先加载插件

EngineCore 在独立 Python 进程中，父进程已经 import/注册的状态不应被假定自动可见，尤其是 spawn 模式。

插件可能注册：

- 平台或设备实现；
- quantization / model loader；
- attention/kernel；
- scheduler 或其他扩展。

因此 proc 端需要在实例化依赖前加载通用插件。

---

## 4. 为什么 Executor 最先

KV Cache 容量不能仅由配置静态计算。必须先有能够在真实设备上执行的模型：

```text
Executor
→ Worker/device/distributed init
→ GPUModelRunner
→ load model
→ get_kv_cache_specs()
→ determine_available_memory()
```

Executor 构造完成后，模型权重已加载，但 KV Cache 尚未完成。

---

## 5. 为什么 KV 初始化在 Scheduler 前

Scheduler 的逻辑账本需要知道：

- 有哪些 KV cache groups；
- 每组 block size/spec；
- 一共有多少 block；
- prefix caching / hybrid cache 如何映射；
- 是否根本没有 KV Cache（某些 encoder/attention-free model）。

这些只有 M8/M9 后才确定。因此：

```text
物理资源事实（A7）
        ↓
逻辑资源管理器（A8）
```

不能先创建 BlockPool 再猜容量。

---

## 6. _initialize_kv_caches 的输出

输入：

- `VllmConfig`；
- 已构造的 `model_executor`；
- Worker 上已加载的模型。

输出：

- 面向 Scheduler 的 `KVCacheConfig`；
- 回写到 `cache_config` 的实际容量字段；
- Worker 上已注册/分配的物理 KV Cache；
- 已完成必要 compile/warmup 的模型执行路径。

详细拆分见 A7。

---

## 7. Scheduler 构造前的 block size 解析

KV cache 可能不是单一 full-attention 类型。EngineCore 会根据生成的 KV config 解析：

- scheduler 使用的 block size；
- prefix-cache hash 使用的 hash block size；
- hybrid/mixed cache groups 的兼容尺寸。

这样 Scheduler 不需要理解每个模型层的底层 tensor layout，只消费规范化后的逻辑配置。

---

## 8. 无 KV Cache 的模型

不是所有 runner 都有生成模型式 KV Cache。若 `kv_cache_groups` 为空：

- capacity 可以为零；
- Scheduler 仍可存在；
- 某些依赖 KV 的优化必须关闭；
- chunked prefill 等功能可能不适用。

因此“EngineCore 初始化一定分配 N 个 GPU blocks”不是普遍事实。

---

## 9. Scheduler 创建

EngineCore 通过：

```text
Scheduler = scheduler_config.get_scheduler_cls()
Scheduler(
  vllm_config,
  kv_cache_config,
  structured_output_manager,
  block_size,
  hash_block_size,
  ...
)
```

来实例化默认或自定义 Scheduler。

Scheduler 内部再创建：

- waiting/running/request map；
- KVCacheManager；
- KVCacheCoordinator；
- BlockPool；
- encoder cache manager；
- KV connector；
- scheduling policy。

M10 必须覆盖这条完整逻辑链，而不仅是 `Scheduler(...)` 刚开始执行。

---

## 10. Scheduler 之后的连接动作

某些组件只有 Scheduler 建好才能继续连接：

- 如果 scheduler 创建了 KV connector，Executor 端要初始化相应 output aggregator；
- 从各 Worker 收集 connector handshake metadata；
- 将 PP/TP-aware metadata 合并给 Scheduler connector；
- 创建 multimodal receiver cache；
- 根据 prefix caching/connector 创建 request block hasher。

因此“Scheduler 是 EngineCore.__init__ 的绝对最后一行”不准确。它是最后一个重量级核心组件，后面还有重要的连接和收尾。

---

## 11. batch queue 与 step_fn

当 `max_concurrent_batches > 1`：

```text
self.batch_queue = deque(...)
self.step_fn = self.step_with_batch_queue
```

否则：

```text
self.batch_queue = None
self.step_fn = self.step
```

batch queue 支持异步 schedule/execute 和 PP 降低 pipeline bubble。启动期只选择并保存 step function；真正执行属于运行期。

---

## 12. request_block_hasher

当 prefix caching 或 KV connector 需要块身份时，EngineCore：

1. 从配置取得 hash 算法；
2. 初始化特殊的 `None` hash；
3. 结合 hash block size 创建 request hasher；
4. 在请求进入 EngineCore 时生成 block hash。

物理 block tensor 不存“完整 prompt 文本”；BlockPool 使用 hash 建立逻辑前缀索引。

---

## 13. GC 与环境变量缓存

启动期创建了大量长期存活对象。末尾：

- `freeze_gc_heap()` 把启动堆标为静态，减少老年代 GC 扫描与延迟抖动；
- 可选 GC debug hook；
- `enable_envs_cache()` 假定此后环境变量不再动态覆盖，减少重复解析。

这些操作应放在大多数持久对象创建后。shutdown 时需要相应 unfreeze/cleanup，避免 in-process 重建时泄漏。

---

## 14. EngineCore 初始化产物

| 产物 | 物理/逻辑 | 生命周期 |
|---|---|---|
| model weights | 物理设备资源 | Engine 生命周期 |
| KV tensors | 物理设备资源 | Engine 生命周期 |
| Scheduler queues | 逻辑状态 | Engine 生命周期 |
| BlockPool blocks | 逻辑账本 | Engine 生命周期 |
| request hasher | 逻辑工具 | Engine 生命周期 |
| structured output manager | 管理器，backend 可延迟 | Engine 生命周期 |
| step_fn | 方法选择 | Engine 生命周期 |
| per-step token budget | 临时逻辑状态 | **不是静态产物** |

---

## 15. 运行期 step 预览

```text
EngineCore.step()
→ scheduler.has_requests()
→ scheduler.schedule()
→ model_executor.execute_model()
→ sample_tokens()
→ scheduler.update_from_output()
→ EngineCoreOutputs
```

这个流程解释 Executor/Scheduler 为何都由 EngineCore 持有，但不属于初始化阶段。

---

## 16. 失败定位

| 构造位置 | 常见根因 |
|---|---|
| plugin load | 插件 import/注册 |
| executor | Worker spawn、CUDA、NCCL、权重 |
| KV init | profiling OOM、容量不足、cache layout |
| scheduler | block/group/connector 配置不一致 |
| connector metadata | P/D、跨 rank 元数据 |
| GC/env 收尾 | 通常不是主要耗时点 |

---

## 17. 阶段完成条件

EngineCore 没有单独新增一个总 milestone；它聚合：

```text
M6 Worker/设备 ready
M7 model ready
M8 KV capacity ready
M9 KV physical/warmup ready
M10 Scheduler/logical managers ready
```

当 M10 及辅助收尾完成，`EngineCore.__init__()` 才返回给 EngineCoreProc。

下一阶段：[A6-Executor-Worker-GPUModelRunner.md](A6-Executor-Worker-GPUModelRunner.md)。
