# A8 · Scheduler → KVCacheManager → BlockPool

> **阶段**：M9 → M10  
> **进程**：EngineCore 子进程  
> **主要源码**：`vllm/v1/core/sched/`、`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/block_pool.py`

---

## 1. 物理层与逻辑层

进入本阶段前，A7 已完成：

```text
GPU/设备上的 KV tensor：N 个 block slots
```

本阶段创建：

```text
Scheduler 的逻辑账本
├── 哪些请求在等待/运行
├── 哪些 block ID 空闲
├── 哪个请求占哪些 block ID
├── 哪个 hash 可复用哪些 block
└── 本 step 能调度多少 token
```

Scheduler 不在运行期频繁 `cudaMalloc`/`cudaFree` KV tensor；它主要移动逻辑 block 的所有权。

---

## 2. Scheduler 构造输入

```text
Scheduler(
  vllm_config,
  kv_cache_config,              # M8/M9 产物
  structured_output_manager,
  include_finished_set,
  log_stats,
  block_size,
  hash_block_size,
)
```

关键依赖：

- `SchedulerConfig`：budget、policy、chunked prefill、async scheduling；
- `CacheConfig`：prefix caching 等；
- `KVCacheConfig`：实际 groups/num blocks；
- `ModelConfig`：runner、max length、多模态；
- connector/encoder/structured output 配置。

---

## 3. Scheduler 的持久状态

### 请求集合

- request ID → Request；
- waiting queue；
- running 状态/集合；
- finished request IDs；
- skipped waiting 或策略相关队列；
- request wave / DP 状态。

具体容器会随版本调整，稳定概念是：

```text
所有请求索引 + admission queue + active requests + completion state
```

### 资源预算配置

- `max_num_scheduled_tokens`；
- `max_num_seqs`；
- max model length；
- encoder budget；
- block size；
- scheduling policy。

这些是持久配置；每 step 的剩余 token budget 是临时状态。

---

## 4. 调度策略

Scheduler 在初始化时选择 policy/queue 实现，例如：

- FCFS；
- priority；
- 版本支持的其他策略。

policy 决定 waiting request 的比较/出队方式，但不能绕过：

- token budget；
- KV block capacity；
- structured output readiness；
- encoder cache；
- KV connector 状态；
- preemption/recompute 规则。

---

## 5. KVCacheManager

Scheduler 通过 KVCacheManager 使用 KV 资源：

```text
Scheduler
└── KVCacheManager
    ├── KVCacheCoordinator
    │   ├── SingleType manager(s)
    │   └── BlockPool
    ├── request → block mapping
    ├── allocate/free/cache
    └── prefix match
```

主要职责：

- 查询请求已有/cached KV；
- 判断新增 token 需要多少 block；
- allocate slots；
- free 或 cache request blocks；
- 处理 sliding-window/hybrid group；
- 为 Scheduler 提供可调度性判断。

它管理的是逻辑资源，不执行 attention kernel。

---

## 6. 为什么需要 Coordinator

简单模型所有层使用同一种 full-attention KV spec，一个统一 manager 即可。

混合模型可能同时包含：

- full attention；
- sliding-window attention；
- local attention；
- Mamba/SSM state；
- 不同 page/block 约束。

Coordinator 把“一个请求的 token 进度”协调到多个 KV groups，使各类型 manager：

- 使用兼容的 block 数；
- 同步 allocate/free；
- 正确计算最小共同可用容量；
- 向 Scheduler 提供统一接口。

因此不能把 `KVCacheManager == BlockPool`。前者是请求级资源服务，后者是 block 级容器。

---

## 7. BlockPool 初始化

概念结构：

```text
BlockPool
├── blocks[0..N-1]
├── free_block_queue
├── cached_block_hash_to_block
├── null_block?
└── group/usage metadata
```

### blocks

每个 `KVCacheBlock` 通常保存：

- block ID；
- ref count；
- block hash；
- token/hash metadata；
- free queue linkage/state。

它不直接持有完整 KV tensor 数据；block ID 映射到 A7 创建的物理 slot。

### free_block_queue

启动时绝大多数 blocks 都为空闲：

```text
head → block 0 → block 1 → ... → block N-1 → tail
```

运行期 allocate 从一端取，free/evict 按策略归还。实现可能使用 intrusive/free queue 以降低对象分配和删除成本。

### cached_block_hash_to_block

prefix caching 启用时：

```text
block hash → 一个或多个 KVCacheBlock
```

用于定位可复用的已计算前缀。hash 命中仍需做必要的一致性/链式匹配，不能把 hash 表简单理解为“hash 命中必复用”。

### null block

某些路径需要一个不参与普通容量的占位 block，用于 padding、无效位置或统一接口。它不是给真实请求分配的普通空闲 block。

---

## 8. block ID 如何连到物理 tensor

简化：

```text
Scheduler allocate logical block_id = 17
            │
            ▼
ModelRunner receives slot mapping
            │
            ▼
attention writes:
kv_tensor[group/layer][17, token_offset, ...]
```

Scheduler 不传递一块 Python tensor 给请求，而是生成 block table/slot mapping，GPU kernel 据此读写预分配存储。

---

## 9. KV cache groups

`KVCacheConfig.kv_cache_groups` 把共享同种 spec/layout 的层组织起来。

每个 group 至少包含：

- spec；
- layer names；
- block/page 语义。

Scheduler 看到的是 group 级逻辑，Worker 看到的是层到物理 tensor 的注册。二者由同一个 M8 配置对应起来。

---

## 10. prefix caching 初始化

静态期：

- 配置 hash 算法；
- 创建 request block hasher；
- 创建 cached-block 索引；
- 所有普通 blocks 初始 free；
- 特殊 hash/null 状态初始化。

运行期：

- prompt 分块并计算 chained hash；
- 查最长 cached prefix；
- 增加 ref count；
- 新算出的完整 block 加入 cache；
- 内存压力下驱逐无引用 cached block。

prefix caching 提高复用率，不增加物理 blocks 总数。

---

## 11. Encoder Cache Manager

多模态/encoder-decoder 路径还会创建 encoder cache 管理器：

- 记录已计算 encoder outputs；
- 受独立 budget 限制；
- Scheduler 决定本 step 哪些 encoder inputs 需要执行；
- GPUModelRunner 持有对应物理/设备侧 cache。

这与 autoregressive KV Cache 是相关但不同的资源。

---

## 12. KV Connector

P/D 分离或外部 KV 传输启用时，Scheduler 创建 connector：

- 决定请求 KV 是本地命中、需 load、需 save；
- 将传输等待纳入调度；
- 请求结束时释放外部/本地传输资源；
- 与 Worker 端 connector/aggregator 交换 metadata。

connector 初始化跨 Scheduler 和 Executor，因此 EngineCore 在 Scheduler 构造后还需要做握手 metadata 连接。

---

## 13. 运行期 schedule 预览

每个 step 概念上：

```text
创建本 step 临时预算/结果桶
→ 先处理 running requests
→ 计算已有 cached/computed tokens
→ KVCacheManager 判断/分配新 blocks
→ 再接纳 waiting requests
→ 处理 preemption / connector / encoder
→ 形成 SchedulerOutput
```

以下不是静态初始化产物：

- 本 step `token_budget`；
- 本 step `scheduled_*` lists；
- 本 step block table 更新；
- `new_step_starts()` 等 step hook。

但它们操作的 Scheduler/KVCacheManager/BlockPool 是 M10 的持久对象。

---

## 14. 容量与并发

可支持并发不是简单的：

```text
num_blocks / max_model_len
```

还受到：

- 每个请求实际 context；
- chunked prefill；
- prefix cache 命中；
- speculative tokens；
- sliding window；
- hybrid groups；
- encoder budget；
- TP/PP；
- scheduler max seqs/tokens

影响。M8 提供理论 cache capacity，Scheduler 在运行期做动态 admission。

---

## 15. 常见误区

### “BlockPool 分配 GPU 显存”

不准确。物理显存在 M9 由 Worker/ModelRunner 分配；BlockPool 在 M10 建逻辑对象。

### “一个 block 对应模型的一层”

不准确。逻辑 block ID 会在相关 layers/groups 的物理 tensors 中对应一组 slot。

### “prefix cache 有一套独立 KV tensor”

不准确。它复用普通 pool 中无引用但保留 hash 的 blocks。

### “Scheduler 初始化就创建了每步 token budget”

不准确。Scheduler 保存上限和持久容器；每 step 的剩余 budget 是临时值。

---

## 16. Milestone

> 🚩 **M10 · Scheduler 与逻辑 KV 管理链就绪**
>
> 请求容器、policy、KVCacheManager、Coordinator、BlockPool、prefix/encoder/connector 等适用结构已经建立。EngineCore 已具备运行 `step_fn` 所需的执行端和调度端。

下一阶段：[A9-辅助组件与可选分支.md](A9-辅助组件与可选分支.md)。
