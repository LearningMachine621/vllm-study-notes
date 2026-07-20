# A7 · KV Cache Profiling、容量、分配与 Warmup

> **阶段**：M7 → M8 → M9  
> **进程**：EngineCore 协调，Worker/GPUModelRunner 执行  
> **主要源码**：`EngineCore._initialize_kv_caches()`、`vllm/v1/core/kv_cache_utils.py`、Worker/ModelRunner KV 接口

---

## 1. 为什么这是两个 milestone

旧式描述常把下面四件事统称为“profiling”：

1. 获取每层 KV 规格；
2. 测量可用于 KV 的显存；
3. 计算 block/group 容量；
4. 创建 KV tensor 并 compile/warmup。

但它们的产物不同：

```text
规格 + 测量 + 计算              物理初始化 + 预热
          │                              │
          ▼                              ▼
     🚩 M8 容量已知                  🚩 M9 可执行
```

发生 OOM 时，区分 M8 前后的失败对定位非常重要。

---

## 2. 总流程

v0.24.0 `EngineCore._initialize_kv_caches()` 可概括为：

```text
register_all_kvcache_specs(vllm_config)
→ model_executor.get_kv_cache_specs()
→ 检查 non-causal/no-KV 等模型特征
→ model_executor.determine_available_memory()
→ get_kv_cache_configs(...)
→ 必要时同步 auto-fit max_model_len
→ generate_scheduler_kv_cache_config(...)
→ 回写 num_gpu_blocks/block_size/capacity
                                      🚩 M8
→ model_executor.initialize_from_config(kv_cache_configs)
  → 注册/创建 KV tensors
  → compile/warmup model
                                      🚩 M9
→ 返回 Scheduler 使用的 KVCacheConfig
```

---

## 3. KVCacheSpec：每层需要什么

Worker/ModelRunner 根据模型层返回 KV spec。spec 描述的是布局需求而不是已分配 tensor，例如：

- full attention；
- sliding-window/local attention；
- Mamba/SSM state；
- hybrid cache；
- block size；
- head 数、head size、dtype；
- 非因果 attention 标志；
- 每个 block 的字节数/形状信息。

多 Worker 返回：

```text
list[dict[layer_name, KVCacheSpec]]
```

EngineCore 统一收集，是因为 Scheduler 和全局容量计算必须看到完整 rank/group 视图。

---

## 4. 模型能力会反向调整配置

如果某些 layer 标记为 non-causal：

- chunked prefill 的因果假设可能不成立；
- prefix caching 的前缀复用假设可能不成立；
- EngineCore 会关闭不兼容能力并记录原因。

无 KV 模型：

- 不执行普通 KV 显存预算；
- available memory 可按零处理；
- 生成空 groups；
- Scheduler 走无 KV 路径。

这体现了 A1 所说的“VllmConfig 仍允许启动期派生写回”。

---

## 5. determine_available_memory()

目标不是读取 `torch.cuda.mem_get_info()` 的某一瞬间，而是估算：

> 在满足用户显存利用率上限并覆盖模型峰值执行开销后，能稳定留给 KV Cache 的字节数。

概念公式：

```text
目标可用显存
= total_gpu_memory × gpu_memory_utilization
 - 模型权重/持久占用
 - profiling forward 的峰值 activation
 - non-torch / CUDA / communicator 占用
 - 需要保留的执行 workspace
```

具体 allocator accounting 和 compile 策略随版本变化，不能把这个概念式当源码逐项减法。

---

## 6. profiling run 做什么

ModelRunner 构造最大或代表性 dummy batch，执行足以暴露峰值的路径：

- language model forward；
- attention workspace；
- sampling/输出相关必要路径；
- 多模态模型的 encoder peak；
- 可能的 draft/spec 路径；
- lazy kernel/CUDA runtime 初始化。

profiling 输入不是用户请求，不进入 Scheduler，也不会保留为运行时 sequence。

---

## 7. 为什么权重大小不够

即便两个模型 checkpoint 大小相同，也可能因以下因素得到不同 KV 容量：

- 层数、KV heads/head size；
- dtype；
- max model length；
- attention backend；
- activation peak；
- TP/PP 切分；
- quantization workspace；
- CUDA/NCCL 版本；
- 多模态 encoder；
- compile/CUDA Graph 配置；
- 当前 GPU 上其他进程占用。

因此默认 `num_gpu_blocks` 必须在目标部署上实测。

---

## 8. 从字节预算到 KVCacheConfig

`get_kv_cache_configs(...)` 结合：

- 每个 Worker 的 available bytes；
- 每层 spec；
- block size；
- page 对齐与 group 约束；
- max model length；
- 用户 override/auto-fit；
- hybrid KV cache manager 策略

生成各 Worker 的物理 KV 配置。

核心关系：

```text
single_block_bytes
= Σ(layer/group 在一个 block 中的 tensor bytes)

num_blocks
≈ floor(available_kv_bytes / single_block_bytes)
```

hybrid/multiple groups 下不能用一个简单的统一 `single_block_bytes` 覆盖全部情况，实际要满足各 group 的共享/对齐约束。

---

## 9. 一个 full-attention block 的概念大小

对某层、未压缩标准 KV：

```text
bytes_per_block_per_layer
= 2                          # K + V
 × block_size                # token slots
 × num_kv_heads_per_rank
 × head_size
 × bytes_per_element
```

再乘本 rank 负责的相关层数，并考虑实际 layout/alignment。

例如 TP 会改变每 rank 的 KV heads，PP 会改变每 rank 的层数，所以每 rank 的 spec/容量必须由 Worker 返回，不能只在前端按全模型公式估算。

---

## 10. auto-fit 与 max_model_len

若配置允许 auto-fit，容量不足以支持原 max model length 时可能下调：

```text
max_model_len_before
→ get_kv_cache_configs()
→ max_model_len_after
```

Worker 早于 profiling 创建，可能缓存了原值。因此 EngineCore 需要通过 collective RPC 把新值同步到所有 Worker。

这不是修改 Hugging Face 模型结构，而是调整 Engine 允许的最大请求长度。

---

## 11. 生成 Scheduler 视图

物理 Worker 配置会规范化为 Scheduler 使用的：

```text
KVCacheConfig
├── num_blocks
├── kv_cache_groups
│   ├── group 0 spec + layer names
│   └── ...
└── block/group metadata
```

随后回写：

- `cache_config.num_gpu_blocks`；
- 有效 `block_size`；
- `kv_cache_size_tokens`；
- `kv_cache_max_concurrency`。

这些是可观测容量，不等于 Scheduler 已经创建 BlockPool。

> 🚩 **M8 · KV 规格、显存预算和容量完成**
>
> 已知道每个 group 的 layout 和可用 block 数，配置派生值已更新；GPU KV tensor 和完整 warmup 尚未保证完成。

---

## 12. initialize_from_config()

EngineCore 对 Executor 调用：

```text
model_executor.initialize_from_config(kv_cache_configs)
```

Executor 将每个 Worker 对应的配置发给它。Worker/ModelRunner 负责：

- 创建/绑定 KV cache tensor；
- 为 attention layer 注册 KV cache；
- 建立 layer-name 到 tensor/group 的映射；
- 初始化相关 backend metadata；
- 校验形状、dtype 和容量；
- compile/warmup 执行模型；
- 按配置捕获 CUDA Graph（若启用）。

v0.24.0 的 EngineCore 日志把这一整段概括为：

```text
profile, create kv cache, warmup model
```

---

## 13. 物理 KV 与逻辑 block

M9 产生物理存储：

```text
GPU tensor storage
[block 0][block 1]...[block N-1]
```

M10 产生逻辑账本：

```text
KVCacheBlock(block_id=0...)
free_block_queue
cached_block_hash_to_block
request → allocated block IDs
```

BlockPool 中的 `KVCacheBlock` 不是一块独立 GPU tensor，它是对物理 slot 的逻辑句柄/元数据。

---

## 14. compile 与 warmup

目的包括：

- 触发 lazy kernel 初始化；
- 建立 torch.compile graph/code cache；
- 预分配固定 buffer；
- 按 capture sizes 运行 warmup；
- 捕获 CUDA Graph；
- 验证最终 KV tensor 能被执行路径正确访问；
- 把大部分一次性开销移出首个用户请求。

不同 compilation level / `enforce_eager` 下会跳过部分动作，但 M9 的通用完成条件是：

> 当前配置要求的物理 KV 注册和执行准备全部完成。

---

## 15. CUDA Graph 与 profiling 的准确关系

不应笼统写成“profiling 一定先精确扣除固定 2GB CUDA Graph 开销”。

更准确的表述：

- vLLM 必须为最终执行路径留下足够显存；
- profiling、KV 分配、compile 和 capture 在一个受控初始化协议中协作；
- compile/capture 的真实行为由 `CompilationConfig`、模型和 backend 决定；
- 固定 “~2GB” 不是跨模型/版本成立的常量；
- M9 才能证明当前配置的 KV + warmup/capture 组合实际成功。

---

## 16. prefix caching 与容量

prefix caching 不创建第二份物理 KV Cache。它增加：

- block hash；
- cached block 索引；
- 引用计数/复用规则；
- eviction 逻辑。

物理 slot 总量仍由 M8/M9 确定，逻辑复用机制在 M10 创建。

---

## 17. 常见失败

| 失败 | 含义 |
|---|---|
| profiling OOM | M8 前，峰值执行无法完成或外部占用变化 |
| no available memory | 预算不足以形成有效容量 |
| max sequence cannot fit | 最小并发/长度需求超容量 |
| KV shape/dtype mismatch | spec 与 ModelRunner/backend 注册不一致 |
| allocate OOM | M8 计算后到 M9 分配时资源状态或估算不成立 |
| compile/capture OOM | KV tensor 已建，但最终执行准备失败，M9 未到 |

排障时记录：

- GPU 型号/总显存；
- `gpu_memory_utilization`；
- TP/PP；
- max model length；
- KV dtype/block size；
- compilation/enforce eager；
- 其他进程显存；
- 最后完成的 milestone。

---

## 18. Milestone

> 🚩 **M8 · KV 容量完成**
>
> specs、available memory、groups、num blocks 和 cache capacity 已确定并回写。

> 🚩 **M9 · KV 物理初始化与执行预热完成**
>
> 所有 Worker 已按 config 创建/注册 KV 存储，并完成当前 compilation 配置要求的 warmup/CUDA Graph 准备。

下一阶段：[A8-Scheduler-KVCacheManager-BlockPool.md](A8-Scheduler-KVCacheManager-BlockPool.md)。
