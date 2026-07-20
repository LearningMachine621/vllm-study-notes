# 07 与 vLLM 差异总结

> nano-vllm 1450 行 vs vLLM v1 ~30000 行

## 一、架构差异总览

```
vLLM v1 架构：
  API Server (FastAPI)
    ↓ ZMQ
  EngineCoreProc (子进程)
    ↓ 函数调用
  EngineCore
    ├── Scheduler (900 行)
    ├── KVCacheManager (400 行)
    ├── ModelRegistry → 任意模型
    └── GPUModelRunner (7000 行)
         ├── TP (NCCL)
         ├── PP (跨节点)
         ├── CUDA Graph
         └── 多模态

nano-vllm 架构：
  LLM (主进程)
    ├── LLMEngine (90 行)
    │     ├── Scheduler (92 行)
    │     ├── BlockManager (120 行)
    │     └── ModelRunner (257 行)
    │           ├── Qwen3ForCausalLM (硬编码)
    │           ├── TP (NCCL + SHM)
    │           └── CUDA Graph
    └── 子进程 (SHM IPC)
```

## 二、逐模块差异

### 2.1 入口层

| | nano-vllm | vLLM |
|--|----------|------|
| 对外接口 | `LLM` (5 行) | `AsyncLLM` (~500 行) |
| API Server | 无 | FastAPI + OpenAI 兼容 |
| 流式输出 | 无 | SSE streaming |
| 并发模型 | 同步阻塞 | async/await |
| IPC | 无（同进程） | ZMQ (REQ/ROUTER) |

### 2.2 引擎层

| | nano-vllm | vLLM |
|--|----------|------|
| EngineCore | `LLMEngine` (90 行) | `EngineCore` (~2100 行) |
| 多进程 | spawn + SHM | spawn + ZMQ |
| DP | 不支持 | 支持 (DPEngineCoreProc) |
| PP | 不支持 | 支持 |
| 状态持久化 | 无 | 有 (EngineCoreProc restart) |

### 2.3 调度器

| | nano-vllm | vLLM |
|--|----------|------|
| 行数 | 92 行 | ~900 行 |
| 队列 | `deque` × 2 | `FCFSQueue` + `PriorityQueue` |
| Prefill 优先 | 是 | 是 |
| Chunked Prefill | 简化版 | 完整版 |
| Preemption | 抢最后一个 | 按 priority 抢 |
| Token Budget | `max_num_batched_tokens` | `max_num_batched_tokens` + 条件编译 |
| 异步调度 | 无 | `async_scheduling` |
| Connector | 无 | DP/PP connector |

### 2.4 KV Cache 管理

| | nano-vllm | vLLM |
|--|----------|------|
| 行数 | 120 行 | ~700 行 |
| 数据结构 | `Block` 对象 + `dict` | `KVCacheManager` + `BlockPool` |
| Prefix Cache | xxhash 链式 | hash |
| Ref Count | 有 | 有 |
| 多模态 KV | 不支持 | 支持 |

### 2.5 模型执行

| | nano-vllm | vLLM |
|--|----------|------|
| 行数 | 257 行 | ~7000 行 |
| 模型 | 硬编码 Qwen3 | 任意 HF 模型 |
| 权重加载 | safetensors + packed mapping | safetensors + GGUF + GPTQ |
| CUDA Graph | 有 | 有（更复杂） |
| TP | NCCL | NCCL |
| PP | 不支持 | 支持 |
| 多模态 | 不支持 | 支持 |
| LoRA | 不支持 | 支持 |

### 2.6 注意力层

| | nano-vllm | vLLM |
|--|----------|------|
| KV 写入 | Triton kernel | CUDA kernel |
| KV 读取 | flash_attn | flash_attn + paged_attention kernel |
| PagedAttention | 有（通过 block_table） | 有 |

### 2.7 线性层

| | nano-vllm | vLLM |
|--|----------|------|
| TP 实现 | Python all_reduce | NCCL kernel |
| 支持类型 | Column/Row/Merged/QKV | 同左 + 更多 |
| weight_loader | 有 | 有 |

## 三、nano-vllm 简化的地方

### 3.1 完全省掉的

| 功能 | 为什么省掉 |
|------|----------|
| API Server | 不需要 HTTP 接口 |
| 异步/流式 | 同步调用更简单 |
| DP/PP | 只支持 TP |
| 多模态 | 只支持文本 |
| LoRA | 不需要适配器 |
| GGUF/GPTQ | 只支持 safetensors |
| Structured Output | 不需要 JSON schema |
| 多模型注册 | 硬编码 Qwen3 |

### 3.2 简化实现的

| 功能 | 简化方式 |
|------|---------|
| IPC | SHM + Event 替代 ZMQ |
| 调度 | deque 替代 PriorityQueue |
| 模型加载 | 简单的 packed mapping |
| 输出处理 | 无（直接返回 token_ids） |
| 错误处理 | 几乎没有 |
| 日志/监控 | 无 |

## 四、nano-vllm 可借鉴的地方

### 4.1 SHM IPC 方案

```python
# 1MB SharedMemory + Event 同步
# 适用于：TP 场景，数据量小（token ids + block table）
# 不适用：大 payload（多模态 features）
```

### 4.2 统一 KV Cache Tensor

```python
# [2, layers, blocks, block_size, kv_heads, head_dim]
# 一次分配，按 layer_id 索引
# 比 vLLM 的分层管理更简单
```

### 4.3 pickle 序列化优化

```python
# prefill: 传完整 token_ids
# decode: 只传 last_token
# 利用 SHM 的有限空间
```

### 4.4 xxhash prefix cache

```python
# 链式 hash: block[i] 的 hash 包含 block[i-1] 的 hash
# 比 SHA 快 10x，碰撞率足够低
```

## 五、面试要点

### 5.1 能讲清楚的

1. **PagedAttention 原理**：block_table → 非连续 KV 存储 → 无碎片
2. **Prefix Cache**：hash 链 → block 级别匹配 → ref_count 共享
3. **CUDA Graph**：录制 → 重放 → 减少 CPU launch 开销
4. **Chunked Prefill**：长 prompt 分多步 → 不阻塞 decode
5. **Preemption**：显存不够 → 抢占最低优先级 → 回到 waiting

### 5.2 能对比的

1. **vLLM vs nano-vllm**：30000 行 vs 1450 行，核心功能 1:1 对应
2. **SHM vs ZMQ**：SHM 简单但有 1MB 限制，ZMQ 灵活但更复杂
3. **flash_attn_varlen_func vs with_kvcache**：prefill 变长 vs decode 固定 bs

### 5.3 能扩展的

1. **DeepSeek-V2 MLA**：改 BlockManager（压缩 KV）+ Attention（latent KV）
2. **Workload-aware Scheduler**：改 Scheduler（priority 分层 + token budget）
3. **Session-aware Prefix Cache**：改 KVCacheManager（session 级 block 保留）

## 六、代码量统计

| 模块 | nano-vllm | vLLM v1 | 比例 |
|------|----------|---------|------|
| 入口 | 5 | ~500 | 1% |
| 引擎 | 90 | ~2100 | 4% |
| 调度 | 92 | ~900 | 10% |
| KV Cache | 120 | ~700 | 17% |
| 模型执行 | 257 | ~7000 | 4% |
| 注意力 | 75 | ~500 | 15% |
| 线性层 | 156 | ~400 | 39% |
| 其他层 | 197 | ~600 | 33% |
| 模型 | 216 | ~2000 | 11% |
| 工具 | 55 | ~500 | 11% |
| **总计** | **1450** | **~30000** | **5%** |

用 5% 的代码实现了 vLLM 80% 的核心功能。
省掉的 20% 是：异步、HTTP、DP/PP、多模态、LoRA、错误处理。
