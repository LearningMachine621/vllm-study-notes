# vLLM 源码理解（前置知识主体）

> 本目录是 vLLM 项目的**理解层**（"学的多、做得少"的那部分），围绕 [[goal]] 组织。
> **目标**：吃透 vLLM serving core，最终能改 scheduler（见 goal 北极星：优先级感知调度 + 过载背压）。

---

## 全局架构

```
[API Server]
  ├── OpenAI protocol / tokenizer / SamplingParams
  └── EngineCoreRequest
          ↓ ZMQ
[Engine Core]
  ├── Request Queue (waiting / running)
  ├── Scheduler  (token budget · prefill/decode/chunked · priority · → KVCacheManager)
  ├── KVCacheManager (prefix cache · block alloc/free · KVCacheCoordinator · BlockPool)
  └── ModelExecutor
          ↓
[GPU Worker]
  ├── ModelRunner (prepare input · attn metadata · block table/slot mapping · forward · sampling)
  └── CUDA kernels / PagedAttention / FlashAttention
          ↓
[Engine Core]  update state · append token · check stop · free KV if finished → output
          ↓
[API Server]  stream response
```

**Request 状态机：**
```
[ WAITING ] ──(资源充足,分配块)──> [ RUNNING ]
     ^                                  |  ^
     |                                  |  |
     | (recompute 策略)                 |  | (资源耗尽,释放块)
     +──────────────────────────────────+  +────> [ PREEMPTED ]
                                            (资源恢复,换入块) |
                                               ↓
[ 任意状态 ] ──(完成/中断/超时)──> [ FINISHED ] (清理所有资源)
```

---

## 目录导航

| 区 | 内容 | 对应 goal 文档 |
|---|---|---|
| [[基础知识/\|基础知识/]] | 进程与对象 · 数据结构 · 有限状态机 | 全局前置 |
| [[静态期-初始化/\|静态期-初始化/]] | A0–A8 启动初始化（主干+扩展已合一） | vLLM_Static_Init |
| [[动态期-请求处理/\|动态期-请求处理/]] | Request 全景/主线 + EngineCore 关键函数 | vLLM_Request_Lifecycle |

> vLLM_Performance_Model 文档：用 [[../../../ccshouldknow/vllm_README\|压测台实验数据]] 喂入（见 goal 末尾实验沉淀）。

---

## 静态期 A0–A8 速览（初始化层级 + 时间顺序）

```
Layer 0  配置生成      VllmConfig
Layer 1  用户入口      AsyncLLM (Renderer · InputProcessor · OutputProcessor · EngineCoreClient)
Layer 2  IPC 传输      AsyncMPClient → DPLBAsyncMPClient · CoreEngineProcManager
Layer 3  引擎进程      EngineCoreProc (ZMQ 守护线程×2) → EngineCore
Layer 4  核心组件      ① Executor → ② KV cache profiling → ③ 辅助结构 → ④ Scheduler
Layer 5a GPU 执行链    Executor → Worker → GPUModelRunner (Sampler · InputBatch · CUDA buffers)
Layer 5b KV 管理链     Scheduler → KVCacheManager → KVCacheCoordinator → BlockPool (blocks · free_queue · hash_map)
```

| 顺序 | 模块 | 原因 |
|---|---|---|
| A0 | VllmConfig | 配置先行，后续模块都读它 |
| A1 | AsyncLLM + 前端 | 用户入口 |
| A2 | EngineCoreClient | IPC 层，启动引擎子进程 |
| A3 | EngineCoreProc | 子进程内先建 ZMQ 通道再建 EngineCore |
| A4 | EngineCore | 内核，按 ①②③④ 创建子模块 |
| A5 | Executor→Worker→GPUModelRunner | ① 最先，profiling 需要 GPUModelRunner |
| A6 | KV Cache Profiling | ② 依赖 A5 的 warmup forward |
| A8 | 辅助结构 | ③ 轻量，不依赖 profiling |
| A7 | Scheduler→KVCacheManager→BlockPool | ④ 最后，依赖 A6 的 kv_cache_config |

---

## ★改造 scheduler 的三处入口（goal 北极星）

| 入口 | 文件 | 看什么 |
|---|---|---|
| 状态机 | [[基础知识/有限状态机\|有限状态机]] | scheduler 的状态转移逻辑 |
| 初始化 | [[静态期-初始化/A7-Scheduler-KVCacheManager-BlockPool\|A7-Scheduler-KVCacheManager-BlockPool]] | scheduler / KVCacheManager / BlockPool 怎么建 |
| 运行 | [[动态期-请求处理/EngineCore关键函数\|EngineCore关键函数]] | step() → schedule() 主循环（token-level decision） |

> 推荐改造方向（详见 goal）：**优先级感知调度 + 过载背压**——fab 场景天然有优先级（告警>问答），且已测出过载拐点。

---

## 学习路线（呼应 goal）

1. **基础**：`基础知识/`（进程与对象 → 数据结构 → 有限状态机）
2. **静态期**：A0 → A8（vLLM 启动时各模块怎么来）
3. **动态期**：Request 全景 → Request 主线 → EngineCore 关键函数（请求怎么跑完一生）
4. **沉淀**：按 goal 的 5 文档收口（Static_Init / Request_Lifecycle / Scheduler_KVCache / Performance_Model / mini-vLLM）

---

## 进度

见 [[goal]] 的 checklist 与「能力阶段路线」。当前：理解🟡 / 实验🟡 / 改造🔲 / 讲述🔲。
