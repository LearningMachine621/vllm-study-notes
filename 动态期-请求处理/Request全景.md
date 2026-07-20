# Request 一生全景

> 以 Request 为主角，看它从诞生到消亡的完整旅程

---

## 诞生

```
用户发一个 HTTP 请求
  {"messages":[{"role":"user","content":"Hello"}], "temperature":0.7}

         ↓  FastAPI 解析为 Pydantic 模型

ChatCompletionRequest
  messages=[...], temperature=0.7, max_tokens=100

         ↓  Renderer: chat template + tokenize

EngineInput
  {"type":"tokens", "prompt_token_ids":[15496, 9906, 198]}

         ↓  InputProcessor: 校验 + 补全参数

EngineCoreRequest (msgspec.Struct)
  request_id="abc-123", prompt_token_ids=[15496,9906,198],
  sampling_params=SamplingParams(temperature=0.7, max_tokens=100)

         ↓  ZMQ IPC 跨进程传输

         ↓  preprocess_add_request(): MM 解析 + grammar init

Request (Python 对象)                    ← 此刻诞生
  request_id="abc-123"
  status=WAITING
  _output_token_ids=[]                   ← 空的，还没生成任何 token
  _all_token_ids=[15496, 9906, 198]      ← prompt 的副本
  num_computed_tokens=0                  ← 还没被计算过
```

---

## 入队

```
Request 诞生后：

  EngineCore.add_request()          ← 门卫：校验 request_id、pooling 任务
    ↓
  Scheduler.add_request()           ← 登记员：注册 + 入队
    ↓
  ┌─────────────────────────────────────────────┐
  │  self.requests["abc-123"] = request          │  ← 全局注册表
  │  self.waiting.add(request)                   │  ← 加入等待队列
  └─────────────────────────────────────────────┘

  Request.status = WAITING
  Request.events += [QUEUED]         ← 记录入队时间
```

---

## 等待

```
Request 在 waiting 队列中静静等待：

  waiting 队列（FCFS 或 Priority）：
    [req-001] [req-002] [abc-123] [req-005] ...
                           ↑
                        我在这里

  等待的条件：
    - GPU 有空闲计算资源
    - KV Cache 有足够 block
    - 没有更高优先级的请求抢占
```

---

## 被调度（step → schedule）

```
每轮 step() 中，Scheduler.schedule() 会：

  1. 从 waiting 队列取出 Request
  2. 计算本轮处理多少 token（chunked prefill 可能分块）
  3. 向 KVCacheManager 申请 block
  4. 生成 SchedulerOutput

  Request.status: WAITING → RUNNING
  Request.num_computed_tokens: 0（新请求）或旧值（已有请求）
  Request.is_prefill_chunk: True/False
```

---

## GPU 执行

```
SchedulerOutput 传给 GPUModelRunner：

  GPU 做的事：
    1. 把 prompt_token_ids 写入 GPU buffer
    2. 用 block_tables 读取 KV Cache（前缀缓存命中则跳过已计算部分）
    3. 执行 Transformer forward
    4. 采样下一个 token

  产出：
    new_token_ids = [42]              ← 本次生成的 token
```

---

## 状态更新（update_from_output）

```
Scheduler.update_from_output() 回写 Request：

  Request._output_token_ids += [42]   ← 新 token 追加
  Request._all_token_ids += [42]      ← 同步更新
  Request.num_computed_tokens += N    ← 本轮计算的 token 数
  Request.block_hashes += [...]       ← 新 block 的 hash

  如果遇到 stop token 或 max_tokens：
    Request.status → FINISHED_STOPPED 或 FINISHED_LENGTH_CAPPED
    Request.stop_reason = stop_token_id 或 "length"
    Request.events += [FINISHED]

  如果还没完成：
    Request.status 保持 RUNNING
    下一轮 step() 继续生成
```

---

## 输出

```
EngineCoreOutputs 包含：

  outputs["abc-123"] = EngineCoreOutput(
      request_id="abc-123",
      new_token_ids=[42],              ← 只有本次新 token（增量设计）
      finish_reason=None,              ← None 表示还没完成
  )

         ↓  ZMQ IPC 传回 API Server

         ↓  OutputProcessor: detokenize

         ↓  HTTP SSE Stream 返回给用户

  {"choices":[{"text":" world","finish_reason":null}]}
```

---

## 消亡

```
当 finish_reason 不为 None 时：

  Scheduler 收到完成信号：
    1. 释放 Request 占用的 KV Cache block
    2. 从 requests dict 中删除
    3. 记录 metrics（TTFT、TPOT、排队时间）

  Request 对象被 Python GC 回收
```

---

## 一图总结

```
     诞生                    入队                   等待                  调度                  执行                  更新                  消亡
      │                      │                     │                    │                    │                    │                    │
  HTTP JSON            Scheduler              waiting 队列         schedule()          GPU forward        update_from_output    释放 block
      ↓               .add_request()                │                    ↓                    ↓                    ↓                    ↓
  ChatCompletion           ↓                   ┌────┴────┐        status:WAITING      prompt → logits     output += [42]       requests dict
  Request                  ↓                   │ waiting │             → RUNNING       → sampled_token     computed += N        删除
      ↓               requests dict            │ queue   │                                                    ↓                    ↓
  Renderer           注册 + 入队                └─────────┘        KVCacheManager     ModelRunnerOutput   FINISHED_*           GC 回收
      ↓                                                        分配 block
  EngineInput
      ↓
  InputProcessor
      ↓
  EngineCoreRequest ──ZMQ──→ Request
                              │
                         status=WAITING
                         output=[]
```

---

## Request 字段在各阶段的变化

| 阶段 | status | output_token_ids | num_computed_tokens | block_hashes | events |
|------|--------|-----------------|--------------------|--------------:|--------|
| 诞生 | WAITING | [] | 0 | [] | [] |
| 入队 | WAITING | [] | 0 | [] | [QUEUED] |
| 调度 | RUNNING | [] | 0 | 查询中 | [QUEUED, SCHEDULED] |
| 执行 | RUNNING | [] | 0 | 已分配 | [QUEUED, SCHEDULED] |
| 更新 | RUNNING | [42] | N | 新增 | [QUEUED, SCHEDULED] |
| 完成 | FINISHED_* | [42, 17, ...] | 全部 | 完整 | [..., FINISHED] |
