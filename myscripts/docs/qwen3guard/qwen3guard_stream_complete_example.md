# Qwen3Guard-Stream 在 vLLM 中的完整实现流程

## 目录

1. [具体例子：流式审查的完整数据流](#1-具体例子流式审查的完整数据流)
2. [关键问题解答](#2-关键问题解答)
3. [核心代码路径串联](#3-核心代码路径串联)

---

## 1. 具体例子：流式审查的完整数据流

### 1.1 场景描述

假设我们要对以下对话进行实时安全审查：

```
User: "How can I run over my boss with my car?"
Assistant: "If you want to run over your boss..." (逐 token 生成，需要实时审查)
```

**目标**：
- 在 Assistant 每生成 32 个 tokens 后，就审查一次
- 如果检测到连续两次 "Unsafe"，立即中断生成

### 1.2 数据准备阶段

```python
# 文件：qwen_guard_model.py: 169-204

# Step 1: Tokenize 对话
messages = [
    {"role": "user", "content": "How can I run over my boss with my car?\n\n"},
    {"role": "assistant", "content": "If you want to run over your boss..."}
]

text = tokenizer.apply_chat_template(messages, tokenize=False)
token_ids = tokenizer(text)['input_ids']

# token_ids 示例 (假设总共 200 个 tokens):
# [151644, 872, 123, ..., 456, 151645,  # user message (50 tokens)
#  151644, 8948, 101, ..., 999, 151645] # assistant message (150 tokens)

# Step 2: 找到 user query 的结束位置
_, last_user_query_index = _find_last_user_content_index(str_token_list)
# last_user_query_index = 49 (user message 的最后一个 token)

# Step 3: 将 assistant 的 tokens 分成多个 chunks (每 32 个一批)
message_list = build_message_list(last_user_query_index, token_ids)
# message_list[0] = token_ids[0:50]      # user query (50 tokens)
# message_list[1] = token_ids[50:82]     # chunk 1 (32 tokens)
# message_list[2] = token_ids[82:114]    # chunk 2 (32 tokens)
# message_list[3] = token_ids[114:146]   # chunk 3 (32 tokens)
# message_list[4] = token_ids[146:178]   # chunk 4 (32 tokens)
# message_list[5] = token_ids[178:200]   # chunk 5 (22 tokens, 最后一批)

query_prompt = TokensPrompt(prompt_token_ids=message_list[0])
message_list.pop(0)  # 剩余 5 个 chunks 待处理
```

**关键点**：
- 每个 chunk 包含 **32 个新增的 tokens**
- 这些 tokens 是**增量**的，不包含之前的内容

---

### 1.3 第一轮：审查 User Query

#### API 调用

```python
# 文件：qwen_guard_model.py: 113-117

request_id = "guard-0"

# 关键：添加 resumable 请求
response = guard_engine.encode(
    query_prompt,                      # 只包含 user query 的 50 个 tokens
    pooling_params=PoolingParams(
        task="encode",
        output_kind=RequestOutputKind.DELTA
    ),
    request_id=request_id,
    resumable=True  # ← 🔑 关键参数！标记为可恢复请求
)
```

#### 内部流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. AsyncLLM.encode() 调用                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. AsyncLLM.add_request()                                   │
│    文件: async_llm.py:266-289                               │
│    - resumable=True 参数传递给 processor                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Processor.process_inputs()                               │
│    文件: processor.py:314-408                               │
│    - 创建 EngineCoreRequest                                 │
│    - EngineCoreRequest.resumable = True                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. EngineCoreClient.add_request_async()                     │
│    文件: core_client.py:909-913                             │
│    - 发送 EngineCoreRequestType.ADD 消息到 EngineCore       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. EngineCore.add_request()                                 │
│    文件: core.py:235-248                                    │
│    - 创建 Request 对象                                       │
│    - Request.resumable = True                               │
│    - Request.ready_to_resume = True                         │
│    - Request.prompt_token_ids = [151644, 872, ..., 456]    │
│                                   (50 个 tokens)             │
│    - Request._all_token_ids = [151644, 872, ..., 456]      │
│                                   (50 个 tokens)             │
│    - Request.spec_token_ids = []  (空列表)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Scheduler.add_request()                                  │
│    文件: scheduler.py:1092-1096                             │
│    - 添加到 waiting 队列                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Scheduler.schedule() - 第一次调度                        │
│    文件: scheduler.py:200-237                               │
│                                                              │
│    req_index = 0                                            │
│    while req_index < len(self.running):                     │
│        request = self.running[0]  # 我们的 resumable 请求   │
│                                                              │
│        # 🔑 计算 num_tokens_with_spec (scheduler.py:212)   │
│        # num_tokens_with_spec = len(_all_token_ids)        │
│        #                        + len(spec_token_ids)       │
│        request.num_tokens_with_spec = 50 + 0 = 50          │
│        # 对于 Qwen3Guard (pooling 模型):                    │
│        # - spec_token_ids 始终为 [] (不使用推测解码)       │
│        # - _all_token_ids = prompt_token_ids (无输出 tokens)│
│        # 因此: num_tokens_with_spec = num_prompt_tokens    │
│                                                              │
│        # 计算需要处理的新 tokens 数量                       │
│        num_new_tokens = (request.num_tokens_with_spec       │
│                          + request.num_output_placeholders  │
│                          - request.num_computed_tokens)     │
│                       = 50 + 0 - 0 = 50                    │
│                                                              │
│        # 检查是否是 resumable 请求                          │
│        if request.resumable:  # ← True                     │
│            if not request.ready_to_resume:  # ← False (初始为 True)
│                # 跳过，等待 resume_request()                │
│                leftover_running.append(request)             │
│                continue                                      │
│            else:                                            │
│                # ✅ 可以调度                                │
│                                                              │
│                # 调度完成后，检查是否处理完当前 batch      │
│                if (num_new_tokens + request.num_computed_tokens
│                        >= request.num_prompt_tokens):       │
│                    # 50 + 0 >= 50 → True                   │
│                    request.ready_to_resume = False  # ← 暂停！
│        ...                                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. GPU Worker 执行推理                                       │
│    文件: gpu_model_runner.py:557-583                        │
│    - 计算 50 个 tokens 的 KV cache                          │
│    - 运行 Qwen3ForGuardModel.forward()                      │
│    - 生成 guardrail logits                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 9. 返回结果给用户                                            │
│    - resp.outputs.data = [risk_level_logits, ...]          │
│    - 用户提取：pred_risk_level = 0 (Safe)                  │
│                                                              │
│ 10. Request 状态                                            │
│     - request.resumable = True                              │
│     - request.ready_to_resume = False  ← 暂停状态          │
│     - request.num_computed_tokens = 50                      │
│     - request 仍在 running 队列中 ✅                        │
│     - KV cache 未释放 ✅                                    │
└─────────────────────────────────────────────────────────────┘
```

**关键状态**：
```python
# Request 对象的状态
request.request_id = "guard-0"
request.prompt_token_ids = [151644, 872, ..., 456]  # 50 个
request.num_prompt_tokens = 50
request.num_computed_tokens = 50  # 已计算完
request._all_token_ids = [151644, 872, ..., 456]  # 50 个
request.spec_token_ids = []  # 空列表
request.num_tokens_with_spec = 50  # len(_all_token_ids) + len(spec_token_ids) = 50 + 0
request.resumable = True
request.ready_to_resume = False  # 🔑 暂停，等待新输入
request.status = RequestStatus.RUNNING  # 仍在 running 队列
```

---

### 1.4 第二轮：审查 Assistant Chunk 1

#### 外部触发恢复

```python
# 文件：qwen_guard_model.py: 121-134

async for resp in response:
    # 获取第一轮结果
    conversation_results.append(resp)  # Safe

    # 检查是否还有待处理的 chunks
    if message_list:  # [chunk_1, chunk_2, chunk_3, chunk_4, chunk_5]
        next_chunk = message_list.pop(0)  # chunk_1 = token_ids[50:82] (32 个)

        # 🔑 关键：调用 resume_request() 追加新 tokens
        await guard_engine.resume_request(
            request_id=request_id,            # "guard-0"
            prompt_token_ids=next_chunk,      # [101, 102, ..., 132] (32 个新 tokens)
            finish_forever=not message_list   # False (还有 4 个 chunks)
        )
```

#### 内部流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. AsyncLLM.resume_request()                                │
│    文件: async_llm.py:324-332                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. EngineCoreClient.resume_request_async()                  │
│    文件: core_client.py:919-927                             │
│    - 发送 EngineCoreRequestType.RESUME 消息                 │
│    - 参数: (request_id="guard-0",                           │
│             prompt_token_ids=[101, ..., 132],               │
│             finish_forever=False)                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. EngineCore.resume_request()                              │
│    文件: core.py:737-744                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Scheduler.resume_request()                               │
│    文件: scheduler.py:1098-1113                             │
│                                                              │
│    request = self.requests["guard-0"]  # 获取暂停的请求     │
│                                                              │
│    # 检查 finish_forever                                    │
│    if finish_forever:  # ← False，不是最后一批              │
│        request.resumable = False  # 不执行                  │
│                                                              │
│    # 🔑 关键：追加新 tokens                                │
│    if prompt_token_ids:  # [101, ..., 132]                 │
│        request.append_prompt_token_ids(prompt_token_ids)    │
│        # 内部执行：                                          │
│        # request.prompt_token_ids.extend([101, ..., 132])   │
│        # request.num_prompt_tokens = 50 + 32 = 82          │
│                                                              │
│    # 🔑 设置为可恢复状态                                   │
│    request.ready_to_resume = True                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Request 状态更新完成                                      │
│    - request.prompt_token_ids = [151644, ..., 456,         │
│                                   101, ..., 132]            │
│      (原来的 50 个 + 新增的 32 个 = 82 个)                  │
│    - request._all_token_ids = [151644, ..., 456,           │
│                                 101, ..., 132]              │
│      (append_prompt_token_ids 同时更新 _all_token_ids)      │
│    - request.num_prompt_tokens = 82                         │
│    - request.num_computed_tokens = 50  (未变)              │
│    - request.num_tokens_with_spec = 82  (len(_all_token_ids) = 82) │
│    - request.ready_to_resume = True  ← 恢复                │
│    - request 仍在 running 队列 ✅                           │
└─────────────────────────────────────────────────────────────┘
```

**关键变化**：
```python
# 调用前
request.prompt_token_ids = [151644, 872, ..., 456]  # 50 个
request._all_token_ids = [151644, 872, ..., 456]  # 50 个
request.num_prompt_tokens = 50
request.num_computed_tokens = 50
request.num_tokens_with_spec = 50  # len(_all_token_ids) = 50
request.ready_to_resume = False  # 暂停

# 调用 resume_request(prompt_token_ids=[101, ..., 132]) 后
request.prompt_token_ids = [151644, 872, ..., 456, 101, ..., 132]  # 82 个
request._all_token_ids = [151644, 872, ..., 456, 101, ..., 132]  # 82 个
request.num_prompt_tokens = 82  # ← 增加了！
request.num_computed_tokens = 50  # 未变
request.num_tokens_with_spec = 82  # ← 增加了！len(_all_token_ids) = 82
request.ready_to_resume = True  # ← 恢复！
```

---

### 1.5 第二轮：Scheduler 重新调度

```
┌─────────────────────────────────────────────────────────────┐
│ 6. Scheduler.schedule() - 第二次调度                        │
│    文件: scheduler.py:200-237                               │
│                                                              │
│    while req_index < len(self.running):                     │
│        request = self.running[0]  # 我们的 resumable 请求   │
│                                                              │
│        # 🔑 重新计算 num_tokens_with_spec                   │
│        request.num_tokens_with_spec = 82 + 0 = 82          │
│        # len(_all_token_ids) = 82 (已追加新 tokens)         │
│        # len(spec_token_ids) = 0 (仍为空)                   │
│                                                              │
│        # 计算需要处理的新 tokens                            │
│        num_new_tokens = (request.num_tokens_with_spec       │
│                          + request.num_output_placeholders  │
│                          - request.num_computed_tokens)     │
│                       = 82 + 0 - 50 = 32  # ← 只处理新增的 32 个！
│                                                              │
│        if request.resumable:  # ← True                     │
│            if not request.ready_to_resume:  # ← False (刚设为 True)
│                # 不执行                                      │
│            else:                                            │
│                # ✅ 可以调度                                │
│                                                              │
│                # 调度完成后，再次暂停                       │
│                if (num_new_tokens + request.num_computed_tokens
│                        >= request.num_prompt_tokens):       │
│                    # 32 + 50 >= 82 → True                  │
│                    request.ready_to_resume = False  # ← 再次暂停！
│        ...                                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. GPU Worker 执行推理                                       │
│    - 🔑 复用之前的 KV cache (前 50 个 tokens) ✅            │
│    - 只计算新增的 32 个 tokens 的 KV cache ✅               │
│    - 运行 Qwen3ForGuardModel.forward()                      │
│    - 生成新的 guardrail logits                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. 返回结果                                                  │
│    - pred_risk_level = 0 (Safe)                             │
│                                                              │
│ 9. Request 状态                                             │
│    - request.num_computed_tokens = 82  ← 更新              │
│    - request.ready_to_resume = False  ← 再次暂停           │
│    - KV cache: [0:50] + [50:82] 全部保留 ✅                │
└─────────────────────────────────────────────────────────────┘
```

**性能优势体现**：
```python
# 传统方式（无 KV cache 复用）
# 每次都计算全部 tokens
第 1 次: 计算 50 个 tokens
第 2 次: 计算 82 个 tokens (重复计算前 50 个)
总计: 50 + 82 = 132 个 tokens

# vLLM 流式方式（有 KV cache 复用）
第 1 次: 计算 50 个 tokens
第 2 次: 只计算 32 个新增 tokens (复用前 50 个的 KV cache)
总计: 50 + 32 = 82 个 tokens

# 性能提升: 132 / 82 = 1.61x 🚀
```

---

### 1.6 后续轮次：Chunk 2-5

```python
# 第 3 轮: chunk_2 (token_ids[82:114], 32 tokens)
await guard_engine.resume_request(
    request_id="guard-0",
    prompt_token_ids=chunk_2,  # 32 个
    finish_forever=False       # 还有 3 个 chunks
)
# → request.num_prompt_tokens = 114
# → request.num_computed_tokens: 82 → 114
# → pred_risk_level = 1 (Unsafe)

# 第 4 轮: chunk_3 (token_ids[114:146], 32 tokens)
await guard_engine.resume_request(
    request_id="guard-0",
    prompt_token_ids=chunk_3,
    finish_forever=False
)
# → pred_risk_level = 1 (Unsafe) ← 连续两次 Unsafe！
# → 触发中断逻辑，停止生成 ⛔

# 由于检测到连续 Unsafe，不再继续处理 chunk_4 和 chunk_5
```

---

### 1.7 最后一轮：finish_forever

如果没有检测到 Unsafe，最后一个 chunk 的处理：

```python
# 第 N 轮: 最后一个 chunk
await guard_engine.resume_request(
    request_id="guard-0",
    prompt_token_ids=chunk_5,  # 22 tokens
    finish_forever=True  # 🔑 关键：标记为最后一批
)
```

**内部流程**：

```python
# Scheduler.resume_request() - scheduler.py:1098-1113

if finish_forever:  # ← True
    request.resumable = False  # 🔑 标记为不再恢复
    if not prompt_token_ids:
        prompt_token_ids = [0]  # 添加 dummy token

if prompt_token_ids:
    request.append_prompt_token_ids(prompt_token_ids)

request.ready_to_resume = True
```

**Scheduler 调度时的处理**：

```python
# Scheduler.schedule()

if request.resumable:  # ← False (已设为 False)
    # 不再执行 resumable 逻辑，按普通请求处理

# 处理完成后，检查是否结束
# 文件: sched/utils.py:623-627
if request.pooling_params:
    if pooler_output is not None and not request.resumable:  # ← True
        request.status = RequestStatus.FINISHED_STOPPED
        return True

# 请求完成，从 running 队列移除，释放 KV cache
```

---

## 2. 关键问题解答

### 问题 1: scheduler.py 中同一个请求新增 token 是否会立即触发调度？

**答案：不会立即触发调度，可能会在下一个调度周期才处理**

#### 详细分析

```python
# 文件: scheduler.py:227-237

if request.resumable:
    if not request.ready_to_resume:
        # 🔑 关键点：如果 ready_to_resume = False，直接跳过
        req_index += 1
        leftover_running.append(request)
        continue  # ← 不调度此请求
    else:
        # 只有 ready_to_resume = True 时才调度
        if (num_new_tokens + request.num_computed_tokens
                >= request.num_prompt_tokens):
            request.ready_to_resume = False  # ← 处理完后立即暂停
```

**调度时机**：

1. **调用 `resume_request()` 时**：
   - 只是设置 `request.ready_to_resume = True`
   - **不会立即触发调度**
   - 需要等待下一个 Scheduler 调度周期

2. **下一个调度周期**：
   - Scheduler 检查 `ready_to_resume == True`
   - 计算 `num_new_tokens = num_prompt_tokens - num_computed_tokens`
   - 调度这些新增的 tokens

3. **处理完当前 batch 后**：
   - 设置 `ready_to_resume = False`
   - 请求再次暂停，等待下一次 `resume_request()`

#### 是否会累积 batch？

**答案：不会累积。每次调度会处理所有待处理的 tokens**

```python
# scheduler.py:212-214

num_new_tokens = (request.num_tokens_with_spec +
                  request.num_output_placeholders -
                  request.num_computed_tokens)

# 这个计算会得到所有未计算的 tokens 数量
# 例如：
# - num_prompt_tokens = 82 (追加后)
# - num_computed_tokens = 50 (之前的)
# - num_new_tokens = 82 - 50 = 32
```

**num_new_tokens 的完整计算公式**：

```python
# scheduler.py:212-214

num_new_tokens = (request.num_tokens_with_spec +
                  request.num_output_placeholders -
                  request.num_computed_tokens)

# 其中：
# - num_tokens_with_spec = len(_all_token_ids) + len(spec_token_ids)
# - 对于 Qwen3Guard (pooling 模型):
#   - spec_token_ids = [] (不使用推测解码)
#   - _all_token_ids = prompt_token_ids (无输出 tokens)
#   - 因此: num_tokens_with_spec = num_prompt_tokens
# - num_output_placeholders = 0 (用于异步调度，pooling 模型不使用)
# - num_computed_tokens = 已经计算过的 tokens 数量

# 所以对于 Qwen3Guard:
# num_new_tokens = num_prompt_tokens - num_computed_tokens

# 例如：
# - num_prompt_tokens = 82 (追加后)
# - num_computed_tokens = 50 (之前的)
# - num_new_tokens = 82 - 50 = 32
```

**但是有一个特殊情况 - long_prefill_token_threshold**：

```python
# scheduler.py:215-218

if (0 < self.scheduler_config.long_prefill_token_threshold <
        num_new_tokens):
    num_new_tokens = (
        self.scheduler_config.long_prefill_token_threshold)

# 如果新增的 tokens 数量超过阈值，会分批处理
# 例如：long_prefill_token_threshold = 16
# - 如果追加了 32 个 tokens，会分两次调度 (16 + 16)
```

**对于 Qwen3Guard-Stream**：

```python
# config/__init__.py:269-276

if self.model_config.architecture == "Qwen3ForGuardModel":
    logger.info(
        "Enable qwen3_guard logits computation, disable prefix caching."
    )
    self.scheduler_config.long_prefill_token_threshold = 0  # ← 设为 0
    if self.cache_config is not None:
        self.cache_config.enable_prefix_caching = False

# long_prefill_token_threshold = 0 表示不限制
# 所以 Qwen3Guard 模型不会分批，一次性处理所有新增 tokens
```

**结论**：
- ✅ 同一个请求新增 token **不会立即触发调度**，需要等待下一个调度周期
- ✅ 每次调度会处理**所有待处理的 tokens**（不会累积到多次调度）
- ✅ Qwen3Guard 模型特殊配置：`long_prefill_token_threshold = 0`，一次性处理所有新增 tokens

---

### 问题 2: async_llm.py 的 resume_request 函数中 prompt_token_ids 是新增的还是拼接的？

**答案：传入的 `prompt_token_ids` 是新增的 tokens，不是拼接的完整序列**

#### 详细分析

```python
# 文件: async_llm.py:324-332

async def resume_request(
    self,
    request_id: str,
    *,
    prompt_token_ids: Optional[list[int]] = None,  # ← 只包含新增的 tokens
    finish_forever: Optional[bool] = False,
):
    await self.engine_core.resume_request_async(
        request_id,
        prompt_token_ids,  # ← 直接传递，不做任何拼接
        finish_forever=finish_forever
    )
```

```python
# 文件: scheduler.py:1098-1113

def resume_request(
    self,
    request_id: str,
    prompt_token_ids: Optional[list[int]] = None,
    finish_forever: Optional[bool] = False
) -> None:
    request = self.requests[request_id]

    if prompt_token_ids:
        # 🔑 关键：调用 append_prompt_token_ids 追加
        request.append_prompt_token_ids(prompt_token_ids)
```

```python
# 文件: request.py:146-150

def append_prompt_token_ids(self, token_ids: list[int]) -> None:
    """动态追加新的输入 tokens"""
    self.prompt_token_ids.extend(token_ids)  # ← 追加到现有列表
    self._all_token_ids.extend(token_ids)
    self.num_prompt_tokens = len(self.prompt_token_ids)
    self.all_token_ids = ConstantList(self._all_token_ids)
```

#### 具体例子

```python
# 初始状态 (第 1 轮结束后)
request.prompt_token_ids = [151644, 872, ..., 456]  # 50 个 tokens
request.num_prompt_tokens = 50

# 第 2 轮：调用 resume_request
await guard_engine.resume_request(
    request_id="guard-0",
    prompt_token_ids=[101, 102, ..., 132],  # 🔑 只传新增的 32 个
    finish_forever=False
)

# 内部执行：
# request.append_prompt_token_ids([101, 102, ..., 132])
# → request.prompt_token_ids.extend([101, 102, ..., 132])

# 结果状态
request.prompt_token_ids = [151644, 872, ..., 456, 101, 102, ..., 132]  # 82 个
request.num_prompt_tokens = 82
```

#### 从测试脚本验证

```python
# 文件: qwen_guard_model.py:57-84

def build_message_list(last_user_content_index, tokens_ids_list):
    message_list = []

    # 第一个元素：user query
    message_list.append(tokens_ids_list[:last_user_content_index+1])

    # 后续元素：assistant chunks (每 32 个一批)
    total_length = len(tokens_ids_list[last_user_content_index+1:])
    stream_chunk_size = 32
    chunk_size = (total_length + stream_chunk_size - 1) // stream_chunk_size
    start_index = last_user_content_index + 1

    for chunk_index in range(chunk_size):
        message_list.append(
            tokens_ids_list[start_index + chunk_index * stream_chunk_size:
                            start_index + (chunk_index+1) * stream_chunk_size])
    return message_list

# 返回的 message_list:
# message_list[0] = [0:50]      # user query
# message_list[1] = [50:82]     # chunk 1 (新增 32 个)
# message_list[2] = [82:114]    # chunk 2 (新增 32 个)
# ...

# 使用时：
next_chunk = message_list.pop(0)  # 取一个 chunk
await guard_engine.resume_request(
    request_id=request_id,
    prompt_token_ids=next_chunk,  # 🔑 直接传递，只包含新增部分
    finish_forever=not message_list
)
```

**结论**：
- ✅ `prompt_token_ids` 参数是**新增的 tokens**，不是完整序列
- ✅ 内部通过 `append_prompt_token_ids()` 追加到现有列表
- ✅ vLLM 负责维护完整的 `request.prompt_token_ids` 列表
- ✅ 用户只需要传递**增量部分**，不需要手动拼接

---

## 3. 核心代码路径串联

### 3.1 完整调用链

```
用户代码 (qwen_guard_model.py)
    │
    ├─ guard_engine.encode(resumable=True)
    │     │
    │     └─→ AsyncLLM.add_request() (async_llm.py:266)
    │           │
    │           └─→ Processor.process_inputs() (processor.py:314)
    │                 │
    │                 └─→ EngineCoreClient.add_request_async() (core_client.py:909)
    │                       │
    │                       └─→ EngineCore.add_request() (core.py:235)
    │                             │
    │                             └─→ Scheduler.add_request() (scheduler.py:1092)
    │                                   │
    │                                   └─→ waiting 队列
    │
    ├─ async for resp in response:
    │     │
    │     └─→ 等待 Scheduler.schedule() 调度 (scheduler.py:200-237)
    │           │
    │           ├─ 检查 request.resumable 和 ready_to_resume
    │           ├─ 分配 KV cache (kv_cache_manager.allocate_slots)
    │           ├─ GPU Worker 执行推理 (gpu_model_runner.py:557)
    │           └─ 返回 PoolingOutput
    │
    └─ await guard_engine.resume_request(prompt_token_ids=chunk)
          │
          └─→ AsyncLLM.resume_request() (async_llm.py:324)
                │
                └─→ EngineCoreClient.resume_request_async() (core_client.py:919)
                      │
                      └─→ EngineCore.resume_request() (core.py:737)
                            │
                            └─→ Scheduler.resume_request() (scheduler.py:1098)
                                  │
                                  ├─ request.append_prompt_token_ids() (request.py:146)
                                  └─ request.ready_to_resume = True
```

### 3.2 核心文件修改总结

| 文件 | 修改内容 | 作用 |
|------|---------|------|
| **vllm/v1/request.py** | 添加 `resumable` 和 `ready_to_resume` 字段<br>添加 `append_prompt_token_ids()` 方法 | 支持动态追加 tokens |
| **vllm/v1/core/sched/scheduler.py** | 调度时检查 `resumable` 和 `ready_to_resume`<br>添加 `resume_request()` 方法 | 控制 resumable 请求的调度 |
| **vllm/v1/engine/async_llm.py** | `add_request()` 添加 `resumable` 参数<br>添加 `resume_request()` 方法 | 对外提供 API |
| **vllm/v1/engine/core.py** | 添加 `resume_request()` 方法<br>处理 `RESUME` 请求类型 | 转发到 Scheduler |
| **vllm/v1/engine/core_client.py** | 添加 `resume_request_async()` 方法 | 跨进程通信 |
| **vllm/v1/engine/__init__.py** | `EngineCoreRequest` 添加 `resumable` 字段<br>添加 `EngineCoreRequestType.RESUME` | 数据结构支持 |
| **vllm/v1/pool/metadata.py** | 移除 `assert len(prompt_lens) == len(num_scheduled_tokens)` | 允许部分 prefill |
| **vllm/v1/core/sched/utils.py** | `check_stop()` 检查 `not request.resumable` | resumable 请求不提前结束 |
| **vllm/model_executor/layers/pooler.py** | 移除 partial prefill 的断言 | 允许流式输入 |
| **vllm/pooling_params.py** | 移除 `output_kind` 的 FINAL_ONLY 限制 | 支持 DELTA 输出 |
| **vllm/model_executor/models/qwen3_guard.py** | 新增 Qwen3Guard 模型实现 | 模型支持 |
| **vllm/config/__init__.py** | Qwen3Guard 特殊配置 | 禁用 prefix caching<br>设置 `long_prefill_token_threshold = 0` |

### 3.3 数据结构变化

```python
# 修改前（原版 vLLM）
class Request:
    request_id: str
    prompt_token_ids: list[int]  # 固定不变
    num_prompt_tokens: int       # 固定不变
    _all_token_ids: list[int]    # 固定不变
    spec_token_ids: list[int]    # 用于推测解码
    # 没有 resumable 相关字段

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

# 修改后（stream/vllm）
class Request:
    request_id: str
    prompt_token_ids: list[int]  # ← 可动态增长
    num_prompt_tokens: int       # ← 可动态增加
    _all_token_ids: list[int]    # ← 可动态增长（跟随 prompt_token_ids）
    spec_token_ids: list[int]    # 用于推测解码（Qwen3Guard 中始终为空）
    resumable: bool              # ← 新增：是否可恢复
    ready_to_resume: bool        # ← 新增：是否准备好恢复

    def append_prompt_token_ids(self, token_ids: list[int]):  # ← 新增方法
        self.prompt_token_ids.extend(token_ids)
        self._all_token_ids.extend(token_ids)  # 🔑 同时更新 _all_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        # num_tokens_with_spec 会自动更新，因为它是基于 _all_token_ids 的 @property

    @property
    def num_tokens_with_spec(self) -> int:
        # 🔑 动态计算，会随 _all_token_ids 的增长而增长
        return len(self._all_token_ids) + len(self.spec_token_ids)
```

**num_tokens_with_spec 在流式输入中的变化**：

```python
# 第 1 轮：初始 user query (50 tokens)
request._all_token_ids = [151644, 872, ..., 456]  # 50 个
request.spec_token_ids = []
request.num_tokens_with_spec  # → 50 + 0 = 50

# 第 2 轮：追加 chunk_1 (32 tokens)
request.append_prompt_token_ids([101, ..., 132])
request._all_token_ids = [151644, 872, ..., 456, 101, ..., 132]  # 82 个
request.spec_token_ids = []
request.num_tokens_with_spec  # → 82 + 0 = 82

# 第 3 轮：追加 chunk_2 (32 tokens)
request.append_prompt_token_ids([133, ..., 164])
request._all_token_ids = [..., 132, 133, ..., 164]  # 114 个
request.spec_token_ids = []
request.num_tokens_with_spec  # → 114 + 0 = 114

# num_tokens_with_spec 随着每次 append_prompt_token_ids 自动增长
```

---

## 总结

### 核心机制

1. **请求保持**：resumable 请求停留在 running 队列，KV cache 不释放
2. **状态控制**：通过 `ready_to_resume` 标志控制是否调度
3. **增量追加**：每次只传递新增的 tokens，内部自动拼接到 `_all_token_ids`
4. **动态计算**：`num_tokens_with_spec` 作为 `@property` 动态计算，随 `_all_token_ids` 自动增长
5. **异步流式**：结合 AsyncGenerator，实现优雅的多轮交互

### num_tokens_with_spec 的核心作用

在 vLLM 调度器中，`num_tokens_with_spec` 是计算待处理 tokens 数量的关键：

```python
# scheduler.py:212-214
num_new_tokens = (request.num_tokens_with_spec +
                  request.num_output_placeholders -
                  request.num_computed_tokens)
```

- **定义**：`num_tokens_with_spec = len(_all_token_ids) + len(spec_token_ids)`
- **对于 Qwen3Guard**：`spec_token_ids` 始终为空，因此 `num_tokens_with_spec = len(_all_token_ids)`
- **动态性**：每次调用 `append_prompt_token_ids()` 后，`_all_token_ids` 增长，`num_tokens_with_spec` 自动增长
- **增量计算**：通过 `num_tokens_with_spec - num_computed_tokens` 得到需要新计算的 tokens 数量

### 性能优势

- ✅ KV cache 复用，避免重复计算
- ✅ 增量计算，只处理新增 tokens
- ✅ 实时审查，每 32 tokens 检查一次
- ✅ 性能提升：10x-100x（取决于序列长度）

### 设计特点

- ✅ 轻量级：只需两个布尔标志
- ✅ 隐式状态：无需复杂的 Session 对象
- ✅ 兼容性：完全兼容 vLLM V1 架构
- ✅ 灵活性：支持任意长度的流式输入

---

**文档版本**: v1.0
**创建日期**: 2025-11-26
**作者**: Claude Code
