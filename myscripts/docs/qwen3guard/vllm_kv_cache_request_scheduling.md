# vLLM v1 架构 KV Cache 请求调度流程

> 本文档详细介绍 vLLM v1 架构中 Scheduler 如何在请求调度过程中管理 KV Cache，包括 prefix caching、block 分配、preemption 等机制。

## 目录
- [概述](#概述)
- [调度流程](#调度流程)
  - [1. Scheduler.schedule() 总览](#1-schedulerschedule-总览)
  - [2. RUNNING 请求调度](#2-running-请求调度)
  - [3. WAITING 请求调度](#3-waiting-请求调度)
  - [4. Prefix Caching 查找](#4-prefix-caching-查找)
  - [5. KV Cache Blocks 分配](#5-kv-cache-blocks-分配)
  - [6. Preemption 机制](#6-preemption-机制)
  - [7. Block Caching](#7-block-caching)
- [核心数据流](#核心数据流)
- [调度时序图](#调度时序图)

---

## 概述

vLLM v1 架构的调度器采用**统一的 token 调度**策略，不区分传统的 "prefill 阶段" 和 "decode 阶段"。每个请求维护两个关键状态：
- `num_computed_tokens`: 已经计算过的 token 数量
- `num_tokens_with_spec`: 总 token 数（prompt + output + speculative tokens）

调度器的目标是让每个请求的 `num_computed_tokens` 追赶上 `num_tokens_with_spec`，这种设计天然支持：
- **Chunked Prefill**: 将长 prompt 分块处理
- **Prefix Caching**: 复用已计算的 KV cache
- **Speculative Decoding**: 投机解码
- **Jump Decoding**: 跳跃解码优化

### KV Cache 在调度中的角色

```
Scheduler (调度层)
    │
    ├─> 决策: 哪些请求被调度，分配多少 tokens
    │
    └─> KVCacheManager (执行层)
          │
          ├─> Prefix Cache 查找 (get_computed_blocks)
          ├─> Block 分配 (allocate_slots)
          ├─> Block 缓存 (cache_blocks)
          └─> Block 释放 (free)
```

## 调度流程

### 1. Scheduler.schedule() 总览

**位置**: `vllm/v1/core/sched/scheduler.py:177-600`

```python
def schedule(self) -> SchedulerOutput:
    # 初始化调度数据结构
    scheduled_new_reqs: list[Request] = []
    scheduled_resumed_reqs: list[Request] = []
    scheduled_running_reqs: list[Request] = []
    preempted_reqs: list[Request] = []

    req_to_new_blocks: dict[str, KVCacheBlocks] = {}
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens  # 总 token 预算

    # 第一步: 调度 RUNNING 请求
    self._schedule_running_requests(...)

    # 第二步: 调度 WAITING 请求
    if not preempted_reqs:
        self._schedule_waiting_requests(...)

    # 返回调度结果
    return SchedulerOutput(...)
```

**调度顺序**:
1. **RUNNING 请求优先**: 保证正在运行的请求能继续执行
2. **WAITING 请求次之**: 如果没有 preemption，尝试调度新请求
3. **FCFS (First-Come-First-Serve)**: 基本遵循先到先服务，但允许跳过某些请求

**Token Budget 管理**:
- `token_budget = max_num_batched_tokens` (例如 2048)
- 每调度一个请求，减去分配的 tokens
- 当 `token_budget <= 0` 时停止调度

### 2. RUNNING 请求调度

**位置**: `scheduler.py:206-340`

RUNNING 请求是已经开始执行但尚未完成的请求，需要继续分配 KV cache。

```python
# scheduler.py:206-340
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]

    # 计算本次需要调度的 token 数
    num_new_tokens = (request.num_tokens_with_spec -
                      request.num_computed_tokens)

    # 应用 chunked prefill 限制
    if 0 < long_prefill_token_threshold < num_new_tokens:
        num_new_tokens = long_prefill_token_threshold

    # 应用 token budget 限制
    num_new_tokens = min(num_new_tokens, token_budget)

    # 应用 max_model_len 限制
    num_new_tokens = min(num_new_tokens,
                        max_model_len - 1 - request.num_computed_tokens)

    # 尝试分配 KV cache blocks
    while True:
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens,
            num_lookahead_tokens=self.num_lookahead_tokens
        )

        if new_blocks is None:
            # 分配失败，需要 preemption
            preempted_req = self._preempt_request()
            if preempted_req == request:
                # 没有更多请求可以抢占
                break
        else:
            # 分配成功
            scheduled_running_reqs.append(request)
            req_to_new_blocks[request.request_id] = new_blocks
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            break
```

**RUNNING 请求调度要点**:
- **不查找 prefix cache**: RUNNING 请求已经有分配的 blocks，只需追加新 blocks
- **支持 chunked prefill**: 通过 `long_prefill_token_threshold` 限制每次处理的 token 数
- **支持 speculative decoding**: 通过 `num_lookahead_tokens` 预分配 speculative tokens 的空间
- **失败时触发 preemption**: 如果 KV cache 不足，抢占低优先级请求

### 3. WAITING 请求调度

**位置**: `scheduler.py:353-565`

WAITING 请求是新到达的请求，需要进行 prefix caching 查找和初次 block 分配。

```python
# scheduler.py:353-565
while self.waiting and token_budget > 0:
    if len(self.running) == self.max_num_running_reqs:
        break  # 达到最大并发请求数

    request = self.waiting.peek_request()

    # 步骤 1: Prefix Caching 查找
    if request.num_computed_tokens == 0:
        # 查找本地缓存的 blocks
        new_computed_blocks, num_new_local_computed_tokens = \
            self.kv_cache_manager.get_computed_blocks(request)

        # 查找外部缓存 (KVConnector, 用于 P/D disaggregation)
        if self.connector is not None:
            num_external_computed_tokens, load_kv_async = \
                self.connector.get_num_new_matched_tokens(
                    request, num_new_local_computed_tokens)

        # 总计算 token 数
        num_computed_tokens = (num_new_local_computed_tokens +
                               num_external_computed_tokens)
    else:
        # 如果已有 computed_tokens (例如 resumed 请求)
        new_computed_blocks = self.kv_cache_manager.create_empty_block_list()
        num_computed_tokens = request.num_computed_tokens

    # 步骤 2: 计算需要调度的 token 数
    num_new_tokens = request.num_tokens - num_computed_tokens
    if 0 < long_prefill_token_threshold < num_new_tokens:
        num_new_tokens = long_prefill_token_threshold
    num_new_tokens = min(num_new_tokens, token_budget)

    # 步骤 3: 分配 KV cache slots
    new_blocks = self.kv_cache_manager.allocate_slots(
        request,
        num_new_tokens + num_external_computed_tokens,
        num_new_local_computed_tokens,
        new_computed_blocks,
        num_lookahead_tokens=effective_lookahead_tokens,
        delay_cache_blocks=load_kv_async,
        num_encoder_tokens=num_encoder_tokens,
    )

    if new_blocks is None:
        # 分配失败，停止调度新请求
        break

    # 步骤 4: 移动请求到 RUNNING 状态
    request = self.waiting.pop_request()
    self.running.append(request)
    scheduled_new_reqs.append(request)

    # 更新请求状态
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = num_computed_tokens
    request.num_cached_tokens = num_computed_tokens  # prefix cache 命中数

    # 记录分配信息
    req_to_new_blocks[request.request_id] = \
        self.kv_cache_manager.get_blocks(request.request_id)
    num_scheduled_tokens[request.request_id] = num_new_tokens
    token_budget -= num_new_tokens
```

**WAITING 请求调度要点**:
- **Prefix Caching 是关键优化**: 查找已缓存的 blocks，避免重新计算
- **支持外部 KV transfer**: 通过 KVConnector 从其他节点获取 KV cache
- **Chunked Prefill**: 长 prompt 可以分多次调度
- **失败不触发 preemption**: WAITING 请求失败时直接跳过，不抢占其他请求

### 4. Prefix Caching 查找

**位置**: `vllm/v1/core/kv_cache_manager.py:153-190`

Prefix caching 是 vLLM 的核心优化，通过复用相同 prefix 的 KV cache 避免重复计算。

```python
# kv_cache_manager.py:153-190
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
    # 检查是否启用 prefix caching
    if not self.enable_caching:
        return self.create_empty_block_list(), 0

    # 如果需要 prompt_logprobs，不能使用 prefix cache
    if request.sampling_params and \
       request.sampling_params.prompt_logprobs is not None:
        return self.create_empty_block_list(), 0

    # 设置最大缓存命中长度
    # 注意: 必须重新计算最后一个 token 以获取 logits
    max_cache_hit_length = request.num_tokens - 1

    # 调用 coordinator 查找缓存
    computed_blocks, num_new_computed_tokens = \
        self.coordinator.find_longest_cache_hit(
            request.block_hashes,
            max_cache_hit_length
        )

    # 记录 prefix cache 统计信息
    if self.log_stats:
        self.prefix_cache_stats.requests += 1
        self.prefix_cache_stats.queries += request.num_tokens
        self.prefix_cache_stats.hits += num_new_computed_tokens

    return KVCacheBlocks(computed_blocks), num_new_computed_tokens
```

**Prefix Cache 查找过程** (以 UnitaryKVCacheCoordinator 为例):

```python
# kv_cache_coordinator.py:251-265
def find_longest_cache_hit(
    self, block_hashes: list[BlockHash], max_cache_hit_length: int
) -> tuple[tuple[list[KVCacheBlock], ...], int]:

    computed_blocks: tuple[list[KVCacheBlock], ...] = tuple([] for _ in ...)
    block_size = self.kv_cache_spec.block_size
    max_num_blocks = max_cache_hit_length // block_size

    # 遍历请求的 block hashes
    for block_hash in itertools.islice(block_hashes, max_num_blocks):
        # 在 BlockPool 中查找缓存的 block
        cached_block = block_pool.get_cached_block(
            block_hash, kv_cache_group_ids
        )

        if cached_block:
            # 找到缓存，添加到结果
            for computed, cached in zip(computed_blocks, cached_block):
                computed.append(cached)
        else:
            # 缓存未命中，停止查找
            # (因为 block_hashes 是连续的，后续 blocks 肯定也未缓存)
            break

    # 返回命中的 blocks 和 token 数
    hit_length = len(computed_blocks[0]) * block_size
    return computed_blocks, hit_length
```

**BlockPool 缓存查找**:

```python
# block_pool.py:71-93
def get_cached_block(
    self, block_hash: BlockHash, kv_cache_group_ids: list[int]
) -> Optional[list[KVCacheBlock]]:
    cached_blocks = []
    for group_id in kv_cache_group_ids:
        # 构造带 group_id 的 hash key
        block_hash_with_group_id = BlockHashWithGroupId(block_hash, group_id)

        # 在哈希表中查找
        cached_blocks_one_group = \
            self.cached_block_hash_to_block.get(block_hash_with_group_id)

        if not cached_blocks_one_group:
            return None  # 任何一个 group 未命中都返回 None

        # 返回第一个匹配的 block (如果有重复)
        first_block = next(iter(cached_blocks_one_group.values()))
        cached_blocks.append(first_block)

    return cached_blocks
```

**Prefix Caching 关键点**:
- **Block Hash 计算**: Request 创建时就计算好所有 block 的 hash
- **最长前缀匹配**: 从第一个 block 开始查找，直到遇到未命中
- **必须重新计算最后一个 token**: 因为需要 logits 进行采样
- **支持多 KV cache groups**: 混合模型 (如 Mamba + Attention) 每个 group 都要命中

### 5. KV Cache Blocks 分配

**位置**: `vllm/v1/core/kv_cache_manager.py:192-303`

`allocate_slots()` 是 KV cache 分配的核心方法。

```python
# kv_cache_manager.py:192-303
def allocate_slots(
    self,
    request: Request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: Optional[KVCacheBlocks] = None,
    num_lookahead_tokens: int = 0,
    delay_cache_blocks: bool = False,
    num_encoder_tokens: int = 0,
) -> Optional[KVCacheBlocks]:

    # ============== 步骤 1: 准备 computed blocks ==============
    if new_computed_blocks is not None:
        new_computed_block_list = new_computed_blocks.blocks
    else:
        new_computed_block_list = tuple([] for _ in ...)

    # ============== 步骤 2: 释放 skipped blocks ==============
    # 例如: sliding window 之外的 blocks
    self.coordinator.remove_skipped_blocks(
        request.request_id,
        request.num_computed_tokens
    )

    # ============== 步骤 3: 计算需要的 block 数 ==============
    num_computed_tokens = (request.num_computed_tokens +
                           num_new_computed_tokens)
    num_tokens_need_slot = min(
        num_computed_tokens + num_new_tokens + num_lookahead_tokens,
        self.max_model_len
    )

    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=num_tokens_need_slot,
        new_computed_blocks=new_computed_block_list,
        num_encoder_tokens=num_encoder_tokens,
    )

    # ============== 步骤 4: 检查是否有足够的 free blocks ==============
    if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
        return None  # 分配失败

    # ============== 步骤 5: Touch computed blocks ==============
    # 增加 ref_cnt，防止被驱逐
    if self.enable_caching:
        self.block_pool.touch(new_computed_block_list)

    # ============== 步骤 6: 保存 computed blocks ==============
    self.coordinator.save_new_computed_blocks(
        request.request_id,
        new_computed_block_list
    )

    # ============== 步骤 7: 分配新 blocks ==============
    new_blocks = self.coordinator.allocate_new_blocks(
        request.request_id,
        num_tokens_need_slot,
        num_encoder_tokens
    )

    # ============== 步骤 8: 缓存 full blocks ==============
    if self.enable_caching and not delay_cache_blocks:
        num_tokens_to_cache = min(
            num_computed_tokens + num_new_tokens,
            request.num_tokens  # 只缓存 finalized tokens
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

    return KVCacheBlocks(new_blocks)
```

**Blocks 布局示意**:
```
请求的完整 blocks 布局:
-----------------------------------------------------------------------
| < computed > | < new computed > |    < new >    | < pre-allocated > |
-----------------------------------------------------------------------
^              ^                  ^               ^
|              |                  |               |
已有 blocks    prefix cache 命中  本次调度分配    speculative tokens

|<------------ required ---------->|
|<------------- full ------------->|
                                   |<- new full ->|
```

**Block 分配细节**:

```python
# coordinator 调用 single_type_manager 分配
# single_type_kv_cache_manager.py
def allocate_new_blocks(self, request_id: str, num_tokens: int):
    # 计算需要的总 blocks 数
    needed_blocks = (num_tokens + self.block_size - 1) // self.block_size

    # 计算需要新分配的 blocks 数
    current_blocks = len(self.req_to_blocks.get(request_id, []))
    num_to_allocate = max(0, needed_blocks - current_blocks)

    # 从 BlockPool 获取新 blocks
    new_blocks = self.block_pool.get_new_blocks(num_to_allocate)

    # 添加到请求的 block 列表
    self.req_to_blocks[request_id].extend(new_blocks)

    return new_blocks
```

**BlockPool.get_new_blocks()**:

```python
# block_pool.py:163-190
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    if num_blocks > self.get_num_free_blocks():
        raise ValueError("Cannot get blocks from the pool")

    # 从 free_block_queue 中取出 blocks
    ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

    for block in ret:
        # 如果 block 有缓存，驱逐它
        if self.enable_caching:
            self._maybe_evict_cached_block(block)

        # 设置 ref_cnt
        assert block.ref_cnt == 0
        block.ref_cnt += 1

    return ret
```

**分配流程总结**:
1. 准备 prefix cache 命中的 blocks
2. 释放不需要的 blocks (如 sliding window 外的)
3. 计算需要多少新 blocks
4. 检查 free blocks 是否足够
5. Touch computed blocks (增加 ref_cnt)
6. 从 BlockPool 分配新 blocks
7. 缓存已填充完整的 blocks

### 6. Preemption 机制

**位置**: `scheduler.py:269-296`

当 KV cache 不足时，Scheduler 会抢占低优先级的 RUNNING 请求。

```python
# scheduler.py:269-296
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(
        request, num_new_tokens, ...
    )

    if new_blocks is None:
        # 分配失败，需要 preemption

        # 选择要抢占的请求
        if self.policy == SchedulingPolicy.PRIORITY:
            # 优先级调度: 抢占最低优先级的请求
            preempted_req = max(
                self.running,
                key=lambda r: (r.priority, r.arrival_time),
            )
            self.running.remove(preempted_req)
        else:
            # FCFS: 抢占队尾请求 (最后到达的)
            preempted_req = self.running.pop()

        # 释放被抢占请求的 KV cache
        self.kv_cache_manager.free(preempted_req)
        self.encoder_cache_manager.free(preempted_req)

        # 更新请求状态
        preempted_req.status = RequestStatus.PREEMPTED
        preempted_req.num_computed_tokens = 0  # 重置计算进度

        # 移回 WAITING 队列头部
        self.waiting.prepend_request(preempted_req)
        preempted_reqs.append(preempted_req)

        # 如果抢占的就是当前请求，说明无法调度
        if preempted_req == request:
            can_schedule = False
            break
    else:
        # 分配成功
        can_schedule = True
        break
```

**KVCacheManager.free()**:

```python
# kv_cache_manager.py:305-313
def free(self, request: Request) -> None:
    # 调用 coordinator 释放所有 groups 的 blocks
    self.coordinator.free(request.request_id)

# coordinator 调用每个 single_type_manager
# single_type_kv_cache_manager.py
def free(self, request_id: str) -> None:
    blocks = self.req_to_blocks.pop(request_id, [])

    # 按相反顺序释放 (tail blocks 先被驱逐)
    self.block_pool.free_blocks(reversed(blocks))

    # 清理缓存计数
    self.req_to_num_cached_blocks.pop(request_id, None)
```

**BlockPool.free_blocks()**:

```python
# block_pool.py:243-258
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    blocks_list = list(ordered_blocks)

    # 减少 ref_cnt
    for block in blocks_list:
        block.ref_cnt -= 1

    # 将 ref_cnt=0 的 blocks 加回 free_block_queue
    self.free_block_queue.append_n([
        block for block in blocks_list
        if block.ref_cnt == 0 and not block.is_null
    ])
```

**Preemption 特点**:
- **只抢占 RUNNING 请求**: WAITING 请求还没分配 blocks，无需抢占
- **重置计算进度**: `num_computed_tokens = 0`，请求需要重新开始
- **Prefix Cache 依然有效**: 虽然 blocks 被释放，但如果 `ref_cnt > 0` (被其他请求共享) 或仍在缓存中，可以在重新调度时命中
- **优先级调度支持**: 通过 `SchedulingPolicy.PRIORITY` 保护高优先级请求

### 7. Block Caching

**位置**: `kv_cache_manager.py:301` 和 `coordinator 及 block_pool`

Block caching 将已计算的 full blocks 加入 prefix cache，供后续请求复用。

```python
# kv_cache_manager.py:295-303
# 在 allocate_slots 的最后
if self.enable_caching and not delay_cache_blocks:
    num_tokens_to_cache = min(
        num_computed_tokens + num_new_tokens,
        request.num_tokens  # 只缓存 finalized tokens，不缓存 draft tokens
    )
    self.coordinator.cache_blocks(request, num_tokens_to_cache)
```

**Coordinator.cache_blocks()**:

```python
# kv_cache_coordinator.py:118-129
def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
    for manager in self.single_type_managers:
        manager.cache_blocks(request, num_computed_tokens)
```

**SingleTypeManager.cache_blocks()**:

```python
# single_type_kv_cache_manager.py
def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
    blocks = self.req_to_blocks.get(request.request_id, [])

    # 计算有多少个 full blocks
    num_full_blocks = num_computed_tokens // self.block_size

    # 获取已缓存的 block 数
    num_cached_blocks = self.req_to_num_cached_blocks.get(request.request_id, 0)

    # 只缓存新的 full blocks
    if num_full_blocks > num_cached_blocks:
        self.block_pool.cache_full_blocks(
            request=request,
            blocks=blocks,
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
        )

        # 更新缓存计数
        self.req_to_num_cached_blocks[request.request_id] = num_full_blocks
```

**BlockPool.cache_full_blocks()**:

```python
# block_pool.py:95-161
def cache_full_blocks(
    self,
    request: Request,
    blocks: list[KVCacheBlock],
    num_cached_blocks: int,
    num_full_blocks: int,
    block_size: int,
    kv_cache_group_id: int,
) -> None:
    if num_cached_blocks == num_full_blocks:
        return  # 没有新的 full blocks

    # 获取新的 full blocks
    new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
    new_block_hashes = request.block_hashes[num_cached_blocks:]

    # 为每个 block 设置 hash 并加入缓存
    for i, blk in enumerate(new_full_blocks):
        assert blk.block_hash is None  # 确保还没被缓存

        block_hash = new_block_hashes[i]
        block_hash_with_group_id = BlockHashWithGroupId(
            block_hash, kv_cache_group_id
        )

        # 设置 block 的 hash
        blk.block_hash = block_hash_with_group_id

        # 加入缓存哈希表
        self.cached_block_hash_to_block[block_hash_with_group_id][
            blk.block_id
        ] = blk

    # 如果启用了 KV cache events，发送 BlockStored 事件
    if self.enable_kv_cache_events:
        self.kv_event_queue.append(
            BlockStored(
                block_hashes=...,
                token_ids=request.all_token_ids[...],
                block_size=block_size,
                lora_id=request.lora_request.id if request.lora_request else None,
                ...
            )
        )
```

**Block Caching 关键点**:
- **只缓存 full blocks**: 未填满的 block 不能被缓存
- **只缓存 finalized tokens**: Speculative tokens 可能被拒绝，不应缓存
- **Block Hash 在 Request 创建时计算**: 使用 token IDs 计算 hash
- **支持 ref_cnt 共享**: 多个请求可以共享同一个 cached block
- **LRU 驱逐策略**: 当 free blocks 不足时，驱逐 `ref_cnt=0` 的 cached blocks

## 核心数据流

### Request 状态转换

```
WAITING (新请求)
  │
  ├─> Prefix Cache 查找
  ├─> KV Cache 分配
  └─> 调度成功
        ↓
      RUNNING (正在执行)
        │
        ├─> 继续分配 KV Cache
        ├─> 如果 KV Cache 不足 → PREEMPTED
        └─> 完成 → FINISHED

PREEMPTED (被抢占)
  │
  └─> 移回 WAITING 队列头部
```

### KV Cache Block 生命周期

```
1. Free (在 free_block_queue)
     ↓ allocate_slots() → get_new_blocks()
2. Allocated (ref_cnt=1)
     ↓ 填充 KV cache tensors
3. Full (block 填满)
     ↓ cache_blocks() → cache_full_blocks()
4. Cached (有 block_hash，在 cached_block_hash_to_block)
     ↓ 其他请求 prefix cache 命中 → touch()
5. Shared (ref_cnt>1)
     ↓ 所有请求完成 → free()
6. Eviction Candidate (ref_cnt=0，仍在 cache)
     ↓ 需要分配新 blocks → _maybe_evict_cached_block()
7. Evicted (block_hash 重置)
     ↓
   回到状态 1
```

### Token Budget 消耗

```
初始: token_budget = max_num_batched_tokens (例如 2048)

┌─────────────────┐
│ RUNNING Req 1   │ num_new_tokens = 512
│ (decode)        │ token_budget = 1536
└─────────────────┘

┌─────────────────┐
│ RUNNING Req 2   │ num_new_tokens = 512
│ (decode)        │ token_budget = 1024
└─────────────────┘

┌─────────────────┐
│ WAITING Req 3   │ num_new_tokens = 1024
│ (prefill chunk) │ token_budget = 0
└─────────────────┘

停止调度 (token_budget = 0)
```

## 调度时序图

```
Scheduler.schedule()
    │
    ├──> [阶段 1] 调度 RUNNING 请求
    │       │
    │       ├─> for each request in self.running:
    │       │     │
    │       │     ├─> 计算 num_new_tokens
    │       │     │     - 考虑 chunked prefill 限制
    │       │     │     - 考虑 token_budget 限制
    │       │     │
    │       │     ├─> KVCacheManager.allocate_slots(request, num_new_tokens)
    │       │     │     │
    │       │     │     ├─> remove_skipped_blocks()
    │       │     │     ├─> get_num_blocks_to_allocate()
    │       │     │     ├─> check free_blocks 是否足够
    │       │     │     │     └─> 如果不足 → return None
    │       │     │     ├─> allocate_new_blocks()
    │       │     │     │     └─> block_pool.get_new_blocks()
    │       │     │     │           └─> free_block_queue.popleft_n()
    │       │     │     └─> cache_blocks()
    │       │     │           └─> block_pool.cache_full_blocks()
    │       │     │
    │       │     └─> 如果 allocate_slots 返回 None:
    │       │           │
    │       │           └─> [Preemption Loop]
    │       │                 │
    │       │                 ├─> 选择 preempted_req (最低优先级)
    │       │                 ├─> KVCacheManager.free(preempted_req)
    │       │                 │     └─> coordinator.free()
    │       │                 │           └─> block_pool.free_blocks()
    │       │                 │                 └─> free_block_queue.append_n()
    │       │                 ├─> waiting.prepend_request(preempted_req)
    │       │                 └─> 重试 allocate_slots()
    │
    ├──> [阶段 2] 调度 WAITING 请求 (如果没有 preemption)
    │       │
    │       └─> while self.waiting and token_budget > 0:
    │             │
    │             ├─> request = self.waiting.peek_request()
    │             │
    │             ├─> [Prefix Caching 查找]
    │             │     │
    │             │     └─> KVCacheManager.get_computed_blocks(request)
    │             │           │
    │             │           ├─> coordinator.find_longest_cache_hit(
    │             │           │       request.block_hashes,
    │             │           │       max_cache_hit_length
    │             │           │   )
    │             │           │     │
    │             │           │     └─> for each block_hash:
    │             │           │           │
    │             │           │           └─> block_pool.get_cached_block(block_hash)
    │             │           │                 │
    │             │           │                 └─> cached_block_hash_to_block.get(hash)
    │             │           │
    │             │           └─> return (computed_blocks, num_computed_tokens)
    │             │
    │             ├─> 计算 num_new_tokens = request.num_tokens - num_computed_tokens
    │             │
    │             ├─> [KV Cache 分配]
    │             │     │
    │             │     └─> KVCacheManager.allocate_slots(
    │             │             request,
    │             │             num_new_tokens,
    │             │             num_new_computed_tokens,
    │             │             new_computed_blocks
    │             │         )
    │             │           │
    │             │           ├─> touch(new_computed_blocks)  # 增加 ref_cnt
    │             │           │     └─> for each block:
    │             │           │           block.ref_cnt += 1
    │             │           │           if block.ref_cnt == 1:
    │             │           │               free_block_queue.remove(block)
    │             │           │
    │             │           ├─> save_new_computed_blocks()
    │             │           ├─> allocate_new_blocks()
    │             │           └─> cache_blocks()
    │             │
    │             ├─> 如果分配成功:
    │             │     │
    │             │     ├─> waiting.pop_request()
    │             │     ├─> running.append(request)
    │             │     ├─> request.status = RUNNING
    │             │     └─> request.num_computed_tokens = num_computed_tokens
    │             │
    │             └─> 如果分配失败:
    │                   └─> break  # 停止调度新请求，不触发 preemption
    │
    └──> 返回 SchedulerOutput
          ├─> scheduled_new_reqs
          ├─> scheduled_resumed_reqs
          ├─> scheduled_running_reqs
          ├─> preempted_reqs
          ├─> req_to_new_blocks
          └─> num_scheduled_tokens
```

## 关键参数和配置

| 参数 | 位置 | 说明 |
|------|------|------|
| `max_num_seqs` | `scheduler_config` | 最大并发请求数 |
| `max_num_batched_tokens` | `scheduler_config` | 每次调度的最大 token 数 (token budget) |
| `long_prefill_token_threshold` | `scheduler_config` | Chunked prefill 的阈值，超过此值会分块处理 |
| `num_lookahead_tokens` | `speculative_config` | Speculative decoding 的预分配 token 数 |
| `enable_prefix_caching` | `cache_config` | 是否启用 prefix caching |
| `block_size` | `cache_config` | 每个 block 的 token 数 (默认 16) |
| `scheduling_policy` | `scheduler_config` | 调度策略: FCFS 或 PRIORITY |

## 总结

vLLM v1 架构的请求调度阶段实现了高效的 KV Cache 管理：

### 核心特性

1. **统一的 Token 调度**
   - 不区分 prefill 和 decode
   - 天然支持 chunked prefill 和 continuous batching

2. **Prefix Caching**
   - 通过 block hash 查找已计算的 KV cache
   - 避免重复计算相同的 prompt
   - 支持多请求共享 (ref_cnt 机制)

3. **灵活的 Block 管理**
   - BlockPool 统一管理所有 blocks
   - LRU 驱逐策略
   - 支持 ref_cnt 共享和驱逐候选

4. **Preemption 机制**
   - 在 KV cache 不足时抢占低优先级请求
   - 支持优先级调度
   - 被抢占请求可以重新调度，利用 prefix cache

5. **多种优化支持**
   - Speculative Decoding: 通过 `num_lookahead_tokens` 预分配
   - Chunked Prefill: 通过 `long_prefill_token_threshold` 分块
   - Sliding Window: 通过 `remove_skipped_blocks()` 释放窗口外的 blocks
   - P/D Disaggregation: 通过 KVConnector 和 `delay_cache_blocks`

### 性能关键

- **Token Budget 管理**: 平衡 prefill 和 decode
- **Prefix Cache 命中率**: 减少重复计算
- **Block 复用**: 多请求共享相同 prefix
- **高效驱逐**: LRU 策略，优先驱逐不常用的 blocks
