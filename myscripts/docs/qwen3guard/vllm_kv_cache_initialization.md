# vLLM v1 架构 KV Cache 初始化流程

> 本文档基于 vLLM v1 架构，使用脚本 `examples/offline_inference/qwen_guard_model.py` 作为入口点，详细介绍 KV Cache 的初始化过程。

## 目录
- [整体架构](#整体架构)
- [初始化流程](#初始化流程)
  - [1. AsyncLLM 初始化](#1-asyncllm-初始化)
  - [2. EngineCoreClient 创建](#2-enginecoreclient-创建)
  - [3. EngineCore 初始化](#3-enginecore-初始化)
  - [4. KV Cache 配置计算](#4-kv-cache-配置计算)
  - [5. Scheduler 初始化](#5-scheduler-初始化)
  - [6. Worker 初始化](#6-worker-初始化)
  - [7. Model Runner 初始化](#7-model-runner-初始化)
- [核心数据结构](#核心数据结构)
- [初始化时序图](#初始化时序图)

---

## 整体架构

vLLM v1 架构的 KV Cache 初始化涉及多个核心组件的协同工作：

```
AsyncLLM (主进程)
  └── EngineCoreClient (IPC通信)
        └── EngineCore (后台进程)
              ├── ModelExecutor
              │     └── Worker (每个GPU一个)
              │           └── GPUModelRunner
              │                 ├── KV Cache Tensors (物理内存)
              │                 └── attention_backend
              │
              └── Scheduler
                    └── KVCacheManager (逻辑管理)
                          ├── coordinator: KVCacheCoordinator
                          │     ├── block_pool: BlockPool
                          │     │     ├── blocks: list[KVCacheBlock]
                          │     │     ├── free_block_queue (双向链表)
                          │     │     └── cached_block_hash_to_block (哈希表)
                          │     │
                          │     └── single_type_managers
                          │           ├── FullAttentionManager
                          │           ├── SlidingWindowManager
                          │           └── MambaManager
                          │
                          └── 高层 API (allocate_slots, free, ...)
```

**分层架构说明**:
- **物理层** (Worker/GPUModelRunner): 实际的 GPU tensor 分配
- **元数据层** (BlockPool): KV cache blocks 的元数据管理
- **逻辑层** (SingleTypeManager): 不同 attention 类型的逻辑
- **协调层** (Coordinator): 多种 KV cache 类型的协调
- **接口层** (KVCacheManager): 统一的高层接口
- **调度层** (Scheduler): 请求调度和资源分配决策

## 初始化流程

### 1. AsyncLLM 初始化

**位置**: `vllm/v1/engine/async_llm.py:53-156`

AsyncLLM 是 v1 架构的主入口，通过 `AsyncLLM.from_engine_args()` 创建：

```python
# examples/offline_inference/qwen_guard_model.py:400-404
def init_guard_engine_v1(engine_loop, engine_args):
    engine_args.runner = "pooling"
    engine_args.disable_log_stats = True
    engine_usage_context = UsageContext.API_SERVER
    return AsyncLLM.from_engine_args(engine_args, usage_context=engine_usage_context)
```

**关键步骤**:
- 创建 `vllm_config` 从 `engine_args` (async_llm.py:228)
- 获取 executor 类 (async_llm.py:229)
- 创建 Processor（输入预处理器）(async_llm.py:118-122)
- 创建 OutputProcessor（输出后处理器）(async_llm.py:125-126)
- **创建 EngineCoreClient** (async_llm.py:129-136) ← KV Cache 初始化的起点

### 2. EngineCoreClient 创建

**位置**: `vllm/v1/engine/core_client.py:84-102`

EngineCoreClient 负责与后台 EngineCore 进程通信，通过 ZMQ 实现 IPC：

```python
# async_llm.py:129-136
self.engine_core = EngineCoreClient.make_async_mp_client(
    vllm_config=vllm_config,
    executor_class=executor_class,
    log_stats=self.log_stats,
    client_addresses=client_addresses,
    client_count=client_count,
    client_index=client_index,
)
```

**关键步骤**:
- 创建 ZMQ 上下文和 socket (core_client.py:431-432)
- **启动 EngineCore 进程** (core_client.py:453-456)：
  ```python
  with launch_core_engines(vllm_config, executor_class, log_stats) as (
      engine_manager, coordinator, addresses):
  ```
- 建立 input/output socket 连接 (core_client.py:469-472)
- 等待所有 engine 发送 ready 消息 (core_client.py:497-504)

### 3. EngineCore 初始化

**位置**: `vllm/v1/engine/core.py:65-130`

EngineCore 是 v1 引擎的核心调度器，运行在独立进程中：

```python
# core.py:65-130
def __init__(self, vllm_config, executor_class, log_stats, executor_fail_callback):
    # 1. 创建 model executor
    self.model_executor = executor_class(vllm_config)  # 第 82 行

    # 2. 初始化 KV Cache
    num_gpu_blocks, num_cpu_blocks, kv_cache_config = \
        self._initialize_kv_caches(vllm_config)  # 第 90-91 行

    # 3. 通知所有 worker 初始化 cache
    self.collective_rpc("initialize_cache",
                        args=(num_gpu_blocks, num_cpu_blocks))  # 第 95-96 行

    # 4. 创建 Scheduler
    self.scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        ...
    )  # 第 123-130 行
```

### 4. KV Cache 配置计算

**位置**: `vllm/v1/engine/core.py:162-220`

`_initialize_kv_caches()` 是 KV Cache 初始化的核心方法：

#### 4.1 获取 KV Cache 规格

```python
# core.py:167
kv_cache_specs = self.model_executor.get_kv_cache_specs()
```

- 通过 `collective_rpc` 调用所有 worker 的 `get_kv_cache_spec()` (executor/abstract.py:86-87)
- Worker 通过 model runner 获取模型的 KV cache 需求 (worker/gpu_worker.py:282-283)
- 返回每个 attention layer 的 KV cache 规格（数据类型、形状等）

#### 4.2 测量可用 GPU 内存

```python
# core.py:182-183
available_gpu_memory = self.model_executor.determine_available_memory()
self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
```

**内存测量过程** (worker/gpu_worker.py:222-280):
1. 清空 CUDA cache 和重置内存统计 (gpu_worker.py:234-235)
2. 执行 dummy forward pass 来 profile 内存使用 (gpu_worker.py:244)
3. 计算可用内存:
   ```python
   available_kv_cache_memory = requested_memory - non_kv_cache_memory
   ```
   其中:
   - `requested_memory = total_memory * gpu_memory_utilization`
   - `non_kv_cache_memory = weights_memory + peak_memory_during_profile`

#### 4.3 计算 KV Cache 配置

```python
# core.py:192-197
kv_cache_configs = [
    get_kv_cache_config(vllm_config, kv_cache_spec_one_worker,
                        available_gpu_memory_one_worker)
    for kv_cache_spec_one_worker, available_gpu_memory_one_worker in
    zip(kv_cache_specs, available_gpu_memory)
]
```

`get_kv_cache_config()` 计算:
- **每个 block 的大小**: `block_size` (例如 16 tokens)
- **总 block 数**: `num_blocks = available_memory / (num_layers * kv_size_per_block)`
- **KV cache groups**: 将相同 KV cache 配置的层分组

#### 4.4 统一所有 worker 的配置

```python
# core.py:202
unify_kv_cache_configs(kv_cache_configs)
```

- 确保所有 worker 使用相同的 block 数量
- 使用最小的 `num_blocks`，确保所有 worker 都能分配成功

#### 4.5 初始化 Model Executor

```python
# core.py:215
self.model_executor.initialize_from_config(kv_cache_configs)
```

这会调用所有 worker 的初始化方法 (executor/abstract.py:66-74):
```python
def initialize_from_config(self, kv_cache_configs):
    self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))
    self.collective_rpc("compile_or_warm_up_model")
```

### 5. Scheduler 初始化

**位置**: `vllm/v1/core/sched/scheduler.py:43-174`

Scheduler 负责请求调度和 KV cache 的逻辑管理：

```python
# scheduler.py:123-174
def __init__(self, vllm_config, kv_cache_config, ...):
    # 设置调度约束
    self.max_num_running_reqs = self.scheduler_config.max_num_seqs  # 第 71 行
    self.max_num_scheduled_tokens = \
        self.scheduler_config.max_num_batched_tokens  # 第 72-73 行

    # 创建 KVCacheManager
    self.kv_cache_manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=self.max_model_len,
        enable_caching=self.cache_config.enable_prefix_caching,
        use_eagle=self.use_eagle,
        log_stats=self.log_stats,
        enable_kv_cache_events=self.enable_kv_cache_events,
        dcp_world_size=self.dcp_world_size,
    )  # 第 166-174 行
```

#### 5.1 KVCacheManager 初始化

**位置**: `vllm/v1/core/kv_cache_manager.py:86-131`

KVCacheManager 是 KV cache 的高层管理器，协调所有 KV cache 相关操作：

```python
# kv_cache_manager.py:86-131
def __init__(self, kv_cache_config, max_model_len, enable_caching, ...):
    self.max_model_len = max_model_len
    self.enable_caching = enable_caching

    # 1. 提取 block_size（所有 group 必须使用相同的 block_size）
    self.block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size

    # DCP (Decode Context Parallel) 支持
    if dcp_world_size > 1:
        self.block_size *= dcp_world_size  # 第 118 行

    # 2. 创建 KVCacheCoordinator
    self.coordinator = get_kv_cache_coordinator(
        kv_cache_config=kv_cache_config,
        max_model_len=self.max_model_len,
        use_eagle=self.use_eagle,
        enable_caching=self.enable_caching,
        enable_kv_cache_events=enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
    )  # 第 120-127 行

    # 3. 获取 BlockPool 的引用
    self.block_pool = self.coordinator.block_pool  # 第 129 行
```

**KVCacheManager 职责**:
- 管理 KV cache blocks 的分配和释放
- 实现 prefix caching（前缀缓存）
- 跟踪每个 request 的 KV cache 使用情况
- 支持 preemption（抢占）和 swapping（交换）

#### 5.2 KVCacheCoordinator 初始化

**位置**: `vllm/v1/core/kv_cache_coordinator.py:15-46`

KVCacheCoordinator 协调不同 KV cache groups 的管理，根据模型配置选择不同的实现：

```python
# kv_cache_coordinator.py:417-440
def get_kv_cache_coordinator(...) -> KVCacheCoordinator:
    if not enable_caching:
        # 禁用 prefix caching 时使用
        return KVCacheCoordinatorNoPrefixCache(...)

    if len(kv_cache_config.kv_cache_groups) == 1:
        # 单一 KV cache group（大多数模型）
        return UnitaryKVCacheCoordinator(...)

    # 混合 KV cache groups（例如 Mamba + Attention 混合模型）
    return HybridKVCacheCoordinator(...)
```

**Coordinator 类型**:

1. **KVCacheCoordinatorNoPrefixCache**:
   - 用于禁用 prefix caching 的场景
   - 不实现缓存查找功能
   - 支持任意数量的 KV cache groups（包括 0 个）

2. **UnitaryKVCacheCoordinator** (最常见):
   - 用于只有一种 KV cache 类型的模型
   - 所有 attention layers 使用相同的 attention 类型（如 full attention）
   - 实现完整的 prefix caching 功能

3. **HybridKVCacheCoordinator**:
   - 用于混合模型（如 Mamba + Attention）
   - 支持多种 KV cache 类型的组合
   - 需要一种类型是 full attention

**Coordinator 初始化过程** (kv_cache_coordinator.py:20-45):
```python
def __init__(self, kv_cache_config, max_model_len, ...):
    # 1. 创建 BlockPool
    self.block_pool = BlockPool(
        kv_cache_config.num_blocks,
        enable_caching,
        enable_kv_cache_events
    )  # 第 33-34 行

    # 2. 为每个 KV cache group 创建对应的 SingleTypeManager
    self.single_type_managers = tuple(
        get_manager_for_kv_cache_spec(
            kv_cache_spec=kv_cache_group.kv_cache_spec,
            block_pool=self.block_pool,
            kv_cache_group_id=i,
            dcp_world_size=dcp_world_size,
        ) for i, kv_cache_group in enumerate(
            self.kv_cache_config.kv_cache_groups)
    )  # 第 38-45 行
```

#### 5.3 BlockPool 初始化

**位置**: `vllm/v1/core/block_pool.py:32-69`

BlockPool 是 KV cache blocks 的底层内存池管理器：

```python
# block_pool.py:32-69
def __init__(self, num_gpu_blocks, enable_caching, enable_kv_cache_events):
    self.num_gpu_blocks = num_gpu_blocks
    self.enable_caching = enable_caching

    # 1. 创建所有 KV cache blocks
    self.blocks = [
        KVCacheBlock(idx) for idx in range(num_gpu_blocks)
    ]  # 第 42-44 行

    # 2. 创建 free block queue（双向链表）
    self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)  # 第 48 行

    # 3. 创建缓存映射表（用于 prefix caching）
    # {(block_hash, group_id): {block_id: block}}
    self.cached_block_hash_to_block = defaultdict(dict)  # 第 59-60 行

    # 4. 预留 null_block（block_id=0 的占位符）
    self.null_block = self.free_block_queue.popleft()  # 第 65 行
    self.null_block.is_null = True  # 第 66 行

    # 5. KV cache events 队列（用于事件追踪）
    self.enable_kv_cache_events = enable_kv_cache_events
    self.kv_event_queue = []  # 第 69 行
```

**BlockPool 核心数据结构**:

1. **blocks**: `list[KVCacheBlock]`
   - 所有 KV cache blocks 的列表
   - 每个 block 有唯一的 `block_id`
   - block 状态：free（未使用）、allocated（已分配）、cached（可被驱逐）

2. **free_block_queue**: `FreeKVCacheBlockQueue`
   - 双向链表结构，按驱逐优先级排序
   - LRU (Least Recently Used) 策略
   - 支持高效的 `popleft()` 和 `append()` 操作

3. **cached_block_hash_to_block**: `dict[BlockHashWithGroupId, dict[int, KVCacheBlock]]`
   - 用于 prefix caching 的哈希表
   - 外层 key: `(block_hash, kv_cache_group_id)`
   - 内层 key: `block_id`
   - 内层 value: `KVCacheBlock` 对象
   - 支持快速查找相同 prefix 的缓存 blocks

4. **null_block**: `KVCacheBlock`
   - 特殊的占位符 block（`block_id=0`）
   - 用于表示不需要 KV cache 的位置（如 sliding window 之外的 tokens）
   - `ref_cnt` 不被维护，需要特殊处理

**KVCacheBlock 结构**:
```python
class KVCacheBlock:
    block_id: int                          # 唯一 ID
    ref_cnt: int                           # 引用计数
    block_hash: Optional[BlockHashWithGroupId]  # 缓存哈希
    is_null: bool                          # 是否是 null block
    # ... 双向链表指针等
```

#### 5.4 SingleTypeKVCacheManager 初始化

**位置**: `vllm/v1/core/single_type_kv_cache_manager.py`

为每种 KV cache 类型创建对应的管理器：

```python
def get_manager_for_kv_cache_spec(kv_cache_spec, block_pool, ...):
    if isinstance(kv_cache_spec, FullAttentionSpec):
        return FullAttentionManager(block_pool, kv_cache_group_id)
    elif isinstance(kv_cache_spec, SlidingWindowSpec):
        return SlidingWindowAttentionManager(...)
    elif isinstance(kv_cache_spec, MambaSpec):
        return MambaManager(...)
    # ...
```

**管理器类型**:
- **FullAttentionManager**: 管理 full attention 的 KV cache
- **SlidingWindowAttentionManager**: 管理 sliding window attention
- **MambaManager**: 管理 Mamba 架构的状态
- **CrossAttentionManager**: 管理 encoder-decoder 的 cross-attention

每个管理器负责：
- `allocate_new_blocks()`: 为请求分配新的 blocks
- `free()`: 释放请求占用的 blocks
- `cache_blocks()`: 缓存已完成的 blocks（prefix caching）
- `find_longest_cache_hit()`: 查找最长的缓存命中
- `remove_skipped_blocks()`: 移除不需要的 blocks（如 sliding window 外的）

### 6. Worker 初始化

**位置**: `vllm/v1/worker/gpu_worker.py:285-296`

每个 GPU 都有一个 Worker 实例，负责实际的模型执行：

```python
# gpu_worker.py:285-296
def initialize_from_config(self, kv_cache_config):
    """分配 GPU KV cache"""
    if self.vllm_config.model_config.enable_sleep_mode:
        # 使用 memory pool 管理
        allocator = CuMemAllocator.get_instance()
        context = allocator.use_memory_pool(tag="kv_cache")
    else:
        context = nullcontext()

    with context:
        self.model_runner.initialize_kv_cache(kv_cache_config)  # 第 296 行
```

### 7. Model Runner 初始化

**位置**: `vllm/v1/worker/gpu_model_runner.py:3376-3414`

GPUModelRunner 执行最终的 KV cache tensor 分配：

```python
# gpu_model_runner.py:3376-3414
def initialize_kv_cache(self, kv_cache_config):
    # 1. 保存配置
    self.kv_cache_config = kv_cache_config  # 第 3384 行

    # 2. 可能重新初始化 input batch
    self.may_reinitialize_input_batch(kv_cache_config)  # 第 3385 行

    # 3. 添加 encoder-only layers
    self.may_add_encoder_only_layers_to_kv_cache_config()  # 第 3386 行

    # 4. 添加 KV sharing layers
    self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)  # 第 3387 行

    # 5. 初始化 attention backend
    self.initialize_attn_backend(kv_cache_config)  # 第 3388 行

    # 6. 初始化 KV cache tensors
    kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)  # 第 3389 行

    # 7. 注册到 KV transfer group（用于 P/D 和 offloading）
    if has_kv_transfer_group():
        get_kv_transfer_group().register_kv_caches(kv_caches)  # 第 3397-3398 行
```

#### 7.1 初始化 KV Cache Tensors

**位置**: `vllm/v1/worker/gpu_model_runner.py:3318-3345`

```python
def initialize_kv_cache_tensors(self, kv_cache_config):
    # 1. 分配原始内存
    kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)

    # 2. 重塑为期望的形状
    kv_caches = self._reshape_kv_cache_tensors(kv_cache_config,
                                               kv_cache_raw_tensors)

    # 3. 设置 cross-layer KV cache sharing
    for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
        kv_caches[layer_name] = kv_caches[target_layer_name]

    # 4. 绑定 KV cache 到持久化列表
    bind_kv_cache(kv_caches, self.compilation_config.static_forward_context,
                  self.kv_caches)

    return kv_caches
```

**KV Cache Tensor 形状**:
- 对于标准 attention: `(2, num_blocks, num_heads, block_size, head_size)`
  - `2` 代表 Key 和 Value
  - `num_blocks` 是计算得到的 block 总数
  - `block_size` 默认为 16
- 对于 MLA (Multi-head Latent Attention): 形状不同，取决于模型架构

## 核心数据结构

### KVCacheConfig

**位置**: `vllm/v1/kv_cache_interface.py`

```python
@dataclass
class KVCacheConfig:
    num_blocks: int                          # 总 block 数
    block_size: int                          # 每个 block 的 token 数（废弃，现在在 spec 中）
    kv_cache_groups: list[KVCacheGroupSpec]  # KV cache 分组
```

**计算公式**:
```python
num_blocks = available_gpu_memory / (num_layers * kv_size_per_block)
kv_size_per_block = block_size * num_heads * head_size * num_kv_groups * dtype_size * 2  # Key + Value
```

### KVCacheGroupSpec

**位置**: `vllm/v1/kv_cache_interface.py`

```python
@dataclass
class KVCacheGroupSpec:
    group_id: int                                      # 组 ID
    kv_cache_spec: Union[AttentionSpec, MambaSpec, ...]  # KV cache 规格
    layer_names: list[str]                            # 该组包含的层名称
```

**KVCacheSpec 类型**:
```python
# 1. Full Attention (最常见)
@dataclass
class FullAttentionSpec:
    block_size: int          # 默认 16
    num_kv_heads: int        # KV heads 数量
    head_size: int           # head 维度
    dtypes: list[torch.dtype]  # [Key dtype, Value dtype]

# 2. Sliding Window Attention
@dataclass
class SlidingWindowSpec:
    block_size: int
    sliding_window: int      # 窗口大小
    # ... 其他 attention 参数

# 3. Chunked Local Attention
@dataclass
class ChunkedLocalAttentionSpec:
    block_size: int
    chunk_size: int          # chunk 大小
    # ...

# 4. Mamba (SSM)
@dataclass
class MambaSpec:
    block_size: int
    state_size: int          # 状态维度
    # ...
```

### KVCacheManager 组件层次

```
KVCacheManager (高层接口)
  └── coordinator: KVCacheCoordinator (协调器)
        ├── block_pool: BlockPool (内存池)
        │     ├── blocks: list[KVCacheBlock]
        │     ├── free_block_queue: FreeKVCacheBlockQueue
        │     └── cached_block_hash_to_block: dict
        │
        └── single_type_managers: tuple[SingleTypeKVCacheManager, ...]
              ├── FullAttentionManager
              ├── SlidingWindowAttentionManager
              ├── MambaManager
              └── CrossAttentionManager
```

### KVCacheBlock

**位置**: `vllm/v1/core/kv_cache_utils.py`

```python
class KVCacheBlock:
    """单个 KV cache block 的元数据"""

    block_id: int                               # 唯一 ID (0 到 num_blocks-1)
    ref_cnt: int                                # 引用计数（被多少个请求使用）
    block_hash: Optional[BlockHashWithGroupId]  # 缓存哈希（用于 prefix caching）
    is_null: bool                               # 是否是 null block（占位符）

    # 双向链表指针（用于 free_block_queue）
    prev: Optional[KVCacheBlock]
    next: Optional[KVCacheBlock]
```

**Block 生命周期**:
```
1. Free (ref_cnt=0, 在 free_block_queue 中)
   ↓ allocate
2. Allocated (ref_cnt≥1, 不在 free_block_queue)
   ↓ 填充完整 + 计算 hash
3. Cached (ref_cnt≥1, block_hash != None, 在 cached_block_hash_to_block)
   ↓ ref_cnt 减至 0
4. Eviction Candidate (ref_cnt=0, 仍在 cached_block_hash_to_block)
   ↓ 被驱逐
5. Evicted (ref_cnt=0, block_hash 被重置)
   回到状态 1
```

### BlockPool 核心方法

```python
class BlockPool:
    def get_new_blocks(num_blocks: int) -> list[KVCacheBlock]:
        """从 free queue 中分配新 blocks"""
        # 1. 从 free_block_queue popleft num_blocks 个 blocks
        # 2. 如果 block 有 hash，驱逐它（_maybe_evict_cached_block）
        # 3. 设置 ref_cnt = 1
        # 4. 返回 blocks

    def free_blocks(ordered_blocks: Iterable[KVCacheBlock]):
        """释放 blocks 回到 free queue"""
        # 1. 减少每个 block 的 ref_cnt
        # 2. 将 ref_cnt=0 的 blocks append 到 free_block_queue

    def touch(blocks: tuple[list[KVCacheBlock], ...]):
        """增加 blocks 的引用计数（prefix cache 命中）"""
        # 1. ref_cnt += 1
        # 2. 如果 ref_cnt 从 0 变为 1，从 free_block_queue 中移除

    def cache_full_blocks(...):
        """缓存已填充完整的 blocks（prefix caching）"""
        # 1. 设置 block.block_hash
        # 2. 添加到 cached_block_hash_to_block
        # 3. 发送 BlockStored 事件（如果启用）

    def get_cached_block(block_hash, kv_cache_group_ids):
        """查找缓存的 block（prefix caching）"""
        # 返回匹配 hash 的第一个 block
```

### KVCacheCoordinator 方法

```python
class KVCacheCoordinator:
    def get_num_blocks_to_allocate(...) -> int:
        """计算需要分配的 block 数"""
        # 累加每个 single_type_manager 的需求

    def allocate_new_blocks(...) -> tuple[list[KVCacheBlock], ...]:
        """为请求分配新 blocks"""
        # 每个 group 调用对应的 manager.allocate_new_blocks()

    def find_longest_cache_hit(...) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """查找最长的 prefix cache 命中"""
        # UnitaryCoordinator: 直接查找
        # HybridCoordinator: 先查 full attention，再查其他类型

    def cache_blocks(request: Request, num_computed_tokens: int):
        """缓存请求的 blocks"""
        # 每个 group 调用对应的 manager.cache_blocks()

    def free(request_id: str):
        """释放请求的所有 blocks"""
        # 每个 group 调用对应的 manager.free()
```

### SingleTypeKVCacheManager 状态

每个 SingleTypeKVCacheManager 维护：

```python
class SingleTypeKVCacheManager:
    block_pool: BlockPool                        # BlockPool 引用
    kv_cache_group_id: int                      # 管理的 group ID

    # 请求 -> blocks 映射
    req_to_blocks: dict[str, list[KVCacheBlock]]

    # 请求 -> 已缓存的 block 数
    req_to_num_cached_blocks: dict[str, int]

    def get_num_blocks_to_allocate(request_id, num_tokens, new_computed_blocks):
        """计算需要新分配多少个 blocks"""
        current_blocks = len(self.req_to_blocks.get(request_id, []))
        needed_blocks = (num_tokens + block_size - 1) // block_size
        return max(0, needed_blocks - current_blocks - len(new_computed_blocks))

    def allocate_new_blocks(request_id, num_tokens):
        """分配新 blocks"""
        num_to_allocate = self.get_num_blocks_to_allocate(...)
        new_blocks = self.block_pool.get_new_blocks(num_to_allocate)
        self.req_to_blocks[request_id].extend(new_blocks)
        return new_blocks
```

## 初始化时序图

```
User Script (qwen_guard_model.py)
    │
    ├─> AsyncLLM.from_engine_args()
    │       │
    │       ├─> EngineCoreClient.make_async_mp_client()
    │       │       │
    │       │       └─> launch_core_engines()  [启动后台进程]
    │       │               │
    │       │               └─> EngineCore.__init__()
    │       │                       │
    │       │                       ├─> model_executor = Executor()
    │       │                       │
    │       │                       ├─> _initialize_kv_caches()
    │       │                       │       │
    │       │                       │       ├─> get_kv_cache_specs()
    │       │                       │       │       └─> [Worker] get_kv_cache_spec()
    │       │                       │       │
    │       │                       │       ├─> determine_available_memory()
    │       │                       │       │       └─> [Worker] 执行 profile_run()
    │       │                       │       │
    │       │                       │       ├─> get_kv_cache_config()
    │       │                       │       │       └─> 计算 num_blocks
    │       │                       │       │
    │       │                       │       ├─> unify_kv_cache_configs()
    │       │                       │       │
    │       │                       │       └─> model_executor.initialize_from_config()
    │       │                       │               │
    │       │                       │               └─> [Worker] initialize_from_config()
    │       │                       │                       │
    │       │                       │                       └─> model_runner.initialize_kv_cache()
    │       │                       │                               │
    │       │                       │                               ├─> initialize_attn_backend()
    │       │                       │                               │
    │       │                       │                               ├─> _allocate_kv_cache_tensors()
    │       │                       │                               │       └─> torch.empty(...)
    │       │                       │                               │
    │       │                       │                               ├─> _reshape_kv_cache_tensors()
    │       │                       │                               │
    │       │                       │                               └─> bind_kv_cache()
    │       │                       │
    │       │                       ├─> collective_rpc("initialize_cache")
    │       │                       │       └─> [Worker] initialize_cache()
    │       │                       │
    │       │                       └─> Scheduler.__init__()
    │       │                               │
    │       │                               └─> KVCacheManager.__init__()
    │       │                                       │
    │       │                                       ├─> 提取 block_size
    │       │                                       │
    │       │                                       ├─> get_kv_cache_coordinator()
    │       │                                       │     │
    │       │                                       │     └─> 选择 Coordinator 类型:
    │       │                                       │           - UnitaryKVCacheCoordinator (最常见)
    │       │                                       │           - HybridKVCacheCoordinator
    │       │                                       │           - KVCacheCoordinatorNoPrefixCache
    │       │                                       │
    │       │                                       └─> KVCacheCoordinator.__init__()
    │       │                                             │
    │       │                                             ├─> BlockPool.__init__()
    │       │                                             │     │
    │       │                                             │     ├─> 创建所有 KVCacheBlock
    │       │                                             │     ├─> 初始化 free_block_queue
    │       │                                             │     ├─> 初始化 cached_block_hash_to_block
    │       │                                             │     └─> 预留 null_block
    │       │                                             │
    │       │                                             └─> 创建 single_type_managers
    │       │                                                   └─> get_manager_for_kv_cache_spec()
    │       │                                                         ├─> FullAttentionManager
    │       │                                                         ├─> SlidingWindowManager
    │       │                                                         └─> MambaManager
    │       │
    │       └─> 创建 Processor, OutputProcessor
    │
    └─> AsyncLLM 准备就绪，可以处理请求
```

## 关键文件位置索引

| 组件 | 文件路径 | 关键方法 |
|------|---------|---------|
| AsyncLLM | `vllm/v1/engine/async_llm.py` | `__init__:53-156`, `from_engine_args:218-240` |
| EngineCoreClient | `vllm/v1/engine/core_client.py` | `make_async_mp_client:84-102`, `MPClient.__init__:417-520` |
| EngineCore | `vllm/v1/engine/core.py` | `__init__:65-130`, `_initialize_kv_caches:162-220` |
| Scheduler | `vllm/v1/core/sched/scheduler.py` | `__init__:43-174` |
| **KVCacheManager** | `vllm/v1/core/kv_cache_manager.py` | `__init__:86-131`, `allocate_slots:192-300`, `get_computed_blocks:153-190` |
| **KVCacheCoordinator** | `vllm/v1/core/kv_cache_coordinator.py` | `__init__:20-45`, `get_kv_cache_coordinator:417-440` |
| **BlockPool** | `vllm/v1/core/block_pool.py` | `__init__:32-69`, `get_new_blocks:163-190`, `free_blocks:243-258`, `touch:227-241` |
| **SingleTypeManager** | `vllm/v1/core/single_type_kv_cache_manager.py` | `FullAttentionManager:249-290`, `get_manager_for_kv_cache_spec` |
| **KVCacheBlock** | `vllm/v1/core/kv_cache_utils.py` | 数据结构定义 |
| Executor | `vllm/v1/executor/abstract.py` | `initialize_from_config:66-74` |
| Worker | `vllm/v1/worker/gpu_worker.py` | `initialize_from_config:285-296`, `determine_available_memory:222-280` |
| GPUModelRunner | `vllm/v1/worker/gpu_model_runner.py` | `initialize_kv_cache:3376-3414`, `initialize_kv_cache_tensors:3318-3345` |

## 总结

vLLM v1 架构的 KV Cache 初始化是一个分层、协同的过程：

1. **AsyncLLM** 创建整体引擎实例
2. **EngineCoreClient** 启动后台进程并建立 IPC 通信
3. **EngineCore** 协调所有组件的初始化
4. **ModelExecutor** 通过 profile 确定可用内存并计算 block 数量
5. **Scheduler** 创建 KVCacheManager 管理 KV cache 的逻辑分配
6. **Worker** 在每个 GPU 上初始化实际的 tensor
7. **GPUModelRunner** 分配和绑定 KV cache tensors

这种设计实现了：
- **内存高效**: 通过 profiling 最大化 KV cache 使用
- **灵活扩展**: 支持多种 attention 类型和 KV cache 配置
- **性能优化**: 支持 prefix caching、KV sharing 等优化技术
- **分布式友好**: 支持 DP、TP、PP 等并行策略
