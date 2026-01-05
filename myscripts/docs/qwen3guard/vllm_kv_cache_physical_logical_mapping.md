# vLLM v1 架构：物理 KV Cache Tensor 与逻辑管理的关联

> 本文档详细解释 vLLM v1 架构中物理 KV Cache tensor（GPU 内存）如何与逻辑管理组件（BlockPool、InputBatch、block_table）关联起来。

## 目录
- [整体架构](#整体架构)
- [三层架构](#三层架构)
  - [物理层：KV Cache Tensors](#物理层kv-cache-tensors)
  - [元数据层：BlockPool](#元数据层blockpool)
  - [执行层：InputBatch 和 BlockTable](#执行层inputbatch-和-blocktable)
- [数据流转](#数据流转)
  - [1. 初始化阶段](#1-初始化阶段)
  - [2. 调度阶段](#2-调度阶段)
  - [3. 执行准备阶段](#3-执行准备阶段)
  - [4. 模型执行阶段](#4-模型执行阶段)
- [关键映射关系](#关键映射关系)
- [完整示例](#完整示例)

---

## 整体架构

vLLM v1 使用三层架构来管理 KV Cache：

```
┌─────────────────────────────────────────────────────────────┐
│                       Scheduler (调度层)                      │
│  - 决定哪些请求被调度                                          │
│  - KVCacheManager 分配 logical blocks                         │
│  - 输出: block_ids (逻辑 ID)                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │ SchedulerOutput
                           │ req_to_new_blocks: {req_id: block_ids}
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   Worker (执行准备层)                          │
│  - GPUModelRunner._update_states()                          │
│  - 将 block_ids 填入 InputBatch.block_table                  │
│  - InputBatch.block_table.compute_slot_mapping()            │
└──────────────────────────┬──────────────────────────────────┘
                           │ slot_mapping
                           │ (token position → KV cache tensor index)
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Attention Backend (模型执行层)                    │
│  - 使用 slot_mapping 和 block_table                          │
│  - 直接索引 KV cache tensors                                 │
│  - kv_cache[slot_mapping[i]] = new_kv                       │
└─────────────────────────────────────────────────────────────┘
```

**核心问题**：
- **block_id** (逻辑): Scheduler/KVCacheManager 分配的抽象 ID (0, 1, 2, ...)
- **tensor index** (物理): KV cache tensor 中的实际索引
- **映射方式**: `slot_mapping = block_id * block_size + block_offset`

## 三层架构

### 物理层：KV Cache Tensors

**位置**: `GPUModelRunner.initialize_kv_cache_tensors()`

物理层是实际的 GPU tensor，存储所有请求的 KV cache。

```python
# gpu_model_runner.py:3318-3345
def initialize_kv_cache_tensors(self, kv_cache_config):
    # 1. 分配原始内存
    kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)

    # 2. 重塑为期望的形状
    kv_caches = self._reshape_kv_cache_tensors(
        kv_cache_config, kv_cache_raw_tensors
    )

    # 3. 绑定到 self.kv_caches
    bind_kv_cache(kv_caches, ..., self.kv_caches)

    return kv_caches
```

**KV Cache Tensor 形状**（以 Full Attention 为例）:

```python
# 对于每个 attention layer
kv_cache_shape = (
    2,              # Key 和 Value
    num_blocks,     # 总 block 数（例如 1000）
    num_heads,      # attention heads 数
    block_size,     # 每个 block 的 token 数（默认 16）
    head_size       # head 维度
)

# 例如：Llama-7B，1000 blocks，block_size=16
kv_cache.shape = (2, 1000, 32, 16, 128)
# 总共可以存储: 1000 * 16 = 16000 tokens
```

**关键点**：
- `num_blocks` 是在初始化时计算的，基于可用 GPU 内存
- `block_id` 的范围是 `[0, num_blocks-1]`
- **block_id 直接作为 tensor 的第二维索引**

**示例**：
```python
# 写入 KV cache (在 attention kernel 中)
block_id = 5
block_offset = 3  # token 在 block 中的位置
slot_index = block_id * block_size + block_offset  # 5 * 16 + 3 = 83

# 访问 KV cache
kv_cache[0, block_id, :, block_offset, :]  # Key
kv_cache[1, block_id, :, block_offset, :]  # Value
```

### 元数据层：BlockPool

**位置**: `vllm/v1/core/block_pool.py`

BlockPool 维护所有 blocks 的元数据，但不直接访问物理 tensor。

```python
# block_pool.py:32-69
class BlockPool:
    def __init__(self, num_gpu_blocks, ...):
        # 创建所有 block 的元数据
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # idx 就是 block_id，范围 [0, num_gpu_blocks-1]

        # Free block queue (LRU)
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Prefix caching 哈希表
        self.cached_block_hash_to_block = defaultdict(dict)
```

**KVCacheBlock 元数据**:

```python
class KVCacheBlock:
    block_id: int           # 对应 tensor 的索引 (0 到 num_blocks-1)
    ref_cnt: int            # 引用计数
    block_hash: Optional[...] # 用于 prefix caching
    is_null: bool           # 是否是 null block (占位符)
    # 双向链表指针
    prev: Optional[KVCacheBlock]
    next: Optional[KVCacheBlock]
```

**关键映射**：
```
BlockPool.blocks[i].block_id == i
→ 对应 kv_cache[0/1, i, :, :, :]
```

### 执行层：InputBatch 和 BlockTable

**位置**: `vllm/v1/worker/gpu_input_batch.py` 和 `vllm/v1/worker/block_table.py`

执行层将逻辑 block_ids 转换为物理 tensor 索引。

#### InputBatch

```python
# gpu_input_batch.py:72-130
class InputBatch:
    def __init__(self, ...):
        # Block table for all KV cache groups
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            block_sizes=block_sizes,  # 每个 KV cache group 的 block_size
        )
```

#### MultiGroupBlockTable

```python
# block_table.py:156-216
class MultiGroupBlockTable:
    """支持多个 KV cache groups (例如 Mamba + Attention 混合模型)"""

    def __init__(self, ...):
        self.block_tables = [
            BlockTable(block_size, max_num_reqs,
                      max_num_blocks_per_req, ...)
            for block_size in block_sizes
        ]
```

#### BlockTable (核心)

```python
# block_table.py:14-154
class BlockTable:
    def __init__(self, block_size, max_num_reqs, max_num_blocks_per_req, ...):
        self.block_size = block_size  # 例如 16

        # 二维表：[max_num_reqs, max_num_blocks_per_req]
        # 存储每个请求的 block_ids
        self.block_table_np = np.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            dtype=np.int32
        )

        # GPU 版本
        self.block_table = torch.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            device=device,
            dtype=torch.int32,
        )

        # Slot mapping: token position → KV cache tensor index
        self.slot_mapping_np = np.zeros(
            max_num_batched_tokens, dtype=np.int64
        )
        self.slot_mapping = torch.zeros(
            max_num_batched_tokens, dtype=torch.int64, device=device
        )
```

**BlockTable 的数据结构示意**：

```
假设有 3 个请求，每个请求最多 4 个 blocks：

block_table_np:
┌───────┬───────┬───────┬───────┐
│ Req 0 │   5   │   7   │   9   │  10  │  (block_ids)
├───────┼───────┼───────┼───────┤
│ Req 1 │   2   │   3   │   0   │   0  │
├───────┼───────┼───────┼───────┤
│ Req 2 │   1   │   4   │   8   │  11  │
└───────┴───────┴───────┴───────┘

num_blocks_per_row = [4, 2, 4]
```

**关键方法**：

```python
def append_row(self, block_ids: list[int], row_idx: int):
    """将新的 block_ids 添加到指定请求的 block table 行"""
    num_blocks = len(block_ids)
    start = self.num_blocks_per_row[row_idx]
    self.num_blocks_per_row[row_idx] += num_blocks
    self.block_table_np[row_idx, start:start + num_blocks] = block_ids

def compute_slot_mapping(self, req_indices, positions):
    """计算 slot_mapping: token position → KV cache tensor index"""
    # 关键公式！
    block_table_indices = (
        req_indices * self.max_num_blocks_per_req +
        positions // self.block_size
    )
    block_numbers = self.block_table_np.ravel()[block_table_indices]
    block_offsets = positions % self.block_size

    # 这就是物理索引的计算！
    slot_mapping = block_numbers * self.block_size + block_offsets
    self.slot_mapping_np[:len(req_indices)] = slot_mapping
```

## 数据流转

### 1. 初始化阶段

```
EngineCore.__init__()
    │
    ├─> _initialize_kv_caches()
    │     │
    │     └─> get_kv_cache_config()
    │           └─> 计算 num_blocks (例如 1000)
    │
    ├─> model_executor.initialize_from_config(kv_cache_configs)
    │     │
    │     └─> [Worker] model_runner.initialize_kv_cache()
    │           │
    │           └─> initialize_kv_cache_tensors()
    │                 └─> 分配 shape=(2, 1000, 32, 16, 128) 的 tensor
    │
    └─> Scheduler.__init__()
          │
          └─> KVCacheManager.__init__()
                │
                └─> KVCacheCoordinator.__init__()
                      │
                      └─> BlockPool.__init__(num_blocks=1000)
                            │
                            └─> 创建 1000 个 KVCacheBlock 对象
                                  block_id: 0, 1, 2, ..., 999
```

**初始化后的状态**：

```
物理层 (Worker):
  kv_cache: Tensor[2, 1000, 32, 16, 128]
  └─> 可以通过 block_id 索引：kv_cache[:, block_id, ...]

元数据层 (Scheduler):
  BlockPool.blocks: [KVCacheBlock(0), KVCacheBlock(1), ..., KVCacheBlock(999)]
  free_block_queue: [0, 1, 2, ..., 999]  # 所有 blocks 都是 free

执行层 (Worker):
  InputBatch.block_table: 空
  └─> 等待 Scheduler 分配 blocks
```

### 2. 调度阶段

```
Scheduler.schedule()
    │
    ├─> [WAITING 请求] get_computed_blocks(request)
    │     └─> coordinator.find_longest_cache_hit()
    │           └─> block_pool.get_cached_block()
    │                 └─> 返回 cached blocks (如果有)
    │
    ├─> KVCacheManager.allocate_slots(request, num_new_tokens)
    │     │
    │     ├─> coordinator.get_num_blocks_to_allocate()
    │     │     └─> 计算需要多少个 blocks
    │     │
    │     ├─> 检查 block_pool.get_num_free_blocks()
    │     │
    │     ├─> block_pool.touch(computed_blocks)
    │     │     └─> 增加 ref_cnt，从 free_queue 移除
    │     │
    │     ├─> coordinator.allocate_new_blocks()
    │     │     │
    │     │     └─> single_type_manager.allocate_new_blocks()
    │     │           │
    │     │           └─> block_pool.get_new_blocks(num_blocks)
    │     │                 │
    │     │                 └─> free_block_queue.popleft_n(num_blocks)
    │     │                       └─> 返回 block_ids: [5, 7, 9, 10]
    │     │
    │     └─> coordinator.cache_blocks()
    │           └─> block_pool.cache_full_blocks()
    │
    └─> 返回 SchedulerOutput
          └─> req_to_new_blocks: {
                "req_1": KVCacheBlocks([[5, 7, 9, 10]])
              }
```

**调度后的状态**：

```
元数据层 (Scheduler):
  BlockPool:
    - blocks[5].ref_cnt = 1
    - blocks[7].ref_cnt = 1
    - blocks[9].ref_cnt = 1
    - blocks[10].ref_cnt = 1
    - free_block_queue: [0, 1, 2, 3, 4, 6, 8, 11, ...]  # 5,7,9,10 已分配

逻辑输出:
  SchedulerOutput.req_to_new_blocks = {
    "req_1": [5, 7, 9, 10]  # block_ids
  }
```

### 3. 执行准备阶段

```
Worker.execute_model(scheduler_output)
    │
    └─> GPUModelRunner.execute_model(scheduler_output)
          │
          ├─> _update_states(scheduler_output)
          │     │
          │     ├─> [新请求] 创建 CachedRequestState
          │     │     │
          │     │     └─> block_ids = scheduler_output.req_to_new_blocks[req_id]
          │     │           └─> [5, 7, 9, 10]
          │     │
          │     ├─> [运行中请求] 更新 block_ids
          │     │     │
          │     │     └─> req_state.block_ids[0].extend(new_block_ids)
          │     │
          │     └─> input_batch.block_table.append_row(block_ids, req_index)
          │           └─> 将 [5, 7, 9, 10] 写入 block_table_np[req_index, :]
          │
          └─> _prepare_inputs(scheduler_output)
                │
                ├─> [步骤 1] commit_block_table()
                │     └─> 将 CPU 的 block_table 复制到 GPU
                │
                ├─> [步骤 2] 计算 positions
                │     │
                │     └─> positions = num_computed_tokens + arange
                │           例如：[32, 33, 34, ..., 47] (16 个 tokens)
                │
                ├─> [步骤 3] compute_slot_mapping(req_indices, positions)
                │     │
                │     │  # 关键映射计算！
                │     │
                │     └─> block_table_indices = (
                │             req_indices * max_num_blocks_per_req +
                │             positions // block_size
                │         )
                │
                │         # 例如：req_index=0, positions=[32,33,34,...,47]
                │         # positions // 16 = [2, 2, 2, ..., 2] (都在第 3 个 block)
                │         # block_table_indices = [0*4 + 2] = [2]
                │         # block_numbers = block_table_np[0, 2] = 9
                │
                │         block_numbers = block_table_np.ravel()[block_table_indices]
                │         # = [9, 9, 9, ..., 9] (16 个)
                │
                │         block_offsets = positions % block_size
                │         # = [0, 1, 2, ..., 15]
                │
                │         slot_mapping = block_numbers * block_size + block_offsets
                │         # = [9*16+0, 9*16+1, ..., 9*16+15]
                │         # = [144, 145, 146, ..., 159]
                │
                └─> [步骤 4] commit_slot_mapping()
                      └─> 将 slot_mapping 复制到 GPU
```

**执行准备后的状态**：

```
执行层 (Worker):
  InputBatch.block_table.block_table_np[0, :] = [5, 7, 9, 10]
  InputBatch.block_table.block_table (GPU) = [5, 7, 9, 10]

  InputBatch.block_table.slot_mapping_np = [144, 145, 146, ..., 159]
  InputBatch.block_table.slot_mapping (GPU) = [144, 145, 146, ..., 159]
```

### 4. 模型执行阶段

```
GPUModelRunner.execute_model() (续)
    │
    ├─> _preprocess()
    │     └─> 准备 input_ids, positions 等
    │
    ├─> model.forward(
    │       input_ids,
    │       positions,
    │       kv_caches=self.kv_caches,  # 物理 tensor
    │       attn_metadata={
    │           "slot_mapping": slot_mapping,  # [144, 145, ..., 159]
    │           "block_table": block_table,    # [[5, 7, 9, 10]]
    │           ...
    │       }
    │   )
    │     │
    │     └─> [Attention Layer]
    │           │
    │           ├─> 计算 Q, K, V
    │           │
    │           ├─> 写入 KV cache:
    │           │     for i in range(num_tokens):
    │           │         slot = slot_mapping[i]  # 144, 145, ...
    │           │         kv_cache[0, slot // 16, :, slot % 16, :] = K[i]
    │           │         kv_cache[1, slot // 16, :, slot % 16, :] = V[i]
    │           │
    │           │     # 等价于：
    │           │     # kv_cache[0, 9, :, 0, :] = K[0]   (slot=144)
    │           │     # kv_cache[0, 9, :, 1, :] = K[1]   (slot=145)
    │           │     # ...
    │           │     # kv_cache[0, 9, :, 15, :] = K[15] (slot=159)
    │           │
    │           └─> 读取 KV cache 进行 attention:
    │                 # 使用 block_table 和 position 读取历史 KV
    │                 for block_idx in range(num_blocks):
    │                     block_id = block_table[req_idx, block_idx]
    │                     K_block = kv_cache[0, block_id, :, :, :]
    │                     V_block = kv_cache[1, block_id, :, :, :]
    │                     # 执行 attention 计算
    │
    └─> _sample()
          └─> 返回 sampled_token_ids
```

## 关键映射关系

### block_id 到 Tensor 索引

**公式 1: 直接索引** (在 attention kernel 中)

```python
# 给定一个 token 的 slot_mapping
slot = slot_mapping[token_idx]

# 计算 block_id 和 block_offset
block_id = slot // block_size
block_offset = slot % block_size

# 访问 KV cache
key = kv_cache[0, block_id, :, block_offset, :]
value = kv_cache[1, block_id, :, block_offset, :]
```

**公式 2: slot_mapping 计算** (在 compute_slot_mapping 中)

```python
# 输入：
#   req_indices: 每个 token 所属的请求索引
#   positions: 每个 token 在请求中的位置

# 步骤 1: 找到 token 在哪个 block
block_index_in_req = positions // block_size

# 步骤 2: 在 block_table 中查找对应的 block_id
block_table_indices = (
    req_indices * max_num_blocks_per_req + block_index_in_req
)
block_ids = block_table_np.ravel()[block_table_indices]

# 步骤 3: 计算 block 内偏移
block_offsets = positions % block_size

# 步骤 4: 计算最终的 slot_mapping
slot_mapping = block_ids * block_size + block_offsets
```

### 完整映射链

```
Request Position (逻辑位置)
    ↓
  position // block_size
    ↓
Block Index in Request (请求内的 block 序号)
    ↓
  block_table[req_idx, block_index]
    ↓
Block ID (全局 block ID)
    ↓
  block_id * block_size + (position % block_size)
    ↓
Slot Mapping (KV cache tensor 的线性索引)
    ↓
  slot // block_size → block_id
  slot % block_size → block_offset
    ↓
Tensor Index (实际的 tensor 索引)
  kv_cache[0/1, block_id, :, block_offset, :]
```

## 完整示例

假设：
- `block_size = 16`
- `num_blocks = 1000`
- `kv_cache.shape = (2, 1000, 32, 16, 128)`

### 场景：处理一个新请求

**请求信息**：
- `req_id = "req_001"`
- `prompt_token_ids = [1, 2, 3, ..., 50]` (50 个 tokens)
- 需要 `ceil(50 / 16) = 4` 个 blocks

#### 步骤 1: Scheduler 分配 blocks

```python
# Scheduler.schedule()
KVCacheManager.allocate_slots(request, num_new_tokens=50)
  → BlockPool.get_new_blocks(4)
    → 返回 block_ids: [123, 456, 789, 234]

SchedulerOutput.req_to_new_blocks["req_001"] = [123, 456, 789, 234]
```

#### 步骤 2: Worker 更新 InputBatch

```python
# GPUModelRunner._update_states()
req_state = CachedRequestState(
    req_id="req_001",
    block_ids=[[123, 456, 789, 234]],  # 一个 KV cache group
    num_computed_tokens=0,
    ...
)

input_batch.block_table.add_row(
    block_ids=([123, 456, 789, 234],),
    row_idx=0  # 假设这是第一个请求
)

# 结果：
# block_table_np[0, :] = [123, 456, 789, 234, 0, 0, ...]
```

#### 步骤 3: 计算 slot_mapping

```python
# GPUModelRunner._prepare_inputs()

# 假设我们调度前 32 个 tokens (前 2 个 blocks)
req_indices = [0, 0, 0, ..., 0]  # 32 个 0
positions = [0, 1, 2, ..., 31]

# compute_slot_mapping()
block_table_indices = (
    [0, 0, ..., 0] * max_num_blocks_per_req +
    [0, 1, 2, ..., 31] // 16
)
# = [0, 0, ..., 0, 1, 1, ..., 1]
#   前16个位置在 block_table[0, 0]
#   后16个位置在 block_table[0, 1]

block_numbers = block_table_np.ravel()[block_table_indices]
# = [123, 123, ..., 123,  # 前 16 个
#    456, 456, ..., 456]  # 后 16 个

block_offsets = [0, 1, 2, ..., 31] % 16
# = [0, 1, 2, ..., 15, 0, 1, 2, ..., 15]

slot_mapping = block_numbers * 16 + block_offsets
# = [123*16+0, 123*16+1, ..., 123*16+15,
#    456*16+0, 456*16+1, ..., 456*16+15]
# = [1968, 1969, ..., 1983,
#    7296, 7297, ..., 7311]
```

#### 步骤 4: 模型执行

```python
# Attention kernel (伪代码)
for i in range(32):
    slot = slot_mapping[i]
    block_id = slot // 16
    block_offset = slot % 16

    # 写入 KV cache
    kv_cache[0, block_id, :, block_offset, :] = K[i]
    kv_cache[1, block_id, :, block_offset, :] = V[i]

# 示例：
# i=0: slot=1968, block_id=123, offset=0
#   → kv_cache[0, 123, :, 0, :] = K[0]
# i=15: slot=1983, block_id=123, offset=15
#   → kv_cache[0, 123, :, 15, :] = K[15]
# i=16: slot=7296, block_id=456, offset=0
#   → kv_cache[0, 456, :, 0, :] = K[16]
```

#### 步骤 5: 下一次调度

假设下次调度剩余的 18 个 tokens (positions 32-49)：

```python
req_indices = [0, 0, ..., 0]  # 18 个 0
positions = [32, 33, ..., 49]  # num_computed_tokens=32

block_table_indices = [0] * max_num_blocks_per_req + [32, 33, ..., 49] // 16
# = [2, 2, ..., 2, 3, 3]
#   前16个在 block_table[0, 2]，后2个在 block_table[0, 3]

block_numbers = block_table_np.ravel()[block_table_indices]
# = [789, 789, ..., 789, 234, 234]

block_offsets = [32, 33, ..., 49] % 16
# = [0, 1, ..., 15, 0, 1]

slot_mapping = block_numbers * 16 + block_offsets
# = [789*16+0, ..., 789*16+15, 234*16+0, 234*16+1]
# = [12624, ..., 12639, 3744, 3745]
```

### 关键观察

1. **block_id 是全局的**: `123, 456, 789, 234` 在整个系统中唯一
2. **block_id 直接作为 tensor 索引**: `kv_cache[0/1, block_id, ...]`
3. **slot_mapping 是线性化的索引**: 方便 attention kernel 快速访问
4. **block_table 存储请求的 block_ids**: 允许按需读取历史 KV cache

## 总结

vLLM v1 架构通过三层设计实现了高效的 KV Cache 管理：

### 物理层
- **KV Cache Tensors**: 实际的 GPU 内存，shape = `(2, num_blocks, ...)`
- **block_id 即 tensor 索引**: 直接映射到 tensor 的第二维

### 元数据层
- **BlockPool**: 管理 block 的分配、释放、缓存
- **KVCacheBlock**: 存储 block 的元数据 (ref_cnt, hash, ...)
- **block_id**: `[0, num_blocks-1]`，与 tensor 索引一致

### 执行层
- **BlockTable**: 存储每个请求的 block_ids
- **slot_mapping**: token position → KV cache tensor 线性索引
- **关键公式**: `slot = block_id * block_size + block_offset`

### 核心优势

1. **简单高效**: block_id 直接作为 tensor 索引，无额外映射开销
2. **灵活管理**: BlockPool 统一管理分配、释放、缓存
3. **Prefix Caching**: 通过 block_hash 复用相同 prefix
4. **多请求共享**: 通过 ref_cnt 支持多个请求共享同一 block
5. **支持各种 attention**: 通过 block_table 和 slot_mapping 灵活支持不同 attention 类型

这种设计使得 vLLM 能够高效地处理大量并发请求，同时最大化 GPU 内存利用率。
