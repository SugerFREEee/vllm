# Qwen3Boundary 模型适配 vLLM 指南

## 目录
1. [项目背景](#项目背景)
2. [适配概述](#适配概述)
3. [核心实现](#核心实现)
4. [自定义模型适配 vLLM 的一般流程](#自定义模型适配-vllm-的一般流程)
5. [详细代码解析](#详细代码解析)
6. [使用示例](#使用示例)
7. [常见问题](#常见问题)

---

## 项目背景

本项目参考 **Qwen3-Guard streaming** 模型，实现了基于流式输入的断句判断功能。核心目标是在流式输入场景下，实时判断每个 token 是否构成完整的句子边界，从而支持实时的句子切分和处理。

为了实现高效的推理，我们将这个自定义模型适配到了 **vLLM** 推理引擎中，使其能够利用 vLLM 的高性能推理能力和流式处理特性。

---

## 适配概述

### 主要适配内容

我们成功将 `Qwen3BoundaryForStreaming` 模型适配到 vLLM，具体包括：

1. **新增模型架构实现** (`qwen3_boundary.py`)
   - 基于 Qwen3 基座模型
   - 添加句子边界判断头
   - 实现流式推理接口

2. **注册模型到 vLLM 系统**
   - 在模型注册表中添加新架构
   - 配置 Pooling 模式支持
   - 优化模型选择逻辑

3. **配置系统集成**
   - 禁用不兼容的优化特性（如 prefix caching）
   - 设置合适的调度参数

4. **提供完整示例代码**
   - 流式输入处理示例
   - 字符级 tokenization
   - 异步推理接口

### 关键特性

- ✅ **流式输入支持**：支持逐字符输入，实时判断句子边界
- ✅ **异步推理**：基于 vLLM V1 AsyncLLM 引擎
- ✅ **Resume 机制**：支持请求恢复和增量输入
- ✅ **高性能**：利用 vLLM 的批处理和优化能力
- ✅ **灵活配置**：通过 config.json 配置边界判断参数

---

## 核心实现

### 1. 模型架构 (`qwen3_boundary.py`)

#### 1.1 句子边界判断头

```python
class SentenceBoundaryHead(nn.Module):
    """Two-layer MLP that consumes LM probabilities."""

    def __init__(self, vocab_size: int, hidden_size: int,
                 num_labels: int, dropout: float):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        return self.layer(probs)
```

**关键设计**：
- 输入：LM 模型输出的词表概率分布
- 结构：两层 MLP，带 GELU 激活和 Dropout
- 输出：句子边界的分类 logits（通常是 2 分类：未完结/完整）

#### 1.2 主模型类

```python
@default_pooling_type("ALL")
class Qwen3BoundaryForStreaming(nn.Module, SupportsPP, VllmModelForPooling):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # 1. 加载基座模型（Qwen3）
        self.model = Qwen3Model(vllm_config=vllm_config,
                               prefix=maybe_prefix(prefix, "model"))

        # 2. 创建 LM Head（用于生成词表概率）
        self.lm_head = ParallelLMHead(...)

        # 3. 创建边界判断头
        self.boundary_head = SentenceBoundaryHead(...)

        # 4. 配置 Pooler（ALL 模式，返回所有 token 的输出）
        self.pooler = DispatchPooler({
            "encode": Pooler.for_encode(
                PoolerConfig(pooling_type="ALL", ...)
            )
        })
```

**关键点**：
- 继承 `VllmModelForPooling`，标识为 Pooling 模型
- 使用 `@default_pooling_type("ALL")` 装饰器，指定默认返回所有 token 的输出
- 支持 Pipeline Parallelism (`SupportsPP`)

#### 1.3 前向传播

```python
def forward(self, input_ids, positions, ...):
    # 1. 通过基座模型获取隐藏状态
    hidden_states = self.model(input_ids, positions, ...)

    # 2. 计算 LM logits 和概率分布
    lm_logits = self.logits_processor(self.lm_head, hidden_states, ...)
    probs = torch.softmax(lm_logits, dim=-1)

    # 3. 通过边界判断头得到边界 logits
    boundary_logits = self.boundary_head(probs)

    # 4. 拼接边界 logits 和隐藏状态返回
    return torch.cat([boundary_logits, hidden_states], dim=-1)
```

**设计思路**：
- 先计算语言模型的输出概率
- 将概率作为特征输入到边界判断头
- 最终输出包含边界判断 logits 和原始隐藏状态

### 2. 模型注册 (`registry.py`)

#### 2.1 注册到模型表

```python
_TEXT_GENERATION_MODELS = {
    ...
    "Qwen3BoundaryForStreaming": ("qwen3_boundary", "Qwen3BoundaryForStreaming"),
}

_EMBEDDING_MODELS = {
    ...
    "Qwen3BoundaryForStreaming": ("qwen3_boundary", "Qwen3BoundaryForStreaming"),
}
```

**说明**：
- 同时注册到文本生成模型和嵌入模型表
- 第一个参数是模块名，第二个是类名

#### 2.2 架构优先级处理

```python
def _prioritize_architectures(self, architectures, model_config):
    """当 runner_type='pooling' 时，优先选择 pooling 模型"""
    runner_type = getattr(model_config, "runner_type", None)
    if runner_type != "pooling":
        return arch_list

    # 将 pooling 模型排在前面
    pooling_archs = [arch for arch in arch_list
                     if is_pooling_model(arch)]
    remaining_archs = [arch for arch in arch_list
                      if not is_pooling_model(arch)]
    return pooling_archs + remaining_archs
```

**作用**：
- 当配置指定 `runner="pooling"` 时，优先选择 pooling 模型架构
- 解决一个 checkpoint 可能支持多个架构的问题

### 3. 配置集成 (`config.py`)

```python
if self.model_config.architecture in (
        "Qwen3ForGuardModel", "Qwen3BoundaryForStreaming"):
    logger.info(
        "Enable Qwen3 guard/boundary logits computation, disable "
        "prefix caching.")
    # 禁用 chunked prefill 优化
    self.scheduler_config.long_prefill_token_threshold = 0
    # 禁用 prefix caching
    if self.cache_config is not None:
        self.cache_config.enable_prefix_caching = False
```

**原因**：
- Guard/Boundary 模型需要对每个 token 都计算输出
- Prefix caching 会复用之前的结果，与流式判断逻辑冲突
- 必须禁用这些优化以保证正确性

### 4. 使用示例 (`qwen_boundary_model.py`)

#### 4.1 流式 Tokenization

```python
def stream_tokenize_text(tokenizer, text: str):
    """模拟流式输入，逐字符 tokenize"""
    chunks, aggregated = [], []
    for ch in text:
        char_token_ids = tokenizer.encode(ch, add_special_tokens=False)
        aggregated.extend(char_token_ids)
        for token_id in char_token_ids:
            chunks.append([token_id])  # 每个 token 一个 chunk
    return chunks, aggregated
```

#### 4.2 请求处理

```python
async def handle_request(engine, request_id, query_prompt,
                        message_list, ...):
    # 1. 初始请求（第一个 token）
    response = engine.encode(
        query_prompt,
        pooling_params=PoolingParams(
            task="encode",
            output_kind=RequestOutputKind.DELTA  # 增量输出
        ),
        request_id=request_id,
        resumable=True  # 支持恢复
    )

    # 2. 流式接收结果
    async for resp in response:
        if resp.outputs.data is not None:
            # 解析边界判断结果
            probs = decode_boundary_output(resp, num_labels)
            preds = torch.argmax(probs, dim=-1)
            # 处理每个 token 的判断结果
            ...

        # 3. 发送下一个 token（Resume）
        if message_list:
            next_chunk = message_list.pop(0)
            await engine.resume_request(
                request_id=request_id,
                prompt_token_ids=next_chunk,
                finish_forever=not message_list
            )
```

**关键流程**：
1. 用第一个 token 初始化请求
2. 异步接收每个 token 的边界判断结果
3. 通过 `resume_request` 不断喂入新的 token
4. 实现真正的流式推理

---

## 自定义模型适配 vLLM 的一般流程

基于我们的实践经验，总结自定义模型适配 vLLM 的通用流程：

### 第一步：分析模型需求

1. **确定模型类型**
   - 生成式模型 (Causal LM)
   - 编码模型 (Encoder-only)
   - Pooling 模型 (用于嵌入、分类等)

2. **识别特殊需求**
   - 是否需要流式输入？
   - 是否需要返回所有 token 的输出？
   - 是否有自定义的输出格式？
   - 是否需要额外的模型头（分类头、检测头等）？

### 第二步：实现模型架构

在 `vllm/model_executor/models/` 下创建新文件（如 `your_model.py`）：

```python
# 1. 选择合适的基类
from .interfaces_base import VllmModelForPooling  # 或其他基类

# 2. 实现模型类
@default_pooling_type("ALL")  # 如果是 pooling 模型
class YourModelForVLLM(nn.Module, VllmModelForPooling):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # a. 加载配置
        config = vllm_config.model_config.hf_config

        # b. 创建基座模型（复用已有实现）
        self.model = ExistingBaseModel(vllm_config, prefix=...)

        # c. 添加自定义的模型头
        self.custom_head = YourCustomHead(...)

        # d. 配置 Pooler（如果需要）
        self.pooler = DispatchPooler({...})

    def forward(self, input_ids, positions, ...):
        # 实现前向传播逻辑
        hidden_states = self.model(...)
        custom_output = self.custom_head(hidden_states)
        return custom_output

    def load_weights(self, weights):
        # 实现权重加载逻辑
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights)
        # 加载自定义头的权重
        self._load_custom_head()
        return loaded
```

**关键点**：
- 尽量复用现有的基座模型实现（如 Qwen3Model、LlamaModel 等）
- 只实现自定义的部分（如特殊的输出头）
- 正确处理权重加载，包括自定义组件的权重

### 第三步：注册模型

修改 `vllm/model_executor/models/registry.py`：

```python
# 1. 在相应的模型字典中注册
_TEXT_GENERATION_MODELS = {
    ...
    "YourModelArchName": ("your_model", "YourModelForVLLM"),
}

# 2. 如果是 pooling/embedding 模型，也注册到这里
_EMBEDDING_MODELS = {
    ...
    "YourModelArchName": ("your_model", "YourModelForVLLM"),
}
```

**注意**：
- `YourModelArchName` 必须与 `config.json` 中的 `architectures` 字段匹配
- 第一个参数是模块名（不含 .py）
- 第二个参数是类名

### 第四步：配置系统集成

修改 `vllm/config/__init__.py`（如果需要特殊配置）：

```python
class VllmConfig:
    def __post_init__(self):
        ...
        # 为你的模型添加特殊配置
        if self.model_config.architecture == "YourModelArchName":
            # 禁用某些优化
            self.cache_config.enable_prefix_caching = False
            # 设置调度参数
            self.scheduler_config.some_param = value
```

**常见配置项**：
- `enable_prefix_caching`: 是否启用前缀缓存
- `enable_chunked_prefill`: 是否启用分块预填充
- `long_prefill_token_threshold`: 长预填充阈值

### 第五步：准备模型配置

在模型目录下的 `config.json` 中添加必要配置：

```json
{
  "architectures": ["YourModelArchName"],
  "model_type": "your_model",
  ...
  "custom_config": {
    "num_labels": 2,
    "hidden_size": 512,
    "head_file": "custom_head.pt"
  }
}
```

### 第六步：编写示例代码

在 `examples/offline_inference/` 下创建示例文件：

```python
from vllm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.pooling_params import PoolingParams

async def main():
    # 1. 创建引擎参数
    engine_args = AsyncEngineArgs(
        model="/path/to/your/model",
        trust_remote_code=True,
        runner="pooling",  # 如果是 pooling 模型
        ...
    )

    # 2. 初始化引擎
    engine = AsyncLLM.from_engine_args(engine_args, ...)

    # 3. 准备输入
    prompt = TokensPrompt(prompt_token_ids=[...])

    # 4. 推理
    response = engine.encode(
        prompt,
        pooling_params=PoolingParams(task="encode"),
        request_id="example-0",
    )

    # 5. 处理输出
    async for resp in response:
        print(resp.outputs.data)
```

### 第七步：测试和验证

1. **单元测试**
   ```bash
   pytest tests/model_executor/test_your_model.py
   ```

2. **功能测试**
   ```bash
   python examples/offline_inference/your_model_example.py \
       --model /path/to/model \
       --trust-remote-code
   ```

3. **性能测试**
   ```bash
   python benchmarks/benchmark_serving.py \
       --model /path/to/model \
       --backend vllm
   ```

---

## 详细代码解析

### 权重加载机制

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
    # 1. 使用 AutoWeightsLoader 自动加载大部分权重
    loader = AutoWeightsLoader(
        self,
        skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
    )
    loaded = loader.load_weights(weights)

    # 2. 手动加载特殊组件的权重
    if self._load_boundary_head():
        loaded |= {
            f"boundary_head.{name}"
            for name in self.boundary_head.state_dict().keys()
        }
    return loaded

def _load_boundary_head(self) -> bool:
    """从单独的文件加载边界判断头的权重"""
    head_path = Path(model_dir) / self.boundary_head_file
    if not head_path.exists():
        logger.warning("Boundary head file not found.")
        return False
    state_dict = torch.load(head_path, map_location="cpu")
    self.boundary_head.load_state_dict(state_dict)
    return True
```

**设计考虑**：
- 大部分权重通过 `AutoWeightsLoader` 自动加载
- 特殊组件（如独立训练的头）需要手动加载
- 支持从 checkpoint 目录下的单独文件加载

### Resume 机制解析

```python
# 初始请求
response = engine.encode(
    query_prompt,  # 只包含第一个 token
    pooling_params=PoolingParams(...),
    request_id="unique-id",
    resumable=True  # 标记为可恢复
)

# 处理流式输入
async for resp in response:
    # 处理当前输出
    process_output(resp)

    # 恢复请求，添加新 token
    await engine.resume_request(
        request_id="unique-id",  # 使用相同的 request_id
        prompt_token_ids=[next_token],  # 新的 token
        finish_forever=is_last_token  # 是否是最后一个 token
    )
```

**工作原理**：
1. 首次调用 `encode` 时设置 `resumable=True`
2. 引擎会保持请求状态（KV cache、隐藏状态等）
3. 通过 `resume_request` 增量添加新 token
4. 引擎只需要计算新 token 的输出，无需重新计算之前的部分
5. `finish_forever=True` 表示这是最后一个 token，请求结束

### 输出格式处理

```python
def decode_boundary_output(resp, num_labels):
    """解析边界判断的输出"""
    if resp.outputs.data is None:
        return None

    # 输出格式：[num_tokens, batch_size, total_dim]
    # total_dim = num_labels + hidden_size
    logits = resp.outputs.data[..., :num_labels].to(torch.float32)

    # 计算概率和预测
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)

    return probs, preds
```

**输出结构**：
- 前 `num_labels` 维是边界判断的 logits
- 后面的维度是原始的隐藏状态（保留用于其他用途）
- 通过 softmax 转换为概率分布

---

## 使用示例

### 基本使用

```bash
python3 examples/offline_inference/qwen_boundary_model.py \
    --model /path/to/Qwen3Boundary-Stream \
    --trust-remote-code \
    --max-num-seqs 1 \
    --max-model-len 2048 \
    --disable-log-stats
```

### Python API 使用

```python
import asyncio
from transformers import AutoTokenizer
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import RequestOutputKind

async def main():
    # 1. 初始化引擎
    engine_args = AsyncEngineArgs(
        model="/path/to/Qwen3Boundary-Stream",
        trust_remote_code=True,
        runner="pooling",
        max_num_seqs=1,
    )
    engine = AsyncLLM.from_engine_args(engine_args, ...)

    # 2. 准备输入（模拟流式）
    tokenizer = AutoTokenizer.from_pretrained(engine_args.model)
    text = "这是一个测试句子。"

    chunks = []
    for char in text:
        token_ids = tokenizer.encode(char, add_special_tokens=False)
        for tid in token_ids:
            chunks.append([tid])

    # 3. 初始请求
    initial_chunk = chunks.pop(0)
    response = engine.encode(
        TokensPrompt(prompt_token_ids=initial_chunk),
        pooling_params=PoolingParams(
            task="encode",
            output_kind=RequestOutputKind.DELTA
        ),
        request_id="test-0",
        resumable=True,
    )

    # 4. 流式处理
    async for resp in response:
        if resp.outputs.data is not None:
            # 解析边界判断结果
            logits = resp.outputs.data[..., :2]
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            print(f"Token boundary prediction: {pred.item()}")

        # 发送下一个 token
        if chunks:
            next_chunk = chunks.pop(0)
            await engine.resume_request(
                request_id="test-0",
                prompt_token_ids=next_chunk,
                finish_forever=not chunks,
            )

    engine.shutdown()

asyncio.run(main())
```

### 配置文件示例

`config.json`:
```json
{
  "architectures": ["Qwen3BoundaryForStreaming"],
  "model_type": "qwen3",
  "vocab_size": 152064,
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "sentence_boundary": {
    "num_labels": 2,
    "hidden_size": 512,
    "dropout": 0.1,
    "head_file": "boundary_head.pt"
  },
  "tie_word_embeddings": false
}
```

---

## 常见问题

### Q1: 为什么需要禁用 prefix caching？

**A**: Prefix caching 会缓存已处理的 token 序列的 KV cache 和输出，对于相同的前缀，会直接复用之前的结果。但对于边界判断模型：
- 每个新 token 的到来都可能改变之前 token 的边界判断
- 需要重新计算所有 token 的输出，不能复用缓存
- 因此必须禁用 prefix caching

### Q2: `resumable=True` 和 `finish_forever` 的区别？

**A**:
- `resumable=True`: 标记请求可以恢复，引擎会保持请求的状态
- `finish_forever=False`: 表示后续还会有新 token 输入，不要结束请求
- `finish_forever=True`: 表示这是最后一个 token，可以结束请求了

### Q3: 为什么要注册到两个模型表？

**A**: Qwen3BoundaryForStreaming 既可以用于文本生成（通过 LM head），也可以用于嵌入/分类任务（通过 boundary head）。注册到两个表可以让 vLLM 在不同场景下都能正确识别和加载模型。

### Q4: 如何处理自定义权重文件？

**A**:
```python
# 1. 在 config.json 中指定权重文件
{
  "custom_config": {
    "head_file": "custom_head.pt"
  }
}

# 2. 在模型中加载
def _load_custom_head(self):
    head_path = Path(model_dir) / self.custom_head_file
    state_dict = torch.load(head_path, map_location="cpu")
    self.custom_head.load_state_dict(state_dict)
```

### Q5: 流式输入时如何控制并发？

**A**: 使用 `asyncio.Semaphore` 控制并发请求数：
```python
limiter = asyncio.Semaphore(max_concurrent_requests)

async def handle_with_limit(request):
    async with limiter:
        await handle_request(request)
```

### Q6: 输出维度不匹配怎么办？

**A**: 检查以下几点：
1. 确保 `num_labels` 配置正确
2. 确保前向传播返回的维度与配置一致
3. 使用 `pooling_type="ALL"` 确保返回所有 token 的输出
4. 检查 boundary head 的输出维度

---

## 总结

本文档详细介绍了如何将 Qwen3BoundaryForStreaming 模型适配到 vLLM，以及自定义模型适配的一般流程。关键要点：

1. **理解模型需求**：明确模型类型、特殊功能、输出格式
2. **复用现有组件**：尽量使用已有的基座模型和工具
3. **正确注册模型**：在注册表中添加，配置优先级
4. **处理特殊配置**：禁用不兼容的优化，设置合适参数
5. **实现流式接口**：使用 Resume 机制支持增量输入
6. **完善文档和示例**：提供清晰的使用说明

遵循这些原则，可以高效地将各种自定义模型适配到 vLLM，充分利用其高性能推理能力。
