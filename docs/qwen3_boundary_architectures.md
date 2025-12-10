# Qwen3Boundary 架构识别差异说明

## 背景

为了让流式句边界模型在 vLLM 中正常以 pooling runner 运行，我们在 `myscripts/export_boundary_model.py` 中为 HuggingFace `config.json` 写入 `architectures` 字段（`myscripts/export_boundary_model.py:60-79`）。近期删除了 `registry._prioritize_architectures` 的重排逻辑（`vllm/model_executor/models/registry.py:685-704`），因此 `architectures` 中各条目出现的顺序会直接影响 vLLM 的模型解析。

## vLLM 如何解析 `architectures`

`ModelConfig._get_default_runner_type` 会按顺序遍历 `architectures` 列表来尝试加载类，并通过 `registry.is_pooling_model()` 判断模型属性（`vllm/config/__init__.py:926-948`）。一旦首先加载到的类不是 pooling 模型，runner 就会被解析为 `generate`。移除了 `_prioritize_architectures` 后，vLLM 不再自动把带 `is_pooling_model` 标记的类放到列表前面，配置文件本身必须保证期望的顺序。

## `architectures` 包含 `Qwen3ForCausalLM` 时

- `config.json` 中若把 `"Qwen3ForCausalLM"` 放在 `"Qwen3BoundaryForStreaming"` 前面（历史版本导出的默认行为），`ModelConfig` 会先命中生成式主干，并把 runner 锁定在 `generate`。
- 结果：加载模型时无法进入 pooling runner，也不会初始化 `pooler_config`，导致句边界头无法被调度，更无法在 streaming guard 流程里输出 boundary logits。
- 这个设置的唯一好处在于，HuggingFace 生态（例如 `AutoModelForCausalLM`）仍能以常规 Qwen3 主干的方式加载模型，忽略我们自定义的 boundary 头。

## `architectures` 只包含 `Qwen3BoundaryForStreaming` 时

- 当前导出脚本会先移除 `"Qwen3ForCausalLM"`，再确保 `"Qwen3BoundaryForStreaming"` 存在（`myscripts/export_boundary_model.py:60-79`），因此 HF config 中仅保留 pooling 架构。
- vLLM 在解析时会直接加载 `Qwen3BoundaryForStreaming`，`runner_type` 被解析为 `pooling`，从而允许 streaming guard/断句逻辑正常工作。
- 副作用是：如果有人尝试用 `AutoModelForCausalLM.from_pretrained()` 加载同一个目录，会因为找不到 `"Qwen3BoundaryForStreaming"` 对应的 HF 实现而失败；如需在 HF 环境下使用，必须显式提供自定义代码。

## 建议

1. **面向 vLLM**：保持当前“仅保留 `Qwen3BoundaryForStreaming`”的导出策略，确保 runner 能自动识别为 pooling。
2. **面向 HF 生态**：如果确实需要继续支持 `AutoModel*` 加载，可以维护两个导出版本，或在配置里加上 `auto_map`，指向自定义的 `Qwen3BoundaryForStreaming` 实现，以取代旧的 `Qwen3ForCausalLM`。
3. **后续开发注意事项**：若再次新增需要 pooling 的自定义架构，应确保它在 `architectures` 列表中的顺序优先，或在 CLI 中显式指定 `--runner pooling` 防止歧义。
