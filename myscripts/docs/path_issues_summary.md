# 原 README.md 路径问题汇总

## 🔴 严重问题

### 1. **工作目录不一致**（第 89-91 行）
```bash
# ❌ 错误
cd vllm
python3 examples/offline_inference/qwen_boundary_model.py \
     --model model/Qwen3Boundary-Stream \
```

**问题**：
- 先 `cd vllm` 进入 vllm 子目录
- 然后使用 `model/Qwen3Boundary-Stream`，但这个路径在父目录

**修正**：
```bash
# ✅ 正确（方案1：在根目录执行）
python3 vllm/examples/offline_inference/qwen_boundary_model.py \
     --model /data/workspace/stream/model/Qwen3Boundary-Stream \

# ✅ 正确（方案2：使用相对路径）
python3 vllm/examples/offline_inference/qwen_boundary_model.py \
     --model ./model/Qwen3Boundary-Stream \
```

### 2. **缺失脚本路径前缀**（第 55, 66, 78, 81 行）
```bash
# ❌ 错误
python prepare_dataset.py \
python train_sentence_boundary.py \
python simple—test.py
python export_boundary_model.py \
```

**问题**：这些脚本实际在 `scripts/` 目录下，不在根目录

**修正**：
```bash
# ✅ 正确
python scripts/prepare_dataset.py \
python scripts/train_sentence_boundary.py \
python scripts/simple_test.py \
python scripts/export_boundary_model.py \
```

---

## ⚠️ 中等问题

### 3. **文件名拼写错误**（第 78 行）
```bash
# ❌ 错误
python simple—test.py
```

**问题**：使用了中文破折号（—）而不是英文连字符或下划线

**修正**：
```bash
# ✅ 正确（根据命名规范应该是下划线）
python scripts/simple_test.py
```

### 4. **反斜杠后有多余空格**（第 82 行）
```bash
# ❌ 错误
--base-model model/Qwen3-0.6B \
#                             ↑ 这里有空格
```

**问题**：反斜杠后的空格可能导致命令续行失败

**修正**：
```bash
# ✅ 正确
--base-model model/Qwen3-0.6B \
```

---

## ℹ️ 改进建议

### 5. **Git clone 目标路径不明确**（第 51 行）
```bash
# 当前写法
git clone https://www.modelscope.cn/datasets/dingkun/chinese_word_segmentation_pku.git
# 需要手动移动到 ./data/chinese_word_segmentation_pku
```

**建议改进**：
```bash
# 直接 clone 到目标位置
git clone https://www.modelscope.cn/datasets/dingkun/chinese_word_segmentation_pku.git data/chinese_word_segmentation_pku
```

### 6. **缺少目录创建步骤**
**建议添加**：
```bash
mkdir -p model
mkdir -p data
```

### 7. **modelscope 命令缺少必要参数**（第 64 行）
```bash
# 当前写法
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./model/Qwen3-0.6B
```

**可能需要添加**：
```bash
# 如果需要指定 revision 或其他参数
modelscope download --model Qwen/Qwen3-0.6B \
    --local_dir ./model/Qwen3-0.6B \
    --revision main
```

---

## 📋 完整的修正清单

| 行号 | 问题类型 | 原内容 | 修正 |
|------|---------|--------|------|
| 55 | 路径错误 | `python prepare_dataset.py` | `python scripts/prepare_dataset.py` |
| 66 | 路径错误 | `python train_sentence_boundary.py` | `python scripts/train_sentence_boundary.py` |
| 67 | 格式问题 | `--pretrained-model model/Qwen3-0.6B\` | `--pretrained-model model/Qwen3-0.6B \` (空格) |
| 78 | 拼写错误 | `python simple—test.py` | `python scripts/simple_test.py` |
| 81 | 路径错误 | `python export_boundary_model.py` | `python scripts/export_boundary_model.py` |
| 82 | 格式问题 | `--base-model model/Qwen3-0.6B \` | 移除反斜杠后的空格 |
| 89-91 | 路径逻辑 | `cd vllm` 然后用 `model/...` | 不 cd，直接用绝对或相对路径 |

---

## 🎯 推荐的执行流程

```bash
# 1. 确保在项目根目录
cd /data/workspace/stream/

# 2. 检查目录结构
ls -la
# 应该看到: vllm/, scripts/, data/, docs/ 等

# 3. 创建必要的目录
mkdir -p model
mkdir -p data

# 4. 按顺序执行命令（都在根目录下）
# 数据准备
git lfs install
git clone https://... data/chinese_word_segmentation_pku
python scripts/prepare_dataset.py ...

# 模型训练
modelscope download ...
python scripts/train_sentence_boundary.py ...
python scripts/simple_test.py ...
python scripts/export_boundary_model.py ...

# vLLM 推理
python3 vllm/examples/offline_inference/qwen_boundary_model.py \
    --model /data/workspace/stream/model/Qwen3Boundary-Stream \
    ...
```

---

## 💡 关键提示

1. **始终在 `/data/workspace/stream/` 执行命令**
2. **使用 `pwd` 确认当前目录**
3. **使用 `ls` 检查文件是否存在再执行**
4. **优先使用绝对路径避免混淆**
5. **注意中英文标点符号的区别**
