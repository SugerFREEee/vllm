## 数据准备
```bash
git lfs install
git clone https://www.modelscope.cn/datasets/dingkun/chinese_word_segmentation_pku.git

# 将 txt 格式转换成 jsonl

python  prepare_dataset.py \
    --input-dir ./data/chinese_word_segmentation_pku \
    --output-dir ./data/pku_bies
```

## 模型训练与组装

```bash
# 下载Qwen3-0.6B
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./model/Qwen3-0.6B
# 训练
python train_sentence_boundary.py \
    --pretrained-model model/Qwen3-0.6B\
    --train-file data/pku_bies/train.bies.jsonl \
    --val-file data/pku_bies/dev.bies.jsonl \
    --output-dir model/head \
    --epochs 100 \
    --max-length 256 \
    --log-interval 100 \
    --batch-size 32 

# 此时训练成的分类头保存下来
# 简单测试一下
python simple—test.py --text "１２月３１日，中共中央总书记发表新年讲话。"

# 将底座和分类头合并成完成模型
python export_boundary_model.py \
    --base-model model/Qwen3-0.6B \ 
    --head-checkpoint model/head \
    --output-dir model/Qwen3Boundary-Stream
```

## vllm推理
```bash
cd vllm
python3 examples/offline_inference/qwen_boundary_model.py \
     --model model/Qwen3Boundary-Stream \
     --trust-remote-code \
     --max-num-seqs 1 \
     --max-model-len 2048 \
     --disable-log-stats
```
