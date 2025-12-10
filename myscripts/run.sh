#! /bin/bash
mkdir -p data
mkdir -p model

git lfs install
git clone https://www.modelscope.cn/datasets/dingkun/chinese_word_segmentation_pku.git data/chinese_word_segmentation_pku
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./model/Qwen3-0.6B

python  prepare_dataset.py \
    --input-dir data/chinese_word_segmentation_pku \
    --output-dir data/pku_bies

python train_sentence_boundary.py \
    --pretrained-model model/Qwen3-0.6B\
    --train-file data/pku_bies/train.bies.jsonl \
    --val-file data/pku_bies/dev.bies.jsonl \
    --output-dir model/head \
    --epochs 1 \
    --max-length 256 \
    --log-interval 100 \
    --batch-size 32 

python export_boundary_model.py \
    --base-model model/Qwen3-0.6B \
    --head-checkpoint model/head/best_boundary_head.pt \
    --output-dir model/Qwen3Boundary-Stream


cd ..
python3 examples/offline_inference/qwen_boundary_model.py \
     --model myscripts/model/Qwen3Boundary-Stream \
     --trust-remote-code \
     --max-num-seqs 1 \
     --max-model-len 2048 \
     --disable-log-stats
