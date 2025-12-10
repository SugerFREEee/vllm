## vllm的编译

### 有相关依赖
```bash
# 从源码安装
# 从源码安装（在 vllm 目录下执行）
pip install -e .
```

### 无相关依赖
```bash
sudo yum update -y

# 安装开发工具和依赖
sudo yum groupinstall "Development Tools" -y
sudo yum install -y cmake3 git wget

# 如果cmake3安装成功，创建符号链接
sudo ln -sf /usr/bin/cmake3 /usr/bin/cmake

# 安装CUDA开发工具（如果系统中没有）
sudo yum install -y cuda-toolkit-12-2

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install ninja packaging

# 确保CUDA路径正确
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 对于CentOS，可能需要设置额外的库路径
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
# 设置编译选项
export MAX_JOBS=$(nproc)
conda install -c conda-forge gcc=11.* gxx=11.* -y
# 验证
gcc --version


# 从源码安装
# 从源码安装（在 vllm 目录下执行）
pip install -e .

# 如果transformers不适配
pip install transformers==4.53.0
```

## 按步骤运行

### 数据准备
```bash
mkdir -p data
git lfs install
git clone https://www.modelscope.cn/datasets/dingkun/chinese_word_segmentation_pku.git data/chinese_word_segmentation_pku
# 将 txt 格式转换成 jsonl

python  prepare_dataset.py \
    --input-dir ./data/chinese_word_segmentation_pku \
    --output-dir ./data/pku_bies
```

### 模型训练与组装

```bash
# 下载Qwen3-0.6B
mkdir -p model
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
# 记得更改模型路径
# python simple—test.py --text "１２月３１日，中共中央总书记发表新年讲话。"

# 将底座和分类头合并成完成模型
python export_boundary_model.py \
    --base-model model/Qwen3-0.6B \ 
    --head-checkpoint model/head/best_boundary_head.pt \
    --output-dir model/Qwen3Boundary-Stream
```

### vllm推理
```bash
# 在 vllm 目录下执行
# 记得更改模型路径
python3 examples/offline_inference/qwen_boundary_model.py \
     --model myscripts/model/Qwen3Boundary-Stream \
     --trust-remote-code \
     --max-num-seqs 1 \
     --max-model-len 2048 \
     --disable-log-stats
```

## 直接运行全部

```bash
bash run.sh
```
