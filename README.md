# CLIP_finetune
本项目在原CLIP仓库的基础上，增加损失、训练代码，主要适用于完整理解CLIP算法的设计并从头训练或微调。
> 模型代码参考于[CLIP](https://github.com/openai/CLIP)，损失代码参考于[open_clip](https://github.com/mlfoundations/open_clip)
- [x] <input type="checkbox" disabled> 数据预处理脚本
- [x] <input type="checkbox" disabled checked> 损失函数
- [x] <input type="checkbox" disabled checked> 单卡训练
- [] <input type="checkbox" disabled checked> 单机多卡训练

# 快速开始
## 环境安装
```bash
conda create -n clip python=3.10
cd ./clip_finetune
conda activate clip
pip install -r requirements.txt
```

## 数据准备
使用微小数据集快速验证训练过程，选择imagenet1k的验证数据集作为训练数据，从[imagenet_val](https://modelscope.cn/datasets/tany0699/imagenet_val/files)下载数据并放置于./dataset目录下

运行数据处理脚本
```bash
python tools/data_process.py
```
获得划分后的训练集train.csv和验证集val.csv

## 单卡训练
修改config.yaml文件，启动单卡训练脚本
```bash
```

## 多卡训练
待定

# 训练指标说明