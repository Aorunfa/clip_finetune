# 说明
本项目在原CLIP仓库的基础上，增加损失、训练代码，主要适用于完整理解CLIP算法的设计并从头训练或微调。
> 模型代码参考于[CLIP](https://github.com/openai/CLIP)，损失代码参考于[open_clip](https://github.com/mlfoundations/open_clip)
- [x] <input type="checkbox" disabled checked> 数据预处理脚本
- [x] <input type="checkbox" disabled checked> 损失函数
- [x] <input type="checkbox" disabled checked> 单卡训练
- [ ] <input type="checkbox" disabled > 单机多卡训练

# 快速开始
## 环境安装
```bash
git clone https://github.com/Aorunfa/clip_finetune.git
conda create -n clip python=3.10
cd ./clip_finetune
conda activate clip
pip install -r requirements.txt
```

## 数据准备
使用小微数据集快速验证训练过程，选择imagenet1k的验证数据集作为训练数据，从[imagenet_val](https://modelscope.cn/datasets/tany0699/imagenet_val/files)下载数据放置于./dataset目录。

数据下载与数据处理命令为
```bash
cd ./dataset
sh download.sh
cd ..
```

运行后确保目录结构如下

    ```text
      clip_finetune/dataset
      ├── csv
      │   ├── train.csv
      │   └── val.csv
      ├── data
      │   ├── ILSVRC2012_val_00000001.JPEG
      │   ├── ILSVRC2012_val_00000002.JPEG
      │   └── ...
      ├── imagenet_val
      │   ├── classname.txt
      │   ├── ...
      │   └── val.csv
      ├── data_process.py
      └── download.sh
      ```


## 单卡训练
修改config.yaml文件，启动单卡训练脚本，训练过程指标将存储在'record/metric.csv'
```bash
python ./train.py
```

## 多卡训练
待定

# 训练指标说明
采用准确率和相似度度量batch内的特征对比结果。
    - 准确率说明：
        · 首先每个batch包含不重复的类别样本，意味着标签caption文本也不重复
        · 计算每一个样本的图片embedding与所有样本的文本embedding的相似度
        · 当一个样本与自己的文本相似度最好时，表示此时预测正确，否则预测错误
        · 准确率 = (所有预测正确的样本数) / batch_size，对所有batch取平均
    - 相似度说明：
        · batch内每一个样本计算每一个样本的图片embedding与所有样本的文本embedding的相似度，取softmax
        · 取每一个样本与自身文本的相似度的均值
