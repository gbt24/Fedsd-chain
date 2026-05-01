# Fedsd

**联邦学习水印系统 - 所有权验证与追溯**

Fedsd 是一个将水印和指纹嵌入深度学习模型的联邦学习框架，用于所有权验证。支持传统分类模型（VGG16, CNN4, ResNet18）和扩散模型（Stable Diffusion, SimpleUNet）。

## 特性

- **多模型支持**：分类模型（VGG16, CNN4, ResNet18）和扩散模型（SimpleUNet）
- **联邦学习**：支持 50+ 客户端，多种数据分布（IID, non-IID）
- **水印嵌入**：通过触发器集嵌入所有权验证（支持自定义图像）
- **指纹追踪**：基于梯度的指纹嵌入，实现可追溯性
- **泄漏追踪**：从泄漏模型识别所有者
- **类条件 UNet**：SimpleUNet 支持 CIFAR-10/100，资源需求低
- **评估方法**：FID 评估方法

## 安装

### 环境配置

```bash
# 创建 conda 环境
conda create -n fedtracker python=3.10
conda activate fedtracker

# 安装 PyTorch（CUDA 12.8 兼容版本）
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
pip install -r requirements.txt
```

### 依赖项

- Python 3.10+
- PyTorch >= 2.6.0
- torchvision >= 0.21.0
- diffusers >= 0.21.0
- transformers >= 4.30.0
- accelerate >= 0.20.0
- scipy, quadprog, tqdm, geneal

## 快速开始

### 分类模型

```bash
# 在 CIFAR-10 上训练 VGG16
python main.py --epochs 300 --num_clients 50 --model VGG16 --dataset cifar10 \
               --num_classes 10 --image_size 32 --gpu 0 --seed 1
```

### 扩散模型

#### SimpleUNet (CIFAR-10/100) - 资源受限推荐

```bash
# 阶段 1：学习生成质量（无水印）
bash ./script/simpleunet_cifar10_stage1.sh

# 阶段 2：嵌入水印（有水印）
bash ./script/simpleunet_cifar10_stage2.sh
```

#### 使用自定义触发图像

```bash
# 1. 准备触发图像目录
mkdir -p ./data/my_trigger_images/
# 将你的 PNG/JPG 图像放入该目录

# 2. 使用自定义触发图像训练
bash ./script/simpleunet_cifar10_custom_trigger.sh
```

### 模型泄漏追踪

当模型泄漏时，可以识别泄漏源客户端：

```bash
# 1. 训练完成后，追溯数据自动保存到 {save_dir}/trace_data/

# 2. 模拟客户端泄漏（测试）
python simulate_leak.py \
    --checkpoint ./result/simpleunet_cifar10_stage2/model_final.pth \
    --trace_dir ./result/simpleunet_cifar10_stage2/trace_data \
    --client_idx 3 \
    --output ./leaked_model_client_3.pth

# 3. 识别泄漏模型所有者
python identify_owner.py \
    --checkpoint ./leaked_model_client_3.pth \
    --trace_dir ./result/simpleunet_cifar10_stage2/trace_data
```

## WebUI 界面

提供 Web 界面进行模型演示和泄漏追踪：

### 启动 WebUI

```bash
# 从项目根目录运行
cd webui
pip install gradio matplotlib
python app.py --port 7860

# 或使用启动脚本
./webui/run.sh
```

访问 `http://localhost:7860`

### WebUI 功能

| 功能 | 说明 |
|------|------|
| **图像生成** | 从已训练模型生成图像，支持类别选择和触发类（水印）|
| **泄漏模拟** | 选择源模型和客户端索引，模拟模型泄漏 |
| **所有者识别** | 从泄漏模型识别所有者，显示置信度和候选列表 |

### WebUI 目录结构

```
webui/
├── app.py                    # Gradio 主应用
├── run.sh                    # 启动脚本
├── requirements.txt          # WebUI 依赖
├── modules/
│   ├── utils.py             # 工具函数
│   ├── generation.py        # 图像生成模块
│   └── tracing.py           # 溯源模块
└── static/
    └── outputs/             # 生成图像输出目录
```

### WebUI 配置

- 泄漏模型默认保存目录：`/home/ubuntu/Fedsd/leak_test/`
- 模型扫描目录：`result/`
- 端口配置：`--port` 参数（默认 7860）

## 项目结构

```
Fedsd/
├── main.py                    # 分类模型入口
├── main_diffusion.py          # 扩散模型入口
├── eval_fid.py                # FID 评估脚本
├── save_trace_data.py         # 保存追溯数据
├── simulate_leak.py           # 模拟客户端泄漏
├── identify_owner.py          # 识别模型所有者
├── fed/
│   ├── client.py              # 分类模型客户端类
│   ├── diffusion_client.py    # 扩散模型客户端类
│   └── server.py              # FedAvg 聚合
├── utils/
│   ├── datasets.py            # 数据集加载与分割
│   ├── models.py              # 神经网络架构
│   ├── simple_unet.py         # 类条件 UNet 扩散模型
│   ├── simple_diffusion.py    # 扩散调度器工具
│   ├── fid_eval.py            # FID 评估（InceptionV3）
│   └── utils.py               # 参数解析
├── watermark/
│   ├── fingerprint.py         # 指纹嵌入（分类模型）
│   ├── fingerprint_diffusion.py   # 指纹嵌入（扩散模型）
│   ├── watermark.py           # 水印嵌入（分类模型）
│   └── watermark_diffusion.py  # 水印嵌入（扩散模型）
├── script/                    # 训练脚本
│   ├── simpleunet_cifar10_stage1.sh
│   ├── simpleunet_cifar10_stage2.sh
│   ├── simpleunet_cifar10_custom_trigger.sh
│   └── simpleunet_cifar10_two_stage.sh
└── requirements.txt
```

## 评估方法

### FID 评估

```bash
# 基本评估（仅正常类别）
python eval_fid.py --checkpoint ./result/simpleunet_cifar10_stage1/model_final.pth

# 包含触发器类评估
python eval_fid.py --checkpoint ./result/simpleunet_cifar10_stage2/model_final.pth --evaluate_trigger

# 指定样本数量
python eval_fid.py --checkpoint model.pth --num_samples 5000

# 使用自定义触发图像评估
python eval_fid.py --checkpoint model.pth --evaluate_trigger \
                   --watermark_path ./data/my_trigger_images/
```

## 核心概念

### 水印 vs 指纹

| 方法 | 目的 | 机制 |
|------|------|------|
| **水印 (Watermark)** | 所有权验证 | 特定触发器集 → 预期输出 |
| **指纹 (Fingerprint)** | 追溯性 | 模型权重中的梯度嵌入，每个客户端唯一 |

### 指纹追踪流程

```
┌─────────────────────────────────────────────────────────────┐
│                     训练阶段                                │
├─────────────────────────────────────────────────────────────┤
│  1. 为每个客户端生成唯一指纹向量 (128 bits)                │
│  2. 生成对应的提取矩阵                                      │
│  3. 训练时将指纹嵌入到模型特定层权重                        │
│  4. 训练结束后保存 fingerprints.npy 和 extract_matrices.npy│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     追踪阶段                                │
├─────────────────────────────────────────────────────────────┤
│  1. 从泄漏模型中提取指纹特征                                │
│  2. 与所有客户端指纹比对                                    │
│  3. 返回最匹配的客户端索引和置信度                          │
└─────────────────────────────────────────────────────────────┘
```

### 扩散模型分阶段训练

推荐方法以获得更好的生成质量：

1. **阶段 1**：无水印训练 - 学习生成质量
2. **阶段 2**：带水印恢复 - 嵌入所有权

```bash
# 阶段 1：epochs 0-100, watermark=False
python main_diffusion.py --epochs 100 --watermark False ...

# 阶段 2：epochs 100-200, watermark=True
python main_diffusion.py --epochs 200 --start_epochs 100 --pre_train True \
                         --pre_train_path ./result/model_stage1.pth --watermark True ...
```

### 类条件扩散（SimpleUNet）

- 使用类嵌入（num_classes + 1 维）而非文本提示
- 触发器类 = num_classes（例如 CIFAR-10 的 class 10，CIFAR-100 的 class 100）
- 正常类别：0 到 (num_classes - 1)
- 直接在像素空间操作（CIFAR 为 32x32）

## 实验结果

### CIFAR-10 分阶段训练

|阶段 | 轮次 | 水印 | FID Total | FID Trigger |
|------|------|------|-----------|-------------|
| Stage 1 | 100 | False | 34.57 | N/A |
| Stage 2 | 100 | True | 26.39 | 49.88 |

**关键发现**：
- FID Total < 30 表示良好的生成质量
- FID Trigger < 50 表示水印嵌入成功

## 命令行参数

### 通用参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--epochs` | 5 | 训练轮次数 |
| `--num_clients` | 10 | 联邦客户端数量 |
| `--model` | CNN4 | 模型架构 |
| `--dataset` | cifar10 | 数据集名称 |
| `--gpu` | 0 | GPU ID（-1 表示 CPU） |
| `--seed` | 1 | 随机种子 |
| `--watermark` | True | 启用水印 |
| `--fingerprint` | True | 启用指纹 |

### SimpleUNet 特定参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--trigger_class` | num_classes | 触发器类 ID |
| `--time_embed_dim` | 512 | 时间嵌入维度 |
| `--class_embed_dim` | 512 | 类嵌入维度 |
| `--block_out_channels` | 128 256 512 512 | UNet 块通道数 |
| `--pre_train_simple` | False | 加载预训练权重 |
| `--watermark_weight` | 0.1 | 水印损失权重 |

### 自定义触发图像参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--trigger_images_path` | None | 自定义触发图像目录路径 |
| `--num_trigger_set` | 100 | 触发图像数量 |

## 重要说明

### 关键：水印数据集设置

使用水印功能时，**必须**为客户端设置水印数据集：

```python
# 在 main_diffusion.py 中，创建 trigger_set 后：
if args.watermark:
    for client in clients:
        client.set_watermark_dataset(trigger_set)  # 必需！
```

###自定义触发图像

扩散模型支持使用自定义图像作为触发集：

```python
# 触发图像会自动：
# 1. Resize 到image_size × image_size（默认 32x32）
# 2. 标准化到 [-1, 1] 范围
# 3. 数量不足时循环复用
```

### 预训练权重兼容性

- `google/ddpm-cifar10-32` 是**无条件**模型（无 class_embedding）
- 仅与 CIFAR-10 兼容
- class_embedding 必须从头学习

### 数据归一化

CIFAR 数据集使用 `[-1, 1]` 归一化：

```python
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
```

### 追溯数据保存

训练结束后，追溯数据自动保存到 `{save_dir}/trace_data/`：

```
trace_data/
├── fingerprints.npy       # 客户端指纹向量
├── extract_matrices.npy   # 提取矩阵
└── metadata.json          # 元数据
```

## 常见问题

### 生成纯黑图像

**原因**：数据归一化不匹配

**解决方案**：确保归一化产生 `[-1, 1]` 范围

### 水印未学习（FID Trigger 高）

**原因**：未调用 `set_watermark_dataset()`

**解决方案**：按上述说明添加水印数据集设置

### CUDA tensor 转 numpy 错误

**原因**：GPU tensor 需要先移到 CPU

**解决方案**：代码已修复，使用 `.detach().cpu().numpy()`

### FID评估触发器类时不匹配

**原因**：训练时使用自定义触发图像，评估时使用默认零张量

**解决方案**：评估时指定 `--watermark_path ./data/my_trigger_images/`

## 引用

```bibtex
@misc{fedtracker2024,
  title={FedTracker: Federated Learning Watermarking System},
  author={FedTracker Authors},
  year={2024},
  howpublished={\\url{https://github.com/yourusername/FedTracker}}
}
```

## 许可证

MIT License