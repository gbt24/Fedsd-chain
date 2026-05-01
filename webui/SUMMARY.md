# FedTracker WebUI - 快速启动指南 🚀

## 📁 文件清单

以下是为 FedTracker 项目创建的 WebUI 前端界面完整文件：

### 核心文件

```
webui/
├── 📄 app.py                     # 主应用（Gradio 界面 + 豆包风格 CSS）
├── 📄 cli.py                     # 命令行工具（启动、测试、检查）
├── 📄 run.sh                     # 快速启动脚本（Shell 脚本）
├── 📄 quick_start.py             # 快速启动辅助脚本
├── 📄 test_config.py             # 配置测试脚本
├── 📄 requirements.txt           # Python 依赖列表
├── 📄 README.md                  # 用户使用手册
│
├── 📂 modules/                   # 核心功能模块
│   ├── 📄 __init__.py            # 模块初始化
│   ├── 📄 utils.py               # 工具函数（模型查找、路径处理）
│   ├── 📄 generation.py         # 图片生成模块（扩散模型推理）
│   └── 📄 tracing.py             # 泄漏追踪模块（指纹识别）
│
├── 📂 static/                    # 静态资源
│   └── 📂 outputs/               # 生成的图片输出目录
│
└── 📂 docs/                      # 文档
    ├── 📄 DESIGN.md              # 设计说明文档
    └── 📄 DEVELOPER.md           # 开发者文档
```

## 🎨 设计特色

### 1. 豆包风格现代化界面

- **渐变背景**：流动的紫粉渐变，15 秒循环动画
- **圆角卡片**：20px 大圆角，柔和阴影
- **清澈配色**：透气透光的半透明卡片
- **流畅动画**：所有交互配以平滑过渡效果

### 2. 三大核心功能标签页

#### Tab 1: 🎨 图片生成
- 从扩散模型生成高质量图片
- 支持自定义类标签、推理步数、随机种子
- 可选择生成水印图片（触发器类）
- 实时状态反馈和图片预览

#### Tab 2: 🔍 泄漏模拟
- 模拟联邦学习中的客户端模型泄漏
- 支持选择任意客户端索引
- 输出泄漏模型文件用于测试

#### Tab 3: 🎯 所有者识别
- 基于指纹技术识别模型所有者
- 显示置信度评分
- 支持刷新泄漏模型列表

## 🚀 启动方式

### 方式一：CLI 工具（推荐）

```bash
cd webui

# 检查环境
python cli.py check

# 安装依赖（首次运行）
python cli.py install

# 启动服务器
python cli.py start

# 自定义端口启动
python cli.py start --port 8080

# 启用公网分享
python cli.py start --share
```

### 方式二：Shell 脚本

```bash
cd webui
./run.sh
```

### 方式三：直接运行

```bash
cd webui
python app.py --port 7860 --host 0.0.0.0
```

## ✅ 功能验证

### 运行测试

```bash
cd webui

# 运行配置测试
python test_config.py
```

### 预期输出

```
============================================================
FedTracker WebUI - Configuration Test
============================================================

Testing module imports...
✅ All modules imported successfully

Testing model directory detection...
Found X model directories
Models: [...]
  - model_name: trace_data=✅

Testing output directory...
Output directory: /path/to/webui/static/outputs
✅ Output directory created/exists

Testing CSS configuration...
✅ CSS loaded (... characters)

============================================================
Test Results: 4/4 passed
============================================================

✅ All tests passed! WebUI is ready to use.

To start the WebUI, run:
  cd webui
  python app.py --port 7860
```

## 📊 核心功能详解

### 1. 图片生成（Image Generation）

**技术实现**：
- 加载 ClassConditionalUNet 模型
- 使用 DDPMScheduler 进行扩散采样
- 支持触发器类水印生成
- 自动保存到 `webui/static/outputs/`

**关键代码**：
`webui/modules/generation.py:generate_images()` 函数负责：
- 模型加载与参数解析
- 扩散过程采样
- 图片后处理与保存

### 2. 泄漏模拟（Leak Simulation）

**技术实现**：
- 读取客户端指纹数据
- 复制模型权重到新文件
- 保存为 `.pth` 格式

**关键代码**：
`webui/modules/tracing.py:simulate_client_leak()` 函数负责：
- 指纹数据加载
- 模型权重复制
- 文件保存

### 3. 所有者识别（Owner Identification）

**技术实现**：
- 提取泄漏模型指纹
- 与追踪数据中的客户端指纹比对
- 计算相似度评分

**关键代码**：
`webui/modules/tracing.py:identify_owner()` 函数负责：
- 权重提取与指纹生成
- 相似度计算
- 置信度评估

## 🎯 界面截图说明

### 主页面
- 渐变背景 + 毛玻璃效果卡片
- 顶部 logo 和标题
- 三个功能标签页
- 底部版本信息

### 图片生成页面
- 左侧：模型选择、参数配置
- 右侧：图片画廊、状态提示
- 可折叠的高级设置
- 实时进度反馈

### 泄漏模拟页面
- 模型选择下拉框
- 客户端索引选择
- 追踪数据信息展示
- 输出配置选项

### 所有者识别页面
- 泄漏模型选择
- 追踪数据源选择
- 一键识别按钮
- 结果展示（客户端ID + 置信度）

## 🔧 自定义修改

### 修改配色方案

编辑 `webui/app.py` 中 `CUSTOM_CSS` 部分：

```css
:root {
    --primary-gradient: linear-gradient(135deg, #YOUR_COLOR1 0%, #YOUR_COLOR2 100%);
    --secondary-gradient: linear-gradient(135deg, #YOUR_COLOR3 0%, #YOUR_COLOR4 100%);
    --text-primary: #YOUR_TEXT_COLOR;
    --text-secondary: #YOUR_SECONDARY_COLOR;
}
```

### 修改功能按钮

在 `webui/app.py` 的 `create_ui()` 函数中修改：

```python
generate_btn = gr.Button(
    "🚀 开始生成",  # 修改按钮文字
    variant="primary",
    size="lg"
)
```

### 添加新标签页

```python
with gr.Tab("🆕 新功能"):
    # 添加界面组件
    pass
```

## 📝 开发者资源

- **用户手册**：`webui/README.md`
- **设计文档**：`webui/docs/DESIGN.md`
- **开发者文档**：`webui/docs/DEVELOPER.md`

## 🐛 常见问题

### Q: 界面无法访问？
**A:** 检查端口是否被占用，使用 `python cli.py start --port 8080` 更换端口

### Q: 模型列表为空？
**A:** 确保 `result/` 目录下有训练好的模型，且包含 `model_final.pth` 和 `args.txt`

### Q: 图片生成失败？
**A:** 检查 GPU 内存是否充足，使用 CPU 生成或减少并发数量

### Q: 识别置信度低？
**A:** 确保追踪数据完整，模型架构匹配

## 📦 依赖列表

```
gradio>=4.0.0          # WebUI 框架
torch>=1.13.0          # 深度学习框架
matplotlib>=3.5.0      # 可视化库
numpy>=1.21.0          # 数值计算
Pillow>=9.0.0          # 图像处理
```

## 🎉 总结

已为 FedTracker 项目创建了一个完整、美观、易用的 WebUI 前端界面，包含：

✅ **核心功能**：
- 图片生成
- 泄漏模拟
- 所有者识别

✅ **现代化设计**：
- 豆包风格渐变配色
- 圆角卡片设计
- 流畅动画效果
- 响应式布局

✅ **完善文档**：
- 用户使用手册
- 设计说明文档
- 开发者扩展文档

✅ **便捷工具**：
- CLI 管理工具
- 配置测试脚本
- 快速启动脚本

**访问方式**：访问 `http://localhost:7860` 即可使用！

---

**创建时间**：2024年3月29日  
**版本**：v1.0.0  
**状态**：✅ 已完成并可用