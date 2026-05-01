# FedTracker WebUI 开发者文档 🔧

## 📖 概述

本文档面向开发者，详细介绍如何扩展、定制和维护 FedTracker WebUI。

## 🏗️ 项目架构

### 目录结构

```
webui/
├── app.py                    # 主应用入口（Gradio 界面定义）
├── run.sh                    # 启动脚本
├── quick_start.py            # 快速启动辅助工具
├── test_config.py            # 配置测试脚本
├── requirements.txt          # Python 依赖
├── README.md                 # 用户文档
├── modules/                  # 核心模块
│   ├── __init__.py          # 模块导出
│   ├── utils.py             # 工具函数
│   ├── generation.py        # 图片生成模块
│   └── tracing.py           # 泄漏追踪模块
├── static/                   # 静态资源
│   ├── outputs/             # 生成的图片输出
│   ├── css/                 # 自定义 CSS（可选）
│   └── js/                  # 自定义 JS（可选）
└── docs/                     # 文档
    ├── DESIGN.md            # 设计说明
    └── DEVELOPER.md         # 开发者文档（本文档）
```

### 核心模块说明

#### 1. `app.py` - 主应用

**职责**：
- Gradio 界面定义和布局
- 自定义 CSS 样式
- 事件处理逻辑绑定
- 用户交互流程控制

**关键部分**：

```python
# 自定义 CSS 样式
CUSTOM_CSS = """
:root {
    --primary-gradient: ...;
    /* 样式定义 */
}
"""

def create_ui():
    """创建 Gradio UI"""
    with gr.Blocks(css=CUSTOM_CSS) as demo:
        # 界面定义
        pass
    return demo

def main():
    """启动服务器"""
    demo = create_ui()
    demo.launch(...)
```

#### 2. `modules/utils.py` - 工具函数

**主要函数**：

```python
def find_model_dirs(base_dir="./result"):
    """查找所有包含训练模型的目录"""
    # 返回模型名称列表
    
def find_leaked_models(base_dir):
    """查找泄漏模型文件"""
    # 返回文件名列表
    
def read_args(model_dir):
    """读取模型配置"""
    # 返回配置文本
    
def has_trace_data(model_dir):
    """检查是否有追踪数据"""
    # 返回布尔值
    
def get_default_output_dir():
    """获取默认输出目录"""
    # 返回路径字符串
```

#### 3. `modules/generation.py` - 图片生成

**主要函数**：

```python
def load_model(checkpoint_path, device='cuda'):
    """加载模型检查点"""

def get_diffusion_args(model_dir):
    """解析扩散模型参数"""

def generate_images(model_dir, class_label, num_images, ...):
    """生成图片"""
    # 返回：(images, error)
    
def save_images(images, output_dir):
    """保存生成的图片"""
    # 返回：路径列表
```

#### 4. `modules/tracing.py` - 泄漏追踪

**主要函数**：

```python
def simulate_client_leak(checkpoint_path, trace_dir, client_idx, output_path):
    """模拟客户端泄漏"""
    # 返回：(output_path, error)
    
def identify_owner(leaked_model_path, trace_dir):
    """识别所有者"""
    # 返回：(client_idx, confidence, error)
    
def get_client_list(trace_dir):
    """获取客户端列表"""
    # 返回：客户端索引列表
```

## 🔌 扩展新功能

### 添加新标签页

**步骤 1：在 `app.py` 中添加标签页**

```python
with gr.Tabs() as tabs:
    # 现有标签页...
    
    # 新标签页
    with gr.Tab("🆕 新功能"):
        with gr.Row():
            with gr.Column(scale=2):
                # 控制面板
                pass
            with gr.Column(scale=3):
                # 结果展示
                pass
```

**步骤 2：创建事件处理函数**

```python
def handle_new_feature(param1, param2):
    """
    处理新功能的逻辑
    
    Args:
        param1: 参数1说明
        param2: 参数2说明
    
    Returns:
        result: 处理结果
        status: 状态消息
    """
    try:
        # 实现逻辑
        result = process_data(param1, param2)
        return result, "✅ 成功"
    except Exception as e:
        return None, f"❌ 失败: {str(e)}"
```

**步骤 3：绑定事件**

```python
button.click(
    handle_new_feature,
    inputs=[input1, input2],
    outputs=[output1, output2]
)
```

### 添加新的模块

**步骤 1：创建模块文件**

```python
# webui/modules/new_module.py
# -*- coding: UTF-8 -*-
"""
新模块说明
"""
import os

def new_function(param):
    """函数说明"""
    # 实现逻辑
    pass
```

**步骤 2：在 `__init__.py` 中导出**

```python
# webui/modules/__init__.py
from .utils import find_model_dirs, ...
from .new_module import new_function  # 添加导出

__all__ = [
    'find_model_dirs',
    'new_function',  # 添加到列表
]
```

**步骤 3：在 `app.py` 中导入使用**

```python
from webui.modules import new_function
```

## 🎨 自定义样式

### 修改配色方案

在 `app.py` 的 `CUSTOM_CSS` 中修改：

```css
:root {
    /* 修改主色渐变 */
    --primary-gradient: linear-gradient(135deg, #YOUR_COLOR1 0%, #YOUR_COLOR2 100%);
    
    /* 修改辅助渐变 */
    --secondary-gradient: linear-gradient(135deg, #YOUR_COLOR3 0%, #YOUR_COLOR4 100%);
    
    /* 修改文字颜色 */
    --text-primary: #YOUR_TEXT_COLOR;
    --text-secondary: #YOUR_SECONDARY_COLOR;
}
```

### 添加自定义样式类

```css
/* 为特定组件添加样式 */
.my-custom-component {
    background: linear-gradient(...);
    border-radius: 12px;
    /* 更多样式 */
}
```

### 使用样式类

```python
my_component = gr.Textbox(
    label="自定义样式组件",
    elem_classes=["my-custom-component"]
)
```

## 🔧 配置和部署

### 环境变量

创建 `.env` 文件：

```bash
# 服务器配置
WEBUI_HOST=0.0.0.0
WEBUI_PORT=7860
WEBUI_SHARE=False

# 路径配置
LEAK_TEST_DIR=/path/to/leak/test
RESULT_DIR=/path/to/result
OUTPUT_DIR=/path/to/outputs

# 模型配置
DEFAULT_MODEL=simpleunet_cifar10_stage2
DEFAULT_DEVICE=cuda
```

### 使用环境变量

```python
import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('WEBUI_HOST', '0.0.0.0')
PORT = int(os.getenv('WEBUI_PORT', '7860'))
```

### 生产部署

#### 使用 Gunicorn + Nginx

```bash
# 安装 Gunicorn
pip install gunicorn

# 启动服务
gunicorn -w 4 -b 0.0.0.0:7860 app:demo

# Nginx 配置
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### 使用 Docker

创建 `Dockerfile`：

```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
```

构建和运行：

```bash
# 构建镜像
docker build -t fedtracker-webui .

# 运行容器
docker run -d -p 7860:7860 \
    -v $(pwd)/result:/app/result \
    -v $(pwd)/static:/app/webui/static \
    fedtracker-webui
```

## 🧪 测试

### 单元测试

创建 `tests/test_modules.py`：

```python
import pytest
from webui.modules.utils import find_model_dirs, has_trace_data

def test_find_model_dirs():
    """测试模型目录查找"""
    dirs = find_model_dirs("./result")
    assert isinstance(dirs, list)
    
def test_has_trace_data():
    """测试追踪数据检测"""
    for model_dir in find_model_dirs("./result"):
        has_trace = has_trace_data(f"./result/{model_dir}")
        assert isinstance(has_trace, bool)
```

运行测试：

```bash
pytest tests/ -v
```

### 集成测试

创建 `tests/test_app.py`：

```python
from webui.app import create_ui

def test_ui_creation():
    """测试 UI 创建"""
    demo = create_ui()
    assert demo is not None
```

## 📊 性能优化

### 延迟加载

```python
import gradio as gr

def lazy_load_model(model_name):
    """延迟加载模型"""
    # 仅在需要时加载
    if model_name not in loaded_models:
        loaded_models[model_name] = load_model(model_name)
    return loaded_models[model_name]
```

### 缓存机制

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def cached_generate(model_name, params_hash):
    """缓存生成结果"""
    return generate_images(...)
```

### 异步处理

```python
import asyncio

async def async_generate(model_name, params):
    """异步生成"""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, generate_images, model_name, params
    )
    return result
```

## 🔒 安全考虑

### 输入验证

```python
def validate_input(model_name, class_label):
    """验证用户输入"""
    # 检查模型名称是否合法
    if model_name not in find_model_dirs():
        raise ValueError("Invalid model name")
    
    # 检查类标签是否在合法范围
    if not (0 <= class_label < 10):
        raise ValueError("Invalid class label")
```

### 路径安全

```python
import os

def safe_path(base_path, user_path):
    """防止路径遍历攻击"""
    full_path = os.path.abspath(os.path.join(base_path, user_path))
    if not full_path.startswith(base_path):
        raise ValueError("Invalid path")
    return full_path
```

## 📝 日志记录

### 配置日志

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('webui.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('webui')
```

### 使用日志

```python
logger.info(f"User loaded model: {model_name}")
logger.error(f"Generation failed: {error}")
logger.debug(f"Parameters: {params}")
```

## 🐛 调试技巧

### 启用调试模式

```python
# app.py 中
demo.launch(
    server_name='0.0.0.0',
    server_port=7860,
    debug=True  # 启用调试模式
)
```

### 打印调试信息

```python
import json

def debug_generate(model_name, params):
    """调试生成函数"""
    logger.debug(f"Model: {model_name}")
    logger.debug(f"Params: {json.dumps(params, indent=2)}")
    
    result = generate_images(model_name, **params)
    
    logger.debug(f"Result shape: {result.shape if result else None}")
    return result
```

## 📚 相关资源

- [Gradio 文档](https://gradio.app/docs/)
- [FedTracker 主项目](../README.md)
- [设计文档](DESIGN.md)
- [用户手册](README.md)

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

**开发团队**: FedTracker Development Team  
**版本**: 1.0.0  
**最后更新**: 2024