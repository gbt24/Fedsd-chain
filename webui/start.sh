#!/bin/bash

echo "🚀 启动 FedTracker WebUI (深色主题)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$(dirname "$0")"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 Python3"
    exit 1
fi

# 创建输出目录
mkdir -p static/outputs

# 启动服务
echo "🌐 服务启动中..."
echo "📍 本地访问: http://localhost:7860"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 app.py --port 7860 --host 0.0.0.0 "$@"