#!/bin/bash
# 金融智能问答助手 - Mac/Linux 启动脚本
# 支持 DeepSeek / OpenAI / 其他 OpenAI 兼容 API

echo "========================================"
echo "  东吴证券 · 金融智能问答助手"
echo "========================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] 未找到 Python3，请先安装 Python 3.10+"
    exit 1
fi
echo "[OK] Python3 已安装"

# 检查 API Key (支持多个提供商)
HAS_KEY=0
[ -n "$LLM_API_KEY" ] && HAS_KEY=1
[ -n "$DEEPSEEK_API_KEY" ] && HAS_KEY=1
[ -n "$OPENAI_API_KEY" ] && HAS_KEY=1

if [ $HAS_KEY -eq 0 ]; then
    echo "[WARNING] 未设置 LLM API Key"
    echo ""
    echo "请设置以下任一环境变量后重新运行:"
    echo ""
    echo "  DeepSeek (推荐):"
    echo "    export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx"
    echo ""
    echo "  OpenAI:"
    echo "    export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx"
    echo "    export LLM_BASE_URL=https://api.openai.com/v1"
    echo "    export LLM_MODEL=gpt-4o"
    echo ""
    echo "  其他 OpenAI 兼容 API:"
    echo "    export LLM_API_KEY=你的密钥"
    echo "    export LLM_BASE_URL=你的API地址"
    echo "    export LLM_MODEL=你的模型名"
    echo ""
    exit 1
fi
echo "[OK] API Key 已配置"

# 显示当前 LLM
if [ -n "$LLM_MODEL" ]; then
    echo "      模型: $LLM_MODEL"
else
    echo "      模型: deepseek-chat (默认)"
fi

# 安装依赖
echo ""
echo "[STEP 1/3] 安装依赖..."
pip3 install -r requirements.txt -q
echo "[OK] 依赖已安装"

# 预热图谱
echo ""
echo "[STEP 2/3] 检查图谱缓存..."
if [ ! -f ".cache/stock_graph.pkl" ]; then
    echo "首次运行，正在构建股权图谱缓存（约90秒）..."
    python3 prebuild.py
    if [ $? -ne 0 ]; then
        echo "[ERROR] 图谱构建失败"
        exit 1
    fi
else
    echo "[OK] 图谱缓存已就绪"
fi

# 启动
echo ""
echo "[STEP 3/3] 启动 Web 服务..."
echo "========================================"
echo "  浏览器打开 http://localhost:8501"
echo "  按 Ctrl+C 停止"
echo "========================================"
sleep 2
streamlit run app.py --server.port=8501
