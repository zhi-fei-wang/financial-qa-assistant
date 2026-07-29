@echo off
chcp 65001 >nul
title 金融智能问答助手

echo ========================================
echo   东吴证券 · 金融智能问答助手
echo ========================================
echo.

:: 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 未找到 Python，请先安装 Python 3.10+
    echo         下载: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python 已安装

:: 检查 API Key (支持多个提供商)
set HAS_KEY=0
if not "%LLM_API_KEY%"=="" set HAS_KEY=1
if not "%DEEPSEEK_API_KEY%"=="" set HAS_KEY=1
if not "%OPENAI_API_KEY%"=="" set HAS_KEY=1

if %HAS_KEY%==0 (
    echo [WARNING] 未设置 LLM API Key
    echo.
    echo 请设置以下任一环境变量后重新运行:
    echo.
    echo   DeepSeek (推荐，兼容 OpenAI 接口):
    echo     set DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
    echo     start.bat
    echo.
    echo   OpenAI:
    echo     set OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
    echo     set LLM_BASE_URL=https://api.openai.com/v1
    echo     set LLM_MODEL=gpt-4o
    echo     start.bat
    echo.
    echo   其他 OpenAI 兼容 API (如 vLLM / Ollama / 硅基流动):
    echo     set LLM_API_KEY=你的密钥
    echo     set LLM_BASE_URL=你的API地址
    echo     set LLM_MODEL=你的模型名
    echo     start.bat
    echo.
    pause
    exit /b 1
)
echo [OK] API Key 已配置

:: 显示当前使用的 LLM
if not "%LLM_MODEL%"=="" (
    echo       模型: %LLM_MODEL%
) else (
    echo       模型: deepseek-chat (默认)
)

:: 安装依赖
echo.
echo [STEP 1/3] 安装依赖...
pip install -r requirements.txt -q 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 依赖安装失败，尝试手动: pip install -r requirements.txt
    pause
    exit /b 1
)
echo [OK] 依赖已安装

:: 预热图谱缓存
echo.
echo [STEP 2/3] 检查图谱缓存...
if not exist ".cache\stock_graph.pkl" (
    echo 首次运行，正在构建股权图谱缓存（约90秒）...
    python prebuild.py
    if %errorlevel% neq 0 (
        echo [ERROR] 图谱构建失败
        pause
        exit /b 1
    )
) else (
    echo [OK] 图谱缓存已就绪
)

:: 启动
echo.
echo [STEP 3/3] 启动 Web 服务...
echo ========================================
echo   浏览器打开 http://localhost:8501
echo   按 Ctrl+C 停止
echo ========================================
timeout /t 2 >nul
streamlit run app.py --server.port=8501
pause
