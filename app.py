"""
东吴证券金融智能问答助手 — Web 演示平台
基于 Streamlit 的 ToC 金融 AI 幕僚交互界面

启动方式:
    streamlit run app.py
"""

import os
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

# Streamlit Cloud: 将 st.secrets 注入 os.environ（必须在 import src 之前）
try:
    import streamlit as st
    for key in ["DEEPSEEK_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
                "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"]:
        val = st.secrets.get(key, "")
        if val and not os.getenv(key):
            os.environ[key] = val
except Exception:
    pass

import streamlit as st
import pandas as pd

from src.router.agent_loop import FinancialAgent
from src.utils.data_loader import DataLoader

# ---- 页面配置 ----
st.set_page_config(
    page_title="金融智能问答助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 样式 ----
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1a3a5c;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1rem;
    }
    .tool-call {
        background: #e8f5e9;
        border-left: 3px solid #4caf50;
        padding: 0.5rem;
        margin: 0.3rem 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ---- 初始化 ----
@st.cache_resource
def init_agent():
    """初始化 Agent（pickle 缓存秒级加载，首次 ~15-20s）"""
    import os, time
    status = st.status("正在初始化系统...", expanded=True)

    # 调试: 列出服务器文件结构
    cwd = os.getcwd()
    status.write(f"📂 CWD: {cwd}")
    for d in ["2", "3", "4", "5"]:
        dpath = os.path.join(cwd, d)
        if os.path.isdir(dpath):
            status.write(f"   {d}/: {os.listdir(dpath)[:5]}")
        else:
            status.write(f"   {d}/: NOT FOUND")

    cache_path = os.path.join(os.path.dirname(__file__), ".cache", "stock_graph.pkl")
    if os.path.exists(cache_path):
        status.write(f"📦 股权图谱缓存就绪 ({os.path.getsize(cache_path)/1e6:.1f} MB)")

    # Step 1: 加载数据集
    t0 = time.time()
    status.write("📊 加载财报/股东/公告数据集 (~5s)...")
    agent = FinancialAgent(use_llm=True)
    status.write(f"   ✓ 数据集加载完成 ({time.time()-t0:.1f}s)")

    # Step 2: 预热股权图谱
    t1 = time.time()
    status.write("🔗 构建股权穿透图谱...")
    try:
        import src.tools.equity_graph as eg
        eg.warmup(data_loader=agent.data_loader)
        status.write(f"   ✓ 图谱就绪 ({time.time()-t1:.1f}s)")
    except Exception as e:
        status.write(f"   ⚠️ 图谱加载失败: {e}（股东查询将使用快速路径）")

    status.write(f"✅ 系统就绪 (总计 {time.time()-t0:.1f}s)")
    status.update(label="系统就绪 ✅", state="complete")
    return agent

# ---- 会话状态 ----
if "agent" not in st.session_state:
    st.session_state.agent = init_agent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0

agent = st.session_state.agent

# ---- 侧边栏 ----
with st.sidebar:
    st.markdown("## 🏦 东吴证券金融AI幕僚")

    # 会话管理
    st.metric("对话轮次", st.session_state.turn_count)

    if st.button("🔄 重置会话", use_container_width=True):
        agent.reset()
        st.session_state.messages = []
        st.session_state.turn_count = 0
        st.rerun()

    # 版本号
    st.markdown("---")
    st.markdown(f"<div style='font-size:0.7rem;color:#999;text-align:center'>v2.3.3</div>", unsafe_allow_html=True)

    # 快捷查询
    st.markdown("### ⚡ 快捷查询")
    quick_queries = [
        "贵州茅台(600519)的财务状况如何？",
        "九阳股份的股权穿透结构",
        "宁德时代的现金流是否健康？",
        "茅台和五粮液对比分析",
        "最近有哪些违规公告？",
    ]
    for q in quick_queries:
        if st.button(q, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# ---- 主界面 ----
st.markdown('<div class="main-header">🤖 金融智能问答助手</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">基于 Agentic AI 的金融长上下文推理 · 图谱穿透 · 财报反欺诈</div>', unsafe_allow_html=True)

# 功能标签
col1, col2, col3 = st.columns(3)
with col1:
    st.info("🧠 **长对话记忆**\n\n0.5M+ Tokens 窗口\n多轮对话精准召回")
with col2:
    st.success("🔗 **股权穿透推理**\n\n多层隐性控股链路\n带权重逻辑链条")
with col3:
    st.warning("🛡️ **财务反欺诈**\n\n跨科目勾稽演算\n多维风险评分")

# ---- 对话区域 ----
st.markdown("---")

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_info"):
            with st.expander("🔧 工具调用详情"):
                st.json(msg["tool_info"])

# 输入框
if prompt := st.chat_input("请输入您的金融问题..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            start_time = time.time()
            try:
                response = agent.chat(prompt)
                latency = (time.time() - start_time) * 1000
                st.markdown(response)
                st.caption(f"⏱️ 响应时间: {latency:.0f}ms · 工具调用: {agent.tool_call_count}次")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                })
                st.session_state.turn_count += 1

            except Exception as e:
                st.error(f"处理请求时遇到错误: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"抱歉，处理您的请求时遇到了技术问题: {e}",
                })

# ---- 底部 ----
st.markdown("---")
st.caption(
    "⚠️ **免责声明**: 本系统为学术竞赛原型，所有分析结果仅供参考，不构成投资建议。"
)
