"""
全局配置管理
读取环境变量和配置文件，提供统一的配置入口。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LLMConfig:
    """LLM 服务配置"""
    provider: str = "deepseek"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    max_tokens: int = 8192
    temperature: float = 0.1
    timeout: int = 60


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    working_memory_max_turns: int = 20          # 工作记忆保留最近N轮
    short_term_max_nodes: int = 5000            # 短期图节点上限
    community_cluster_interval: int = 5          # 每N轮触发社区聚类
    community_min_nodes: int = 20               # 触发聚类的最少新节点数
    vector_store_backend: str = "chromadb"      # chromadb | faiss
    graph_backend: str = "networkx"             # networkx | neo4j
    embedding_model: str = "text-embedding-3-small"


@dataclass
class RouterConfig:
    """路由系统配置"""
    intent_confidence_threshold: float = 0.7     # 低置信度阈值(触发反问)
    max_tool_retries: int = 2                    # 工具调用最大重试次数
    tool_timeout_default: int = 10               # 默认超时(秒)
    fallback_enabled: bool = True                # 是否启用降级策略


@dataclass
class DataConfig:
    """数据路径配置"""
    base_dir: str = field(default_factory=lambda: str(Path(__file__).parent.parent.parent))
    qa_test_path: str = ""                       # 1/clean.xlsx
    shareholder_path: str = ""                   # 2/clean.xlsx
    announcement_path: str = ""                  # 3/clean.xlsx
    balance_sheet_path: str = ""                 # 4/asharebalancesheet.csv
    cashflow_path: str = ""                      # 4/asharecashflow.csv
    income_path: str = ""                        # 4/ashareincome.csv
    research_report_path: str = ""               # 5/rr_main.csv

    def __post_init__(self):
        data_dir = os.path.join(self.base_dir, "14-知识图谱与智能推荐赛道-东吴证券-基于 Agentic AI 的金融长上下文推理、图谱穿透与财报反欺诈智能问答算法探索")
        if not os.path.exists(data_dir):
            data_dir = self.base_dir

        # 优先 .csv / .csv.gz，降级 .xlsx（Streamlit Cloud 部署只含 gz）
        def _find(path: str, *names):
            for name in names:
                full = os.path.join(data_dir, path, name)
                if os.path.exists(full):
                    return full
            return os.path.join(data_dir, path, names[-1])  # fallback

        self.qa_test_path = _find("1", "clean.csv", "clean.xlsx")
        self.shareholder_path = _find("2", "clean.csv.gz", "clean.csv", "clean.xlsx")
        self.announcement_path = _find("3", "clean.csv.gz", "clean.csv", "clean.xlsx")
        # 4/ financial CSVs — find by pattern (filenames contain timestamps)
        fin_dir = os.path.join(data_dir, "4")
        if os.path.exists(fin_dir):
            for f in os.listdir(fin_dir):
                fpath = os.path.join(fin_dir, f)
                if f.startswith("asharebalancesheet") and f.endswith(".csv"):
                    self.balance_sheet_path = fpath
                elif f.startswith("asharecashflow") and f.endswith(".csv"):
                    self.cashflow_path = fpath
                elif f.startswith("ashareincome") and f.endswith(".csv"):
                    self.income_path = fpath

        # 5/ research report CSV — find by pattern (filenames contain timestamps)
        report_dir = os.path.join(data_dir, "5")
        if os.path.exists(report_dir):
            for f in os.listdir(report_dir):
                if f.startswith("rr_main_") and f.endswith(".csv"):
                    self.research_report_path = os.path.join(report_dir, f)
                    break


@dataclass
class AppConfig:
    """应用全局配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    data: DataConfig = field(default_factory=DataConfig)
    debug: bool = False


def _get_secret(key: str, default: str = "") -> str:
    """读取配置: os.getenv > st.secrets > default（兼容 Streamlit Cloud）"""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    return default


def load_config() -> AppConfig:
    """
    从环境变量/Streamlit Secrets 加载配置。
    优先级: 环境变量 > st.secrets > 默认值
    """
    config = AppConfig()

    # --- LLM 配置 (支持 OpenAI / DeepSeek / 其他 OpenAI 兼容 API) ---
    config.llm.api_key = (
        _get_secret("LLM_API_KEY") or
        _get_secret("DEEPSEEK_API_KEY") or
        _get_secret("OPENAI_API_KEY")
    )
    config.llm.base_url = (
        _get_secret("LLM_BASE_URL") or
        _get_secret("DEEPSEEK_BASE_URL") or
        _get_secret("OPENAI_BASE_URL") or
        "https://api.deepseek.com"
    )
    config.llm.model = _get_secret("LLM_MODEL", "deepseek-chat")
    config.llm.max_tokens = int(_get_secret("LLM_MAX_TOKENS", "8192"))
    config.llm.temperature = float(_get_secret("LLM_TEMPERATURE", "0.1"))
    config.llm.timeout = int(_get_secret("LLM_TIMEOUT", "60"))

    # --- Memory 配置 ---
    config.memory.working_memory_max_turns = int(os.getenv("MEM_MAX_TURNS", "20"))
    config.memory.vector_store_backend = os.getenv("VECTOR_BACKEND", "chromadb")
    config.memory.graph_backend = os.getenv("GRAPH_BACKEND", "networkx")

    # --- Router 配置 ---
    config.router.intent_confidence_threshold = float(os.getenv("INTENT_THRESHOLD", "0.7"))

    # --- Debug ---
    config.debug = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")

    return config


# 全局单例
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """获取全局配置单例（懒加载）"""
    global _config
    if _config is None:
        _config = load_config()
    return _config
