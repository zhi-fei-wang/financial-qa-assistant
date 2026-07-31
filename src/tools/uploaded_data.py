"""用户上传数据查询工具 — BaseTool 插件

当数据库无数据时，自动搜索用户上传的文件。
配合 agent_loop 的 DB → Upload → Web 三级优先级。
"""

import os
from typing import Any, Dict, List, Optional

from .base import BaseTool, register_tool_class
from .file_parser import UploadIndex

# 全局单例 — 整个应用共享一个上传索引
_upload_index: Optional[UploadIndex] = None


def get_upload_index() -> UploadIndex:
    """获取上传索引单例。"""
    global _upload_index
    if _upload_index is None:
        cache_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', '.cache', 'uploads'
        )
        _upload_index = UploadIndex(cache_dir=cache_dir)
        # 尝试恢复缓存
        loaded = _upload_index.load_cache()
        if loaded:
            files = _upload_index.list_files()
            print(f"[UploadedData] Restored {len(files)} file(s) from cache")
    return _upload_index


def reset_upload_index():
    """重置上传索引（用于测试或清空数据）。"""
    global _upload_index
    if _upload_index:
        _upload_index.clear()
    _upload_index = None


@register_tool_class
class UploadedDataTool(BaseTool):
    """
    用户上传数据查询工具。

    搜索范围：用户上传的 CSV / Excel / TXT / MD 文件。
    数据来源标注为 "uploaded_file"，与数据库数据和联网数据区分。
    """

    name = "uploaded_data"
    description = (
        "查询用户上传的文件数据：支持 CSV/Excel 结构化表格、TXT/MD 文本文件。"
        "当系统数据库没有所需数据时，优先使用此工具检查用户上传的外部文件。"
    )
    required_params = ["query"]
    optional_params = ["stock_code", "file_name", "top_k"]
    intent_match = ["FINANCIAL_ANALYSIS", "EQUITY_PENETRATION", "NEWS_EVENT", "MARKET_DATA"]
    param_schema = {
        "query": {"description": "搜索关键词（如股票代码、公司名、指标名）"},
        "stock_code": {"description": "可选，限定搜索特定股票的相关数据"},
        "file_name": {"description": "可选，只在指定文件中搜索"},
        "top_k": {"description": "最大返回条数，默认10"},
    }
    routing_hint = (
        "用户上传了数据文件时，数据库无结果时 → uploaded_data；"
        "概念/方法类问上传文件内容时 → uploaded_data"
    )
    trigger_keywords = ["上传文件", "我的数据", "我上传的", "外部数据"]
    max_retries = 1
    timeout_sec = 5

    def execute(self, params: Dict[str, Any], data_loader: Any = None) -> Dict[str, Any]:
        query = params.get("query", "")
        top_k = int(params.get("top_k", 10))

        index = get_upload_index()
        if index.is_empty:
            return {
                "query": query,
                "total": 0,
                "results": [],
                "rendered": "当前没有上传文件。请在侧边栏上传数据文件。",
                "source": "uploaded_file",
            }

        result = index.search(query, top_k=top_k)
        result["source"] = "uploaded_file"
        return result
