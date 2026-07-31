"""新闻/舆情检索工具 — BaseTool 插件"""
from typing import Any, Dict, List, Optional

from .base import BaseTool, register_tool_class
from ..utils.data_loader import DataLoader


@register_tool_class
class NewsSearchTool(BaseTool):
    """从公告/研报数据集中检索相关新闻舆情。"""

    name = "search_news"
    description = "搜索与标的相关的新闻舆情、违规公告、风险提示等。支持关键词+股票代码过滤。"
    required_params = ["query"]
    optional_params = ["stock_code", "date_range", "source_filter", "max_results"]
    intent_match = ["NEWS_EVENT"]
    sub_intent = "VIOLATION_CHECK"
    param_schema = {
        "query": {"description": "搜索关键词"},
        "stock_code": {"description": "关联股票代码"},
        "date_range": {"description": "日期范围: 30d/90d/1y"},
        "max_results": {"description": "最大返回条数，默认20"},
    }
    routing_hint = "用户问违规/处罚/风险/公告/监管 → search_news"
    trigger_keywords = [
        "违规", "处罚", "公告", "监管", "监管措施", "风险提示",
        "舆情", "事件", "利好", "利空", "被查", "被罚",
    ]
    max_retries = 2
    timeout_sec = 8

    def execute(self, params: Dict[str, Any], data_loader: Any = None) -> Dict[str, Any]:
        query = params.get("query", "")
        stock_code = params.get("stock_code", "")
        max_results = int(params.get("max_results", 20))

        loader = data_loader or DataLoader()

        try:
            df = loader.load_announcements()

            # 按 stock_code 过滤
            matched = df
            if stock_code:
                if "stock_code" in df.columns:
                    matched = df[df["stock_code"] == stock_code]
                elif "s_info_windcode" in df.columns:
                    matched = df[df["s_info_windcode"].str.contains(stock_code, na=False)]

            # 关键词标题匹配
            if query and "n_info_title" in matched.columns:
                kw_matched = matched[matched["n_info_title"].str.contains(query, na=False)]
                if len(kw_matched) > 0:
                    matched = kw_matched

            if len(matched) > 0:
                results = []
                for _, row in matched.head(max_results).iterrows():
                    results.append({
                        "title": str(row.get("n_info_title", "")),
                        "date": str(row.get("ann_dt", "")),
                        "stock_code": str(row.get("stock_code", row.get("s_info_windcode", ""))),
                        "source": "公司公告",
                    })

                rendered_lines = [
                    f"## 新闻/公告检索",
                    f"查询: {query}",
                    f"股票: {stock_code or 'ALL'}",
                    f"结果: {len(results)} 条",
                    "",
                ]
                for i, r in enumerate(results):
                    rendered_lines.append(f"{i+1}. **{r['title'][:80]}** | {r['date']}")

                return {
                    "query": query,
                    "stock_code": stock_code or "ALL",
                    "total": len(results),
                    "articles": results,
                    "rendered": "\n".join(rendered_lines),
                    "source": "dataset",
                }
            else:
                return {
                    "query": query,
                    "stock_code": stock_code or "ALL",
                    "total": 0,
                    "articles": [],
                    "rendered": f"## 新闻/公告检索\n未找到相关公告（查询: {query}, 股票: {stock_code or 'ALL'}）。\n建议尝试更宽泛的关键词或去掉股票限制。",
                    "source": "dataset",
                }
        except Exception as e:
            return {
                "query": query,
                "stock_code": stock_code or "ALL",
                "total": 0,
                "articles": [],
                "rendered": f"新闻检索出错: {e}",
                "source": "error",
            }
