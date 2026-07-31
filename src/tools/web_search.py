"""联网搜索工具 — BaseTool 插件

使用 DuckDuckGo 免费搜索，无需 API Key。
触发条件：
  1. 数据库无对应时间数据 → 自动联网
  2. 概念解释/方法指导类问题 → 自动联网
结果标注 "web" 来源，与数据库数据明确区分。
"""

import re
import urllib.request
import urllib.parse
import urllib.error
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

from .base import BaseTool, register_tool_class


class _DuckDuckGoParser(HTMLParser):
    """解析 DuckDuckGo HTML 搜索结果"""

    def __init__(self):
        super().__init__()
        self.results: List[Dict[str, str]] = []
        self._current: Dict[str, str] = {}
        self._in_result = False
        self._in_link = False
        self._in_snippet = False
        self._text_buf = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "")

        if tag == "div" and "result" in cls:
            self._in_result = True
            self._current = {}

        if self._in_result:
            if tag == "a" and "result__a" in cls:
                self._in_link = True
                self._current["url"] = attrs_dict.get("href", "")
            elif tag == "a" and "result__snippet" in cls:
                self._in_snippet = True

    def handle_endtag(self, tag):
        if self._in_result and tag == "div":
            if self._current.get("title"):
                self.results.append(dict(self._current))
            self._in_result = False
            self._current = {}
        if tag == "a":
            self._in_link = False
            self._in_snippet = False

    def handle_data(self, data):
        if self._in_result and self._in_link:
            self._current["title"] = (self._current.get("title", "") + data).strip()
        elif self._in_result and self._in_snippet:
            self._current["snippet"] = (self._current.get("snippet", "") + data).strip()


def _search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """DuckDuckGo HTML 搜索，免费无限制。"""
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        parser = _DuckDuckGoParser()
        parser.feed(html)
        return parser.results[:max_results]
    except Exception as e:
        print(f"[WebSearch] DuckDuckGo search failed: {e}")
        return []


def _search_web(query: str, max_results: int = 5, finance_boost: bool = True) -> Dict[str, Any]:
    """
    执行联网搜索。

    Args:
        query: 搜索关键词
        max_results: 最大返回数
        finance_boost: 是否添加金融搜索增强词

    Returns:
        结构化搜索结果
    """
    # 金融搜索增强
    if finance_boost:
        enhanced = f"{query} 金融 财经"
    else:
        enhanced = query

    results = _search_duckduckgo(enhanced, max_results)

    # 渲染 Markdown
    rendered_lines = [
        "## 🌐 联网搜索结果\n",
        f"搜索关键词: {query}",
        f"结果数: {len(results)}\n",
    ]
    for i, r in enumerate(results, 1):
        title = r.get("title", "无标题")[:100]
        snippet = r.get("snippet", "")[:200]
        url = r.get("url", "")
        rendered_lines.append(f"{i}. **{title}**\n")
        if snippet:
            rendered_lines.append(f"   {snippet}\n")
        if url:
            rendered_lines.append(f"   🔗 {url}\n")

    rendered_lines.append("\n*以上信息来自互联网公开来源，请核实准确性*")

    return {
        "query": query,
        "total": len(results),
        "results": results,
        "rendered": "\n".join(rendered_lines),
        "source": "web",
    }


# =========================================================================
# WebSearchTool
# =========================================================================


@register_tool_class
class WebSearchTool(BaseTool):
    """
    联网搜索工具 — 当数据库无数据或用户问概念/方法时使用。

    触发场景:
      1. 数据库无对应时间数据（如 2024 年报未入库）
      2. 概念解释/方法指导类问题（如"如何开户""什么是牛市"）
      3. 实时行情（如"今天股价"）
      4. 最新新闻事件
    """

    name = "web_search"
    description = (
        "联网搜索：当数据库没有所需数据时，搜索互联网获取最新信息。"
        "适用场景：概念解释、方法指导、实时行情、最新新闻、数据库外的时间段数据。"
    )
    required_params = ["query"]
    optional_params = ["max_results"]
    intent_match = ["NEWS_EVENT", "MARKET_DATA", "CHITCHAT", "FINANCIAL_ANALYSIS", "EQUITY_PENETRATION"]
    param_schema = {
        "query": {"description": "搜索关键词（自动增强为金融搜索）"},
        "max_results": {"description": "最大返回条数，默认5"},
    }
    routing_hint = (
        "数据库无数据时 → web_search（联网搜索）；"
        "概念解释/方法指导类问题 → web_search；"
        "实时行情 → web_search"
    )
    trigger_keywords = [
        "如何", "怎么", "教程", "开户", "什么是", "概念", "定义",
        "最新消息", "最新新闻", "新闻", "今天", "近日",
    ]
    max_retries = 1
    timeout_sec = 10

    def execute(self, params: Dict[str, Any], data_loader: Any = None) -> Dict[str, Any]:
        query = params.get("query", "")
        max_results = int(params.get("max_results", 5))
        # 概念/方法类不用金融增强，让搜索结果更通用
        concept_keywords = ["如何", "怎么", "教程", "什么是", "概念", "定义", "方法", "步骤"]
        is_concept = any(kw in query for kw in concept_keywords)
        return _search_web(query, max_results=max_results, finance_boost=not is_concept)
