"""联网搜索工具 — BaseTool 插件

使用 DuckDuckGo 免费搜索，无需 API Key。
触发条件：
  1. 数据库无对应时间数据 → 自动联网
  2. 概念解释/方法指导类问题 → 自动联网
结果标注 "web" 来源，与数据库数据明确区分。
"""

import json
import re
import urllib.request
import urllib.parse
import urllib.error
from typing import Any, Dict, List, Optional

from .base import BaseTool, register_tool_class




def _search_duckduckgo_api(query: str) -> List[Dict[str, str]]:
    """DuckDuckGo Instant Answer API (JSON, 更可靠)."""
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
        "q": query, "format": "json", "no_html": "1", "skip_disambig": "1",
    })
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = []
        # Abstract
        abstract = data.get("Abstract", "") or data.get("AbstractText", "")
        if abstract:
            results.append({
                "title": data.get("Heading", query),
                "snippet": abstract[:300],
                "url": data.get("AbstractURL", data.get("AbstractSource", "")),
            })
        # Related topics
        for topic in data.get("RelatedTopics", [])[:4]:
            if isinstance(topic, dict):
                results.append({
                    "title": topic.get("Text", "")[:100] or topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", "")[:300],
                    "url": topic.get("FirstURL", ""),
                })
        return results
    except Exception as e:
        print(f"[WebSearch] DDG API failed: {e}")
        return []


def _search_bing(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Bing HTML 搜索（国内可访问）。"""
    url = "https://www.bing.com/search?" + urllib.parse.urlencode({"q": query}, quote_via=urllib.parse.quote)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9",
    })
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[WebSearch] Bing failed: {e}")
        return []

    results = []
    # 提取 h2 中 a 标签的链接和标题
    links = re.findall(
        r'<h2[^>]*>.*?<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
        html, re.DOTALL
    )
    # 提取摘要
    snippets = re.findall(
        r'<p[^>]*>(.*?)</p>', html, re.DOTALL
    )

    for i, (href, title_raw) in enumerate(links[:max_results]):
        title = re.sub(r'<[^>]+>', '', title_raw).strip()
        snippet = ""
        if i < len(snippets):
            snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()
        if title and len(title) > 3:
            results.append({
                "title": title[:120],
                "snippet": snippet[:300],
                "url": href,
            })
    return results


def _search_web(query: str, max_results: int = 5, finance_boost: bool = True) -> Dict[str, Any]:
    """
    执行联网搜索。Bing 优先（国内可用），DuckDuckGo API 备选。
    """
    if finance_boost:
        enhanced = f"{query} 金融 财经"
    else:
        enhanced = query

    # 第一通道: Bing（国内可直接访问）
    results = _search_bing(enhanced, max_results)
    search_engine = "Bing"

    # 第二通道: DuckDuckGo API（海外网络备选）
    if not results:
        results = _search_duckduckgo_api(enhanced)
        search_engine = "DuckDuckGo API"

    # 渲染 Markdown
    rendered_lines = [
        "## 🌐 联网搜索结果\n",
        f"搜索关键词: {query}",
        f"搜索引擎: {search_engine}",
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
