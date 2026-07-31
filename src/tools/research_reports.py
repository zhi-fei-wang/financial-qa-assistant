"""
券商研报检索工具 (P0: 赛题缺口 — 研报数据接入)

为 ~55K 篇券商研报提供 BM25 关键词检索 + 股票代码精确过滤，
返回研报标题、摘要、评级、券商、行业分类等结构化信息。

作为 search_reports 工具注册到 ToolExecutor。
"""

import re
from typing import Any, Dict, List, Optional

from ..memory.signal_fusion import BM25Scorer
from ..utils.data_loader import DataLoader


class ResearchReportSearch:
    """券商研报 BM25 检索器"""

    def __init__(self, loader: Optional[DataLoader] = None):
        self.loader = loader or DataLoader()
        self._df = None
        self._bm25 = BM25Scorer()
        self._indexed = False
        self._stock_index: Dict[str, List[int]] = {}  # stock_code → row indices

    def _ensure_loaded(self):
        """延迟加载 + 构建 BM25 索引"""
        if self._df is not None:
            return
        try:
            df = self.loader.load_research_reports()
        except Exception as e:
            print(f"[ResearchReport] Load error: {e}")
            return

        # 过滤无效行
        df = df.dropna(subset=["title", "abstract"], how="all")
        self._df = df.reset_index(drop=True)
        print(f"[ResearchReport] Loaded {len(self._df)} reports")

        # 构建 BM25 索引
        docs = {}
        for i, row in self._df.iterrows():
            title = str(row.get("title", ""))
            abstract = str(row.get("abstract", ""))
            sec_name = str(row.get("sec_name", ""))
            org = str(row.get("org_name", ""))
            docs[f"r_{i}"] = f"{title} {abstract} {sec_name} {org}"

            # 构建股票代码索引
            code = str(row.get("sec_code", "")).strip()
            if code:
                code_norm = code.zfill(6)
                if code_norm not in self._stock_index:
                    self._stock_index[code_norm] = []
                self._stock_index[code_norm].append(i)

        self._bm25.index(docs)
        self._indexed = True
        print(f"[ResearchReport] BM25 indexed {len(docs)} documents, {len(self._stock_index)} stocks")

    def search(
        self,
        query: str = "",
        stock_code: str = "",
        industry: str = "",
        max_results: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        研报检索主入口。

        Args:
            query: 关键词查询（标题+摘要）
            stock_code: 股票代码过滤
            industry: 行业过滤（L1/L2/L3）
            max_results: 最大返回数
        """
        self._ensure_loaded()
        if self._df is None:
            return []

        # 先按股票代码过滤
        candidate_indices = set()
        if stock_code:
            code_norm = self._normalize_code(stock_code)
            candidate_indices.update(self._stock_index.get(code_norm, []))
            # 也按名称模糊匹配
            if not candidate_indices:
                for i, row in self._df.iterrows():
                    code = str(row.get("sec_code", ""))
                    name = str(row.get("sec_name", ""))
                    if code_norm in code or stock_code in name:
                        candidate_indices.add(i)

        # 按行业过滤
        if industry and not candidate_indices:
            for i, row in self._df.iterrows():
                ind_text = (
                    str(row.get("industry_l1", "")) + " " +
                    str(row.get("industry_l2", "")) + " " +
                    str(row.get("industry_l3", ""))
                )
                if industry.lower() in ind_text.lower():
                    candidate_indices.add(i)

        # BM25 检索
        if query:
            candidates = [f"r_{i}" for i in candidate_indices] if candidate_indices else None
            ranked = self._bm25.top_k(query, k=max_results, doc_ids=candidates)
            if not candidate_indices:
                # 全量检索
                scored_indices = {int(k.split("_")[1]): s for k, s in ranked}
            else:
                scored_indices = {int(k.split("_")[1]): s for k, s in ranked}
        elif candidate_indices:
            # 只有股票代码过滤，无 query → 按日期排序
            scored_indices = {i: 1.0 for i in list(candidate_indices)[:max_results]}
        else:
            return []

        # 组装结果
        results = []
        for idx, score in sorted(scored_indices.items(), key=lambda x: -x[1])[:max_results]:
            row = self._df.iloc[idx]
            results.append({
                "title": str(row.get("title", "")),
                "date": str(row.get("publish_date", row.get("write_date", ""))),
                "org": str(row.get("org_name", "")),
                "author": str(row.get("author", "")),
                "rating": str(row.get("rating_org", "")),
                "rating_change": str(row.get("rating_change", "")),
                "industry": f"{row.get('industry_l1', '')}/{row.get('industry_l2', '')}",
                "abstract": str(row.get("abstract", ""))[:300],
                "sec_name": str(row.get("sec_name", "")),
                "sec_code": str(row.get("sec_code", "")),
                "source": "券商研报",
                "score": round(score, 3),
            })

        return results

    def search_stock(
        self, stock_code: str, max_results: int = 10
    ) -> Dict[str, Any]:
        """按股票代码查询研报（不依赖关键词）"""
        self._ensure_loaded()
        code_norm = self._normalize_code(stock_code)
        indices = self._stock_index.get(code_norm, [])
        if not indices:
            return {"stock_code": stock_code, "total": 0, "reports": []}

        # 按日期排序取最新
        rows = []
        for i in indices[:50]:
            row = self._df.iloc[i]
            rows.append((i, str(row.get("publish_date", ""))))

        rows.sort(key=lambda x: x[1], reverse=True)

        reports = []
        for i, _ in rows[:max_results]:
            row = self._df.iloc[i]
            reports.append({
                "title": str(row.get("title", "")),
                "date": str(row.get("publish_date", "")),
                "org": str(row.get("org_name", "")),
                "author": str(row.get("author", "")),
                "rating": f"{row.get('rating_org', '')}({row.get('rating_change', '')})",
                "abstract": str(row.get("abstract", ""))[:300],
                "source": "券商研报",
            })

        # 统计评级分布
        ratings = {}
        for i in indices:
            r = str(self._df.iloc[i].get("rating_org", ""))
            ratings[r] = ratings.get(r, 0) + 1

        return {
            "stock_code": stock_code,
            "total": len(indices),
            "rating_distribution": ratings,
            "reports": reports,
        }

    @staticmethod
    def _normalize_code(code: str) -> str:
        code = str(code).strip().upper()
        for suffix in [".SH", ".SZ", ".BJ", ".N", ".HK"]:
            if code.endswith(suffix):
                code = code[:-len(suffix)]
        return code.zfill(6)


# ---------- 延迟加载单例 ----------

_report_search: Optional[ResearchReportSearch] = None


def get_report_search() -> ResearchReportSearch:
    global _report_search
    if _report_search is None:
        _report_search = ResearchReportSearch()
    return _report_search


# ---------- 注册为工具 ----------

class SearchReportsToolImpl:
    """研报检索 Skill 实现 (P0 新增)"""

    name = "search_reports"
    description = (
        "券商研报检索: 搜索约5.5万篇券商研报，支持关键词+股票代码+行业过滤。"
        "返回研报标题/摘要/评级/券商/行业分类。"
    )
    required_params = ["query"]
    optional_params = ["stock_code", "industry", "max_results"]

    @staticmethod
    def execute(params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        stock_code = params.get("stock_code", "")
        industry = params.get("industry", "")
        max_results = int(params.get("max_results", 15))

        searcher = get_report_search()
        reports = searcher.search(query, stock_code, industry, max_results)

        # 渲染
        rendered_parts = [f"## 券商研报检索: {query or stock_code}\n"]
        rendered_parts.append(f"数据来源: 真实研报数据集 (5/rr_main_*.csv) | 共 {len(reports)} 篇\n")

        for i, r in enumerate(reports, 1):
            rendered_parts.append(
                f"{i}. **{r['title']}**\n"
                f"   [{r['date']}] {r['org']} | {r['rating']} | {r['industry']}\n"
                f"   {r['abstract'][:200]}...\n"
            )

        rendered = "\n".join(rendered_parts)

        return {
            "query": query,
            "stock_code": stock_code,
            "total": len(reports),
            "reports": reports,
            "rendered": rendered,
            "source": "dataset",
        }


class SearchReportsByStockToolImpl:
    """按股票查研报实现 (P0 新增)"""

    name = "search_reports_by_stock"
    description = "查询某只股票的所有研报，按日期排序，附带评级分布。"
    required_params = ["stock_code"]
    optional_params = ["max_results"]

    @staticmethod
    def execute(params: Dict[str, Any]) -> Dict[str, Any]:
        stock_code = params.get("stock_code", "")
        max_results = int(params.get("max_results", 10))

        searcher = get_report_search()
        result = searcher.search_stock(stock_code, max_results)

        # 渲染
        rendered_parts = [f"## {stock_code} 研报概况\n"]
        rendered_parts.append(f"数据来源: 真实研报数据集 | 共 {result['total']} 篇\n")

        if result.get("rating_distribution"):
            rendered_parts.append(f"评级分布: {result['rating_distribution']}\n")

        for i, r in enumerate(result["reports"], 1):
            rendered_parts.append(f"{i}. [{r['date']}] {r['org']} | {r['rating']}\n   {r['title'][:100]}\n")

        rendered = "\n".join(rendered_parts)

        # 启发 2: ResultEnvelope
        env = {
            "conclusion": f"共 {result['total']} 篇研报覆盖 {stock_code}",
            "evidence": [{"claim": f"{r['date']} {r['org']}: {r['title'][:80]}", "source": "dataset", "data": {}} for r in result["reports"][:5]],
            "confidence": 1.0,
            "limitations": ["研报摘要约300字，完整报告不可获取"],
            "metadata": {"skill_name": "search_reports_by_stock"},
        }

        return {
            **result,
            "rendered": rendered,
            "source": "dataset",
            "envelope": env,
        }


# =========================================================================
# BaseTool 包装器
# =========================================================================

from .base import BaseTool, register_tool_class


@register_tool_class
class SearchReportsTool(BaseTool):
    """券商研报检索。"""
    name = "search_reports"
    description = (
        "券商研报检索: 搜索约5.5万篇券商研报，支持关键词+股票代码+行业过滤。"
        "返回研报标题/摘要/评级/券商/行业分类。"
    )
    required_params = ["query"]
    optional_params = ["stock_code", "industry", "max_results"]
    intent_match = ["NEWS_EVENT", "FINANCIAL_ANALYSIS"]
    param_schema = {
        "query": {"description": "搜索关键词（标题+摘要）"},
        "stock_code": {"description": "股票代码过滤"},
        "industry": {"description": "行业过滤（如'电力设备'）"},
        "max_results": {"description": "最大返回数，默认15"},
    }
    routing_hint = "用户问研报/券商观点/行业分析 → search_reports"
    trigger_keywords = ["研报", "券商", "研究报告", "行业分析"]
    max_retries = 2
    timeout_sec = 10

    def execute(self, params, data_loader=None):
        return SearchReportsToolImpl.execute(params)


@register_tool_class
class SearchReportsByStockTool(BaseTool):
    """按股票查研报。"""
    name = "search_reports_by_stock"
    description = "查询某只股票的所有研报，按日期排序，附带评级分布。"
    required_params = ["stock_code"]
    optional_params = ["max_results"]
    intent_match = ["NEWS_EVENT", "FINANCIAL_ANALYSIS"]
    param_schema = {
        "stock_code": {"description": "股票代码"},
        "max_results": {"description": "最大返回数，默认10"},
    }
    routing_hint = "用户要某只股票的研报 → search_reports_by_stock"
    trigger_keywords = ["研报", "券商评级"]
    max_retries = 1
    timeout_sec = 8

    def execute(self, params, data_loader=None):
        return SearchReportsByStockToolImpl.execute(params)

TASK_SEARCH_SKILLS = [SearchReportsTool, SearchReportsByStockTool]
