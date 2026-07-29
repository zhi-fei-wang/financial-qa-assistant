"""新闻/舆情检索工具"""

from typing import Any, Dict, List, Optional

from ..utils.data_loader import DataLoader


class NewsSearchTool:
    """
    新闻舆情检索工具。
    从数据集 3/ (公司公告) 和 5/ (研报) 中检索相关内容。
    """

    def __init__(self, loader: Optional[DataLoader] = None):
        self.loader = loader or DataLoader()
        self._announcements_df = None
        self._reports_df = None
        self._report_column_name: Optional[str] = None

    def _load_data(self):
        """延迟加载数据"""
        if self._announcements_df is None:
            try:
                self._announcements_df = self.loader.load_announcements()
            except Exception as e:
                print(f"[NewsSearch] Announcements load error: {e}")

        if self._reports_df is None:
            try:
                self._reports_df = self.loader.load_research_reports()
            except Exception as e:
                print(f"[NewsSearch] Reports load error: {e}")

    def search(
        self,
        query: str,
        stock_code: Optional[str] = None,
        date_range: Optional[str] = None,
        max_results: int = 20,
    ) -> Dict[str, Any]:
        """
        搜索与标的相关的新闻/公告/研报。

        Args:
            query: 搜索关键词
            stock_code: 关联股票代码
            date_range: 日期范围 (未实现，预留)
            max_results: 最大返回条数

        Returns:
            {"query": ..., "articles": [...], "total": ...}
        """
        self._load_data()
        articles = []

        # 1. 从公告中搜索
        if self._announcements_df is not None:
            df = self._announcements_df
            if stock_code and "stock_code" in df.columns:
                df = df[df["stock_code"] == self._normalize(stock_code)]

            if "n_info_title" in df.columns:
                query_terms = query.lower().split()
                for _, row in df.iterrows():
                    title = str(row.get("n_info_title", ""))
                    # 关键词匹配
                    if any(term.lower() in title.lower() for term in query_terms) or (stock_code and len(df) <= 50):
                        articles.append({
                            "title": title,
                            "date": str(row.get("ann_dt", "")),
                            "stock_code": str(row.get("stock_code", stock_code or "")),
                            "source": "公司公告",
                        })
                        if len(articles) >= max_results:
                            break

        # 2. 从研报中搜索
        if self._reports_df is not None and len(articles) < max_results:
            df = self._reports_df

            if "title" in df.columns and "abstract" in df.columns:
                query_terms = query.lower().split()
                for _, row in df.iterrows():
                    title = str(row.get("title", ""))
                    abstract = str(row.get("abstract", ""))
                    text = title + " " + abstract
                    if any(term.lower() in text.lower() for term in query_terms):
                        articles.append({
                            "title": title,
                            "date": str(row.get("write_date", row.get("publish_date", ""))),
                            "org": str(row.get("org_name", "")),
                            "abstract": abstract[:200],
                            "source": "券商研报",
                        })
                        if len(articles) >= max_results:
                            break

        return {
            "query": query,
            "stock_code": stock_code,
            "total": len(articles),
            "articles": articles[:max_results],
            "source": "dataset",
        }

    @staticmethod
    def _normalize(code: str) -> str:
        import pandas as pd
        if pd.isna(code):
            return ""
        code = str(code).strip().upper()
        for suffix in [".SH", ".SZ", ".BJ"]:
            if code.endswith(suffix):
                code = code[:-len(suffix)]
        return code.zfill(6)
