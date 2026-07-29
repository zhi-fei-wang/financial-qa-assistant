"""财务报表查询工具"""

from typing import Any, Dict, List, Optional

import pandas as pd

from ..utils.data_loader import DataLoader


class FinancialDB:
    """
    财务报表数据库查询工具。
    从数据集 4/ 中查询资产负债表、利润表、现金流量表。
    """

    def __init__(self, loader: Optional[DataLoader] = None):
        self.loader = loader or DataLoader()
        self._cache: Dict[str, pd.DataFrame] = {}

    def query(
        self,
        stock_code: str,
        report_period: Optional[str] = None,
        statement_type: str = "balance_sheet",
        indicators: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        查询财务报表数据。

        Args:
            stock_code: 6位股票代码
            report_period: 报告期，如 2024Q1、2023
            statement_type: balance_sheet / income / cashflow
            indicators: 关注的科目列表（None=全部）

        Returns:
            {"stock_code": ..., "report_period": ..., "data": {...}}
        """
        df = self._load_statement(statement_type)
        if df is None:
            return {"error": f"无法加载 {statement_type} 数据"}

        # 按股票代码筛选
        normalized_code = self._normalize(stock_code)
        if "stock_code" in df.columns:
            matched = df[df["stock_code"] == normalized_code]
        elif "s_info_windcode" in df.columns:
            matched = df[df["s_info_windcode"].apply(self._normalize) == normalized_code]
        else:
            return {"error": "无法找到股票代码列"}

        if len(matched) == 0:
            return {
                "stock_code": normalized_code,
                "report_period": report_period,
                "data": {},
                "note": f"未找到股票 {normalized_code} 的 {statement_type} 数据",
            }

        # 按报告期筛选
        if report_period:
            if "report_period" in matched.columns:
                period_str = str(report_period).replace("Q", "-")
                matched = matched[matched["report_period"].astype(str).str.contains(
                    report_period[:4] if len(report_period) >= 4 else report_period,
                    na=False
                )]

        # 取最新一条
        if "report_period" in matched.columns:
            matched = matched.sort_values("report_period", ascending=False)

        row = matched.iloc[0].to_dict()

        # 筛选指标
        if indicators:
            filtered = {}
            for ind in indicators:
                if ind in row:
                    filtered[ind] = row[ind]
            data = filtered
        else:
            # 返回主要财务科目（排除元数据列）
            meta_cols = {"object_id", "s_info_windcode", "wind_code", "ann_dt",
                        "report_period", "statement_type", "crncy_code",
                        "stock_code", "comp_type_code", "actual_ann_dt"}
            data = {k: v for k, v in row.items() if k not in meta_cols and not pd.isna(v)}

        return {
            "stock_code": normalized_code,
            "report_period": str(row.get("report_period", report_period or "N/A")),
            "statement_type": statement_type,
            "data": data,
            "source": "dataset",
        }

    def get_financial_indicators(
        self, stock_code: str, indicators: List[str]
    ) -> Dict[str, Any]:
        """
        跨报表查询指定指标（自动路由到正确的报表类型）。
        """
        # 根据指标名判断应该查哪个表
        bs_indicators = {"monetary_cap", "inventories", "tot_assets", "tot_liab",
                        "inventories", "acct_rcv", "tot_cur_assets"}
        income_indicators = {"tot_oper_rev", "oper_rev", "net_profit", "tot_profit"}
        cf_indicators = {"cash_recp_sg_and_rs", "net_cash_flows_oper_act"}

        results = {}
        for ind in indicators:
            if ind in bs_indicators:
                stmt = "balance_sheet"
            elif ind in income_indicators:
                stmt = "income"
            elif ind in cf_indicators:
                stmt = "cashflow"
            else:
                # 不确定，全部查一遍
                stmt = "balance_sheet"

            result = self.query(stock_code, statement_type=stmt, indicators=[ind])
            if "data" in result:
                results.update(result["data"])

        return {"stock_code": stock_code, "indicators": results, "source": "dataset"}

    def _load_statement(self, stmt_type: str) -> Optional[pd.DataFrame]:
        """加载报表（带缓存）"""
        if stmt_type in self._cache:
            return self._cache[stmt_type]

        try:
            if stmt_type == "balance_sheet":
                df = self.loader.load_balance_sheet()
            elif stmt_type == "income":
                df = self.loader.load_income()
            elif stmt_type == "cashflow":
                df = self.loader.load_cashflow()
            else:
                return None

            self._cache[stmt_type] = df
            return df
        except Exception as e:
            print(f"[FinancialDB] Load error: {e}")
            return None

    @staticmethod
    def _normalize(code: str) -> str:
        """统一股票代码格式"""
        if pd.isna(code):
            return ""
        code = str(code).strip().upper()
        for suffix in [".SH", ".SZ", ".BJ"]:
            if code.endswith(suffix):
                code = code[:-len(suffix)]
        return code.zfill(6)
