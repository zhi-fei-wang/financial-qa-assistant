"""
财务数据提取器
从三大报表中提取和标准化关键财务科目，为规则引擎提供统一接口。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..utils.data_loader import DataLoader


@dataclass
class FinancialSnapshot:
    """单期财务快照：从三大报表中提取的关键科目"""
    stock_code: str
    report_period: str

    # 资产负债表
    monetary_cap: float = 0.0          # 货币资金
    inventories: float = 0.0           # 存货
    acct_rcv: float = 0.0              # 应收账款
    notes_rcv: float = 0.0             # 应收票据
    prepay: float = 0.0                # 预付款项
    acct_payable: float = 0.0          # 应付账款
    tot_cur_assets: float = 0.0        # 流动资产合计
    tot_assets: float = 0.0            # 资产总计
    tot_cur_liab: float = 0.0          # 流动负债合计
    tot_liab: float = 0.0              # 负债合计
    goodwill: float = 0.0              # 商誉
    fix_assets: float = 0.0            # 固定资产
    intang_assets: float = 0.0         # 无形资产
    tot_equity: float = 0.0            # 股东权益合计

    # 利润表
    oper_rev: float = 0.0              # 营业收入
    tot_oper_rev: float = 0.0          # 营业总收入
    oper_cost: float = 0.0             # 营业成本
    tot_oper_cost: float = 0.0         # 营业总成本
    oper_profit: float = 0.0           # 营业利润
    tot_profit: float = 0.0            # 利润总额
    net_profit: float = 0.0            # 净利润
    inc_tax: float = 0.0               # 所得税
    less_oper_exp: float = 0.0         # 销售费用
    less_admin_exp: float = 0.0        # 管理费用
    less_fin_exp: float = 0.0          # 财务费用
    oper_exp: float = 0.0              # 营业支出

    # 现金流量表
    net_cash_flows_oper: float = 0.0   # 经营性净现金流
    net_cash_flows_inv: float = 0.0    # 投资性净现金流
    net_cash_flows_fin: float = 0.0    # 筹资性净现金流

    # 元数据
    ann_dt: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)


class FinancialDataExtractor:
    """
    从数据集中提取和标准化关键财务数据。
    支持单期快照和多期对比。
    """

    # 列名映射：标准化字段 → 三个表中的实际列名
    COLUMN_MAP = {
        # 资产负债表
        "monetary_cap":      ("balance_sheet", "monetary_cap"),
        "inventories":       ("balance_sheet", "inventories"),
        "acct_rcv":          ("balance_sheet", "acct_rcv"),
        "notes_rcv":         ("balance_sheet", "notes_rcv"),
        "prepay":            ("balance_sheet", "prepay"),
        "acct_payable":      ("balance_sheet", "acct_payable"),
        "tot_cur_assets":    ("balance_sheet", "tot_cur_assets"),
        "tot_assets":        ("balance_sheet", "tot_assets"),
        "tot_cur_liab":      ("balance_sheet", "tot_cur_liab"),
        "tot_liab":          ("balance_sheet", "tot_liab"),
        "goodwill":          ("balance_sheet", "goodwill"),
        "fix_assets":        ("balance_sheet", "fix_assets"),
        "intang_assets":     ("balance_sheet", "intang_assets"),
        "tot_equity":        ("balance_sheet", "tot_shrhldr_eqy_incl_min_int"),

        # 利润表
        "oper_rev":          ("income", "oper_rev"),
        "tot_oper_rev":      ("income", "tot_oper_rev"),
        "oper_cost":         ("income", "less_oper_cost"),
        "tot_oper_cost":     ("income", "tot_oper_cost"),
        "oper_profit":       ("income", "oper_profit"),
        "tot_profit":        ("income", "tot_profit"),
        "net_profit":        ("income", "net_profit_excl_min_int_inc"),
        "inc_tax":           ("income", "inc_tax"),
        "less_oper_exp":     ("income", "less_oper_exp"),
        "less_admin_exp":    ("income", "less_admin_exp"),
        "less_fin_exp":      ("income", "less_fin_exp"),
        "oper_exp":          ("income", "oper_exp"),

        # 现金流量表
        "net_cash_flows_oper": ("cashflow", "net_cash_flows_oper_act"),
        "net_cash_flows_inv":  ("cashflow", "net_cash_flows_inv_act"),
        "net_cash_flows_fin":  ("cashflow", "net_cash_flows_fin_act"),
    }

    def __init__(self, loader: Optional[DataLoader] = None):
        self.loader = loader or DataLoader()
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_snapshot(self, stock_code: str, report_period: Optional[str] = None) -> FinancialSnapshot:
        """
        获取某只股票的单期财务快照。

        Args:
            stock_code: 6位股票代码
            report_period: 报告期（None=最新一期）

        Returns:
            FinancialSnapshot
        """
        snapshot = FinancialSnapshot(
            stock_code=self._norm(stock_code),
            report_period=report_period or "latest",
        )

        for field_name, (stmt_type, col_name) in self.COLUMN_MAP.items():
            df = self._load_statement(stmt_type)
            if df is None:
                continue

            # 筛选股票
            if "stock_code" in df.columns:
                matched = df[df["stock_code"] == self._norm(stock_code)]
            elif "s_info_windcode" in df.columns:
                matched = df[df["s_info_windcode"].apply(self._norm) == self._norm(stock_code)]
            else:
                continue

            if len(matched) == 0:
                continue

            # 筛选报告期
            if report_period:
                if "report_period" in matched.columns:
                    rp_str = str(report_period)
                    matched = matched[matched["report_period"].astype(str).str.contains(
                        rp_str[:4] if len(rp_str) >= 4 else rp_str, na=False
                    )]

            if len(matched) == 0:
                continue

            # 取最新一条
            if "report_period" in matched.columns:
                matched = matched.sort_values("report_period", ascending=False)

            row = matched.iloc[0]
            val = row.get(col_name, 0)

            if pd.notna(val):
                setattr(snapshot, field_name, float(val))

            # 记录元数据
            if not snapshot.report_period or snapshot.report_period == "latest":
                rp = row.get("report_period", "")
                snapshot.report_period = str(rp)[:10] if pd.notna(rp) else ""
            if not snapshot.ann_dt:
                ad = row.get("ann_dt", "")
                snapshot.ann_dt = str(ad)[:10] if pd.notna(ad) else ""

            # 保存原始数据
            snapshot.raw_data[field_name] = {
                "value": float(val) if pd.notna(val) else 0.0,
                "source": stmt_type,
                "column": col_name,
            }

        return snapshot

    def get_multi_period(
        self, stock_code: str, periods: int = 5
    ) -> List[FinancialSnapshot]:
        """
        获取多期财务快照（用于趋势分析）。

        Args:
            stock_code: 6位股票代码
            periods: 返回最近 N 期

        Returns:
            按时间倒序排列的快照列表
        """
        snapshots = []

        # 从资产负债表中提取所有报告期
        df = self._load_statement("balance_sheet")
        if df is None:
            return snapshots

        if "stock_code" in df.columns:
            matched = df[df["stock_code"] == self._norm(stock_code)]
        elif "s_info_windcode" in df.columns:
            matched = df[df["s_info_windcode"].apply(self._norm) == self._norm(stock_code)]
        else:
            return snapshots

        if "report_period" in matched.columns:
            matched = matched.sort_values("report_period", ascending=False)

        # 去重报告期
        seen_periods = set()
        for _, row in matched.iterrows():
            rp = str(row.get("report_period", ""))
            if rp and rp not in seen_periods:
                seen_periods.add(rp)
                snapshot = self.get_snapshot(stock_code, rp)
                snapshots.append(snapshot)
                if len(snapshots) >= periods:
                    break

        return snapshots

    def _load_statement(self, stmt_type: str) -> Optional[pd.DataFrame]:
        """带缓存的报表加载"""
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
            print(f"[DataExtractor] Load error for {stmt_type}: {e}")
            return None

    @staticmethod
    def _norm(code: str) -> str:
        if pd.isna(code):
            return ""
        code = str(code).strip().upper()
        for suffix in [".SH", ".SZ", ".BJ"]:
            if code.endswith(suffix):
                code = code[:-len(suffix)]
        return code.zfill(6)
