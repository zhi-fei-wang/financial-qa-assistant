"""财务报表查询工具 — BaseTool 插件
从真实数据集查询资产负债表/利润表/现金流量表，输出多期全景视图。
"""
from typing import Any, Dict, Optional

import pandas as pd

from .base import BaseTool, register_tool_class

# 中文字段名映射
FIELD_CN = {
    "tot_oper_rev": "营业总收入", "oper_rev": "营业收入",
    "less_oper_cost": "营业成本(不含三费)", "tot_oper_cost": "营业总成本(含三费)",
    "oper_profit": "营业利润", "tot_profit": "利润总额",
    "net_profit_incl_min_int_inc": "净利润(含少数股东)",
    "net_profit_excl_min_int_inc": "净利润(不含少数股东)",
    "monetary_cap": "货币资金", "inventories": "存货",
    "tot_assets": "总资产", "tot_liab": "总负债",
    "tot_shrhdr_eqy_incl_min_int": "股东权益",
    "tot_cur_assets": "流动资产", "tot_cur_liab": "流动负债",
    "goodwill": "商誉", "tradable_fin_assets": "交易性金融资产",
    "notes_rcv": "应收票据",
    "cash_cash_equ_beg_period": "期初现金",
    "cash_cash_equ_end_period": "期末现金",
    "net_cash_flows_oper_act": "经营活动现金流净额",
    "net_cash_flows_inv_act": "投资活动现金流净额",
    "net_cash_flows_fnc_act": "筹资活动现金流净额",
}

CORE_FIELDS = {
    "income": ["tot_oper_rev", "oper_rev", "less_oper_cost", "tot_oper_cost",
               "oper_profit", "tot_profit",
               "net_profit_incl_min_int_inc", "net_profit_excl_min_int_inc"],
    "balance_sheet": ["monetary_cap", "tradable_fin_assets", "notes_rcv",
                      "inventories", "tot_assets", "tot_liab",
                      "tot_shrhdr_eqy_incl_min_int", "tot_cur_assets",
                      "tot_cur_liab", "goodwill"],
    "cashflow": ["cash_cash_equ_beg_period", "cash_cash_equ_end_period",
                 "net_cash_flows_oper_act", "net_cash_flows_inv_act",
                 "net_cash_flows_fnc_act"],
}

OVERVIEW_COLS = {
    "income": ["tot_oper_rev", "less_oper_cost", "net_profit_excl_min_int_inc"],
    "balance_sheet": ["tot_assets", "tot_liab", "inventories"],
    "cashflow": ["net_cash_flows_oper_act", "cash_cash_equ_end_period"],
}


def _fmt_amount(v) -> str:
    """格式化金额为可读形式"""
    if v is None:
        return "-"
    try:
        v = float(v)
        if abs(v) >= 1e8:
            return f"{v/1e8:.1f}亿"
        elif abs(v) >= 1e4:
            return f"{v/1e4:.0f}万"
        else:
            return f"{v:,.0f}"
    except (ValueError, TypeError):
        return "-"


def _build_overview_table(all_matched: pd.DataFrame, cols: list) -> str:
    """构建全景视图 Markdown 表格"""
    if not cols:
        return "(无数据)"
    header = "| 报告期 |" + "|".join(FIELD_CN.get(c, c) for c in cols) + "|"
    sep = "|" + "|".join("------" for _ in range(len(cols) + 1)) + "|"
    lines = [header, sep]
    for _, r in all_matched.iterrows():
        rp_v = str(r["report_period"])
        if rp_v.endswith("0331"):
            rp_v += "(Q1)"
        elif rp_v.endswith("0630"):
            rp_v += "(Q2)"
        elif rp_v.endswith("0930"):
            rp_v += "(Q3)"
        elif rp_v.endswith("1231"):
            rp_v += "(年报)"
        vals = [_fmt_amount(r.get(c)) for c in cols]
        lines.append("| " + rp_v + " |" + "|".join(vals) + "|")
    return "\n".join(lines)


def _build_multi_year_table(all_matched: pd.DataFrame, cols: list) -> str:
    """构建多期对比 Markdown 表格"""
    if not cols or len(all_matched) < 2:
        return ""
    lines = ["\n## 📊 全报告期对比表\n"]
    header = "| 报告期 |" + "|".join(FIELD_CN.get(c, c) for c in cols) + "| 类型 |"
    sep = "|" + "|".join("------" for _ in range(len(cols) + 2)) + "|"
    lines.append(header)
    lines.append(sep)
    for _, r in all_matched.iterrows():
        rp_v = str(r["report_period"])
        tag = ""
        if rp_v.endswith("1231"):
            tag = "年报"
        elif rp_v.endswith("0630"):
            tag = "中报"
        elif rp_v.endswith("0331"):
            tag = "Q1"
        elif rp_v.endswith("0930"):
            tag = "Q3"
        vals = [_fmt_amount(r.get(c)) for c in cols]
        lines.append("| " + rp_v + " |" + "|".join(vals) + f"| {tag} |")
    lines.append("")
    return "\n".join(lines)


@register_tool_class
class QueryFinancialTool(BaseTool):
    """查询上市公司财务报表数据。"""

    name = "query_financial_statement"
    description = "查询上市公司财务报表数据，包括资产负债表、利润表、现金流量表。"
    required_params = ["stock_code", "report_period"]
    optional_params = ["statement_type", "indicators"]
    intent_match = ["FINANCIAL_ANALYSIS"]
    sub_intent = "STATEMENT_QUERY"
    param_schema = {
        "stock_code": {"description": "6位股票代码"},
        "report_period": {"description": "报告期，如 2024Q1 或 2023"},
        "statement_type": {"description": "报表类型: balance_sheet/income/cashflow"},
        "indicators": {"description": "指定指标列表，逗号分隔"},
    }
    routing_hint = (
        "用户问财务/营收/利润/现金流/资产/负债 → query_financial_statement；"
        "务必传 statement_type（营收→income, 资产→balance_sheet, 现金流→cashflow）；"
        "对比类取年报(1231)对齐；年报÷365 季报÷90"
    )
    trigger_keywords = [
        "财报", "财务", "ROE", "ROA", "利润率", "存货周转",
        "现金流", "净利润", "营收", "资产负债", "毛利率", "净利率",
        "每股收益", "总资产", "货币资金", "应收账款", "存货", "商誉",
    ]
    max_retries = 1
    timeout_sec = 10

    def execute(self, params: Dict[str, Any], data_loader: Any = None) -> Dict[str, Any]:
        stock_code = params.get("stock_code", "")
        statement_type = params.get("statement_type", "balance_sheet")
        requested_period = params.get("report_period", "")

        if data_loader is None:
            return {
                "stock_code": stock_code,
                "statement_type": statement_type,
                "error": "data_loader 未提供",
                "source": "error",
            }

        try:
            # 加载数据
            load_map = {
                "balance_sheet": data_loader.load_balance_sheet,
                "income": data_loader.load_income,
                "cashflow": data_loader.load_cashflow,
            }
            load_fn = load_map.get(statement_type, data_loader.load_balance_sheet)
            df = load_fn()

            # 按股票代码筛选
            if "s_info_windcode" in df.columns:
                matched = df[df["s_info_windcode"].str.contains(str(stock_code), na=False)]
            elif "stock_code" in df.columns:
                matched = df[df["stock_code"].astype(str) == str(stock_code)]
            else:
                matched = pd.DataFrame()

            if len(matched) == 0:
                return {
                    "stock_code": stock_code,
                    "statement_type": statement_type,
                    "error": f"未找到股票代码 {stock_code} 的{statement_type}数据",
                    "source": "dataset",
                }

            all_matched = matched.sort_values("report_period", ascending=False)
            raw = None
            rp_str = ""

            # 指定报告期 → 精准匹配
            if requested_period and requested_period.strip():
                rp_clean = (requested_period
                            .replace("Q1", "0331").replace("Q2", "0630")
                            .replace("Q3", "0930").replace("Q4", "1231"))
                if len(rp_clean) == 4:
                    rp_clean = rp_clean + "1231"
                period_match = all_matched[all_matched["report_period"].astype(str).str.startswith(rp_clean[:6])]
                if len(period_match) > 0:
                    latest = period_match.iloc[0]
                    rev = latest.get("tot_oper_rev", 0) or 0
                    cost = latest.get("tot_oper_cost", 0) or 0
                    if rev > 0 and cost > 0 and rev < cost / 5:
                        latest = all_matched.iloc[0]
                    raw = latest.to_dict()
                else:
                    raw = all_matched.iloc[0].to_dict()
            else:
                raw = all_matched.iloc[0].to_dict()

            # 修正 report_period 显示
            if raw:
                rp = raw.get("report_period", "")
                try:
                    if hasattr(rp, 'strftime'):
                        rp_str = rp.strftime('%Y%m%d')
                    else:
                        rp_str = str(int(rp))
                except (ValueError, TypeError):
                    rp_str = str(rp).split('.')[0].replace('-', '')[:8]

            # 核心字段
            core = CORE_FIELDS.get(statement_type, list(raw.keys())[:20])
            summary = {k: raw.get(k) for k in core if k in raw and pd.notna(raw.get(k))}
            summary["report_period"] = rp_str
            summary["stock_code"] = stock_code

            # 全景视图
            overview_cols = OVERVIEW_COLS.get(statement_type, [])
            overview_cols = [c for c in overview_cols if c in all_matched.columns]
            overview_lines = ["", "## 该股票所有可用报告期一览", "",
                              _build_overview_table(all_matched, overview_cols),
                              "",
                              "以上数据可直接计算对比。选取两股票都有的相同报告期，年报优先。"]

            # 多期对比表
            year_cols = {
                "income": ["tot_oper_rev", "net_profit_excl_min_int_inc", "less_oper_cost", "oper_profit"],
                "balance_sheet": ["tot_assets", "tot_liab", "inventories", "monetary_cap", "tot_shrhdr_eqy_incl_min_int"],
                "cashflow": ["net_cash_flows_oper_act", "net_cash_flows_inv_act", "net_cash_flows_fnc_act", "cash_cash_equ_end_period"],
            }.get(statement_type, [])
            year_cols = [c for c in year_cols if c in all_matched.columns]
            multi_year_table = ""
            if not (requested_period and requested_period.strip()):
                multi_year_table = _build_multi_year_table(all_matched, year_cols)

            # 渲染
            rendered_lines = multi_year_table.split("\n") if multi_year_table else []
            rendered_lines += [
                "",
                "---",
                f"数据集: 真实财务报表 (4/{statement_type}.csv)",
                f"股票代码: {stock_code}",
                f"最新报告期: {rp_str}",
                "",
            ]
            for k, v in summary.items():
                if k not in ("stock_code", "report_period") and v is not None:
                    label = FIELD_CN.get(k, k)
                    rendered_lines.append(f"  {label}: {v:,.0f}")

            return {
                "stock_code": stock_code,
                "statement_type": statement_type,
                "report_period": rp_str,
                "data": raw,
                "summary": summary,
                "rendered": "\n".join(overview_lines + rendered_lines),
                "overview": "\n".join(overview_lines),
                "available_periods": sorted(all_matched["report_period"].astype(str).unique(), reverse=True),
                "source": "dataset",
            }

        except Exception as e:
            return {
                "stock_code": stock_code,
                "statement_type": statement_type,
                "report_period": "N/A",
                "error": str(e),
                "source": "error",
            }
