"""
工具执行器
统一的工具执行接口，支持超时控制、结果格式化、Mock 数据回退。
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .tool_registry import ToolMeta


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    tool_name: str
    data: Any = None
    error: str = ""
    execution_time_ms: float = 0.0
    attempts: int = 1
    fallback_used: bool = False
    raw_output: str = ""


class ToolExecutor:
    """
    工具执行器：负责实际调用工具并返回结构化结果。
    当前阶段使用 Mock 数据（数据集中的真实数据），后续可替换为真实 API。
    """

    def __init__(self, data_loader=None):
        """
        Args:
            data_loader: DataLoader 实例，用于 Mock 数据查询
        """
        self.data_loader = data_loader
        # _executors 保留作为向后兼容的降级路径。
        # 新工具通过 BaseTool 插件注册（tool.executor 优先）。
        self._executors = {
            # 所有工具已迁移到 BaseTool。
            # 以下保留仅用于 agent_loop._supplement_query 的快速路径：
            "query_financial_statement": self._mock_financial_statement,  # 保留：_supplement_query 使用
            "control_summary": self._exec_control_summary,  # 保留：_supplement_query 使用
            "search_news": self._mock_news_search,  # 保留：_supplement_query 使用
        }

    def execute(self, tool: ToolMeta, params: Dict[str, Any]) -> ToolResult:
        """
        执行工具调用。

        优先使用 tool.executor（BaseTool 插件方式），
        其次查找 self._executors（传统方式，向后兼容）。

        Args:
            tool: 工具元数据
            params: 调用参数

        Returns:
            ToolResult: 执行结果
        """
        start_time = time.time()

        # 参数校验
        missing = [p for p in tool.required_params if p not in params or not params[p]]
        if missing:
            return ToolResult(
                success=False,
                tool_name=tool.name,
                error=f"缺少必要参数: {missing}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # 执行 —— 优先使用 tool.executor（BaseTool 插件方式）
        executor = tool.executor  # BaseTool._make_executor() 设置的闭包
        if executor:
            try:
                data = executor(params, data_loader=self.data_loader)
                return ToolResult(
                    success=True,
                    tool_name=tool.name,
                    data=data,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    tool_name=tool.name,
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

        # 降级: 查找 _executors 字典（传统方式，向后兼容）
        executor = self._executors.get(tool.name)
        if executor:
            try:
                data = executor(params)
                return ToolResult(
                    success=True,
                    tool_name=tool.name,
                    data=data,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    tool_name=tool.name,
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000,
                )

        return ToolResult(
            success=False,
            tool_name=tool.name,
            error=f"工具 '{tool.name}' 未注册执行器",
            execution_time_ms=(time.time() - start_time) * 1000,
        )

    # ---- Mock 执行器（使用数据集模拟） ----

    def _mock_market_data(self, params: Dict) -> Dict:
        """行情数据查询——系统无实时行情数据，诚实告知"""
        stock_code = params.get("stock_code", "")
        return {
            "stock_code": stock_code,
            "error": "NO_REALTIME_DATA",
            "rendered": (
                f"## 行情查询: {stock_code}\n\n"
                "**无法提供实时行情数据**。\n\n"
                "当前系统仅包含以下历史数据：\n"
                "- 财务报表（2023Q4~2026Q1）\n"
                "- 股东持股明细\n"
                "- 公司公告\n"
                "- 券商研报\n\n"
                "如需查询股价/涨跌幅/换手率/主力资金等实时行情，请使用东方财富、同花顺等行情软件。\n"
                "如需查询财务数据或股东信息，我可以帮您查询。"
            ),
            "source": "system",
            "note": "系统无实时行情API接入",
        }

    def _mock_financial_statement(self, params: Dict) -> Dict:
        """Mock 财务报表查询（优先使用真实数据集）"""
        stock_code = params.get("stock_code", "")
        statement_type = params.get("statement_type", "balance_sheet")

        if self.data_loader:
            try:
                if statement_type == "balance_sheet":
                    df = self.data_loader.load_balance_sheet()
                elif statement_type == "income":
                    df = self.data_loader.load_income()
                elif statement_type == "cashflow":
                    df = self.data_loader.load_cashflow()
                else:
                    df = self.data_loader.load_balance_sheet()

                # 按股票代码筛选（优先 s_info_windcode: "600519.SH" 格式，兼容纯6位代码）
                if "s_info_windcode" in df.columns:
                    matched = df[df["s_info_windcode"].str.contains(str(stock_code), na=False)]
                elif "stock_code" in df.columns:
                    matched = df[df["stock_code"].astype(str) == str(stock_code)]
                else:
                    matched = pd.DataFrame()

                if len(matched) > 0:
                    requested_period = params.get("report_period", "")
                    all_matched = matched.sort_values("report_period", ascending=False)
                    rp_str = ""
                    raw = None

                    # 如果指定了具体报告期，精准匹配
                    if requested_period and requested_period.strip():
                        rp_clean = requested_period.replace("Q1","0331").replace("Q2","0630").replace("Q3","0930").replace("Q4","1231")
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
                        # 未指定报告期 → 返回最新期用于摘要
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

                    # 提取核心字段（避免100+字段淹没 LLM）
                    core_fields = {
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
                    # 中文字段名映射
                    FIELD_CN = {
                        # 利润表
                        "tot_oper_rev": "营业总收入", "oper_rev": "营业收入",
                        "less_oper_cost": "营业成本(不含三费)", "tot_oper_cost": "营业总成本(含三费)",
                        "oper_profit": "营业利润",
                        "tot_profit": "利润总额", "oper_exp": "营业费用",
                        "net_profit_incl_min_int_inc": "净利润(含少数股东)",
                        "net_profit_excl_min_int_inc": "净利润(不含少数股东)",
                        # 资产负债表
                        "monetary_cap": "货币资金", "inventories": "存货",
                        "tot_assets": "总资产", "tot_liab": "总负债",
                        "tot_shrhdr_eqy_incl_min_int": "股东权益",
                        "tot_cur_assets": "流动资产", "tot_cur_liab": "流动负债",
                        "goodwill": "商誉", "tradable_fin_assets": "交易性金融资产",
                        "notes_rcv": "应收票据",
                        # 现金流量表
                        "cash_cash_equ_beg_period": "期初现金",
                        "cash_cash_equ_end_period": "期末现金",
                        "net_cash_flows_oper_act": "经营活动现金流净额",
                        "net_cash_flows_inv_act": "投资活动现金流净额",
                        "net_cash_flows_fnc_act": "筹资活动现金流净额",
                    }
                    core = core_fields.get(statement_type, list(raw.keys())[:20])
                    summary = {k: raw.get(k) for k in core if k in raw and pd.notna(raw.get(k))}
                    summary["report_period"] = rp_str
                    summary["stock_code"] = stock_code

                    # === 全景视图：所有可用报告期 ===
                    overview_cols = {
                        "income": ["tot_oper_rev", "less_oper_cost", "net_profit_excl_min_int_inc"],
                        "balance_sheet": ["tot_assets", "tot_liab", "inventories"],
                        "cashflow": ["net_cash_flows_oper_act", "cash_cash_equ_end_period"],
                    }.get(statement_type, [])
                    overview_cols = [c for c in overview_cols if c in all_matched.columns]
                    overview_lines = ["", "## 该股票所有可用报告期一览", ""]
                    if overview_cols:
                        header = "| 报告期 |" + "|".join(FIELD_CN.get(c, c) for c in overview_cols) + "|"
                        sep = "|" + "|".join("------" for _ in range(len(overview_cols)+1)) + "|"
                        overview_lines.append(header)
                        overview_lines.append(sep)
                        for _, r in all_matched.iterrows():
                            rp_v = str(r["report_period"])
                            if rp_v.endswith("0331"): rp_v += "(Q1)"
                            elif rp_v.endswith("0630"): rp_v += "(Q2)"
                            elif rp_v.endswith("0930"): rp_v += "(Q3)"
                            elif rp_v.endswith("1231"): rp_v += "(年报)"
                            vals = []
                            for c in overview_cols:
                                v = r.get(c)
                                if v and pd.notna(v) and v != 0:
                                    v = float(v)
                                    if abs(v) >= 1e8: vals.append(f"{v/1e8:.1f}亿")
                                    elif abs(v) >= 1e4: vals.append(f"{v/1e4:.0f}万")
                                    else: vals.append(f"{v:,.0f}")
                                else: vals.append("-")
                            overview_lines.append("| " + rp_v + " |" + "|".join(vals) + "|")
                        overview_lines.append("")
                        overview_lines.append("以上数据可直接计算对比。选取两股票都有的相同报告期，年报优先。")
                    else:
                        overview_lines.append("(无数据)")

                    # === 多期对比表（未指定年份时，一次返回所有期） ===
                    multi_year_lines = []
                    if not requested_period or not requested_period.strip():
                        year_cols = {
                            "income": ["tot_oper_rev", "net_profit_excl_min_int_inc",
                                       "less_oper_cost", "oper_profit"],
                            "balance_sheet": ["tot_assets", "tot_liab", "inventories",
                                              "monetary_cap", "tot_shrhdr_eqy_incl_min_int"],
                            "cashflow": ["net_cash_flows_oper_act", "net_cash_flows_inv_act",
                                         "net_cash_flows_fnc_act", "cash_cash_equ_end_period"],
                        }.get(statement_type, [])
                        year_cols = [c for c in year_cols if c in all_matched.columns]
                        if year_cols and len(all_matched) >= 2:
                            multi_year_lines.append("\n## 📊 全报告期对比表\n")
                            header = "| 报告期 |" + "|".join(FIELD_CN.get(c, c) for c in year_cols) + "| 类型 |"
                            sep = "|" + "|".join("------" for _ in range(len(year_cols)+2)) + "|"
                            multi_year_lines.append(header)
                            multi_year_lines.append(sep)
                            for _, r in all_matched.iterrows():
                                rp_v = str(r["report_period"])
                                tag = ""
                                if rp_v.endswith("1231"): tag = "年报"
                                elif rp_v.endswith("0630"): tag = "中报"
                                elif rp_v.endswith("0331"): tag = "Q1"
                                elif rp_v.endswith("0930"): tag = "Q3"
                                vals = []
                                for c in year_cols:
                                    v = r.get(c)
                                    if v and pd.notna(v) and float(v) != 0:
                                        v = float(v)
                                        if abs(v) >= 1e8: vals.append(f"{v/1e8:.2f}亿")
                                        elif abs(v) >= 1e4: vals.append(f"{v/1e4:.0f}万")
                                        else: vals.append(f"{v:,.0f}")
                                    else: vals.append("-")
                                multi_year_lines.append("| " + rp_v + " |" + "|".join(vals) + f"| {tag} |")
                            multi_year_lines.append("")

                    # 生成 LLM 友好的 rendered 文本 — 多期对比表在最前面
                    rendered_lines = [
                        "",
                        "---",
                        f"数据集: 真实财务报表 (4/{statement_type}.csv)",
                        f"股票代码: {stock_code}",
                        f"最新报告期: {rp_str}",
                        "",
                    ]
                    # 插入多期对比表
                    if multi_year_lines:
                        rendered_lines = multi_year_lines + rendered_lines
                    for k, v in summary.items():
                        if k not in ("stock_code", "report_period") and v is not None:
                            label = FIELD_CN.get(k, k)
                            rendered_lines.append(f"  {label}: {v:,.0f}")

                    return {
                        "stock_code": stock_code,
                        "statement_type": statement_type,
                        "report_period": rp_str,
                        "data": raw,           # 完整原始数据
                        "summary": summary,     # 核心字段摘要
                        "rendered": "\n".join(overview_lines + rendered_lines),
                        "overview": "\n".join(overview_lines),
                        "available_periods": sorted(all_matched["report_period"].astype(str).unique(), reverse=True),
                        "source": "dataset",
                    }
                else:
                    return {
                        "stock_code": stock_code,
                        "statement_type": statement_type,
                        "error": f"未找到股票代码 {stock_code} 的{statement_type}数据",
                        "source": "dataset",
                    }
            except Exception as e:
                print(f"[ToolExecutor] Real data query failed: {e}")

        return {
            "stock_code": stock_code,
            "statement_type": statement_type,
            "report_period": "N/A",
            "data": {"note": "Mock 数据，未找到真实数据"},
            "source": "mock",
        }

    def _exec_equity_penetration(self, params: Dict) -> Dict:
        """Task 2: 股权穿透查询"""
        try:
            from ..tools.equity_graph import EquityPenetrationSkill
            return EquityPenetrationSkill.execute(params)
        except Exception as e:
            return {"error": str(e), "source": "task2_error"}

    def _exec_event_trace(self, params: Dict) -> Dict:
        """Task 2: 事件溯源查询"""
        try:
            from ..tools.equity_graph import EventTraceSkill
            return EventTraceSkill.execute(params)
        except Exception as e:
            return {"error": str(e), "source": "task2_error"}

    def _exec_control_summary(self, params: Dict) -> Dict:
        """Task 2: 控股摘要查询（快速路径优先用DataFrame）"""
        stock_code = params.get("stock_code", "")
        try:
            # 快速路径：直接从DataFrame查询（~3s vs 140s图构建）
            if self.data_loader:
                df = self.data_loader.load_shareholder_data()
                matched = df[df["stock_code"].astype(str).str.contains(str(stock_code), na=False)]
                if len(matched) > 0:
                    # 按股东名称去重，取最新报告期的持股比例
                    if "s_holder_enddate" in matched.columns:
                        matched = matched.sort_values("s_holder_enddate", ascending=False)
                    matched_unique = matched.drop_duplicates(subset=["s_holder_name"], keep="first")
                    top10 = matched_unique.sort_values("s_holder_pct", ascending=False).head(10)
                    holders = []
                    total_pct = 0
                    for _, row in top10.iterrows():
                        h = {
                            "name": str(row.get("s_holder_name", "")),
                            "pct": float(row.get("s_holder_pct", 0)),
                            "type": str(row.get("holder_type", "未知")),
                        }
                        holders.append(h)
                        total_pct += h["pct"]

                    rendered_lines = [
                        f"数据来源: 真实股东数据集 (2/clean.xlsx)",
                        f"股票代码: {stock_code}",
                        f"股东总数: {len(matched)}",
                        f"Top10 持股集中度: {total_pct:.1f}%",
                        "",
                        "前十大大股东:",
                    ]
                    for i, h in enumerate(holders):
                        rendered_lines.append(f"  {i+1}. {h['name'][:40]} | 持股 {h['pct']:.2f}% | {h['type']}")

                    return {
                        "stock_code": stock_code,
                        "total_holders": len(matched),
                        "top5_concentration": sum(h["pct"] for h in holders[:5]),
                        "top10_concentration": total_pct,
                        "top_holders": holders,
                        "rendered": "\n".join(rendered_lines),
                        "source": "dataset",
                        "method": "fast_dataframe",  # 标记快速路径
                    }

            # 降级：走完整图构建路径（首次约90s，后续缓存）
            from ..tools.equity_graph import ControlSummarySkill
            return ControlSummarySkill.execute(params)
        except Exception as e:
            return {"error": str(e), "source": "task2_error"}

    def _exec_financial_anomaly(self, params: Dict) -> Dict:
        """Task 3: 财务异象甄别"""
        try:
            from ..tools.financial_anomaly import FinancialAnomalySkill
            return FinancialAnomalySkill.execute(params)
        except Exception as e:
            return {"error": str(e), "source": "task3_error"}

    def _exec_multi_period(self, params: Dict) -> Dict:
        """Task 3: 多期对比分析"""
        try:
            from ..tools.financial_anomaly import MultiPeriodAnalysisSkill
            return MultiPeriodAnalysisSkill.execute(params)
        except Exception as e:
            return {"error": str(e), "source": "task3_error"}

    def _exec_search_reports(self, params: Dict) -> Dict:
        """P0: 券商研报检索（数据集 5）"""
        try:
            from ..tools.research_reports import SearchReportsToolImpl
            return SearchReportsToolImpl.execute(params)
        except Exception as e:
            return {"error": str(e), "source": "research_reports_error"}

    def _exec_search_reports_by_stock(self, params: Dict) -> Dict:
        """P0: 按股票查研报"""
        try:
            from ..tools.research_reports import SearchReportsByStockToolImpl
            return SearchReportsByStockToolImpl.execute(params)
        except Exception as e:
            return {"error": str(e), "source": "research_reports_error"}

    def _mock_news_search(self, params: Dict) -> Dict:
        """Mock 新闻检索（使用数据集 3 中的公告数据）"""
        query = params.get("query", "")
        stock_code = params.get("stock_code", "")

        if self.data_loader:
            try:
                df = self.data_loader.load_announcements()

                # 如果有 stock_code 则筛选
                matched = df
                if stock_code:
                    if "stock_code" in df.columns:
                        matched = df[df["stock_code"] == stock_code]
                    elif "s_info_windcode" in df.columns:
                        matched = df[df["s_info_windcode"].str.contains(stock_code, na=False)]

                # 如果有 query 关键词，做标题匹配
                if query and "n_info_title" in matched.columns:
                    kw_matched = matched[matched["n_info_title"].str.contains(query, na=False)]
                    if len(kw_matched) > 0:
                        matched = kw_matched

                if len(matched) > 0:
                    results = []
                    for _, row in matched.head(10).iterrows():
                        results.append({
                            "title": str(row.get("n_info_title", "")),
                            "date": str(row.get("ann_dt", "")),
                            "stock_code": str(row.get("stock_code", row.get("s_info_windcode", ""))),
                        })
                    return {
                        "query": query,
                        "stock_code": stock_code or "ALL",
                        "total": len(results),
                        "articles": results,
                        "source": "dataset",
                    }
            except Exception as e:
                print(f"[ToolExecutor] News search failed: {e}")

        return {
            "query": query,
            "stock_code": stock_code or "ALL",
            "total": 0,
            "articles": [],
            "source": "mock",
        }

    def _mock_calculator(self, params: Dict) -> Dict:
        """Mock 财务计算器"""
        expression = params.get("expression", "")
        return {
            "expression": expression,
            "result": "N/A (Mock)",
            "note": "财务计算器需集成具体数值",
        }

    def format_result_for_llm(self, result: ToolResult) -> str:
        """将工具执行结果格式化为 LLM 可读文本"""
        if not result.success:
            return f"[工具调用失败] {result.tool_name}: {result.error}"

        data = result.data or {}
        return json.dumps(data, ensure_ascii=False, indent=2, default=self._json_serializer)

    @staticmethod
    def _json_serializer(obj):
        """处理 pandas/numpy 类型 → JSON 可序列化"""
        import pandas as pd
        import numpy as np
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return str(obj)
