"""
研判报告生成器
将风险评分和规则检测结果组合为结构化诊断报告，支持 Markdown 格式输出。
"""

from typing import Any, Dict, List, Optional

from ..llm import get_llm_client
from .data_extractor import FinancialSnapshot
from .risk_scorer import RiskScore
from .rule_engine import RuleResult


class ReportGenerator:
    """
    财务健康度研判报告生成器。
    支持规则化报告和 LLM 增强报告两种模式。
    """

    def __init__(self, use_llm: bool = True):
        self.llm = get_llm_client() if use_llm else None
        self.use_llm = use_llm

    def generate(
        self,
        risk_score: RiskScore,
        rule_results: List[RuleResult],
        snapshot: FinancialSnapshot,
        prev_snapshot: Optional[FinancialSnapshot] = None,
    ) -> str:
        """
        生成完整研判报告。

        报告结构：
        1. 总体健康度评分
        2. 多维雷达图数据
        3. 预警详情（A/B级）
        4. 数据对比
        5. 可能的造假模式分析
        6. 综合建议
        """
        # 基础报告
        base_report = self._generate_base_report(risk_score, rule_results, snapshot, prev_snapshot)

        # LLM 增强（为报告添加更专业的分析语言）
        if self.use_llm and self.llm and risk_score.risk_level in ("high", "critical"):
            llm_analysis = self._llm_enhance_report(risk_score, rule_results, snapshot)
            if llm_analysis:
                base_report += f"\n\n---\n## 🤖 AI 深度分析\n\n{llm_analysis}"

        return base_report

    def _generate_base_report(
        self, rs: RiskScore, rule_results: List[RuleResult],
        s: FinancialSnapshot, prev: Optional[FinancialSnapshot]
    ) -> str:
        """生成基础结构化报告"""
        # 风险等级图标
        level_icons = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
        level_names = {"low": "低风险", "medium": "关注", "high": "高风险", "critical": "严重风险"}
        icon = level_icons.get(rs.risk_level, "⚪")
        level_name = level_names.get(rs.risk_level, "未知")

        lines = [
            f"# 📊 财务健康度研判报告",
            f"",
            f"**股票代码**: {rs.stock_code}  |  **报告期**: {rs.report_period}  |  **评级**: {icon} {level_name}",
            f"",
            f"---",
            f"",
            f"## 一、综合评分",
            f"",
            f"| 维度 | 健康分 | 等级 |",
            f"|------|--------|------|",
            f"| 💰 盈利能力 | {rs.profitability_score:.1f}/100 | {self._score_to_level(rs.profitability_score)} |",
            f"| 📦 资产质量 | {rs.asset_quality_score:.1f}/100 | {self._score_to_level(rs.asset_quality_score)} |",
            f"| 💵 现金流质量 | {rs.cashflow_quality_score:.1f}/100 | {self._score_to_level(rs.cashflow_quality_score)} |",
            f"| 🏦 偿债能力 | {rs.solvency_score:.1f}/100 | {self._score_to_level(rs.solvency_score)} |",
            f"| ⚙️ 运营效率 | {rs.operational_score:.1f}/100 | {self._score_to_level(rs.operational_score)} |",
            f"| **综合评分** | **{rs.overall_score:.1f}/100** | **{level_name}** |",
            f"",
            f"规则检测: {rs.total_rules} 条 | 未通过: {rs.failed_rules} 条 | 高危预警: {len(rs.critical_warnings)} 条",
            f"",
            f"---",
            f"",
            f"## 二、关键财务数据",
            f"",
            f"| 项目 | 数值 |",
            f"|------|------|",
            f"| 营业收入 | {max(s.oper_rev, s.tot_oper_rev)/1e8:.2f} 亿 |",
            f"| 净利润 | {s.net_profit/1e8:.2f} 亿 |",
            f"| 总资产 | {s.tot_assets/1e8:.2f} 亿 |",
            f"| 存货 | {s.inventories/1e8:.2f} 亿 |",
            f"| 应收账款 | {(s.acct_rcv + s.notes_rcv)/1e8:.2f} 亿 |",
            f"| 经营现金流 | {s.net_cash_flows_oper/1e8:.2f} 亿 |",
            f"| 资产负债率 | {s.tot_liab/max(s.tot_assets,1)*100:.1f}% |",
            f"",
            f"---",
            f"",
        ]

        # 预警详情
        failed_rules = [r for r in rule_results if not r.passed]
        if failed_rules:
            lines.append(f"## 三、预警详情（{len(failed_rules)} 条异常）")
            lines.append("")

            for rr in failed_rules:
                lines.append(rr.render())
                lines.append("")

        # 高危预警
        if rs.critical_warnings:
            lines.append("---")
            lines.append("")
            lines.append("## 四、高危预警项 ⚠️")
            lines.append("")
            for wr in rs.critical_warnings:
                lines.append(f"- **{wr.rule_name}**: {wr.detail}")
                if wr.possible_fraud:
                    lines.append(f"  > 风险提示: {wr.possible_fraud}")
                lines.append("")

        # 多期对比（如有）
        if prev:
            lines.append("---")
            lines.append("")
            lines.append("## 五、环比变化")
            lines.append("")
            lines.append("| 项目 | 上期 | 本期 | 变动 |")
            lines.append("|------|------|------|------|")
            items = [
                ("营业收入", prev.oper_rev or prev.tot_oper_rev, s.oper_rev or s.tot_oper_rev),
                ("净利润", prev.net_profit, s.net_profit),
                ("存货", prev.inventories, s.inventories),
                ("应收账款", prev.acct_rcv + prev.notes_rcv, s.acct_rcv + s.notes_rcv),
                ("经营现金流", prev.net_cash_flows_oper, s.net_cash_flows_oper),
            ]
            for name, prev_val, curr_val in items:
                if prev_val > 0 or curr_val > 0:
                    change = ((curr_val - prev_val) / max(abs(prev_val), 1)) * 100
                    arrow = "↑" if change > 0 else "↓"
                    lines.append(
                        f"| {name} | {prev_val/1e8:.2f}亿 | {curr_val/1e8:.2f}亿 | {arrow}{abs(change):.1f}% |"
                    )

        # 综合建议
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 六、综合建议")
        lines.append("")

        if rs.risk_level == "low":
            lines.append("该财报未发现明显财务造假信号，财务健康度良好。建议持续跟踪关键指标变化。")
        elif rs.risk_level == "medium":
            lines.append(f"该财报存在 {rs.failed_rules} 项需关注的异象指标，建议对上述预警项进行深入核查。重点关注现金流质量和资产周转效率。")
        elif rs.risk_level == "high":
            lines.append(f"⚠️ 该财报存在 {rs.failed_rules} 项异常指标（其中 {len(rs.critical_warnings)} 项高危），存在财务造假风险。建议：")
            lines.append("1. 核实收入确认政策，检查应收账款回款情况")
            lines.append("2. 对存货进行实质性盘点验证")
            lines.append("3. 审计关联交易和资金往来")
        else:
            lines.append(f"🔴 **严重预警**：该财报存在 {rs.failed_rules} 项异常，{len(rs.critical_warnings)} 项高危。财务造假风险极高。建议：")
            lines.append("1. 立即启动专项审计")
            lines.append("2. 核实所有重大交易的商业实质")
            lines.append("3. 检查是否存在表外负债")

        lines.append("")
        lines.append("> ⚠️ **免责声明**: 本报告由 AI 系统自动生成，仅供研究参考，不构成投资建议。")

        return "\n".join(lines)

    def _llm_enhance_report(
        self, rs: RiskScore, rule_results: List[RuleResult],
        snapshot: FinancialSnapshot
    ) -> Optional[str]:
        """LLM 增强：为高风险报告添加深度分析"""
        if not self.llm:
            return None

        failed = [r for r in rule_results if not r.passed]
        failed_text = "\n".join(
            f"- {r.rule_name}: {r.detail}\n  潜在风险: {r.possible_fraud}"
            for r in failed[:5]
        )

        prompt = f"""你是一位资深财务审计专家。请基于以下财务异象检测结果，提供专业的深度分析。

## 公司信息
股票代码: {rs.stock_code}
报告期: {rs.report_period}
综合健康度: {rs.overall_score:.1f}/100 ({rs.risk_level})

## 关键数据
- 营业收入: {max(snapshot.oper_rev, snapshot.tot_oper_rev)/1e8:.2f}亿
- 净利润: {snapshot.net_profit/1e8:.2f}亿
- 存货: {snapshot.inventories/1e8:.2f}亿
- 经营现金流: {snapshot.net_cash_flows_oper/1e8:.2f}亿

## 检测到的异常
{failed_text}

请提供：
1. 这些异象之间是否存在关联？是否构成系统性的造假模式？
2. 最可能的造假手法是什么？（如: 虚增收入+少转成本、关联交易输送、体外资金循环等）
3. 建议审计师优先核查哪3个关键点？

请用专业但简洁的中文回答，不超过400字。"""

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.2,
            )
            return response.strip()
        except Exception as e:
            print(f"[ReportGen] LLM enhancement failed: {e}")
            return None

    @staticmethod
    def _score_to_level(score: float) -> str:
        if score >= 80:
            return "✅ 健康"
        elif score >= 60:
            return "⚠️ 关注"
        elif score >= 40:
            return "🟠 异常"
        else:
            return "🔴 严重"

    def generate_short(self, risk_score: RiskScore) -> str:
        """生成简短摘要（用于 Agent 对话回复中嵌入）"""
        level_names = {"low": "低风险", "medium": "需关注", "high": "高风险", "critical": "严重风险"}
        return (
            f"**[{risk_score.stock_code}] 财务健康度: {risk_score.overall_score:.0f}/100 "
            f"({level_names.get(risk_score.risk_level, '未知')})** | "
            f"{risk_score.total_rules}项检测, {risk_score.failed_rules}项异常"
            + (f", {len(risk_score.critical_warnings)}项高危" if risk_score.critical_warnings else "")
        )
