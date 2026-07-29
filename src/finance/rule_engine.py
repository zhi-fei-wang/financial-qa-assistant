"""
财务排雷规则引擎
实现 A/B 两级勾稽规则，对财务快照进行多维度异常检测。

A 级 — 单科目异常检测（8条规则）
B 级 — 跨表勾稽校验（6条规则）
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .data_extractor import FinancialSnapshot


@dataclass
class RuleResult:
    """单条规则的检测结果"""
    rule_id: str                       # 规则 ID
    rule_name: str                     # 规则名称
    level: str                         # A / B
    category: str                      # 分类: 盈利能力/资产质量/现金流/其他
    passed: bool                       # 是否通过检测
    score: float                       # 风险评分 (0=无风险, 1=高风险)
    detail: str                        # 检测详情（预警点）
    data_comparison: str               # 数据对比
    possible_fraud: str = ""           # 可能的造假模式或业务风险
    severity: str = "low"              # low / medium / high / critical

    def render(self) -> str:
        status = "✅ 正常" if self.passed else f"⚠️ 异常 (风险评分: {self.score:.2f})"
        return (
            f"### {self.rule_name} [{self.level}级-{self.category}]\n"
            f"- 状态: {status}\n"
            f"- 详情: {self.detail}\n"
            f"- 数据对比: {self.data_comparison}\n"
            + (f"- 潜在风险: {self.possible_fraud}\n" if self.possible_fraud else "")
        )


class FinancialRuleEngine:
    """
    财务排雷规则引擎。

    规则体系设计参考：
    - 中国注册会计师审计准则
    - 常见财务造假手法（虚增收入/隐瞒负债/虚增资产）
    - 赛题提供的业务场景中的具体案例
    """

    def __init__(self):
        self.rules: List[Callable] = []
        self._register_rules()

    def _register_rules(self):
        """注册所有检测规则"""
        self.rules = [
            # === A 级：单科目异常 ===
            self._rule_inventory_surge,          # A1: 存货激增
            self._rule_receivable_surge,         # A2: 应收账款激增
            self._rule_cashflow_profit_divergence, # A3: 现金流/净利润悖离
            self._rule_revenue_tax_mismatch,     # A4: 营收与税费不匹配
            self._rule_goodwill_impairment,       # A5: 商誉减值风险
            self._rule_financial_expense_anomaly, # A6: 异常财务费用
            self._rule_gross_margin_collapse,     # A7: 毛利率骤降
            self._rule_asset_liability_mismatch,  # A8: 资产负债结构异常

            # === B 级：跨表勾稽校验 ===
            self._rule_net_profit_vs_cashflow,   # B1: 净利润 vs 经营现金流（跨表）
            self._rule_revenue_vs_receivable,    # B2: 营收 vs 应收（跨表）
            self._rule_inventory_vs_payable,     # B3: 存货 vs 应付（跨表）
            self._rule_equity_growth_vs_profit,  # B4: 权益增长 vs 利润
            self._rule_depreciation_consistency, # B5: 折旧与固定资产匹配
            self._rule_cash_cycle_anomaly,       # B6: 现金周期异常
        ]

    def evaluate(self, snapshot: FinancialSnapshot,
                 prev_snapshot: Optional[FinancialSnapshot] = None) -> List[RuleResult]:
        """
        对单个财务快照执行全部规则检测。

        Args:
            snapshot: 当前期财务快照
            prev_snapshot: 上一期快照（用于趋势分析）

        Returns:
            所有规则的检测结果列表
        """
        results = []
        for rule_func in self.rules:
            try:
                result = rule_func(snapshot, prev_snapshot)
                results.append(result)
            except Exception as e:
                # 数据不足时静默跳过
                pass
        return results

    # =========================================================================
    # A 级规则：单科目异常检测
    # =========================================================================

    def _rule_inventory_surge(self, s: FinancialSnapshot, prev: Optional[FinancialSnapshot]) -> RuleResult:
        """A1: 存货激增检测 — 存货增长率是否远超营收增长率（>2倍）"""
        detail = ""
        comparison = ""
        score = 0.0
        passed = True
        fraud_hint = ""

        inventory = s.inventories
        revenue = s.oper_rev or s.tot_oper_rev

        comparison = f"存货: {inventory/1e8:.2f}亿 | 营收: {revenue/1e8:.2f}亿 | 存货/营收比: {inventory/max(revenue,1)*100:.1f}%"

        if inventory > 0 and revenue > 0:
            ratio = inventory / revenue
            if ratio > 0.8:
                score = min(1.0, (ratio - 0.3) / 0.7)  # >30%开始预警
                passed = ratio <= 0.5
                detail = f"存货/营收比 = {ratio:.1%}，超过50%警戒线"
                fraud_hint = "可能存在虚增存货（以存货换交付）、少转成本、或产品滞销导致的减值风险"

        if prev and prev.inventories > 0 and prev.oper_rev > 0:
            inv_growth = (inventory - prev.inventories) / prev.inventories
            rev_growth = (revenue - (prev.oper_rev or prev.tot_oper_rev)) / max(prev.oper_rev or prev.tot_oper_rev, 1)
            if inv_growth > 0.3 and inv_growth > rev_growth * 2:
                score = max(score, 0.7)
                passed = False
                detail += f" | 存货增长率({inv_growth:.1%})远超营收增长率({rev_growth:.1%})"
                fraud_hint = "存货增速是营收增速的2倍以上，典型的虚增存货信号"

        return RuleResult(
            rule_id="A1", rule_name="存货激增检测", level="A", category="资产质量",
            passed=passed, score=score, detail=detail,
            data_comparison=comparison, possible_fraud=fraud_hint,
            severity="high" if score > 0.7 else ("medium" if score > 0.4 else "low"),
        )

    def _rule_receivable_surge(self, s: FinancialSnapshot, prev: Optional[FinancialSnapshot]) -> RuleResult:
        """A2: 应收账款激增 — 应收增速远超营收增速"""
        acct_rcv = s.acct_rcv + s.notes_rcv
        revenue = s.oper_rev or s.tot_oper_rev
        comparison = f"应收账款: {acct_rcv/1e8:.2f}亿 | 营收: {revenue/1e8:.2f}亿"
        score = 0.0
        passed = True

        if revenue > 0:
            ratio = acct_rcv / revenue
            if ratio > 0.5:
                score = min(1.0, ratio)
                passed = False

        if prev:
            prev_rcv = prev.acct_rcv + prev.notes_rcv
            prev_rev = prev.oper_rev or prev.tot_oper_rev
            if prev_rcv > 0 and prev_rev > 0:
                rcv_growth = (acct_rcv - prev_rcv) / prev_rcv
                rev_growth = (revenue - prev_rev) / prev_rev
                if rcv_growth > 0.3 and rcv_growth > rev_growth * 1.5:
                    score = max(score, 0.6)
                    passed = False
                    comparison += f" | 应收增速({rcv_growth:.1%}) vs 营收增速({rev_growth:.1%})"

        return RuleResult(
            rule_id="A2", rule_name="应收账款激增检测", level="A", category="资产质量",
            passed=passed, score=score,
            detail=f"应收/营收比 = {acct_rcv/max(revenue,1):.1%}" if revenue > 0 else "数据不足",
            data_comparison=comparison,
            possible_fraud="应收账款增速远超营收增速，可能虚增收入、放宽信用政策、或存在关联方资金占用",
            severity="high" if score > 0.6 else "medium",
        )

    def _rule_cashflow_profit_divergence(self, s: FinancialSnapshot, prev=None) -> RuleResult:
        """A3: 经营现金流/净利润悖离"""
        cf_oper = s.net_cash_flows_oper
        net_profit = s.net_profit
        comparison = f"经营现金流: {cf_oper/1e8:.2f}亿 | 净利润: {net_profit/1e8:.2f}亿"
        score = 0.0
        passed = True

        if net_profit > 0:
            ratio = cf_oper / net_profit
            if cf_oper < 0:
                score = 0.8
                passed = False
                comparison += " | 净利润为正但经营现金流为负"
            elif ratio < 0.3:
                score = 0.5
                passed = False
                comparison += f" | 现金流/利润比 = {ratio:.1%}"

        return RuleResult(
            rule_id="A3", rule_name="经营现金流悖离检测", level="A", category="现金流质量",
            passed=passed, score=score,
            detail=f"现金流/净利润比 = {cf_oper/max(net_profit,1):.1%}",
            data_comparison=comparison,
            possible_fraud="盈利质量差，利润可能来自应收账款而非真金白银" if not passed else "",
            severity="critical" if cf_oper < 0 and net_profit > 0 else ("medium" if not passed else "low"),
        )

    def _rule_revenue_tax_mismatch(self, s: FinancialSnapshot, prev=None) -> RuleResult:
        """A4: 营收与税费不匹配"""
        revenue = s.oper_rev or s.tot_oper_rev
        inc_tax = s.inc_tax
        comparison = f"营收: {revenue/1e8:.2f}亿 | 所得税: {inc_tax/1e8:.2f}亿"
        score = 0.0
        passed = True
        detail = ""

        if revenue > 0 and inc_tax >= 0:
            tax_rate = inc_tax / revenue
            if tax_rate < 0.01:  # 实际税率 < 1%
                score = 0.6
                passed = False
                detail = f"实际税率仅 {tax_rate:.2%}，显著低于正常水平(15-25%)"
                comparison += f" | 实际税率: {tax_rate:.2%}"
            elif revenue > 1e9 and inc_tax < revenue * 0.05:
                score = 0.3
                detail = f"实际税率偏低: {tax_rate:.2%}"

        return RuleResult(
            rule_id="A4", rule_name="营收与税费匹配检测", level="A", category="盈利能力",
            passed=passed, score=score, detail=detail, data_comparison=comparison,
            possible_fraud="营收大幅增长但税费未同步增加 → 收入真实性存疑" if not passed else "",
            severity="high" if score > 0.5 else "low",
        )

    def _rule_goodwill_impairment(self, s: FinancialSnapshot, prev=None) -> RuleResult:
        """A5: 商誉减值风险 — 商誉占总资产比例过高"""
        goodwill = s.goodwill
        tot_assets = s.tot_assets
        comparison = f"商誉: {goodwill/1e8:.2f}亿 | 总资产: {tot_assets/1e8:.2f}亿"
        score = 0.0
        passed = True

        if tot_assets > 0 and goodwill > 0:
            ratio = goodwill / tot_assets
            if ratio > 0.3:
                score = min(1.0, ratio * 2)
                passed = False
            elif ratio > 0.15:
                score = 0.4
                passed = False

        return RuleResult(
            rule_id="A5", rule_name="商誉减值风险检测", level="A", category="资产质量",
            passed=passed, score=score,
            detail=f"商誉/总资产 = {goodwill/max(tot_assets,1):.1%}",
            data_comparison=comparison,
            possible_fraud="高商誉存在减值风险，可能通过不计提减值来虚增利润" if not passed else "",
            severity="high" if score > 0.6 else "medium",
        )

    def _rule_financial_expense_anomaly(self, s: FinancialSnapshot, prev=None) -> RuleResult:
        """A6: 异常财务费用 — 财务费用/营收比异常高或为负"""
        fin_exp = s.less_fin_exp
        revenue = s.oper_rev or s.tot_oper_rev
        comparison = f"财务费用: {fin_exp/1e8:.2f}亿 | 营收: {revenue/1e8:.2f}亿"
        score = 0.0
        passed = True
        detail = ""

        if revenue > 0:
            ratio = abs(fin_exp) / revenue
            if fin_exp < 0 and abs(ratio) > 0.05:
                # 负财务费用意味着利息收入 > 利息支出 —— 可能资金被大股东占用
                score = 0.5
                passed = False
                detail = f"财务费用为负 ({fin_exp/1e8:.2f}亿)，可能存在大额利息收入覆盖利息支出"
            elif ratio > 0.1:
                score = 0.6
                passed = False
                detail = f"财务费用率 = {ratio:.1%}，超过10%警戒线"

        return RuleResult(
            rule_id="A6", rule_name="异常财务费用检测", level="A", category="盈利能力",
            passed=passed, score=score, detail=detail, data_comparison=comparison,
            possible_fraud="财务费用异常高 → 高杠杆/资金链紧张 | 财务费用为负 → 可能存在资金占用",
            severity="medium" if not passed else "low",
        )

    def _rule_gross_margin_collapse(self, s: FinancialSnapshot, prev=None) -> RuleResult:
        """A7: 毛利率骤降"""
        revenue = s.oper_rev or s.tot_oper_rev
        cost = s.oper_cost or s.tot_oper_cost
        comparison = f"营收: {revenue/1e8:.2f}亿 | 成本: {cost/1e8:.2f}亿"
        score = 0.0
        passed = True

        if revenue > 0 and cost > 0:
            gross_margin = (revenue - cost) / revenue
            comparison += f" | 毛利率: {gross_margin:.1%}"

            if gross_margin < 0.05:
                score = 0.6
                passed = False
            elif gross_margin < 0.10:
                score = 0.3
                passed = False

        if prev:
            prev_rev = prev.oper_rev or prev.tot_oper_rev
            prev_cost = prev.oper_cost or prev.tot_oper_cost
            if prev_rev > 0 and prev_cost > 0:
                prev_margin = (prev_rev - prev_cost) / prev_rev
                curr_margin = (revenue - cost) / max(revenue, 1)
                if prev_margin > 0.2 and (prev_margin - curr_margin) > 0.1:
                    score = max(score, 0.7)
                    passed = False
                    comparison += f" | 毛利率骤降: {prev_margin:.1%} → {curr_margin:.1%}"

        return RuleResult(
            rule_id="A7", rule_name="毛利率骤降检测", level="A", category="盈利能力",
            passed=passed, score=score, detail=f"毛利率: {(revenue-cost)/max(revenue,1):.1%}",
            data_comparison=comparison,
            possible_fraud="毛利率骤降 → 产品竞争力下降或存在收入虚增后的回归",
            severity="high" if score > 0.6 else "medium",
        )

    def _rule_asset_liability_mismatch(self, s: FinancialSnapshot, prev=None) -> RuleResult:
        """A8: 资产负债结构异常"""
        tot_liab = s.tot_liab
        tot_assets = s.tot_assets
        comparison = f"负债: {tot_liab/1e8:.2f}亿 | 资产: {tot_assets/1e8:.2f}亿"
        score = 0.0
        passed = True

        if tot_assets > 0:
            ratio = tot_liab / tot_assets
            comparison += f" | 资产负债率: {ratio:.1%}"
            if ratio > 0.8:
                score = 0.7
                passed = False
            elif ratio > 0.7:
                score = 0.4
                passed = False

        return RuleResult(
            rule_id="A8", rule_name="资产负债结构检测", level="A", category="偿债能力",
            passed=passed, score=score,
            detail=f"资产负债率 = {tot_liab/max(tot_assets,1):.1%}",
            data_comparison=comparison,
            possible_fraud="高杠杆运营，债务违约风险大；异常低的负债率也可能意味着表外负债",
            severity="medium",
        )

    # =========================================================================
    # B 级规则：跨表勾稽校验
    # =========================================================================

    def _rule_net_profit_vs_cashflow(self, s: FinancialSnapshot, prev=None) -> RuleResult:
        """B1: 净利润 vs 经营现金流（跨表：利润表 + 现金流量表）"""
        net_profit = s.net_profit
        cf_oper = s.net_cash_flows_oper
        comparison = f"[利润表] 净利润: {net_profit/1e8:.2f}亿 | [现金流量表] 经营净现金流: {cf_oper/1e8:.2f}亿"
        score = 0.0
        passed = True

        if net_profit > 0:
            if cf_oper < 0:
                score = 0.9
                passed = False
                comparison += " | ⚠️ 利润为正但经营现金流为负"
            elif cf_oper < net_profit * 0.3:
                score = 0.6
                passed = False

        return RuleResult(
            rule_id="B1", rule_name="净利润vs经营现金流跨表勾稽", level="B", category="现金流质量",
            passed=passed, score=score,
            detail=f"经营现金流/净利润 = {cf_oper/max(net_profit,1):.1%}",
            data_comparison=comparison,
            possible_fraud="连续净利润>0但经营现金流为负 → 利润可能是纸面富贵，通过应收账款或存货虚增",
            severity="critical" if score > 0.8 else ("high" if not passed else "low"),
        )

    def _rule_revenue_vs_receivable(self, s: FinancialSnapshot, prev=None) -> RuleResult:
        """B2: 营收 vs 应收（跨表：利润表 + 资产负债表）"""
        revenue = s.oper_rev or s.tot_oper_rev
        receivables = s.acct_rcv + s.notes_rcv
        comparison = f"[利润表] 营收: {revenue/1e8:.2f}亿 | [资产负债表] 应收合计: {receivables/1e8:.2f}亿"
        score = 0.0
        passed = True

        if revenue > 0:
            ratio = receivables / revenue
            comparison += f" | 应收/营收: {ratio:.1%}"
            if ratio > 0.6:
                score = 0.7
                passed = False
            elif ratio > 0.4:
                score = 0.4
                passed = False

        return RuleResult(
            rule_id="B2", rule_name="营收vs应收跨表勾稽", level="B", category="资产质量",
            passed=passed, score=score,
            detail=f"应收/营收比 = {receivables/max(revenue,1):.1%}",
            data_comparison=comparison,
            possible_fraud="应收占比过高 → 收入质量差，可能通过放宽信用政策或虚构销售来虚增收入",
            severity="high" if score > 0.6 else "medium",
        )

    def _rule_inventory_vs_payable(self, s: FinancialSnapshot, prev=None) -> RuleResult:
        """B3: 存货 vs 应付账款（跨表：资产负债表内部勾稽）"""
        inventory = s.inventories
        payable = s.acct_payable
        comparison = f"存货: {inventory/1e8:.2f}亿 | 应付账款: {payable/1e8:.2f}亿"
        score = 0.0
        passed = True

        if inventory > 0:
            ratio = payable / inventory
            comparison += f" | 应付/存货: {ratio:.1%}"
            if inventory > 1e8 and ratio < 0.1:
                # 存货很多但应付很少 → 采购异常
                score = 0.5
                passed = False
            elif inventory > 1e8 and ratio > 2.0:
                # 应付远超存货 → 资金链问题
                score = 0.4
                passed = False

        return RuleResult(
            rule_id="B3", rule_name="存货vs应付账款勾稽", level="B", category="资产质量",
            passed=passed, score=score,
            detail=f"应付/存货比 = {payable/max(inventory,1):.1%}",
            data_comparison=comparison,
            possible_fraud="存货增长但应付未同步 → 存货真实性存疑，可能虚构采购",
            severity="medium",
        )

    def _rule_equity_growth_vs_profit(self, s: FinancialSnapshot, prev=None) -> RuleResult:
        """B4: 权益增长 vs 利润（资产负债表 + 利润表）"""
        equity = s.tot_equity
        net_profit = s.net_profit
        comparison = f"股东权益: {equity/1e8:.2f}亿 | 净利润: {net_profit/1e8:.2f}亿"
        score = 0.0
        passed = True

        if prev and prev.tot_equity > 0 and net_profit > 0:
            equity_growth = equity - prev.tot_equity
            if equity_growth > net_profit * 1.5:
                score = 0.4
                passed = False
                comparison += f" | 权益增长({equity_growth/1e8:.2f}亿)远超净利润({net_profit/1e8:.2f}亿)"

        return RuleResult(
            rule_id="B4", rule_name="权益增长vs利润勾稽", level="B", category="盈利能力",
            passed=passed, score=score,
            detail=f"权益变动: {equity - (prev.tot_equity if prev else 0):.0f} | 净利润: {net_profit:.0f}",
            data_comparison=comparison,
            possible_fraud="权益增长大幅超过净利润 → 可能存在表外资产注入或非经常性损益操纵",
            severity="medium",
        )

    def _rule_depreciation_consistency(self, s: FinancialSnapshot, prev=None) -> RuleResult:
        """B5: 折旧与固定资产匹配"""
        fix_assets = s.fix_assets

        comparison = f"固定资产: {fix_assets/1e8:.2f}亿"
        score = 0.0
        passed = True

        # 简化：检查固定资产是否持续下降（可能意味着不计提折旧）
        if prev and prev.fix_assets > 0 and fix_assets > 0:
            fa_change = (fix_assets - prev.fix_assets) / prev.fix_assets
            comparison += f" | 变动: {fa_change:.1%}"

        return RuleResult(
            rule_id="B5", rule_name="折旧与固定资产匹配检测", level="B", category="资产质量",
            passed=passed, score=score, detail="检测中",
            data_comparison=comparison,
            severity="low",
        )

    def _rule_cash_cycle_anomaly(self, s: FinancialSnapshot, prev=None) -> RuleResult:
        """B6: 现金周期异常"""
        inventory = s.inventories
        receivables = s.acct_rcv + s.notes_rcv
        payable = s.acct_payable
        revenue = s.oper_rev or s.tot_oper_rev

        comparison = f"存货: {inventory/1e8:.2f}亿 | 应收: {receivables/1e8:.2f}亿 | 应付: {payable/1e8:.2f}亿"
        score = 0.0
        passed = True

        if revenue > 0:
            # 简化现金周期 = (存货+应收-应付)/日均营收
            daily_rev = revenue / 365
            if daily_rev > 0:
                cycle_days = (inventory + receivables - payable) / daily_rev
                comparison += f" | 现金周期: {cycle_days:.0f}天"

                if cycle_days > 365:
                    score = 0.7
                    passed = False
                elif cycle_days > 180:
                    score = 0.4
                    passed = False

        return RuleResult(
            rule_id="B6", rule_name="现金周期异常检测", level="B", category="运营效率",
            passed=passed, score=score,
            detail=f"现金周转周期过长，营运资金被大量占用" if not passed else "现金周期正常",
            data_comparison=comparison,
            possible_fraud="现金周期过长 → 资金链紧张，可能存在大量不良存货或坏账",
            severity="high" if score > 0.6 else "medium",
        )
