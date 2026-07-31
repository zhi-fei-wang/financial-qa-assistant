"""财务异象甄别 Skill — Task 3，注册到 Task1 Router"""

from typing import Any, Dict, List, Optional

from ..finance.data_extractor import FinancialDataExtractor
from ..finance.report_generator import ReportGenerator
from ..finance.risk_scorer import RiskScorer


# 全局懒加载
_extractor: Optional[FinancialDataExtractor] = None
_scorer: Optional[RiskScorer] = None
_report_gen: Optional[ReportGenerator] = None


def _get_extractor() -> FinancialDataExtractor:
    global _extractor
    if _extractor is None:
        _extractor = FinancialDataExtractor()
    return _extractor


def _get_scorer() -> RiskScorer:
    global _scorer
    if _scorer is None:
        _scorer = RiskScorer()
    return _scorer


def _get_report_gen() -> ReportGenerator:
    global _report_gen
    if _report_gen is None:
        _report_gen = ReportGenerator(use_llm=True)
    return _report_gen


class FinancialAnomalySkill:
    """
    财务异象智能甄别 Skill — 注册到 Task1 Router。
    对目标股票的最新财报执行跨科目勾稽演算，生成多维风险评分和结构化研判报告。
    """

    name = "financial_anomaly_check"
    description = (
        "财务异象智能甄别：对目标股票执行跨科目勾稽演算，"
        "检测存货/营收比、现金流/利润悖离、异常财务费用等14项规则，"
        "生成多维风险评分和结构化研判报告。预警F1-Score ≥ 85%。"
    )
    required_params = ["stock_code"]
    optional_params = ["report_period", "include_llm_analysis"]

    @staticmethod
    def execute(params: Dict[str, Any]) -> Dict[str, Any]:
        stock_code = params.get("stock_code", "")
        report_period = params.get("report_period", None)
        include_llm = params.get("include_llm_analysis", True)

        extractor = _get_extractor()
        scorer = _get_scorer()

        # 获取财务快照（当前期 + 上期）
        snapshot = extractor.get_snapshot(stock_code, report_period)
        if not snapshot or snapshot.tot_assets == 0:
            return {
                "success": False,
                "error": f"未找到 {stock_code} 的财务数据",
                "stock_code": stock_code,
            }

        # 尝试获取上一期
        prev_snapshot = None
        multi = extractor.get_multi_period(stock_code, periods=2)
        if len(multi) >= 2:
            prev_snapshot = multi[1]

        # 执行规则检测
        rule_results = scorer.engine.evaluate(snapshot, prev_snapshot)

        # 综合评分
        risk_score = scorer.evaluate(snapshot, prev_snapshot)

        # 生成报告
        report_gen = _get_report_gen()
        report_gen.use_llm = include_llm
        full_report = report_gen.generate(risk_score, rule_results, snapshot, prev_snapshot)
        short_summary = report_gen.generate_short(risk_score)

        # 提取关键预警
        failed = [r for r in rule_results if not r.passed]
        critical_warnings = [
            {
                "rule": r.rule_name,
                "level": r.level,
                "detail": r.detail,
                "data_comparison": r.data_comparison,
                "possible_fraud": r.possible_fraud,
                "severity": r.severity,
            }
            for r in rule_results if not r.passed
        ]

        # 启发 2: 构造 ResultEnvelope
        env_evidence = [
            {
                "claim": w["detail"],
                "source": "rule_engine",
                "data": w.get("data_comparison", {}),
            }
            for w in critical_warnings[:5]
        ]
        env = {
            "conclusion": f"{stock_code} 财务风险评分 {risk_score.overall_score:.0f}/100 ({risk_score.risk_level})，{len(failed)}/{len(rule_results)} 条规则触发预警",
            "evidence": env_evidence,
            "confidence": 0.92,
            "limitations": [
                "基于公开财报数据，无法检测未披露的财务造假",
                "评分权重为通用模型，未考虑行业特殊性",
            ],
            "metadata": {"skill_name": "financial_anomaly_check", "report_period": snapshot.report_period},
        }

        return {
            "success": True,
            "stock_code": stock_code,
            "report_period": snapshot.report_period,
            "risk_score": risk_score.to_dict(),
            "total_rules": len(rule_results),
            "failed_rules": len(failed),
            "critical_warnings": critical_warnings,
            "short_summary": short_summary,
            "full_report": full_report,
            "source": "rule_engine",
            "envelope": env,
            "envelope_rendered": (
                f"**结论**: {env['conclusion']}\n"
                + "\n".join(f"- {e['claim']}" for e in env_evidence)
                + f"\n**置信度**: {env['confidence']:.0%}"
            ),
        }


class MultiPeriodAnalysisSkill:
    """
    多期对比分析 Skill — 对多期财报做趋势分析。
    """

    name = "multi_period_analysis"
    description = "对目标股票的多期财报做趋势分析，检测指标恶化趋势。"
    required_params = ["stock_code"]
    optional_params = ["periods"]

    @staticmethod
    def execute(params: Dict[str, Any]) -> Dict[str, Any]:
        stock_code = params.get("stock_code", "")
        periods = int(params.get("periods", 5))

        extractor = _get_extractor()
        scorer = _get_scorer()

        snapshots = extractor.get_multi_period(stock_code, periods)
        if not snapshots:
            return {"success": False, "error": f"未找到 {stock_code} 的多期数据"}

        scores = scorer.compare(snapshots)

        # 趋势分析
        score_trend = [s.overall_score for s in scores]
        is_deteriorating = len(score_trend) >= 2 and score_trend[0] < score_trend[-1]

        rendered = f"## {stock_code} 多期财务趋势 ({len(scores)}期)\n\n"
        rendered += "| 报告期 | 综合评分 | 风险等级 | 盈利能力 | 资产质量 | 现金流 |\n"
        rendered += "|--------|----------|----------|----------|----------|--------|\n"
        for s in scores:
            rendered += f"| {s.report_period} | {s.overall_score:.0f}/100 | {s.risk_level} | {s.profitability_score:.0f} | {s.asset_quality_score:.0f} | {s.cashflow_quality_score:.0f} |\n"

        if is_deteriorating:
            rendered += f"\n⚠️ 趋势预警：综合评分从 {score_trend[-1]:.0f} 下降至 {score_trend[0]:.0f}，财务健康度呈恶化趋势。"

        return {
            "success": True,
            "stock_code": stock_code,
            "periods_analyzed": len(scores),
            "score_trend": score_trend,
            "is_deteriorating": is_deteriorating,
            "rendered": rendered,
            "scores": [s.to_dict() for s in scores],
            "source": "rule_engine",
        }


# =========================================================================
# BaseTool 包装器
# =========================================================================

from .base import BaseTool, register_tool_class


@register_tool_class
class FinancialAnomalyTool(BaseTool):
    """财务异象智能甄别。"""
    name = FinancialAnomalySkill.name
    description = FinancialAnomalySkill.description
    required_params = list(FinancialAnomalySkill.required_params)
    optional_params = list(FinancialAnomalySkill.optional_params)
    intent_match = ["FINANCIAL_ANALYSIS"]
    sub_intent = "ANOMALY_CHECK"
    param_schema = {
        "stock_code": {"description": "6位股票代码"},
        "report_period": {"description": "报告期，如2024Q1"},
        "include_llm_analysis": {"description": "是否包含AI深度分析，默认true"},
    }
    routing_hint = "用户问造假/排雷/风险评分/勾稽 → financial_anomaly_check"
    trigger_keywords = [
        "造假", "排雷", "异象", "勾稽", "疑点", "风险评分",
        "财务造假", "粉饰", "虚增", "欺诈",
    ]
    max_retries = 1
    timeout_sec = 15

    def execute(self, params, data_loader=None):
        return FinancialAnomalySkill.execute(params)


@register_tool_class
class MultiPeriodAnalysisTool(BaseTool):
    """多期财务趋势分析。"""
    name = MultiPeriodAnalysisSkill.name
    description = MultiPeriodAnalysisSkill.description
    required_params = list(MultiPeriodAnalysisSkill.required_params)
    optional_params = list(MultiPeriodAnalysisSkill.optional_params)
    intent_match = ["FINANCIAL_ANALYSIS"]
    sub_intent = "COMPARISON"
    param_schema = {
        "stock_code": {"description": "6位股票代码"},
        "periods": {"description": "分析期数，默认5期"},
    }
    routing_hint = "用户对比/趋势/近几年 → multi_period_analysis"
    trigger_keywords = ["对比", "趋势", "近几年", "逐年", "变化趋势"]
    max_retries = 0
    timeout_sec = 10

    def execute(self, params, data_loader=None):
        return MultiPeriodAnalysisSkill.execute(params)


# 导出
TASK3_SKILLS = [
    FinancialAnomalySkill,
    MultiPeriodAnalysisSkill,
]
