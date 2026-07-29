"""
多维风险评分引擎
将14条规则的检测结果聚合为多维风险评分，生成总体"财务健康度"评分。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .data_extractor import FinancialSnapshot
from .rule_engine import FinancialRuleEngine, RuleResult


@dataclass
class RiskScore:
    """多维风险评分结果"""
    stock_code: str
    report_period: str

    # 分维度评分 (0-100, 越高越健康)
    profitability_score: float = 100.0     # 盈利能力
    asset_quality_score: float = 100.0     # 资产质量
    cashflow_quality_score: float = 100.0  # 现金流质量
    solvency_score: float = 100.0          # 偿债能力
    operational_score: float = 100.0       # 运营效率

    # 综合评分
    overall_score: float = 100.0           # 综合财务健康度 (0-100)
    risk_level: str = "low"                # low / medium / high / critical

    # 详情
    total_rules: int = 0
    failed_rules: int = 0
    critical_warnings: List[RuleResult] = field(default_factory=list)
    dimension_breakdown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "stock_code": self.stock_code,
            "report_period": self.report_period,
            "overall_score": round(self.overall_score, 1),
            "risk_level": self.risk_level,
            "dimensions": {
                "盈利能力": round(self.profitability_score, 1),
                "资产质量": round(self.asset_quality_score, 1),
                "现金流质量": round(self.cashflow_quality_score, 1),
                "偿债能力": round(self.solvency_score, 1),
                "运营效率": round(self.operational_score, 1),
            },
            "total_rules": self.total_rules,
            "failed_rules": self.failed_rules,
            "critical_warnings": len(self.critical_warnings),
        }


class RiskScorer:
    """
    多维风险评分引擎。

    将规则检测结果按维度聚合，计算加权评分。
    """

    # 维度 → 权重 (综合评分权重)
    DIMENSION_WEIGHTS = {
        "盈利能力": 0.25,
        "资产质量": 0.25,
        "现金流质量": 0.25,
        "偿债能力": 0.15,
        "运营效率": 0.10,
    }

    # category → dimension mapping
    CATEGORY_DIMENSION = {
        "盈利能力": "profitability_score",
        "资产质量": "asset_quality_score",
        "现金流质量": "cashflow_quality_score",
        "偿债能力": "solvency_score",
        "运营效率": "operational_score",
    }

    def __init__(self, engine: Optional[FinancialRuleEngine] = None):
        self.engine = engine or FinancialRuleEngine()

    def evaluate(
        self,
        snapshot: FinancialSnapshot,
        prev_snapshot: Optional[FinancialSnapshot] = None,
    ) -> RiskScore:
        """
        对财务快照执行完整评分。

        Args:
            snapshot: 当前期财务快照
            prev_snapshot: 上一期快照

        Returns:
            RiskScore
        """
        # 执行所有规则
        rule_results = self.engine.evaluate(snapshot, prev_snapshot)

        # 按维度聚合
        dim_scores: Dict[str, List[float]] = {}
        for rr in rule_results:
            dim_key = self.CATEGORY_DIMENSION.get(rr.category, "asset_quality_score")
            if dim_key not in dim_scores:
                dim_scores[dim_key] = []
            # 转换: risk_score (0-1) → health_score (100-0)
            dim_scores[dim_key].append(rr.score)

        # 计算各维度健康分 (取平均风险分 → 转换为健康分)
        dimension_health = {}
        for dim_key in self.CATEGORY_DIMENSION.values():
            scores = dim_scores.get(dim_key, [])
            if scores:
                avg_risk = sum(scores) / len(scores)
                dimension_health[dim_key] = round(100 * (1 - avg_risk), 1)
            else:
                dimension_health[dim_key] = 100.0

        # 计算综合健康分（加权）
        overall = 0.0
        for cat_name, weight in self.DIMENSION_WEIGHTS.items():
            dim_key = self.CATEGORY_DIMENSION.get(cat_name, "")
            overall += dimension_health.get(dim_key, 100.0) * weight
        overall = round(overall, 1)

        # 确定风险等级
        if overall >= 80:
            risk_level = "low"
        elif overall >= 60:
            risk_level = "medium"
        elif overall >= 40:
            risk_level = "high"
        else:
            risk_level = "critical"

        # 收集严重预警
        critical = [r for r in rule_results if r.severity in ("critical", "high") and not r.passed]
        failed = [r for r in rule_results if not r.passed]

        return RiskScore(
            stock_code=snapshot.stock_code,
            report_period=snapshot.report_period,
            profitability_score=dimension_health.get("profitability_score", 100),
            asset_quality_score=dimension_health.get("asset_quality_score", 100),
            cashflow_quality_score=dimension_health.get("cashflow_quality_score", 100),
            solvency_score=dimension_health.get("solvency_score", 100),
            operational_score=dimension_health.get("operational_score", 100),
            overall_score=overall,
            risk_level=risk_level,
            total_rules=len(rule_results),
            failed_rules=len(failed),
            critical_warnings=critical,
            dimension_breakdown={
                dim: {
                    "health_score": dimension_health.get(dim, 100),
                    "rules_count": len(dim_scores.get(dim, [])),
                    "failed_count": len([s for s in dim_scores.get(dim, []) if s > 0.4]),
                }
                for dim in self.CATEGORY_DIMENSION.values()
            },
        )

    def compare(self, snapshots: List[FinancialSnapshot]) -> List[RiskScore]:
        """多期对比评分"""
        scores = []
        for i, snap in enumerate(snapshots):
            prev = snapshots[i + 1] if i + 1 < len(snapshots) else None
            scores.append(self.evaluate(snap, prev))
        return scores
