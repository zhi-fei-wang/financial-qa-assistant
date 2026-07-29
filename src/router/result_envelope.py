"""
ResultEnvelope — 统一的 Skill 输出信封 (启发 2: 结果携带证据)

每个 Skill 的 execute() 返回的 data dict 统一包含 evidence / conclusion / confidence / limitations。
LLM 在生成回复时优先消费 evidence 数组，确保每个结论都有数据支撑。

用法:
    from .result_envelope import ResultEnvelope, Evidence
    env = ResultEnvelope(
        conclusion="检测到2条财务异常",
        evidence=[
            Evidence(claim="存货激增35%但营收仅增5%", source="rule_engine", data={"inventory_growth": 0.35, "revenue_growth": 0.05}),
            Evidence(claim="经营性现金流为负", source="dataset", data={"net_cashflow": -5.2e8}),
        ],
        confidence=0.88,
        limitations=["仅覆盖最近2个季度", "行业基准数据缺失"],
    )
    return env.to_dict()
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Evidence:
    """单条证据"""
    claim: str                          # 一句话声明（如"存货增速35%远超营收增速5%"）
    source: str                         # 数据来源: "dataset" | "graph" | "rule_engine" | "llm" | "user"
    data: Dict[str, Any] = field(default_factory=dict)  # 支撑数据（具体数值）
    reference: str = ""                 # 引用路径（如表名/文件名/节点ID）
    confidence: float = 1.0             # 本条证据的置信度 [0, 1]


@dataclass
class ResultEnvelope:
    """
    统一的 Skill 输出信封。

    字段:
        conclusion:  1-3句话的结论 (供 LLM 快速消费)
        evidence:    证据列表 (每条 conclusion 都应有对应 evidence)
        confidence:  整体置信度 [0, 1]
        limitations: 已知的数据/方法局限性
        metadata:    额外元信息 (skill_name, execution_time等)
    """
    conclusion: str = ""
    evidence: List[Evidence] = field(default_factory=list)
    confidence: float = 0.0
    limitations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conclusion": self.conclusion,
            "evidence": [
                {
                    "claim": e.claim,
                    "source": e.source,
                    "data": e.data,
                    "reference": e.reference,
                    "confidence": e.confidence,
                }
                for e in self.evidence
            ],
            "confidence": self.confidence,
            "limitations": self.limitations,
            "metadata": self.metadata,
        }

    def render_for_llm(self) -> str:
        """渲染为 LLM 友好的 Markdown 文本"""
        parts = []
        if self.conclusion:
            parts.append(f"**结论**: {self.conclusion}")
        if self.evidence:
            parts.append(f"**证据** ({len(self.evidence)}条):")
            for i, e in enumerate(self.evidence, 1):
                parts.append(f"  {i}. {e.claim} [来源: {e.source}]")
                if e.data:
                    parts.append(f"     数据: {e.data}")
        if self.confidence > 0:
            parts.append(f"**置信度**: {self.confidence:.0%}")
        if self.limitations:
            parts.append(f"**局限**: {'; '.join(self.limitations)}")
        return "\n".join(parts)

    @classmethod
    def from_skill_result(cls, original_data: Dict[str, Any], skill_name: str = "") -> Dict[str, Any]:
        """
        将已有 Skill 的返回数据包装为带 envelope 的格式。
        向后兼容：不影响已有 rendered 等字段。
        """
        envelope = cls(
            conclusion=original_data.get("conclusion", ""),
            evidence=original_data.get("evidence", []),
            confidence=original_data.get("confidence", 0.0),
            limitations=original_data.get("limitations", []),
            metadata={"skill_name": skill_name, **original_data.get("metadata", {})},
        )
        result = dict(original_data)  # 保留所有原有字段
        result["envelope"] = envelope.to_dict()
        result["envelope_rendered"] = envelope.render_for_llm()
        return result


def make_envelope(
    conclusion: str,
    evidence_items: List[Dict[str, Any]],
    confidence: float = 0.85,
    limitations: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    快捷构造器 — 用于 Skill 内部快速生成 envelope。

    Usage:
        from .result_envelope import make_envelope
        return {
            **original_result,
            **make_envelope(
                "检测到3条异常",
                [{"claim": "存货激增", "source": "rule_engine", "data": {...}}],
                confidence=0.88,
            )
        }
    """
    env = ResultEnvelope(
        conclusion=conclusion,
        evidence=[Evidence(**e) for e in evidence_items],
        confidence=confidence,
        limitations=limitations or [],
        metadata=metadata or {},
    )
    return {
        "envelope": env.to_dict(),
        "envelope_rendered": env.render_for_llm(),
    }
