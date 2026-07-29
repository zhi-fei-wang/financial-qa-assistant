"""
事件簇因果推理模块 (P3: 赛题缺口 — 事件发展的因果/时序逻辑)

为 EventClusterer 的输出添加因果关系判断:
  1. 时序分析: A 事件在 B 之前 → 可能因果
  2. 实体重叠: 共享关键实体 → 可能关联
  3. 类型链条: 违规→减持→股价异动 → 因果链
  4. LLM 因果判断: 对高置信度候选对做因果推断

用法:
    reasoner = CausalReasoner(use_llm=True)
    causal_links = reasoner.analyze(clusters, timeline)
    # → [{"cause": cluster_A, "effect": cluster_B, "confidence": 0.85, "reasoning": "..."}]
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..llm import get_llm_client


# 事件类型的因果链模式
CAUSAL_CHAINS = [
    # (前因类型, 后果类型, 置信度)
    ("监管处罚", "股权变动", 0.6),       # 处罚 → 大股东减持
    ("股权质押", "风险提示", 0.7),       # 高质押 → 风险
    ("ST/退市", "股权变动", 0.8),        # ST → 股东撤离
    ("业绩下滑", "风险提示", 0.65),       # 业绩 → 风险
    ("违规调查", "人事变动", 0.55),       # 调查 → 高管离职
    ("监管处罚", "ST/退市", 0.5),         # 处罚 → ST
    ("收购兼并", "人事变动", 0.5),         # 收购 → 管理层变更
    ("借贷担保", "风险提示", 0.4),        # 担保 → 风险
]


@dataclass
class CausalLink:
    """因果关系链"""
    cause_id: int
    effect_id: int
    cause_name: str
    effect_name: str
    confidence: float               # [0, 1]
    reasoning: str                  # 因果推理说明
    evidence: List[str] = field(default_factory=list)  # 支撑证据
    type: str = "temporal"          # "temporal" | "entity_overlap" | "causal_chain" | "llm"


class CausalReasoner:
    """事件簇因果推理器"""

    def __init__(self, use_llm: bool = True):
        self.llm = get_llm_client() if use_llm else None
        self.use_llm = use_llm

    def analyze(
        self,
        clusters: List[Any],   # List[EventCluster]
        timeline: Any = None,  # Optional[Timeline]
    ) -> List[Dict[str, Any]]:
        """
        分析事件簇之间的因果关系。

        Returns:
            [{"cause": ..., "effect": ..., "confidence": ..., "reasoning": ..., "type": ...}, ...]
        """
        if len(clusters) < 2:
            return []

        links = []

        # Step 1: 时序分析 (基于日期)
        temporal_links = self._temporal_analysis(clusters)
        links.extend(temporal_links)

        # Step 2: 实体重叠分析
        entity_links = self._entity_overlap_analysis(clusters)
        links.extend(entity_links)

        # Step 3: 因果链模式匹配
        chain_links = self._causal_chain_match(clusters)
        links.extend(chain_links)

        # Step 4: LLM 因果判断 (高置信度候选)
        if self.use_llm and self.llm and links:
            verified_links = self._llm_verify(links, clusters)
            return verified_links

        return links

    def _temporal_analysis(self, clusters: List[Any]) -> List[Dict[str, Any]]:
        """时序分析: A 的结束日期在 B 的开始日期之前 → 可能因果关系"""
        links = []
        for i, ci in enumerate(clusters):
            for j, cj in enumerate(clusters):
                if i >= j:
                    continue
                try:
                    ci_end = ci.end_date
                    cj_start = cj.start_date
                    if ci_end and cj_start and ci_end <= cj_start:
                        links.append({
                            "cause_id": ci.cluster_id,
                            "effect_id": cj.cluster_id,
                            "cause_name": ci.name,
                            "effect_name": cj.name,
                            "confidence": 0.4,
                            "reasoning": f"「{ci.name}」({ci_end})先于「{cj.name}」({cj_start})发生，可能存在时序因果",
                            "type": "temporal",
                        })
                except (AttributeError, TypeError):
                    continue
        return links

    def _entity_overlap_analysis(self, clusters: List[Any]) -> List[Dict[str, Any]]:
        """实体重叠分析: 共享关键实体 → 可能关联"""
        links = []
        for i, ci in enumerate(clusters):
            for j, cj in enumerate(clusters):
                if i >= j:
                    continue
                try:
                    ci_ents = set(ci.key_entities)
                    cj_ents = set(cj.key_entities)
                    overlap = ci_ents & cj_ents
                    if len(overlap) >= 2:
                        links.append({
                            "cause_id": ci.cluster_id,
                            "effect_id": cj.cluster_id,
                            "cause_name": ci.name,
                            "effect_name": cj.name,
                            "confidence": min(0.7, 0.3 + len(overlap) * 0.15),
                            "reasoning": f"共享关键实体: {', '.join(list(overlap)[:3])}，可能存在关联",
                            "type": "entity_overlap",
                        })
                except (AttributeError, TypeError):
                    continue
        return links

    def _causal_chain_match(self, clusters: List[Any]) -> List[Dict[str, Any]]:
        """因果链模式匹配"""
        links = []
        for i, ci in enumerate(clusters):
            for j, cj in enumerate(clusters):
                if i >= j:
                    continue
                try:
                    ci_cat = getattr(ci, "category", "")
                    cj_cat = getattr(cj, "category", "")
                    for cause_type, effect_type, conf in CAUSAL_CHAINS:
                        if cause_type in ci_cat and effect_type in cj_cat:
                            links.append({
                                "cause_id": ci.cluster_id,
                                "effect_id": cj.cluster_id,
                                "cause_name": ci.name,
                                "effect_name": cj.name,
                                "confidence": conf,
                                "reasoning": f"因果链模式: {cause_type} → {effect_type}",
                                "type": "causal_chain",
                            })
                except (AttributeError, TypeError):
                    continue
        return links

    def _llm_verify(
        self, candidates: List[Dict], clusters: List[Any]
    ) -> List[Dict[str, Any]]:
        """LLM 验证 + 提升置信度"""
        verified = []
        for link in candidates[:10]:  # 最多验证10对
            prompt = (
                f"判断两个金融事件之间是否存在因果关系:\n"
                f"事件A [{link['cause_name']}]: {link.get('reasoning', '')}\n"
                f"事件B [{link['effect_name']}]\n"
                f'输出 JSON: {{"causal": true/false, "confidence": 0.0-1.0, "reasoning": "..."}}'
            )
            try:
                result = self.llm.chat_with_json_output(prompt, temperature=0.0)
                if result.get("causal"):
                    link["confidence"] = max(link["confidence"], result.get("confidence", 0.5))
                    link["reasoning"] = result.get("reasoning", link["reasoning"])
                    link["type"] = link.get("type", "temporal") + "+llm"
                    verified.append(link)
                elif link["confidence"] >= 0.65:  # 即使 LLM 否认，高模式置信度的仍保留
                    verified.append(link)
            except Exception:
                if link["confidence"] >= 0.5:
                    verified.append(link)

        verified.sort(key=lambda x: x["confidence"], reverse=True)
        return verified

    def format_causal_report(
        self, links: List[Dict[str, Any]], max_links: int = 5
    ) -> str:
        """生成因果关系报告"""
        if not links:
            return "（未发现显著的因果关系）"

        parts = ["### 事件因果关系分析\n"]
        for i, link in enumerate(links[:max_links], 1):
            conf = link["confidence"]
            conf_label = "强" if conf >= 0.7 else ("中" if conf >= 0.5 else "弱")
            parts.append(
                f"{i}. **{link['cause_name']}** → **{link['effect_name']}** "
                f"({conf_label}因果, 置信度 {conf:.0%})\n"
                f"   推理: {link['reasoning']}\n"
            )

        return "\n".join(parts)
