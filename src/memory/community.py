"""
社区发现与摘要管理
基于 GraphRAG 思想：对图谱做社区聚类 → LLM 生成层级摘要 → 用于跨轮次检索
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm import get_llm_client
from ..llm.prompts import COMMUNITY_SUMMARY_PROMPT
from .knowledge_graph import KnowledgeGraph


@dataclass
class CommunitySummary:
    """社区摘要数据结构"""
    community_id: int
    level: int = 0                # 0=实体级, 1=社区级, 2=会话级
    community_name: str = ""
    summary: str = ""
    key_entities: List[str] = field(default_factory=list)
    dialogue_phase: str = ""      # 早期 / 中期 / 后期
    topic_category: str = ""      # 话题分类
    updated_at: float = 0.0


class CommunityManager:
    """
    社区发现与摘要管理器。
    周期性触发社区聚类，并为每个社区生成 LLM 摘要。
    """

    def __init__(self, use_llm: bool = True):
        self.llm = get_llm_client() if use_llm else None
        self.use_llm = use_llm
        self.communities: Dict[int, CommunitySummary] = {}  # community_id → summary
        self._last_node_count = 0  # 上次聚类时的节点数

    def run(self, graph: KnowledgeGraph, force: bool = False) -> Dict[int, List[str]]:
        """
        执行社区发现（Louvain 算法）。

        Args:
            graph: 知识图谱
            force: 强制重新聚类

        Returns:
            {community_id: [node_ids]}
        """
        min_nodes = 5  # 最小节点数阈值
        if graph.node_count < min_nodes:
            return {-1: list(graph.G.nodes())}

        new_nodes = graph.node_count - self._last_node_count
        if not force and new_nodes < 10 and self._last_node_count > 0:
            return {}  # 新节点太少，不触发聚类

        self._last_node_count = graph.node_count
        return graph.detect_communities()

    def regenerate_summaries(
        self, graph: KnowledgeGraph, communities: Optional[Dict[int, List[str]]] = None
    ) -> List[CommunitySummary]:
        """
        为每个社区重新生成 LLM 摘要。

        Args:
            graph: 知识图谱
            communities: 社区划分结果（None 时自动聚类）

        Returns:
            新生成的社区摘要列表
        """
        if communities is None:
            communities = graph.detect_communities()

        new_summaries = []
        for comm_id, node_ids in communities.items():
            if len(node_ids) < 3:
                continue  # 太小的社区跳过

            summary = self._generate_community_summary(graph, comm_id, node_ids)
            if summary:
                self.communities[comm_id] = summary
                new_summaries.append(summary)

        return new_summaries

    def _generate_community_summary(
        self, graph: KnowledgeGraph, comm_id: int, node_ids: List[str]
    ) -> Optional[CommunitySummary]:
        """为一个社区生成 LLM 摘要"""
        # 收集社区信息
        node_info = []
        for nid in node_ids:
            data = graph.G.nodes.get(nid, {})
            ntype = data.get("type", "Unknown")
            name = data.get("name", nid)
            summary = data.get("summary", "")
            node_info.append(f"  [{ntype}] {name}: {summary}")

        # 收集内部边信息
        edge_info = []
        for u, v, d in graph.G.edges(data=True):
            if u in node_ids and v in node_ids:
                edge_info.append(f"  {u} --[{d.get('type', 'RELATED')}]--> {v}")

        if not self.use_llm or not self.llm:
            # 无 LLM fallback：基于统计的摘要
            return CommunitySummary(
                community_id=comm_id,
                community_name=f"Topic_{comm_id}",
                summary=f"社区 {comm_id}：包含 {len(node_ids)} 个节点",
                key_entities=self._extract_key_entities_heuristic(graph, node_ids),
            )

        try:
            prompt = COMMUNITY_SUMMARY_PROMPT.format(
                community_nodes="\n".join(node_info[:30]),   # 截断防止超长
                community_edges="\n".join(edge_info[:20]),
            )
            result = self.llm.chat_with_json_output(
                user_prompt=prompt, temperature=0.1, max_retries=1
            )
            return CommunitySummary(
                community_id=comm_id,
                community_name=result.get("community_name", f"Topic_{comm_id}"),
                summary=result.get("summary", ""),
                key_entities=result.get("key_entities", []),
                dialogue_phase=result.get("dialogue_phase", ""),
                topic_category=result.get("topic_category", ""),
            )
        except Exception as e:
            print(f"[CommunityManager] LLM summary failed for community {comm_id}: {e}")
            return CommunitySummary(
                community_id=comm_id,
                community_name=f"Topic_{comm_id}",
                summary=f"社区 {comm_id}：包含 {len(node_ids)} 个节点",
                key_entities=self._extract_key_entities_heuristic(graph, node_ids),
            )

    def get_community_summary(self, community_id: int) -> Optional[CommunitySummary]:
        """获取某个社区的摘要"""
        return self.communities.get(community_id)

    def get_all_summaries(self) -> List[CommunitySummary]:
        """获取所有社区摘要"""
        return list(self.communities.values())

    def get_context_text(self, max_communities: int = 5) -> str:
        """将所有社区摘要组装为文本格式（注入 LLM 上下文）"""
        summaries = sorted(self.communities.values(),
                          key=lambda s: len(s.key_entities), reverse=True)
        lines = []
        for s in summaries[:max_communities]:
            lines.append(f"[{s.community_name}] {s.summary}")
            if s.key_entities:
                lines.append(f"  关键实体: {', '.join(s.key_entities)}")
        return "\n".join(lines)

    @staticmethod
    def _extract_key_entities_heuristic(graph: KnowledgeGraph, node_ids: List[str]) -> List[str]:
        """启发式提取社区关键实体（基于度中心性）"""
        node_degrees = []
        for nid in node_ids:
            ntype = graph.G.nodes.get(nid, {}).get("type", "")
            if ntype == "ConversationTurn":
                continue  # Turn 节点不作为关键实体
            degree = graph.G.degree(nid)
            name = graph.G.nodes.get(nid, {}).get("name", nid)
            node_degrees.append((nid, name, degree))

        node_degrees.sort(key=lambda x: x[2], reverse=True)
        return [f"{name}({nid})" for nid, name, _ in node_degrees[:5]]
