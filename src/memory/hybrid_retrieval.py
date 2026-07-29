"""
混合检索：图谱遍历 + 向量检索 + 社区摘要 + 多信号融合 四路融合
这是记忆模块最核心的检索接口，替代传统的纯向量检索。

Priority 1 升级: 集成 SignalFusion 多信号融合评分引擎
  Final Score = α × graph_bfs + β × bm25 + γ × entity_match + δ × vector
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .community import CommunityManager, CommunitySummary
from .knowledge_graph import KnowledgeGraph
from .signal_fusion import SignalFusion


@dataclass
class MemoryContext:
    """检索返回的记忆片段"""
    source: str           # "graph" | "vector" | "community" | "fact"
    turn_id: str
    content: str
    score: float          # 0-1 相关性分数
    hop_distance: int = 0
    entities_matched: List[str] = field(default_factory=list)


class HybridRetriever:
    """
    混合检索器：融合四种检索策略，SignalFusion 统一评分后重排序。

    检索优先级（从高到低）：
    1. 图谱遍历 — 确保结构相关的记忆不丢失
    2. 社区摘要 — 确保话题上下文完整
    3. Fact 事实检索 — 精确原子化事实 (Priority 4)
    4. 向量检索 — 兜底语义相关
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        vector_store=None,  # Optional[VectorStore] — lazy import
        community: Optional[CommunityManager] = None,
        signal_fusion: Optional[SignalFusion] = None,
    ):
        self.graph = graph
        self.vector_store = vector_store
        self.community = community or CommunityManager(use_llm=False)
        self.sf = signal_fusion or SignalFusion()
        self._fusion_ready = False  # 首次检索时延迟构建 BM25 索引

    def hybrid_search(
        self,
        query: str,
        query_entities: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[MemoryContext]:
        """
        混合检索主入口。

        Args:
            query: 用户查询文本
            query_entities: 从 query 中已提取的实体 ID 列表
            top_k: 返回的记忆片段数

        Returns:
            排序后的记忆上下文列表
        """
        # 延迟构建 BM25 索引（首次检索时）
        self._ensure_fusion_index()

        all_contexts: List[MemoryContext] = []

        # 路 1: 图谱遍历检索（实体 → 邻居 → Turn）
        if query_entities:
            graph_results = self._graph_search(query_entities)
            all_contexts.extend(graph_results)

        # 路 2: 社区摘要检索（BM25 增强）
        community_results = self._community_search(query)
        all_contexts.extend(community_results)

        # 路 3: Fact 事实检索 (Priority 4 新增)
        fact_results = self._fact_search(query, query_entities)
        all_contexts.extend(fact_results)

        # 路 4: 向量检索（如果可用）
        if self.vector_store:
            vector_results = self._vector_search(query)
            all_contexts.extend(vector_results)

        # 多信号融合排序（Priority 1）
        merged = self._fusion_rerank(query, all_contexts, query_entities, top_k)
        return merged

    def _ensure_fusion_index(self):
        """延迟构建 SignalFusion 的 BM25+实体索引"""
        if self._fusion_ready:
            return

        # 收集 Turn 摘要作为 BM25 文档
        docs = {}
        entity_docs = {}
        for node, data in self.graph.G.nodes(data=True):
            if data.get("type") == "ConversationTurn":
                doc_id = node
                summary = data.get("summary", "")
                if summary:
                    docs[doc_id] = summary

                    # 从 MENTIONS 边收集关联实体
                    for pred in self.graph.G.predecessors(node):
                        pred_data = self.graph.G.nodes.get(pred, {})
                        ent_name = pred_data.get("name", pred)
                        if ent_name not in entity_docs:
                            entity_docs[ent_name] = []
                        if doc_id not in entity_docs[ent_name]:
                            entity_docs[ent_name].append(doc_id)

        if docs:
            self.sf.build_index(docs, entity_docs)
        self._fusion_ready = True

    def _graph_search(self, entity_ids: List[str], max_depth: int = 2) -> List[MemoryContext]:
        """图谱遍历检索：基于实体找关联的对话轮次（Priority 1: SignalFusion 评分）"""
        related_turns = self.graph.get_related_turns(entity_ids, max_depth=max_depth)
        contexts = []
        for turn in related_turns:
            # 原启发式得分 → 作为 graph_score 输入 SignalFusion
            graph_score = max(0.35, 1.0 - turn["hop_distance"] * 0.25)
            contexts.append(MemoryContext(
                source="graph",
                turn_id=turn["turn_id"],
                content=turn["content"],
                score=graph_score,
                hop_distance=turn["hop_distance"],
                entities_matched=[e for e in entity_ids if e.lower() in str(turn).lower()],
            ))
        return contexts

    def _community_search(self, query: str) -> List[MemoryContext]:
        """社区摘要检索：BM25 增强的关键词匹配 (Priority 1 升级)"""
        if not self.community or not self.community.communities:
            return []

        contexts = []
        query_lower = query.lower()

        for comm_id, summary in self.community.communities.items():
            summary_text = summary.summary + " " + " ".join(summary.key_entities)

            # BM25 关键词评分
            bm25_score = 0.0
            if self._fusion_ready:
                bm25_score = self.sf.bm25.score(query, f"community_{comm_id}")
            else:
                # fallback: 简单重叠计数
                overlap = sum(1 for kw in query_lower.split() if kw.lower() in summary_text.lower())
                bm25_score = min(0.8, overlap * 0.2)

            entity_hit = any(
                any(kw.lower() in ke.lower() for kw in query_lower.split())
                for ke in summary.key_entities
            )

            score = max(bm25_score, 0.3 + (entity_hit and 0.1 or 0))
            if score > 0.2:  # 过滤低相关的
                contexts.append(MemoryContext(
                    source="community",
                    turn_id=f"community_{comm_id}",
                    content=f"[{summary.community_name}] {summary.summary}",
                    score=min(0.85, score),
                    entities_matched=summary.key_entities,
                ))
        return contexts

    def _fact_search(self, query: str, query_entities: Optional[List[str]] = None) -> List[MemoryContext]:
        """Fact 事实检索 (Priority 4 新增): 精确原子化事实匹配"""
        contexts = []
        facts = self.graph.search_facts(query, limit=15)

        for f in facts:
            # 实体匹配 boost
            entity_hit = 0
            if query_entities:
                fact_text_lower = f["fact_text"].lower()
                entity_hit = sum(
                    1 for e in query_entities
                    if e.lower() in fact_text_lower
                )

            # BM25 评分
            bm25_score = self.sf.bm25.score(query, f["fact_id"]) if self._fusion_ready else 0.0
            entity_score = min(1.0, entity_hit * 0.3) if query_entities else 0.0
            confidence = f.get("confidence", 0.5)

            score = max(bm25_score, entity_score, confidence * 0.5)
            if score > 0.15:
                contexts.append(MemoryContext(
                    source="fact",
                    turn_id=f["fact_id"],
                    content=f"[{f['category']}] {f['fact_text']}",
                    score=min(0.9, score),
                    entities_matched=[],
                ))
        return contexts

    def _vector_search(self, query: str) -> List[MemoryContext]:
        """向量检索：语义相似度搜索"""
        if not self.vector_store:
            return []

        try:
            results = self.vector_store.search(query, top_k=5)
            contexts = []
            for i, result in enumerate(results):
                score = max(0.1, 1.0 - i * 0.15)
                contexts.append(MemoryContext(
                    source="vector",
                    turn_id=result.get("id", ""),
                    content=result.get("content", result.get("text", "")),
                    score=score,
                ))
            return contexts
        except Exception as e:
            print(f"[HybridRetriever] Vector search failed: {e}")
            return []

    def _fusion_rerank(
        self,
        query: str,
        contexts: List[MemoryContext],
        query_entities: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[MemoryContext]:
        """多信号融合重排序 (Priority 1 提升)"""
        # 按 turn_id 去重
        seen: Dict[str, MemoryContext] = {}
        for ctx in contexts:
            key = ctx.turn_id
            if key in seen:
                # 保留分数更高的
                if ctx.score > seen[key].score:
                    seen[key] = ctx
                # 图检索比事实检索优先级更高
                source_prio = {"graph": 0, "community": 1, "fact": 2, "vector": 3}
                if source_prio.get(ctx.source, 9) < source_prio.get(seen[key].source, 9):
                    seen[key] = ctx
            else:
                seen[key] = ctx

        if not self._fusion_ready:
            # 无索引时回退：按来源优先级 + 分数排序
            source_priority = {"graph": 0, "community": 1, "fact": 2, "vector": 3}
            ranked = sorted(
                seen.values(),
                key=lambda c: (source_priority.get(c.source, 9), -c.score),
            )
            return ranked[:top_k]

        # 有索引时：使用 SignalFusion 融合评分
        doc_scores = {ctx.turn_id: ctx.score for ctx in seen.values()}
        entity_names = query_entities if query_entities else []

        fused = self.sf.fuse_batch(query, doc_scores, entity_names)

        # 更新每个 context 的分数为融合分数
        for ctx in seen.values():
            ctx.score = fused.get(ctx.turn_id, ctx.score)

        ranked = sorted(seen.values(), key=lambda c: -c.score)
        return ranked[:top_k]

    # 保留旧方法名作为别名，向后兼容
    def _merge_and_rerank(self, contexts: List[MemoryContext], top_k: int) -> List[MemoryContext]:
        """向后兼容别名 → _fusion_rerank"""
        return self._fusion_rerank("", contexts, None, top_k)

    def format_as_text(self, contexts: List[MemoryContext]) -> str:
        """将检索结果格式化为 LLM 可读的上下文文本"""
        if not contexts:
            return "（无相关历史记忆）"

        lines = ["## 记忆检索结果\n"]
        for i, ctx in enumerate(contexts, 1):
            source_label = {"graph": "📊图谱", "community": "📝话题摘要", "fact": "💡事实", "vector": "🔍语义"}.get(ctx.source, "❓")
            lines.append(f"{i}. [{source_label}] {ctx.content}")
            if ctx.entities_matched:
                lines.append(f"   关联实体: {', '.join(ctx.entities_matched[:5])}")
        return "\n".join(lines)
