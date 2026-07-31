"""
记忆管理总控 (MemoryManager)
这是任务一的核心类，统筹三级记忆系统的所有操作。

数据流:
  用户输入 Qₜ → 实体抽取 → 图谱检索 → 向量检索 → 记忆融合 → 注入 LLM 上下文
                                                                    ↓
  用户收到回复 ← [路由+生成] ←──────────────────────────────────┘
"""

from typing import Any, Dict, List, Optional, Tuple

from ..llm import get_llm_client
from ..llm.prompts import TURN_SUMMARY_PROMPT
from .community import CommunityManager, CommunitySummary
from .entity_extractor import EntityExtractor
from .fact_extractor import FactExtractor
from .hybrid_retrieval import HybridRetriever, MemoryContext
from .knowledge_graph import KnowledgeGraph
from .signal_fusion import SignalFusion
from .working_memory import TurnRecord, WorkingMemory


class MemoryManager:
    """
    记忆管理总控。

    对外暴露的核心接口：
      - process_turn() — 每轮对话后调用，更新所有记忆
      - retrieve() — 检索相关记忆
      - get_context_for_llm() — 组装注入 LLM 的上下文
    """

    def __init__(
        self,
        graph_backend: str = "networkx",
        vector_backend: Optional[str] = None,  # chromadb | faiss | None(暂不使用)
        use_llm: bool = True,
        max_working_turns: int = 20,
        use_fact_extraction: bool = True,  # Priority 4: 启用结构化 Fact 提取
    ):
        self.llm = get_llm_client() if use_llm else None
        self.use_llm = use_llm
        self.use_fact_extraction = use_fact_extraction

        # 三个核心组件
        self.working = WorkingMemory(max_turns=max_working_turns)
        self.graph = KnowledgeGraph(backend=graph_backend)
        self.extractor = EntityExtractor(use_llm=use_llm)
        self.fact_extractor = FactExtractor(use_llm=use_llm)  # Priority 4 新增
        self.community = CommunityManager(use_llm=use_llm)

        # 多信号融合引擎 (Priority 1 新增)
        self.signal_fusion = SignalFusion()

        # 向量存储（可选，在没有本地 Embedding 模型时先跳过）
        self.vector_store = None
        if vector_backend:
            self._init_vector_store(vector_backend)

        # 混合检索器（传入 SignalFusion）
        self.retriever = HybridRetriever(
            self.graph, self.vector_store, self.community,
            signal_fusion=self.signal_fusion,
        )

        # 统计
        self.turn_count = 0
        self.total_entities_extracted = 0
        self.total_facts_extracted = 0

    def _init_vector_store(self, backend: str):
        """初始化向量存储（延迟导入）"""
        try:
            if backend == "chromadb":
                import chromadb
                self.vector_store = chromadb.Client()
                print("[MemoryManager] ChromaDB vector store initialized")
            elif backend == "faiss":
                print("[MemoryManager] FAISS not yet implemented, skipping vector store")
        except ImportError:
            print(f"[MemoryManager] {backend} not available, vector search disabled")

    # ---- 核心入口 ----

    def add_turn(
        self,
        user_query: str,
        agent_response: str = "",
        tool_results: Optional[List[Dict]] = None,
        intent: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        每轮对话后调用 — 核心入口 (Priority 4 升级: Fact 提取替代 LLM 摘要)。

        工作流程：
        1. 实体+事实联合抽取（一次 LLM 调用完成）
        2. 图谱增量更新（实体 + 关系 + Fact 节点）
        3. 工作记忆更新
        4. SignalFusion 索引更新
        5. 周期性社区聚类
        """
        self.turn_count += 1
        turn_id = f"turn_{self.turn_count}"
        import time

        # === Step 1: 实体 + 事实联合抽取 ===
        if self.use_fact_extraction and self.use_llm:
            # Priority 4: 一次 LLM 调用完成实体+事实+摘要
            extraction = self.fact_extractor.extract(
                user_query, agent_response, tool_results or []
            )
            entities = extraction.get("entities", [])
            relations = extraction.get("relations", [])
            facts = extraction.get("facts", [])
            turn_summary = extraction.get("turn_summary", user_query[:100])
            self.total_facts_extracted += len(facts)
        else:
            # 回退到原有流程：实体抽取 + LLM 摘要
            entities, relations = self.extractor.extract(user_query, agent_response)
            facts = []
            turn_summary = self._summarize_turn(user_query, agent_response, tool_results or [])

        entity_ids = [e["id"] for e in entities]
        self.total_entities_extracted += len(entities)

        # === Step 2: 图谱增量更新 ===
        self.graph.upsert_entities(entities)
        self.graph.upsert_relations(relations)

        # Step 2b: Fact 节点写入 (Priority 4)
        for fact in facts:
            self.graph.add_fact_node(
                fact_id=fact.get("id", ""),
                fact_text=fact.get("text", ""),
                entities=fact.get("entities", []),
                turn_id=turn_id,
                category=fact.get("category", "data"),
                confidence=fact.get("confidence", 0.85),
                timestamp=time.time(),
            )

        # === Step 3: 添加 Turn 节点到图谱 ===
        self.graph.add_turn_node(
            turn_id=turn_id,
            summary=turn_summary,
            entities=entity_ids,
            intent=intent,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        # === Step 4: 工作记忆更新 ===
        self.working.add(
            user_query=user_query,
            agent_response=agent_response,
            tool_results=tool_results or [],
            entities=entity_ids,
            intent=intent,
            summary=turn_summary,
            metadata=metadata or {},
        )

        # === Step 5: SignalFusion 索引增量更新 ===
        self.signal_fusion.update_index(
            doc_id=turn_id,
            text=turn_summary,
            entities=[e.get("name", "") for e in entities],
        )

        # === Step 6: 检查溢出 → 降级老轮次 ===
        overflow = self.working.get_overflow()
        for old_turn in overflow:
            self._archive_turn(old_turn)

        # === Step 7: 周期性社区聚类 ===
        if self.turn_count % 5 == 0 and self.graph.node_count >= 20:
            communities = self.community.run(self.graph)
            if communities:
                self.community.regenerate_summaries(self.graph, communities)

        return turn_id

    # ---- 检索接口 ----

    def retrieve(self, query: str, top_k: int = 10) -> List[MemoryContext]:
        """
        检索相关记忆 — 在生成回复前调用。

        Args:
            query: 用户当前输入
            top_k: 返回的记忆片段数
        """
        # 先从 query 提取实体（用于图谱检索）
        entities, _ = self.extractor.extract(query)
        entity_ids = [e["id"] for e in entities]

        return self.retriever.hybrid_search(
            query=query,
            query_entities=entity_ids,
            top_k=top_k,
        )

    def get_context_for_llm(self, query: str, max_turns: int = 10) -> str:
        """
        组装注入 LLM 的完整上下文文本。

        Returns:
            格式化的上下文字符串，包含：
            - 混合检索结果
            - 社区摘要
            - 最近工作记忆
        """
        parts = []

        # 1. 检索结果
        retrieved = self.retrieve(query, top_k=8)
        parts.append(self.retriever.format_as_text(retrieved))

        # 2. 社区摘要
        community_text = self.community.get_context_text(max_communities=5)
        if community_text:
            parts.append(f"\n## 对话话题结构\n{community_text}")

        # 3. 最近工作记忆
        recent_text = self.working.to_context_text(max_turns=max_turns)
        if recent_text:
            parts.append(f"\n## 最近对话\n{recent_text}")

        return "\n".join(parts)

    # ---- 内部方法 ----

    def _summarize_turn(self, user_query: str, agent_response: str,
                        tool_results: List[Dict]) -> str:
        """生成单轮对话的简短摘要"""
        if self.use_llm and self.llm:
            try:
                tools_str = str(tool_results)[:500] if tool_results else "无"
                prompt = TURN_SUMMARY_PROMPT.format(
                    user_query=user_query[:300],
                    agent_response=agent_response[:500],
                    tool_results=tools_str,
                )
                summary = self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.0,
                )
                return summary.strip()
            except Exception:
                pass

        # Fallback: 截断文本作为摘要
        return user_query[:100] + ("..." if len(user_query) > 100 else "")

    def _archive_turn(self, turn: TurnRecord):
        """归档旧轮次：文本降级为摘要，保留图结构"""
        # 图结构已保留在 KnowledgeGraph 中
        # 这里可以进一步压缩：将 Turn 节点的完整文本替换为摘要
        if turn.turn_id in self.graph.G:
            self.graph.G.nodes[turn.turn_id]["summary"] = turn.summary
            # 可选：清除完整文本以节省内存
            self.graph.G.nodes[turn.turn_id].pop("full_text", None)

    # ---- 查询接口 ----

    def get_entity_context(self, entity_name: str) -> Dict[str, Any]:
        """获取某个实体在对话中的完整上下文"""
        entity_id = f"stock_{entity_name}"  # 简单推理
        if entity_id not in self.graph.G:
            # 尝试搜索
            matches = self.graph.search_entities(entity_name)
            if not matches:
                return {"found": False, "history": [], "neighbors": []}
            entity_id = matches[0]

        return {
            "found": True,
            "entity_id": entity_id,
            "entity_data": dict(self.graph.G.nodes.get(entity_id, {})),
            "history": self.graph.get_entity_history(entity_id),
            "neighbors": self.graph.get_neighbors(entity_id, depth=2),
        }

    def summary(self) -> str:
        """系统状态摘要"""
        return (
            f"MemoryManager:\n"
            f"  - Working: {self.working.turn_count} turns (max {self.working.max_turns})\n"
            f"  - Graph: {self.graph.node_count} nodes, {self.graph.edge_count} edges\n"
            f"  - Communities: {len(self.community.communities)} clusters\n"
            f"  - Entities extracted: {self.total_entities_extracted}\n"
            f"  - Vector store: {'enabled' if self.vector_store else 'disabled'}"
        )
