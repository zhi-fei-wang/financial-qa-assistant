"""
知识图谱管理 (GraphRAG 核心)
基于 NetworkX 的对话实体关系图，支持增量写入、邻居查询、社区发现。
"""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx


class KnowledgeGraph:
    """
    对话知识图谱：存储实体节点和关系边。
    设计灵感来自 GraphRAG，将每一轮对话建模为图节点和边。

    节点类型:
      - Stock: 股票
      - Person: 人物
      - Indicator: 金融指标
      - Event: 事件
      - Organization: 机构
      - Report: 财报/研报
      - ConversationTurn: 对话轮次
      - Topic: 话题簇
      - Fact: 原子化事实 (Priority 4 新增)

    边类型:
      - MENTIONS: 轮次→实体(提及)
      - NEXT: 轮次→轮次(对话顺序)
      - BELONGS_TO: 事实→轮次 | 轮次→话题
      - RELATED_TO: 实体↔实体
      - COMPARES_WITH: 实体↔实体(对比)
      - AFFECTED_BY: 实体→事件
    """

    def __init__(self, backend: str = "networkx"):
        self.backend = backend
        self.G = nx.DiGraph()  # 有向图
        self._entity_index: Dict[str, str] = {}  # entity_name → node_id (快速查找)

    # ---- 数据写入 ----

    def upsert_entities(self, entities: List[Dict]) -> List[str]:
        """增量添加/更新实体节点"""
        node_ids = []
        for entity in entities:
            node_id = entity.get("id", "")
            if not node_id:
                continue
            self.G.add_node(node_id, **entity)
            self._entity_index[entity.get("name", "").lower()] = node_id
            node_ids.append(node_id)
        return node_ids

    def upsert_relations(self, relations: List[Dict]):
        """增量添加/更新关系边"""
        for rel in relations:
            source = rel.get("source", "")
            target = rel.get("target", "")
            rel_type = rel.get("type", "RELATED_TO")
            if source and target:
                if source not in self.G:
                    self.G.add_node(source, type="Entity", name=source)
                if target not in self.G:
                    self.G.add_node(target, type="Entity", name=target)
                self.G.add_edge(source, target, type=rel_type, **{k: v for k, v in rel.items() if k not in ("source", "target", "type")})

    def add_turn_node(self, turn_id: str, summary: str, entities: List[str],
                      intent: str = "", timestamp: float = 0.0,
                      metadata: Dict[str, Any] = None) -> str:
        """添加对话轮次节点并连接实体。metadata 可包含 source_type 等标注。"""
        node_attrs = dict(type="ConversationTurn", summary=summary,
                          intent=intent, timestamp=timestamp)
        if metadata:
            node_attrs.update(metadata)
        self.G.add_node(turn_id, **node_attrs)

        # 连接实体
        for entity_id in entities:
            if entity_id in self.G:
                self.G.add_edge(turn_id, entity_id, type="MENTIONS")

        # 连接上一轮
        turn_nodes = self._get_turn_nodes()
        if len(turn_nodes) >= 2:
            prev_turn = turn_nodes[-2]  # 倒数第二个是上一轮
            self.G.add_edge(prev_turn, turn_id, type="NEXT")

        return turn_id

    def add_entity_relation(self, entity_a: str, entity_b: str, rel_type: str):
        """直接添加实体间关系"""
        if entity_a in self.G and entity_b in self.G:
            self.G.add_edge(entity_a, entity_b, type=rel_type)

    # ---- 图查询 ----

    def get_entity_history(self, entity_id: str) -> List[Dict]:
        """获取某个实体在对话中的历史提及（通过 Turn 节点）"""
        if entity_id not in self.G:
            return []

        results = []
        # 入边：哪些 Turn 提到了该实体
        for pred in self.G.predecessors(entity_id):
            pred_node = self.G.nodes.get(pred, {})
            if pred_node.get("type") == "ConversationTurn":
                results.append({
                    "turn_id": pred,
                    "summary": pred_node.get("summary", ""),
                    "timestamp": pred_node.get("timestamp", 0),
                })

        # 按时间排序
        results.sort(key=lambda x: x.get("timestamp", 0))
        return results

    def get_neighbors(self, entity_id: str, depth: int = 1) -> List[Dict]:
        """获取实体的邻居节点（支持多跳）"""
        if entity_id not in self.G:
            return []

        neighbors = []
        current_layer = {entity_id}
        visited = {entity_id}

        for d in range(depth):
            next_layer = set()
            for node in current_layer:
                for neighbor in set(self.G.predecessors(node)) | set(self.G.successors(node)):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_layer.add(neighbor)
                        node_data = self.G.nodes.get(neighbor, {})
                        edge_data = self.G.get_edge_data(node, neighbor) or self.G.get_edge_data(neighbor, node) or {}
                        neighbors.append({
                            "node_id": neighbor,
                            "node_type": node_data.get("type", "Unknown"),
                            "name": node_data.get("name", neighbor),
                            "hop_distance": d + 1,
                            "relation_type": edge_data.get("type", ""),
                        })
            current_layer = next_layer

        return neighbors

    def get_related_turns(self, entity_ids: List[str], max_depth: int = 2) -> List[Dict]:
        """获取与一组实体相关的所有对话轮次（图谱遍历）"""
        all_turns = {}

        for entity_id in entity_ids:
            if entity_id not in self.G:
                continue

            # BFS 遍历，收集 Turn 节点
            visited = {entity_id}
            queue = [(entity_id, 0)]

            while queue:
                node, depth = queue.pop(0)
                if depth > max_depth:
                    continue

                # 如果是 Turn 节点，收集
                node_data = self.G.nodes.get(node, {})
                if node_data.get("type") == "ConversationTurn":
                    if node not in all_turns:
                        all_turns[node] = {
                            "turn_id": node,
                            "content": node_data.get("summary", ""),
                            "hop_distance": depth,
                            "intent": node_data.get("intent", ""),
                        }

                # 探索邻居
                for neighbor in set(self.G.predecessors(node)) | set(self.G.successors(node)):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

        # 按距离排序
        return sorted(all_turns.values(), key=lambda x: x["hop_distance"])

    def search_entities(self, query: str) -> List[str]:
        """模糊搜索实体节点（按名称）"""
        query_lower = query.lower()
        matches = []
        for node, data in self.G.nodes(data=True):
            name = str(data.get("name", "")).lower()
            if query_lower in name or query_lower in node.lower():
                matches.append(node)
        return matches

    # ---- Fact 节点 (Priority 4: 结构化记忆) ----

    def add_fact_node(
        self,
        fact_id: str,
        fact_text: str,
        entities: List[str] = None,
        turn_id: str = "",
        category: str = "data",
        confidence: float = 0.85,
        timestamp: float = 0.0,
    ) -> str:
        """
        添加 Fact 节点并自动连接到 Turn 和实体。

        Args:
            fact_id: 事实唯一 ID
            fact_text: 事实陈述文本
            entities: 涉及的实体名称列表
            turn_id: 所属对话轮次
            category: 类别 (data/analysis/query/preference)
            confidence: 置信度 [0, 1]
            timestamp: 时间戳

        Returns:
            fact_id
        """
        self.G.add_node(fact_id, type="Fact", fact_text=fact_text,
                        category=category, confidence=confidence,
                        timestamp=timestamp)

        # 连接到 Turn 节点（BELONGS_TO 边）
        if turn_id and turn_id in self.G:
            self.G.add_edge(fact_id, turn_id, type="BELONGS_TO")

        # 连接到实体节点（RELATED_TO 边）
        entities = entities or []
        for entity_name in entities:
            # 尝试在图中找到对应实体节点
            entity_nodes = [
                n for n, d in self.G.nodes(data=True)
                if d.get("name", "").lower() == entity_name.lower()
                or n.lower() == entity_name.lower()
            ]
            for en in entity_nodes[:3]:  # 最多连接3个
                self.G.add_edge(fact_id, en, type="RELATED_TO")

        return fact_id

    def get_facts(
        self,
        entity_name: Optional[str] = None,
        turn_id: Optional[str] = None,
        category: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 20,
    ) -> List[Dict]:
        """
        查询 Fact 节点。

        Args:
            entity_name: 按实体名称筛选
            turn_id: 按轮次筛选
            category: 按类别筛选
            min_confidence: 最低置信度
            limit: 返回数量上限

        Returns:
            Fact 节点列表（按时间戳降序）
        """
        facts = []
        for node, data in self.G.nodes(data=True):
            if data.get("type") != "Fact":
                continue

            # 筛选条件
            if entity_name and entity_name.lower() not in str(data.get("name", "")).lower():
                continue
            if category and data.get("category") != category:
                continue
            if data.get("confidence", 0) < min_confidence:
                continue

            # 检查是否关联到指定 turn
            if turn_id:
                connected = False
                for pred in self.G.predecessors(node):
                    if pred == turn_id:
                        connected = True
                        break
                for succ in self.G.successors(node):
                    if succ == turn_id:
                        connected = True
                        break
                if not connected:
                    continue

            facts.append({
                "fact_id": node,
                "fact_text": data.get("fact_text", ""),
                "category": data.get("category", ""),
                "confidence": data.get("confidence", 0),
                "timestamp": data.get("timestamp", 0),
                "entities": [
                    self.G.nodes.get(n, {}).get("name", n)
                    for n in set(self.G.predecessors(node)) | set(self.G.successors(node))
                    if self.G.nodes.get(n, {}).get("type") not in ("Fact", "ConversationTurn")
                ][:5],
            })

        facts.sort(key=lambda f: f["timestamp"], reverse=True)
        return facts[:limit]

    def get_recent_facts(self, limit: int = 10) -> List[Dict]:
        """获取最近的 Fact 节点"""
        return self.get_facts(limit=limit, min_confidence=0.5)

    def search_facts(self, query: str, limit: int = 10) -> List[Dict]:
        """按文本关键词搜索 Fact"""
        query_lower = query.lower()
        matching = []
        for node, data in self.G.nodes(data=True):
            if data.get("type") != "Fact":
                continue
            fact_text = str(data.get("fact_text", "")).lower()
            if query_lower in fact_text:
                matching.append({
                    "fact_id": node,
                    "fact_text": data.get("fact_text", ""),
                    "category": data.get("category", ""),
                    "confidence": data.get("confidence", 0),
                    "timestamp": data.get("timestamp", 0),
                })
        matching.sort(key=lambda f: f["timestamp"], reverse=True)
        return matching[:limit]

    # ---- 社区发现 ----

    def detect_communities(self) -> Dict[int, List[str]]:
        """
        使用 Louvain 算法(NetworkX内置)进行社区发现。
        返回 {community_id: [node_ids]}
        """
        # 转为无向图进行社区发现
        undirected = self.G.to_undirected()

        # 过滤：只保留有足够连接的节点
        if undirected.number_of_nodes() < 5:
            return {-1: list(undirected.nodes())}

        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(undirected, seed=42)
            return {i: list(comm) for i, comm in enumerate(communities)}
        except ImportError:
            # Fallback: connected components
            communities = list(nx.connected_components(undirected))
            return {i: list(comm) for i, comm in enumerate(communities)}

    def get_community_info(self, community_id: int = -1) -> Dict[str, Any]:
        """获取社区的结构化信息（节点列表 + 内部边）"""
        communities = self.detect_communities()
        if community_id not in communities:
            return {"nodes": [], "edges": [], "summary": ""}

        nodes = communities[community_id]
        subgraph = self.G.subgraph(nodes)
        edges = list(subgraph.edges(data=True))

        # 统计节点类型分布
        type_counts = defaultdict(int)
        for n in nodes:
            ntype = self.G.nodes.get(n, {}).get("type", "Unknown")
            type_counts[ntype] += 1

        return {
            "community_id": community_id,
            "nodes": [(n, self.G.nodes.get(n, {})) for n in nodes],
            "edges": [(u, v, d) for u, v, d in edges],
            "size": len(nodes),
            "type_distribution": dict(type_counts),
        }

    # ---- 工具方法 ----

    def _get_turn_nodes(self) -> List[str]:
        """获取所有 Turn 节点（按时间戳排序）"""
        turn_nodes = [
            (n, d.get("timestamp", 0))
            for n, d in self.G.nodes(data=True)
            if d.get("type") == "ConversationTurn"
        ]
        turn_nodes.sort(key=lambda x: x[1])
        return [n for n, _ in turn_nodes]

    @property
    def node_count(self) -> int:
        return self.G.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self.G.number_of_edges()

    @property
    def turn_count(self) -> int:
        return len(self._get_turn_nodes())

    def to_dict(self) -> Dict:
        """导出为字典（用于序列化）"""
        return {
            "nodes": [(n, d) for n, d in self.G.nodes(data=True)],
            "edges": [(u, v, d) for u, v, d in self.G.edges(data=True)],
        }

    def summary(self) -> str:
        """打印图谱摘要"""
        type_counts = Counter(d.get('type', '?') for _, d in self.G.nodes(data=True))
        return (
            f"KnowledgeGraph: {self.node_count} nodes, {self.edge_count} edges, "
            f"{self.turn_count} turns | types={dict(type_counts)}"
        )
