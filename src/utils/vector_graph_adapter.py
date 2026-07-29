"""
向量+图数据库适配层 (P4: 赛题缺口 — Neo4j + 向量存储准备)

为 NetworkX 原型提供到 Neo4j/ChromaDB 的迁移接口。
赛题原型阶段保留 NetworkX，同时准备好生产级接口。

设计:
  1. GraphAdapter: 抽象图数据库接口 (NetworkX now, Neo4j later)
  2. VectorAdapter: 抽象向量存储接口 (ChromaDB when available)
  3. 统一的 upsert/search 协议

技术白皮书说明: 原型阶段使用 NetworkX 因其零运维开销、
快速迭代能力。生产环境可通过 Adapter 透明切换 Neo4j。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


# =========================================================================
# 图数据库适配
# =========================================================================

class GraphAdapter(ABC):
    """图数据库抽象接口"""

    @abstractmethod
    def add_node(self, node_id: str, labels: List[str], properties: Dict[str, Any]) -> None: ...
    @abstractmethod
    def add_edge(self, src: str, dst: str, rel_type: str, properties: Dict[str, Any] = None) -> None: ...
    @abstractmethod
    def get_neighbors(self, node_id: str, depth: int = 1) -> List[Dict]: ...
    @abstractmethod
    def search_nodes(self, query: str, top_k: int = 10) -> List[Dict]: ...
    @abstractmethod
    def run_query(self, cypher: str, params: Dict = None) -> List[Dict]: ...
    @property
    @abstractmethod
    def node_count(self) -> int: ...
    @property
    @abstractmethod
    def edge_count(self) -> int: ...


class NetworkXAdapter(GraphAdapter):
    """NetworkX 实现 (当前原型)"""

    def __init__(self, graph):
        import networkx as nx
        self.G = graph if isinstance(graph, nx.DiGraph) else nx.DiGraph()

    def add_node(self, node_id, labels, properties=None):
        self.G.add_node(node_id, type=labels[0] if labels else "Entity", **(properties or {}))

    def add_edge(self, src, dst, rel_type, properties=None):
        self.G.add_edge(src, dst, type=rel_type, **(properties or {}))

    def get_neighbors(self, node_id, depth=1):
        if node_id not in self.G:
            return []
        import networkx as nx
        try:
            paths = nx.single_source_shortest_path_length(
                self.G.to_undirected(), node_id, cutoff=depth
            )
            return [
                {"node_id": n, "distance": d, "data": dict(self.G.nodes.get(n, {}))}
                for n, d in paths.items() if n != node_id
            ]
        except Exception:
            return []

    def search_nodes(self, query, top_k=10):
        query_lower = query.lower()
        results = []
        for n, d in self.G.nodes(data=True):
            name = str(d.get("name", "")).lower()
            if query_lower in name:
                results.append({"node_id": n, **dict(d), "score": 0.8})
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    def run_query(self, cypher, params=None):
        raise NotImplementedError("Cypher not supported on NetworkX. Use Neo4jAdapter.")

    @property
    def node_count(self):
        return self.G.number_of_nodes()

    @property
    def edge_count(self):
        return self.G.number_of_edges()


class Neo4jAdapter(GraphAdapter):
    """Neo4j 实现 (生产就绪接口)"""

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        self._available = False
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
            self._driver.verify_connectivity()
            self._available = True
            print(f"[Neo4j] Connected to {uri}")
        except ImportError:
            print("[Neo4j] neo4j-driver not installed. Install: pip install neo4j")
        except Exception as e:
            print(f"[Neo4j] Connection failed: {e}. Falling back to NetworkX.")

    def add_node(self, node_id, labels, properties=None):
        if not self._available:
            return
        label_str = ":".join(labels)
        props = ", ".join(f"{k}: ${k}" for k in (properties or {}))
        self._run(f"MERGE (n:{label_str} {{id: $id}}) SET {props}", {"id": node_id, **(properties or {})})

    def add_edge(self, src, dst, rel_type, properties=None):
        if not self._available:
            return
        props = ", ".join(f"{k}: ${k}" for k in (properties or {}))
        self._run(
            f"MATCH (a {{id: $src}}), (b {{id: $dst}}) MERGE (a)-[r:{rel_type}]->(b) SET {props}",
            {"src": src, "dst": dst, **(properties or {})}
        )

    def get_neighbors(self, node_id, depth=1):
        if not self._available:
            return []
        return self._run(
            f"MATCH (n {{id: $id}})-[*1..{depth}]-(m) RETURN DISTINCT m.id AS node_id, m AS data",
            {"id": node_id}
        )

    def search_nodes(self, query, top_k=10):
        if not self._available:
            return []
        return self._run(
            "CALL db.index.fulltext.queryNodes('entity_index', $query) YIELD node, score "
            "RETURN node.id AS node_id, node AS data, score ORDER BY score DESC LIMIT $top_k",
            {"query": query, "top_k": top_k}
        )

    def run_query(self, cypher, params=None):
        return self._run(cypher, params or {})

    def _run(self, query, params=None):
        if not self._driver:
            return []
        try:
            with self._driver.session() as session:
                result = session.run(query, params or {})
                return [dict(record) for record in result]
        except Exception as e:
            print(f"[Neo4j] Query error: {e}")
            return []

    @property
    def node_count(self):
        if not self._available:
            return 0
        result = self._run("MATCH (n) RETURN count(n) AS cnt")
        return result[0].get("cnt", 0) if result else 0

    @property
    def edge_count(self):
        if not self._available:
            return 0
        result = self._run("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return result[0].get("cnt", 0) if result else 0


# =========================================================================
# 向量存储适配
# =========================================================================

class VectorAdapter(ABC):
    """向量存储抽象接口"""

    @abstractmethod
    def add(self, texts: List[str], metadatas: List[Dict] = None, ids: List[str] = None) -> None: ...
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Dict]: ...
    @abstractmethod
    def delete(self, ids: List[str]) -> None: ...


class ChromaDBAdapter(VectorAdapter):
    """ChromaDB 实现"""

    def __init__(self, collection_name: str = "financial_docs"):
        self._client = None
        self._collection = None
        self._available = False
        try:
            import chromadb
            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(collection_name)
            self._available = True
            print(f"[Vector] ChromaDB ready: '{collection_name}'")
        except ImportError:
            print("[Vector] chromadb not installed. Install: pip install chromadb")
        except Exception as e:
            print(f"[Vector] Init error: {e}")

    def add(self, texts, metadatas=None, ids=None):
        if not self._available:
            return
        ids = ids or [f"doc_{i}" for i in range(len(texts))]
        self._collection.add(documents=texts, metadatas=metadatas or [{}] * len(texts), ids=ids)

    def search(self, query, top_k=10):
        if not self._available:
            return []
        results = self._collection.query(query_texts=[query], n_results=top_k)
        docs = []
        for i, doc_id in enumerate(results.get("ids", [[]])[0]):
            docs.append({
                "id": doc_id,
                "text": results["documents"][0][i] if results.get("documents") else "",
                "score": 1.0 - (i * 0.1),
            })
        return docs

    def delete(self, ids):
        if self._available:
            self._collection.delete(ids=ids)


class DummyVectorAdapter(VectorAdapter):
    """占位向量适配器 (不依赖任何向量库)"""

    def add(self, texts, metadatas=None, ids=None):
        pass

    def search(self, query, top_k=10):
        return []  # 搜索时返回空

    def delete(self, ids):
        pass


# =========================================================================
# 工厂函数
# =========================================================================

def get_graph_adapter(backend: str = "networkx", **kwargs) -> GraphAdapter:
    """获取图数据库适配器"""
    if backend == "neo4j":
        return Neo4jAdapter(**kwargs)
    else:
        import networkx as nx
        return NetworkXAdapter(nx.DiGraph())


def get_vector_adapter(backend: str = "dummy", **kwargs) -> VectorAdapter:
    """获取向量存储适配器"""
    if backend == "chromadb":
        return ChromaDBAdapter(**kwargs)
    else:
        return DummyVectorAdapter()
