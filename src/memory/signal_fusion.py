"""
多信号融合评分引擎 (Priority 1)
灵感来自 Mem0 v3.0 的 multi-signal retrieval + LightRAG 的 mix 模式。

核心公式: Final Score = α × graph_bfs + β × bm25 + γ × entity_match + δ × vector
默认权重: α=0.35, β=0.30, γ=0.25, δ=0.10

BM25 为轻量自实现（无外部依赖），向量信号为预留接口。
"""

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


class BM25Scorer:
    """
    轻量级 BM25 实现（无需外部依赖）。
    用于对 Turn 摘要/社区摘要/实体名称做关键词检索评分。

    BM25 公式: score(D, Q) = Σ IDF(qi) × (tf(qi, D) × (k1 + 1)) / (tf(qi, D) + k1 × (1 - b + b × |D|/avgdl))
    默认: k1=1.5, b=0.75
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._documents: Dict[str, str] = {}          # doc_id → text
        self._tokens: Dict[str, List[str]] = {}        # doc_id → tokenized list
        self._doc_freq: Dict[str, int] = defaultdict(int)  # term → document frequency
        self._avgdl: float = 0.0
        self._N: int = 0

    def index(self, documents: Dict[str, str]):
        """
        构建 BM25 索引。

        Args:
            documents: {doc_id: text} — 例如 {turn_id: turn_summary, community_id: community_summary}
        """
        self._documents = documents
        self._tokens = {}
        self._doc_freq = defaultdict(int)
        total_len = 0

        for doc_id, text in documents.items():
            tokens = self._tokenize(text)
            self._tokens[doc_id] = tokens
            total_len += len(tokens)
            unique_terms = set(tokens)
            for term in unique_terms:
                self._doc_freq[term] += 1

        self._N = len(documents)
        self._avgdl = total_len / max(self._N, 1)

    def add_document(self, doc_id: str, text: str):
        """增量添加单个文档到索引"""
        self._documents[doc_id] = text
        tokens = self._tokenize(text)
        self._tokens[doc_id] = tokens

        unique_terms = set(tokens)
        for term in unique_terms:
            self._doc_freq[term] += 1

        self._N += 1
        total_len = sum(len(t) for t in self._tokens.values())
        self._avgdl = total_len / max(self._N, 1)

    def score(self, query: str, doc_id: str) -> float:
        """计算单个 query 对单个文档的 BM25 分数"""
        if doc_id not in self._tokens:
            return 0.0

        query_tokens = self._tokenize(query)
        doc_tokens = self._tokens[doc_id]
        doc_len = len(doc_tokens)
        score_sum = 0.0

        # 计算词频
        tf_counter = defaultdict(int)
        for t in doc_tokens:
            tf_counter[t] += 1

        for qt in query_tokens:
            tf = tf_counter.get(qt, 0)
            if tf == 0:
                continue
            df = self._doc_freq.get(qt, 0)
            if df == 0:
                continue

            # IDF
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)

            # BM25 term score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self._avgdl))
            score_sum += idf * (numerator / denominator)

        return score_sum

    def score_batch(self, query: str, doc_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """批量计算 query 对所有（或指定）文档的 BM25 分数"""
        target_ids = doc_ids or list(self._tokens.keys())
        return {did: self.score(query, did) for did in target_ids}

    def top_k(self, query: str, k: int = 10, doc_ids: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """返回 BM25 分数最高的 top-K 文档"""
        scores = self.score_batch(query, doc_ids)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """中文友好的分词：按字+按词（简单 2-gram）混合"""
        text = text.lower().strip()
        # 英文/数字：保留原词
        # 中文：2-gram 字符级 token
        tokens = []
        # 提取英文词和数字
        eng_tokens = re.findall(r'[a-zA-Z0-9_]+', text)
        tokens.extend(eng_tokens)
        # 中文 2-gram
        chinese_chars = re.sub(r'[a-zA-Z0-9_\s]', '', text)
        for i in range(len(chinese_chars) - 1):
            tokens.append(chinese_chars[i:i + 2])
        for ch in chinese_chars:
            tokens.append(ch)  # 单字也保留
        return tokens


class SignalFusion:
    """
    多信号融合评分引擎。

    融合公式:
        Final Score = α × graph_bfs + β × bm25 + γ × entity_match + δ × vector

    默认权重 (可配置):
        α (graph_weight)     = 0.35  图谱 BFS 遍历得分
        β (bm25_weight)      = 0.30  BM25 关键词匹配得分
        γ (entity_weight)    = 0.25  实体精确匹配加权
        δ (vector_weight)    = 0.10  向量语义相似得分（预留）
    """

    # 实体精确匹配的加权因子（匹配到一个实体即加分）
    ENTITY_MATCH_BOOST = 0.15  # 每匹配一个实体的额外加权

    def __init__(
        self,
        graph_weight: float = 0.35,
        bm25_weight: float = 0.30,
        entity_weight: float = 0.25,
        vector_weight: float = 0.10,
    ):
        assert abs(graph_weight + bm25_weight + entity_weight + vector_weight - 1.0) < 0.01, \
            f"权重之和必须为 1.0，当前: {graph_weight + bm25_weight + entity_weight + vector_weight}"
        self.graph_weight = graph_weight
        self.bm25_weight = bm25_weight
        self.entity_weight = entity_weight
        self.vector_weight = vector_weight

        self.bm25 = BM25Scorer()
        self._entity_index: Dict[str, Set[str]] = defaultdict(set)  # entity_name → {doc_id, ...}
        self._initialized = False

    # =========================================================================
    # 索引管理
    # =========================================================================

    def build_index(
        self,
        documents: Dict[str, str],
        entity_to_docs: Optional[Dict[str, List[str]]] = None,
    ):
        """
        构建多信号索引。

        Args:
            documents: {doc_id: text} — Turn 摘要或社区摘要
            entity_to_docs: {entity_name: [doc_id, ...]} — 实体到文档的映射
        """
        # BM25 索引
        self.bm25.index(documents)

        # 实体索引
        if entity_to_docs:
            for entity, doc_ids in entity_to_docs.items():
                self._entity_index[entity.lower()].update(doc_ids)

        self._initialized = True

    def update_index(
        self,
        doc_id: str,
        text: str,
        entities: Optional[List[str]] = None,
    ):
        """增量更新索引（每轮对话后调用）"""
        self.bm25.add_document(doc_id, text)
        if entities:
            for ent in entities:
                self._entity_index[ent.lower()].add(doc_id)
        self._initialized = True

    # =========================================================================
    # 融合评分
    # =========================================================================

    def fuse(
        self,
        query: str,
        doc_id: str,
        graph_score: float = 0.0,
        query_entities: Optional[List[str]] = None,
        vector_score: float = 0.0,
    ) -> float:
        """
        计算单个文档的多信号融合得分。

        Args:
            query: 用户查询文本
            doc_id: 文档 ID（如 turn_id 或 community_id）
            graph_score: 图谱 BFS 遍历得分 [0, 1]
            query_entities: 从 query 中提取的实体列表
            vector_score: 向量相似度得分 [0, 1]（预留）

        Returns:
            融合得分 [0, 1]
        """
        if not self._initialized:
            # 索引未构建时，回退到纯图得分
            return graph_score

        # 1. BM25 得分
        bm25_raw = self.bm25.score(query, doc_id)
        # 归一化：取 batch 中的相对排名（简化：用 sigmoid-like 缩放）
        bm25_score = math.tanh(bm25_raw * 2.0)  # 压缩到 [0, ~0.96]

        # 2. 实体匹配得分
        entity_score = self._entity_match_score(query_entities, doc_id) if query_entities else 0.0

        # 3. 融合
        final = (
            self.graph_weight * graph_score +
            self.bm25_weight * bm25_score +
            self.entity_weight * entity_score +
            self.vector_weight * vector_score
        )

        return min(1.0, max(0.0, final))

    def fuse_batch(
        self,
        query: str,
        doc_graph_scores: Dict[str, float],
        query_entities: Optional[List[str]] = None,
        vector_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        批量融合评分。

        Args:
            query: 用户查询
            doc_graph_scores: {doc_id: graph_score}
            query_entities: 查询实体列表
            vector_scores: {doc_id: vector_score}（预留）

        Returns:
            {doc_id: fused_score}
        """
        results = {}
        for doc_id, graph_score in doc_graph_scores.items():
            vec_s = (vector_scores or {}).get(doc_id, 0.0)
            results[doc_id] = self.fuse(
                query, doc_id, graph_score, query_entities, vec_s
            )
        return results

    # =========================================================================
    # 实体匹配
    # =========================================================================

    def _entity_match_score(self, query_entities: List[str], doc_id: str) -> float:
        """计算实体匹配得分 [0, 1]"""
        if not query_entities:
            return 0.0

        matched = 0
        for ent in query_entities:
            ent_lower = ent.lower()
            if doc_id in self._entity_index.get(ent_lower, set()):
                matched += 1

        # 匹配比例 × boost
        ratio = matched / len(query_entities)
        return min(1.0, ratio * (1 + self.ENTITY_MATCH_BOOST * matched))

    def find_docs_by_entity(self, entity_name: str) -> Set[str]:
        """通过实体名查找相关文档"""
        return self._entity_index.get(entity_name.lower(), set())

    # =========================================================================
    # 查询辅助
    # =========================================================================

    def rank_with_fusion(
        self,
        query: str,
        candidates: List[Tuple[str, float]],  # [(doc_id, graph_score), ...]
        query_entities: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        对候选文档进行融合排序。

        Args:
            query: 用户查询
            candidates: [(doc_id, graph_score), ...]
            query_entities: 查询实体
            top_k: 返回数量

        Returns:
            按融合得分降序的 [(doc_id, fused_score), ...]
        """
        doc_scores = {did: gs for did, gs in candidates}
        fused = self.fuse_batch(query, doc_scores, query_entities)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    @property
    def doc_count(self) -> int:
        return self.bm25._N


# =============================================================================
# 测试
# =============================================================================

def test_signal_fusion():
    """快速验证 SignalFusion 功能"""
    sf = SignalFusion()

    # 构建索引
    docs = {
        "turn_1": "茅台营收增长15%，净利润增长12%，存货周转天数120天",
        "turn_2": "宁德时代存货异常增加，经营现金流为负，ROE下降",
        "turn_3": "五粮液现金流质量优秀，毛利率稳定在70%以上",
        "turn_4": "万科资产负债率85%，有息负债率高，现金流压力大",
    }
    entity_docs = {
        "贵州茅台": ["turn_1"],
        "宁德时代": ["turn_2"],
        "五粮液": ["turn_3"],
        "万科": ["turn_4"],
        "存货周转率": ["turn_1"],
        "经营现金流": ["turn_2", "turn_3", "turn_4"],
        "600519": ["turn_1"],
        "300750": ["turn_2"],
    }
    sf.build_index(docs, entity_docs)

    # 测试 1: 纯图得分 + BM25 + 实体匹配
    query = "茅台营收"
    doc_id = "turn_1"
    score = sf.fuse(query, doc_id, graph_score=0.8, query_entities=["贵州茅台", "600519"])
    print(f"测试1: query='{query}' doc={doc_id} score={score:.3f} (期望 > 0.7)")
    assert score > 0.7, f"期望 > 0.7, 实际 {score:.3f}"

    # 测试 2: 无实体匹配 + 低 BM25
    query2 = "比亚迪销量"
    doc_id2 = "turn_1"
    score2 = sf.fuse(query2, doc_id2, graph_score=0.6, query_entities=["比亚迪"])
    print(f"测试2: query='{query2}' doc={doc_id2} score={score2:.3f} (期望 < 0.5)")
    assert score2 < 0.5, f"期望 < 0.5, 实际 {score2:.3f}"

    # 测试 3: 批量排序
    candidates = [("turn_1", 0.7), ("turn_2", 0.8), ("turn_3", 0.6)]
    ranked = sf.rank_with_fusion("现金流质量", candidates, query_entities=["经营现金流"], top_k=3)
    print(f"测试3: ranked={[(did, f'{s:.3f}') for did, s in ranked]}")
    # turn_3 应该排在前面（包含"现金流质量优秀" + 实体匹配）
    assert ranked[0][0] == "turn_3", f"期望 turn_3 排名第一，实际 {ranked[0][0]}"

    # 测试 4: BM25 单测
    bm25 = BM25Scorer()
    bm25.index({"d1": "hello world test", "d2": "world world foo"})
    s1 = bm25.score("hello", "d1")
    s2 = bm25.score("hello", "d2")
    print(f"测试4: BM25('hello', d1)={s1:.3f}, BM25('hello', d2)={s2:.3f}")
    assert s1 > s2, f"BM25 d1 应大于 d2"

    print("\n✓ 所有 SignalFusion 测试通过")


if __name__ == "__main__":
    test_signal_fusion()
