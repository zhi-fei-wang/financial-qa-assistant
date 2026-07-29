"""
OpenIE 实体关系抽取 (P2: 赛题缺口 — 从公告/研报非结构化文本中抽取实体+关系)

受 HippoRAG 启发: LLM 从非结构化文本中抽取出 (subject, predicate, object) 三元组，
写入 KnowledgeGraph 作为实体节点和关系边。弥补当前 EventClusterer 只能读标题的不足。

用法:
    extractor = OpenIEExtractor(use_llm=True)
    triples = extractor.extract_from_text("某公司公告: X公司向Y公司转让5%股权...")
    # → [("X公司", "转让股权给", "Y公司"), ...]
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from ..llm import get_llm_client


OPENIE_PROMPT = """你是一个金融关系抽取器。从以下文本中提取实体和关系三元组。

## 关系类型
- CONTROLS: A控制B (控股/实际控制)
- TRANSFERS_TO: A向B转让/出售 (股权/资产)
- ACQUIRES_FROM: A从B收购/买入
- PLEDGES: A质押股权给B
- SUED_BY: A被B处罚/起诉/问询
- OWNS: A持有B的X%股份
- IS_CEO_OF: A是B的高管/法人代表/董事
- VIOLATES: A违规/受罚
- COOPERATES_WITH: A与B合作/关联

## 输入文本
{text}

## 输出格式
{{
  "triples": [
    {{"subject": "A公司", "predicate": "CONTROLS", "object": "B公司", "weight": 0.95}},
    ...
  ],
  "entities": [
    {{"name": "A公司", "type": "Company"}},
    {{"name": "张三", "type": "Person"}},
    ...
  ]
}}

Respond ONLY with valid JSON."""


class OpenIEExtractor:
    """从金融文本中抽取实体和关系三元组"""

    def __init__(self, use_llm: bool = True):
        self.llm = get_llm_client() if use_llm else None
        self.use_llm = use_llm
        self.stats = {"texts_processed": 0, "triples_extracted": 0, "llm_calls": 0}

    def extract_from_text(
        self, text: str, max_triples: int = 20
    ) -> Dict[str, Any]:
        """
        从一段文本中抽取实体和三元组。

        Args:
            text: 输入文本（公告正文/研报摘要等）
            max_triples: 最大三元组数

        Returns:
            {"triples": [...], "entities": [...]}
        """
        if not text or len(text) < 10:
            return {"triples": [], "entities": []}

        self.stats["texts_processed"] += 1

        if self.use_llm and self.llm:
            self.stats["llm_calls"] += 1
            try:
                return self._llm_extract(text, max_triples)
            except Exception as e:
                print(f"[OpenIE] LLM failed: {e}, falling back to rules")
                return self._rule_extract(text)
        else:
            return self._rule_extract(text)

    def extract_from_batch(
        self, texts: List[str]
    ) -> List[Dict[str, Any]]:
        """批量抽取"""
        results = []
        for text in texts:
            result = self.extract_from_text(text)
            results.append(result)
            if len(results) % 10 == 0:
                print(f"[OpenIE] Processed {len(results)}/{len(texts)} texts, "
                      f"{self.stats['triples_extracted']} triples so far")
        return results

    def enrich_knowledge_graph(
        self, graph, triples: List[Dict], source_text_id: str = ""
    ) -> int:
        """
        将抽取的三元组写入 KnowledgeGraph。

        Args:
            graph: KnowledgeGraph 实例
            triples: extract_from_text 的 triples 输出
            source_text_id: 来源文本 ID (如 "ann_00345")

        Returns:
            新增节点数
        """
        nodes_added = 0
        for t in triples:
            subj = t.get("subject", "")
            obj = t.get("object", "")
            pred = t.get("predicate", "RELATED_TO")
            weight = t.get("weight", 1.0)

            if not subj or not obj:
                continue

            # 添加实体节点
            for name in [subj, obj]:
                node_id = f"entity_{name.replace(' ', '_')[:50]}"
                if node_id not in graph.G:
                    graph.G.add_node(node_id, type="Entity", name=name, source=source_text_id)
                    nodes_added += 1

            # 添加关系边
            subj_id = f"entity_{subj.replace(' ', '_')[:50]}"
            obj_id = f"entity_{obj.replace(' ', '_')[:50]}"
            if subj_id in graph.G and obj_id in graph.G:
                graph.G.add_edge(subj_id, obj_id, type=pred, weight=weight, source=source_text_id)

        return nodes_added

    def _llm_extract(self, text: str, max_triples: int) -> Dict[str, Any]:
        """LLM 抽取"""
        prompt = OPENIE_PROMPT.format(text=text[:2000])
        result = self.llm.chat_with_json_output(prompt, temperature=0.0)

        triples = result.get("triples", [])[:max_triples]
        entities = result.get("entities", [])
        self.stats["triples_extracted"] += len(triples)

        return {"triples": triples, "entities": entities}

    def _rule_extract(self, text: str) -> Dict[str, Any]:
        """规则 fallback: 从公告标题做简单匹配"""
        import re

        triples = []
        entities = set()

        # 简单的正则匹配
        patterns = [
            (r"(\S+公司).*?被(\S+).*?(处罚|罚款|警告|立案|调查|谴责)", "SUED_BY"),
            (r"(\S+公司).*?质押(\S+).*?股份", "PLEDGES"),
            (r"(\S+).*?减持(\S+公司).*?(\d+\.?\d*)%", "TRANSFERS_TO"),
            (r"(\S+).*?增持(\S+公司).*?(\d+\.?\d*)%", "ACQUIRES_FROM"),
            (r"(\S+).*?(控股|控制|持有)(\S+公司).*?(\d+\.?\d*)%", "CONTROLS"),
            (r"(\S+).*?(任|担任|出任)(\S+公司).*?(董事|总经理|CEO|董事长|总裁|监事)", "IS_CEO_OF"),
        ]

        for pattern, pred in patterns:
            matches = re.findall(pattern, text)
            for m in matches:
                subj = m[0].strip()
                obj = m[1].strip() if len(m) >= 2 else "未知"
                triples.append({"subject": subj, "predicate": pred, "object": obj, "weight": 0.5})
                entities.add(subj)
                entities.add(obj)

        return {
            "triples": triples[:10],
            "entities": [{"name": e, "type": "Company"} for e in entities],
        }


# 全局单例
_extractor: Optional[OpenIEExtractor] = None


def get_openie_extractor(use_llm: bool = True) -> OpenIEExtractor:
    global _extractor
    if _extractor is None:
        _extractor = OpenIEExtractor(use_llm=use_llm)
    return _extractor
