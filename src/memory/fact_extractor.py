"""
结构化 Fact 提取器 (Priority 4)
灵感来自 Mem0 v3.0 的 ADD-only 事实提取模型。

核心思路:
- 替代 LLM 轮次摘要 → 提取原子化事实 (Fact)
- 每个 Fact 是一条独立的、可独立检索的陈述
- ADD-only 模式: 新 Fact 自然覆盖旧 Fact，无需 UPDATE/DELETE
- Fact 通过 BELONGS_TO 边连接到 Turn 节点和实体节点

减少 LLM 调用: 将"实体抽取 + 轮次摘要"合并为一次 Fact 提取调用。
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from ..llm import get_llm_client
from ..llm.prompts import ENTITY_EXTRACTION_PROMPT

# ==============================================================================
# Fact 提取专用 Prompt（与实体抽取合并，减少一次 LLM 调用）
# ==============================================================================

FACT_EXTRACTION_PROMPT = """你是一个金融对话分析器。从以下对话中同时提取**实体、关系和原子化事实**。

## 实体类型
- Stock: 股票
- Person: 人物
- Indicator: 金融指标
- Event: 事件
- Organization: 机构
- Report: 财报/研报

## 关系类型
- MENTIONS: 提及
- COMPARES_WITH: 对比
- ASKS_ABOUT: 询问
- CONCERNS: 担忧/关注

## Fact (原子化事实)
每个 fact 是一条独立的、可独立检索的陈述，包含:
- text: 事实陈述文本（一句完整的话）
- entities: 涉及的关键实体名称列表
- category: "analysis"(分析结论) | "data"(数据事实) | "query"(用户意图) | "preference"(用户偏好)

## 输入对话
用户问题: {user_query}
助手回答: {agent_response}
工具调用: {tool_results}

## 输出格式
{{
  "entities": [
    {{"id": "stock_600519", "type": "Stock", "name": "贵州茅台", "code": "600519"}}
  ],
  "relations": [
    {{"source": "turn_N", "target": "stock_600519", "type": "MENTIONS"}}
  ],
  "facts": [
    {{
      "id": "fact_1",
      "text": "贵州茅台最新季度营收达到500亿元",
      "entities": ["贵州茅台", "营收"],
      "category": "data",
      "confidence": 0.95
    }}
  ],
  "turn_summary": "一句简洁中文总结（50字以内）"
}}

Respond ONLY with valid JSON."""


class FactExtractor:
    """
    结构化 Fact 提取器。

    用法:
        extractor = FactExtractor(use_llm=True)
        result = extractor.extract(user_query, agent_response, tool_results)
        # result = {entities, relations, facts, turn_summary}
    """

    def __init__(self, use_llm: bool = True):
        self.llm = get_llm_client() if use_llm else None
        self.use_llm = use_llm
        self.stats = {"facts_extracted": 0, "llm_calls": 0, "fallback_calls": 0}

    def extract(
        self,
        user_query: str,
        agent_response: str = "",
        tool_results: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        一次 LLM 调用同时完成: 实体抽取 + 事实提取 + 轮次摘要。

        Args:
            user_query: 用户输入
            agent_response: 助手回复
            tool_results: 工具调用结果列表

        Returns:
            {
                "entities": List[entity_dict],
                "relations": List[relation_dict],
                "facts": List[fact_dict],
                "turn_summary": str,
            }
        """
        if self.use_llm and self.llm:
            self.stats["llm_calls"] += 1
            try:
                return self._llm_extract(user_query, agent_response, tool_results)
            except Exception as e:
                print(f"[FactExtractor] LLM extraction failed: {e}, falling back")
                self.stats["fallback_calls"] += 1
                return self._rule_extract(user_query, agent_response, tool_results)
        else:
            self.stats["fallback_calls"] += 1
            return self._rule_extract(user_query, agent_response, tool_results)

    def _llm_extract(
        self,
        user_query: str,
        agent_response: str,
        tool_results: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """LLM 模式"""
        tools_str = self._format_tool_results(tool_results) if tool_results else "无"

        prompt = FACT_EXTRACTION_PROMPT.format(
            user_query=user_query[:500],
            agent_response=agent_response[:800],
            tool_results=tools_str[:500],
        )

        result = self.llm.chat_with_json_output(
            user_prompt=prompt,
            temperature=0.0,
            max_retries=1,
        )

        # 丰富 facts 的时间戳和 ID
        facts = result.get("facts", [])
        for i, f in enumerate(facts):
            if "id" not in f or not f["id"]:
                f["id"] = self._generate_fact_id(f.get("text", ""), i)
            f["timestamp"] = time.time()
            if "confidence" not in f:
                f["confidence"] = 0.85

        self.stats["facts_extracted"] += len(facts)
        return result

    def _rule_extract(
        self,
        user_query: str,
        agent_response: str = "",
        tool_results: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """规则 fallback：沿用原有实体抽取逻辑 + 从文本切分事实"""
        # 实体抽取（复用原有逻辑）
        entities = _rule_extract_entities(user_query, agent_response)
        relations = _rule_extract_relations(entities)

        # 事实提取：简单的句子切分
        facts = []
        full_text = f"{user_query} {agent_response}"
        # 按句子切分
        sentences = _split_sentences(full_text)
        for i, sent in enumerate(sentences):
            if len(sent) < 5:
                continue
            sent_entities = [e["name"] for e in entities if e.get("name", "") in sent]
            facts.append({
                "id": self._generate_fact_id(sent, i),
                "text": sent.strip()[:200],
                "entities": sent_entities[:5],
                "category": "data" if any(kw in sent for kw in ["亿", "万", "%", "元", "增长", "下降"]) else "query",
                "confidence": 0.5,  # 规则提取置信度较低
                "timestamp": time.time(),
            })

        turn_summary = user_query[:100] + ("..." if len(user_query) > 100 else "")
        self.stats["facts_extracted"] += len(facts)

        return {
            "entities": entities,
            "relations": relations,
            "facts": facts,
            "turn_summary": turn_summary,
        }

    @staticmethod
    def _format_tool_results(tool_results: List[Dict]) -> str:
        """格式化工具结果为提取用文本"""
        parts = []
        for r in tool_results:
            if isinstance(r, dict):
                rendered = r.get("rendered", r.get("data", {}).get("rendered", ""))
                if rendered:
                    parts.append(str(rendered)[:300])
                else:
                    data = r.get("data", r)
                    parts.append(json.dumps(data, ensure_ascii=False, default=str)[:300])
        return "\n".join(parts)

    @staticmethod
    def _generate_fact_id(text: str, index: int) -> str:
        """生成确定性 Fact ID"""
        content = text[:100].encode("utf-8")
        hash_suffix = hashlib.md5(content).hexdigest()[:8]
        return f"fact_{index}_{hash_suffix}"


# ==============================================================================
# 辅助函数
# ==============================================================================

def _rule_extract_entities(user_query: str, agent_response: str = "") -> List[Dict]:
    """规则式实体抽取（移除 LLM 依赖的简化版）"""
    import re
    from .entity_extractor import STOCK_ALIASES, INDICATOR_ALIASES

    entities = []
    text = f"{user_query} {agent_response}"

    # 股票代码
    for match in re.finditer(r'\b(\d{6})\b', text):
        code = match.group(1)
        entities.append({"id": f"stock_{code}", "type": "Stock", "name": code, "code": code})

    # 已知股票名称
    for name, code in STOCK_ALIASES.items():
        if name in text and len(name) >= 2:
            eid = f"stock_{code}"
            if not any(e["id"] == eid for e in entities):
                entities.append({"id": eid, "type": "Stock", "name": name, "code": code})

    # 指标
    for alias, canonical in INDICATOR_ALIASES.items():
        if alias in text:
            eid = f"indicator_{re.sub(r'[^a-zA-Z0-9_一-鿿]', '_', canonical)[:50]}"
            if not any(e["id"] == eid for e in entities):
                entities.append({"id": eid, "type": "Indicator", "name": canonical})

    return entities


def _rule_extract_relations(entities: List[Dict]) -> List[Dict]:
    """从实体列表构建基础关系"""
    relations = []
    for entity in entities:
        relations.append({"source": "current_turn", "target": entity["id"], "type": "MENTIONS"})
    return relations


def _split_sentences(text: str) -> List[str]:
    """简单的中英文分句"""
    import re
    # 按句号、问号、感叹号、换行分句
    sentences = re.split(r'[。！？\n]+', text)
    return [s.strip() for s in sentences if s.strip()]


# ==============================================================================
# 测试
# ==============================================================================

def test_fact_extraction():
    """快速验证 FactExtractor（无 LLM 模式）"""
    fe = FactExtractor(use_llm=False)

    # 测试规则模式
    result = fe.extract(
        user_query="贵州茅台600519的营收是多少？去年是1200亿，今年增长15%",
        agent_response="根据最新财报，贵州茅台2025年营收约1380亿元，同比增长15%。",
        tool_results=[{"data": {"rendered": "## 财报概览\n营收: 1380亿, 增速: 15%"}}],
    )

    assert "entities" in result, "缺少 entities"
    assert "facts" in result, "缺少 facts"
    assert "turn_summary" in result, "缺少 turn_summary"
    assert len(result["facts"]) > 0, "应至少提取一个 fact"
    assert len(result["entities"]) > 0, "应至少提取一个 entity"

    print(f"✓ entities: {len(result['entities'])} 个")
    print(f"✓ facts: {len(result['facts'])} 个")
    for f in result["facts"]:
        print(f"  [{f['category']}] {f['text'][:80]}... (confidence={f['confidence']})")
    print(f"✓ turn_summary: {result['turn_summary'][:80]}")
    print(f"✓ stats: {fe.stats}")

    print("\n✓ FactExtractor 测试通过")


if __name__ == "__main__":
    test_fact_extraction()
