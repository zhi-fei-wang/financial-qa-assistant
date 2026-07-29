"""
任务一集成测试
验证记忆系统、路由系统、Agent 主循环的端到端功能。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.router.agent_loop import FinancialAgent
from src.router.intent_classifier import IntentClassifier
from src.memory.entity_extractor import EntityExtractor
from src.memory.knowledge_graph import KnowledgeGraph
from src.memory.memory_manager import MemoryManager


def test_intent_classification():
    """测试意图分类准确率"""
    print("\n=== Test 1: Intent Classification ===")
    classifier = IntentClassifier(use_llm=True)

    test_cases = [
        ("帮我查茅台股价", "MARKET_DATA"),
        ("茅台ROE最近五年变化", "FINANCIAL_ANALYSIS"),
        ("九阳股份股权穿透", "EQUITY_PENETRATION"),
        ("茅台最近有违规公告吗", "NEWS_EVENT"),
        ("你好，介绍一下你自己", "CHITCHAT"),
        ("计算近三年平均毛利率", "CALCULATION"),
        ("存货周转率异常，有没有虚增", "FINANCIAL_ANALYSIS"),
    ]

    correct = 0
    for query, expected in test_cases:
        result = classifier.classify(query)
        ok = result.intent == expected
        if ok:
            correct += 1
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} [{result.intent:20s}] {query[:50]}")

    accuracy = correct / len(test_cases)
    print(f"  Intent Accuracy: {accuracy:.1%} ({correct}/{len(test_cases)})")
    assert accuracy >= 0.7, f"Intent accuracy too low: {accuracy}"
    return accuracy


def test_entity_extraction():
    """测试实体抽取"""
    print("\n=== Test 2: Entity Extraction ===")
    extractor = EntityExtractor(use_llm=True)

    test_cases = [
        ("贵州茅台的ROE最近五年变化", ["stock_600519", "indicator_ROE"]),
        ("宁德时代存货周转率", ["stock_300750", "indicator_存货周转率"]),
        ("王旭宁控股九阳股份", ["person_王旭宁", "stock_002242"]),
    ]

    for query, expected_ids in test_cases:
        entities, relations = extractor.extract(query)
        extracted_ids = [e["id"] for e in entities]
        matched = [eid for eid in expected_ids if eid in extracted_ids]
        print(f"  Query: {query}")
        print(f"    Extracted: {extracted_ids}")
        print(f"    Matched: {len(matched)}/{len(expected_ids)}")
        assert len(matched) >= len(expected_ids) * 0.5, f"Too few entities matched"

    print("  Entity Extraction: PASSED")


def test_knowledge_graph():
    """测试图谱构建和查询"""
    print("\n=== Test 3: Knowledge Graph ===")
    kg = KnowledgeGraph()

    # 添加实体
    entities = [
        {"id": "stock_600519", "type": "Stock", "name": "贵州茅台", "code": "600519"},
        {"id": "stock_000858", "type": "Stock", "name": "五粮液", "code": "000858"},
        {"id": "indicator_ROE", "type": "Indicator", "name": "ROE"},
    ]
    kg.upsert_entities(entities)
    assert kg.node_count == 3

    # 添加轮次
    kg.add_turn_node("turn_1", "茅台ROE查询", ["stock_600519", "indicator_ROE"])
    kg.add_turn_node("turn_2", "五粮液对比", ["stock_000858", "stock_600519"])
    assert kg.turn_count == 2

    # 测试邻居查询
    neighbors = kg.get_neighbors("stock_600519", depth=1)
    print(f"  茅台 1-hop neighbors: {len(neighbors)}")
    assert len(neighbors) > 0

    # 测试实体历史
    history = kg.get_entity_history("stock_600519")
    print(f"  茅台 history: {len(history)} turns")
    assert len(history) == 2

    # 测试社区发现
    communities = kg.detect_communities()
    print(f"  Communities: {len(communities)}")
    assert len(communities) > 0

    print("  Knowledge Graph: PASSED")


def test_agent_chat():
    """测试 Agent 端到端对话"""
    print("\n=== Test 4: Agent Chat ===")
    agent = FinancialAgent(use_llm=True)

    # 多轮对话
    queries = [
        "贵州茅台的ROE水平怎么样？",
        "刚才说的茅台，存货周转率是否正常？",
    ]

    for i, q in enumerate(queries, 1):
        resp = agent.chat(q)
        assert resp, "Empty response"
        assert len(resp) > 20, f"Response too short: {len(resp)} chars"
        print(f"  Turn {i}: {q[:40]}... → {len(resp)} chars")

    # 验证记忆系统
    mem_summary = agent.get_memory_summary()
    print(f"  Memory: {mem_summary.split(chr(10))[1].strip()}")

    # 验证路由统计
    stats = agent.get_router_stats()
    print(f"  Router: {stats}")

    assert agent.turn_count == 2
    assert agent.memory.graph.node_count > 0

    print("  Agent Chat: PASSED")


def test_memory_recall():
    """测试跨轮次记忆召回"""
    print("\n=== Test 5: Cross-Turn Memory Recall ===")
    agent = FinancialAgent(use_llm=True)

    # 第1轮：建立上下文
    agent.chat("茅台(600519)2025年资产负债表的存货是574.57亿元")

    # 第2轮：测试召回
    memories = agent.memory.retrieve("茅台存货多少", top_k=5)
    print(f"  Retrieved {len(memories)} memory contexts for '茅台存货多少'")
    assert len(memories) > 0, "No memory retrieved"

    # 检查图检索结果
    graph_results = [m for m in memories if m.source == "graph"]
    print(f"  Graph results: {len(graph_results)}")
    assert len(graph_results) > 0, "No graph results — entity not linked"

    # 检查实体上下文
    entity_ctx = agent.memory.get_entity_context("贵州茅台")
    if entity_ctx["found"]:
        print(f"  茅台 entity: {len(entity_ctx['history'])} turns history, "
              f"{len(entity_ctx['neighbors'])} neighbors")

    print("  Memory Recall: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("  任务一 — 集成测试套件")
    print("=" * 60)

    try:
        test_entity_extraction()
        test_knowledge_graph()
        test_intent_classification()
        test_memory_recall()
        test_agent_chat()

        print("\n" + "=" * 60)
        print("  [OK] 所有集成测试通过!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
