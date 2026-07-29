"""任务二集成测试套件"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.graph_builder import StockGraphBuilder
from src.graph.equity_engine import EquityPenetrationEngine
from src.graph.chain_builder import ChainBuilder
from src.graph.event_clusterer import EventClusterer
from src.tools.equity_graph import EquityPenetrationSkill, EventTraceSkill, ControlSummarySkill


def test_graph_construction():
    """测试图谱构建"""
    print("\n=== Test 1: Graph Construction ===")
    builder = StockGraphBuilder()
    G = builder.build_from_shareholders(nrows=100000)

    assert G.number_of_nodes() > 10000, "Graph too small"
    assert G.number_of_edges() > 50000, "Too few edges"

    stock_count = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "Stock")
    holder_count = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "Shareholder")
    print(f"  Stocks: {stock_count}, Holders: {holder_count}")

    # 验证茅台
    assert "600519" in G, "600519 not in graph"
    holders = builder.get_direct_holders("600519")
    assert len(holders) > 5, "Too few holders for 600519"
    print(f"  600519 holders: {len(holders)}")
    print("  Graph Construction: PASSED")


def test_equity_penetration():
    """测试股权穿透"""
    print("\n=== Test 2: Equity Penetration ===")
    builder = StockGraphBuilder()
    builder.build_from_shareholders(nrows=100000)
    engine = EquityPenetrationEngine(builder.G)

    # 基本穿透
    chains = engine.penetrate("600519", "upstream", max_depth=5, min_ratio=0.5)
    assert len(chains) > 0, "No chains returned"
    print(f"  600519 chains: {len(chains)}")

    # 控股权摘要
    summary = engine.get_control_summary("600519")
    assert summary["total_holders"] > 0
    assert summary["top5_concentration"] > 30
    print(f"  Top5 concentration: {summary['top5_concentration']:.1f}%")

    # 实际控制人
    uc = engine.find_ultimate_controller("600519")
    assert uc is not None, "No ultimate controller"
    assert uc.depth >= 1
    print(f"  Ultimate controller depth: {uc.depth}")
    print("  Equity Penetration: PASSED")


def test_skill_execution():
    """测试 Skill 封装"""
    print("\n=== Test 3: Skill Execution ===")

    # 股权穿透 Skill
    result = EquityPenetrationSkill.execute({
        "target_entity": "600519",
        "max_depth": 5,
        "min_ratio": 0.5,
        "direction": "upstream",
    })
    assert result["total_chains"] > 0
    assert len(result["rendered"]) > 100
    print(f"  EquityPenetrationSkill: {result['total_chains']} chains")

    # 控股摘要 Skill
    result = ControlSummarySkill.execute({"stock_code": "600519"})
    assert result["total_holders"] > 0
    print(f"  ControlSummarySkill: {result['total_holders']} holders")
    print("  Skill Execution: PASSED")


def test_event_clustering():
    """测试事件簇聚合"""
    print("\n=== Test 4: Event Clustering ===")
    builder = StockGraphBuilder()
    builder.build_from_shareholders(nrows=50000)
    builder.build_from_announcements(nrows=2000)
    clusterer = EventClusterer(builder.G, use_llm=False)

    # 获取公告
    anns = clusterer.get_announcements()
    print(f"  Total announcements: {len(anns)}")

    if len(anns) >= 3:
        clusters = clusterer.cluster(anns)
        print(f"  Clusters: {len(clusters)}")
        if clusters:
            print(f"  Largest cluster: {clusters[0].name} ({clusters[0].event_count} events)")

    # 时间线
    timeline = clusterer.build_timeline("600519")
    print(f"  Timeline events for 600519: {len(timeline.events)}")
    print("  Event Clustering: PASSED")


def test_chain_builder():
    """测试多跳链路构建"""
    print("\n=== Test 5: Chain Builder ===")
    builder = StockGraphBuilder()
    builder.build_from_shareholders(nrows=100000)
    cb = ChainBuilder(builder.G, use_llm=False)

    # 跨层连接
    n = cb.build_cross_links()
    print(f"  Cross-links: {n}")

    # 综合穿透
    result = cb.comprehensive_penetration("600519", max_depth=5)
    print(f"  Comprehensive: {result['summary']}")
    print("  Chain Builder: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("  Task 2 — Integration Test Suite")
    print("=" * 60)

    tests = [
        test_graph_construction,
        test_equity_penetration,
        test_skill_execution,
        test_event_clustering,
        test_chain_builder,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{len(tests)} tests passed")
    print(f"{'=' * 60}")
