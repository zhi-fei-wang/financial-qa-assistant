"""股权穿透查询工具 — Task 2 完整版，注册到 Task1 Router"""

import os
import pickle
import time
from typing import Any, Dict, List, Optional

from ..graph.chain_builder import ChainBuilder
from ..graph.equity_engine import EquityPenetrationEngine
from ..graph.graph_builder import StockGraphBuilder
from ..utils.data_loader import DataLoader


# 全局单例（支持 pickle 缓存）
_graph_builder: Optional[StockGraphBuilder] = None
_equity_engine: Optional[EquityPenetrationEngine] = None
_chain_builder: Optional[ChainBuilder] = None
_warmed_up: bool = False

# pickle 缓存路径
_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '.cache')
_GRAPH_CACHE = os.path.join(_CACHE_DIR, 'stock_graph.pkl')


def warmup(data_loader=None, nrows=200000):
    """
    预热：一次性构建图和引擎，消除首次查询的冷启动延迟。

    优先从 pickle 缓存加载（秒级），缓存不存在时才构建（~90s）并保存缓存。
    应在 Agent 初始化时调用。

    Args:
        data_loader: DataLoader 实例，用于加载数据
        nrows: 加载的股东数据行数
    """
    global _graph_builder, _equity_engine, _chain_builder, _warmed_up
    if _warmed_up:
        return

    t0 = time.time()

    # 尝试从 pickle 缓存加载
    if os.path.exists(_GRAPH_CACHE):
        try:
            with open(_GRAPH_CACHE, 'rb') as f:
                _graph_builder = pickle.load(f)
            _equity_engine = EquityPenetrationEngine(_graph_builder.G)
            _equity_engine._graph_builder = _graph_builder
            _chain_builder = ChainBuilder(_graph_builder.G, use_llm=True)
            _warmed_up = True
            print(f"[Warmup] Graph loaded from cache in {time.time()-t0:.1f}s")
            return
        except Exception as e:
            print(f"[Warmup] Cache load failed: {e}, rebuilding...")

    # 从头构建图
    _graph_builder = StockGraphBuilder(loader=data_loader)
    _graph_builder.build_from_shareholders(nrows=nrows)
    _graph_builder.build_from_announcements(nrows=3000)

    # 构建跨层连接
    cb = ChainBuilder(_graph_builder.G, use_llm=True)
    n_links = cb.build_cross_links(min_confidence=0.85)

    _equity_engine = EquityPenetrationEngine(_graph_builder.G)
    _equity_engine._graph_builder = _graph_builder
    _chain_builder = cb
    _warmed_up = True

    print(f"[Warmup] Graph built in {time.time()-t0:.1f}s ({n_links} cross-links)")

    # 保存 pickle 缓存
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_GRAPH_CACHE, 'wb') as f:
            pickle.dump(_graph_builder, f)
        print(f"[Warmup] Graph cached to {_GRAPH_CACHE}")
    except Exception as e:
        print(f"[Warmup] Failed to cache graph: {e}")


def _get_graph_builder() -> StockGraphBuilder:
    global _graph_builder
    if _graph_builder is None:
        warmup()
    return _graph_builder


def _get_equity_engine() -> EquityPenetrationEngine:
    global _equity_engine
    if _equity_engine is None:
        warmup()
    return _equity_engine


def _get_chain_builder() -> ChainBuilder:
    global _chain_builder
    if _chain_builder is None:
        warmup()
    return _chain_builder


class EquityPenetrationSkill:
    """
    股权穿透 Skill — 注册到 Task1 Router。
    支持：向上溯源实际控制人、向下穿透参股公司、同业股东对比。
    """

    name = "equity_penetration"
    description = (
        "股权穿透查询，输出多层控股链。支持向上追溯实际控制人、"
        "向下穿透参股公司、同业股东交叉对比。深度>3层的准确率≥85%。"
    )
    required_params = ["target_entity"]
    optional_params = ["max_depth", "min_ratio", "direction"]

    @staticmethod
    def execute(params: Dict[str, Any]) -> Dict[str, Any]:
        target = params.get("target_entity", "")
        max_depth = int(params.get("max_depth", 5))
        min_ratio = float(params.get("min_ratio", 0.5))
        direction = params.get("direction", "upstream")

        # Step 1: 图谱数据穿透
        engine = _get_equity_engine()
        chains = engine.penetrate(target, direction, max_depth, min_ratio)

        # Step 2: 综合穿透分析（跨层连接 + LLM补全）
        cb = _get_chain_builder()
        comprehensive = cb.comprehensive_penetration(target, max_depth)

        # Step 3: 渲染结果 — 区分数据来源
        complete = [c for c in chains if c.is_complete]
        depths = set(c.depth for c in chains)
        max_d = max(depths) if depths else 0

        # 构建详细渲染文本
        rendered_parts = []
        rendered_parts.append(f"## {target} 股权穿透分析")
        rendered_parts.append(f"数据来源: 真实股东数据集 (2/clean.xlsx) | 穿透方向: {direction} | 最大深度: {max_depth}层\n")

        # 3a: 真实数据链路
        if chains:
            rendered_parts.append(f"### 数据链路 ({len(chains)}条，最深{max_d}层)")
            for i, c in enumerate(chains[:10], 1):
                rendered_parts.append(f"链路{i} (控股权{c.cumulative_control:.2f}%, 深度{c.depth}层):")
                segments = []
                for hop in c.path:
                    segments.append(f"  L{hop.level}: {hop.from_name} → {hop.to_name} ({hop.ratio:.2f}%)")
                rendered_parts.append("\n".join(segments))
                if c.is_complete:
                    rendered_parts.append(f"  ✓ 已穿透到底: {c.termination_reason}")
                elif c.depth < max_depth:
                    rendered_parts.append(f"  → 数据断层: 中间层实体可能为非上市公司，数据库无其股东信息")
            rendered_parts.append("")
        else:
            rendered_parts.append(f"### 数据链路\n未在数据库中找到 {target} 的股东数据。请确认股票代码是否正确。\n")

        # 3b: LLM 知识补全（仅当数据不完整时）
        llm_info = comprehensive.get("llm_enhanced", {})
        if llm_info and llm_info.get("inferred_chains"):
            rendered_parts.append("### AI知识补全 (基于LLM预训练知识)")
            rendered_parts.append("> ⚠️ 以下信息来自AI模型知识库，非实时工商数据，仅供参考。\n")
            for inferred in llm_info.get("inferred_chains", []):
                entity = inferred.get("entity", "?")
                holders = inferred.get("known_holders", [])
                conf = inferred.get("confidence", 0)
                rendered_parts.append(f"**{entity}** (置信度: {conf:.0%})")
                for h in holders:
                    rendered_parts.append(f"  - {h.get('name', '?')}: {h.get('pct', '?')}% ({h.get('type', '?')}) — 来源: {h.get('source', 'AI知识')}")
                rendered_parts.append("")

        # 3c: 公告线索
        ann_clues = comprehensive.get("announcement_clues", [])
        if ann_clues:
            rendered_parts.append("### 公告中的股权线索")
            for clue in ann_clues[:5]:
                rendered_parts.append(f"- {clue}")
            rendered_parts.append("")

        if not chains and not llm_info.get("inferred_chains") and not ann_clues:
            rendered_parts.append("### 分析")
            rendered_parts.append("当前数据集中未找到该实体的股权穿透信息。可能原因：")
            rendered_parts.append("1. 股票代码不在数据集覆盖范围内")
            rendered_parts.append("2. 该实体为非上市公司，股东信息未公开披露")
            rendered_parts.append("3. 数据集中该股票的股东数据缺失\n")
            rendered_parts.append("建议：尝试输入6位数字股票代码（如002242），或使用公司全称查询。")

        rendered = "\n".join(rendered_parts)

        # 启发 2: 构造 ResultEnvelope（证据 + 结论 + 置信度）
        evidence_items = []
        for c in chains[:5]:
            path_str = " → ".join(
                f"{h.from_name}({h.ratio:.1f}%)" for h in c.path
            )
            evidence_items.append({
                "claim": f"链路: {path_str} → {c.path[-1].to_name}, 累积控股权{c.cumulative_control:.2f}%",
                "source": "graph",
                "data": {"depth": c.depth, "cumulative_control": c.cumulative_control},
                "confidence": 1.0 if c.is_complete else 0.7,
            })

        confidence = 0.9 if len(complete) > 0 else (0.6 if chains else 0.3)
        limitations = []
        if max_d < max_depth:
            limitations.append(f"穿透深度受限于数据({max_d}层)，中间非上市公司无数据")
        if llm_info and llm_info.get("inferred_chains"):
            limitations.append("AI知识补全的信息来自LLM预训练知识，非实时工商数据")

        env = {
            "conclusion": f"共发现{len(chains)}条链路，最深{max_d}层，{len(complete)}条完整穿透到底",
            "evidence": evidence_items,
            "confidence": confidence,
            "limitations": limitations,
            "metadata": {"skill_name": "equity_penetration", "direction": direction},
        }

        return {
            "target": target,
            "direction": direction,
            "max_depth": max_depth,
            "total_chains": len(chains),
            "complete_chains": len(complete),
            "max_depth_reached": max_d,
            "chains": [
                {
                    "path": [{"from": h.from_name, "to": h.to_name, "ratio": h.ratio, "level": h.level}
                            for h in c.path],
                    "cumulative_control": c.cumulative_control,
                    "depth": c.depth,
                    "is_complete": c.is_complete,
                }
                for c in chains[:10]
            ],
            "llm_enhanced": bool(llm_info and llm_info.get("inferred_chains")),
            "announcement_clues_count": len(ann_clues),
            "rendered": rendered,
            "source": "graph",
            "envelope": env,
            "envelope_rendered": (
                f"**结论**: {env['conclusion']}\n"
                + "\n".join(f"- {e['claim']} [来源: {e['source']}]" for e in evidence_items[:5])
                + f"\n**置信度**: {env['confidence']:.0%}"
            ),
        }


class EventTraceSkill:
    """
    事件溯源 Skill — 注册到 Task1 Router。
    查询标的公司的舆情事件脉络，输出事件簇分类和时间线。
    """

    name = "event_trace"
    description = (
        "查询标的公司的舆情事件脉络，输出事件簇分类和时间线。"
        "支持事件类型筛选和股-舆对齐分析。"
    )
    required_params = ["stock_code"]
    optional_params = ["event_type", "date_range", "max_events"]

    @staticmethod
    def execute(params: Dict[str, Any]) -> Dict[str, Any]:
        stock_code = params.get("stock_code", "")
        max_events = int(params.get("max_events", 20))

        builder = _get_graph_builder()
        from ..graph.event_clusterer import EventClusterer
        clusterer = EventClusterer(builder.G, use_llm=True)

        # 获取该股票的公告
        anns = clusterer.get_announcements(stock_code)
        anns = anns[:max_events]

        # 聚类
        clusters = clusterer.cluster(anns, stock_code)

        # 时间线
        timeline = clusterer.build_timeline(stock_code)

        # 股-舆对齐
        alignments = clusterer.align_equity_with_news(stock_code)

        # 渲染
        rendered_parts = [f"## {stock_code} 事件脉络分析\n"]
        for c in clusters:
            rendered_parts.append(f"### {c.name} ({c.event_count}条)")
            rendered_parts.append(f"{c.summary}")
            rendered_parts.append(f"时间: {c.start_date} ~ {c.end_date}")
            if c.key_entities:
                rendered_parts.append(f"关键实体: {', '.join(c.key_entities[:5])}")
            rendered_parts.append("")

        if alignments:
            rendered_parts.append("### 股-舆关联分析")
            for al in alignments[:5]:
                rendered_parts.append(f"- 公告「{al['news'].get('description', '')[:50]}...」")
                for ee in al.get("prior_equity_events", []):
                    rendered_parts.append(f"  ← {ee['days_before']}天前: {ee.get('description', '')}")

        return {
            "stock_code": stock_code,
            "total_announcements": len(anns),
            "clusters": [
                {
                    "name": c.name,
                    "summary": c.summary,
                    "event_count": c.event_count,
                    "start_date": c.start_date,
                    "end_date": c.end_date,
                    "key_entities": c.key_entities,
                }
                for c in clusters
            ],
            "timeline_events": len(timeline.events),
            "alignment_count": len(alignments),
            "rendered": "\n".join(rendered_parts),
            "source": "graph",
        }


class ControlSummarySkill:
    """
    控股摘要 Skill — 快速获取某股票的控制权概况。
    """

    name = "control_summary"
    description = "获取某只股票的控股权摘要：实际控制人、Top 5 股东、股权集中度。"

    required_params = ["stock_code"]
    optional_params = []

    @staticmethod
    def execute(params: Dict[str, Any]) -> Dict[str, Any]:
        stock_code = params.get("stock_code", "")
        engine = _get_equity_engine()
        summary = engine.get_control_summary(stock_code)
        summary["rendered"] = (
            f"## {stock_code} 控股权摘要\n"
            f"- 股东总数: {summary['total_holders']}\n"
            f"- Top5 集中度: {summary['top5_concentration']:.1f}%\n"
            f"- 实际控制人: {summary['ultimate_controller']}\n"
            f"- Top 股东:\n"
            + "\n".join(
                f"  {h['name'][:30]} ({h['pct']:.2f}%, {h['type']})"
                for h in summary['top_holders']
            )
        )
        return summary


# 导出：供 Task1 Router 注册使用
TASK2_SKILLS = [
    EquityPenetrationSkill,
    EventTraceSkill,
    ControlSummarySkill,
]
