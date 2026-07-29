"""
事件簇聚合器 + 时间线构建器 + 股-舆对齐
Phase 2: 舆情事件簇聚合与动态脉络推演
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from ..llm import get_llm_client


@dataclass
class EventCluster:
    """事件簇"""
    cluster_id: int
    name: str                         # LLM 生成的簇名
    summary: str                       # LLM 生成的摘要 (2-3句)
    events: List[Dict[str, Any]]      # 包含的公告/事件
    timeline: List[Dict[str, Any]]    # 时间排序的事件列表
    key_entities: List[str]           # 关键实体
    start_date: str
    end_date: str
    event_count: int = 0
    category: str = ""                # 事件类别


@dataclass
class Timeline:
    """统一时间线"""
    stock_code: str
    events: List[Dict[str, Any]]      # 所有事件（股权+公告）
    clusters: List[EventCluster]      # 事件簇
    causal_links: List[Tuple[int, int]] = field(default_factory=list)  # 因果关系对


class EventClusterer:
    """
    事件簇聚合器：构建公告共现图 → Leiden/Louvain 聚类 → LLM 摘要 → 时间线
    """

    # 公告类型码→类别映射
    FCODE_CATEGORY = {
        '5507060000': '监管处罚', '5506040000': '风险提示', '5502010000': 'ST/退市',
        '5507200000': '股权变动', '5230000000': '权益变动', '5203000000': '质押冻结',
        '5507230000': '收购兼并', '5507220000': '资产重组', '5507240000': '借贷担保',
        '5506050000': '重大合同', '5219000000': '回购股权', '5507270000': '人事变动',
        '5506220000': '员工持股', '5507040000': '关联交易', '5507260000': '政策影响',
        '5508000000': '监管函件', '5406000000': '业绩预告', '5107000000': '利润分配',
    }

    def __init__(self, graph: nx.DiGraph, use_llm: bool = True):
        self.G = graph
        self.llm = get_llm_client() if use_llm else None
        self.use_llm = use_llm

    # =========================================================================
    # 获取公告列表
    # =========================================================================

    def get_announcements(
        self, stock_code: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """从图谱中提取公告节点"""
        anns = []
        for node, data in self.G.nodes(data=True):
            if data.get("type") != "Announcement":
                continue
            if stock_code and data.get("stock_code") != stock_code:
                continue
            anns.append({
                "ann_id": node,
                "title": data.get("title", ""),
                "date": data.get("date", ""),
                "stock_code": data.get("stock_code", ""),
                "fcodes": data.get("fcodes", []),
                "fcode_names": data.get("fcode_names", []),
            })
        anns.sort(key=lambda a: a["date"])
        return anns

    # =========================================================================
    # 事件共现图构建
    # =========================================================================

    def build_cooccurrence_graph(
        self, announcements: List[Dict[str, Any]]
    ) -> nx.Graph:
        """
        构建公告共现图：
        - 两条公告共享同一股票 → 边权重+2
        - 两条公告在30天内 → 边权重+(1 - days/30)
        - 两条公告同类型 → 边权重+0.5
        """
        G = nx.Graph()
        n = len(announcements)

        for ann in announcements:
            G.add_node(ann["ann_id"], **ann)

        for i in range(n):
            for j in range(i + 1, n):
                a = announcements[i]
                b = announcements[j]
                weight = 0.0

                # 共享股票
                if a["stock_code"] == b["stock_code"]:
                    weight += 2.0

                # 时间邻近
                try:
                    from datetime import datetime
                    da = datetime.strptime(a["date"], "%Y-%m-%d")
                    db = datetime.strptime(b["date"], "%Y-%m-%d")
                    days = abs((da - db).days)
                    if days <= 30:
                        weight += max(0, 1.0 - days / 30.0)
                except (ValueError, KeyError):
                    pass

                # 同类型
                if set(a.get("fcodes", [])) & set(b.get("fcodes", [])):
                    weight += 0.5

                if weight > 0:
                    G.add_edge(a["ann_id"], b["ann_id"], weight=weight)

        return G

    # =========================================================================
    # 聚类
    # =========================================================================

    def cluster(
        self,
        announcements: Optional[List[Dict]] = None,
        stock_code: Optional[str] = None,
        resolution: float = 0.8,
    ) -> List[EventCluster]:
        """
        主入口：公告 → 事件簇。

        Args:
            announcements: 公告列表 (None=从图谱获取全部)
            stock_code: 筛选特定股票
            resolution: Louvain 分辨率参数 (越大簇越多越小)

        Returns:
            事件簇列表
        """
        if announcements is None:
            announcements = self.get_announcements(stock_code)

        if len(announcements) < 3:
            return []

        # Step 1: 构建共现图
        cg = self.build_cooccurrence_graph(announcements)
        if cg.number_of_edges() == 0:
            # 无共现关系：按股票代码简易分组
            return self._simple_grouping(announcements)

        # Step 2: Louvain 聚类
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(cg, seed=42, resolution=resolution)

        # Step 3: 为每个簇生成摘要
        clusters = []
        for cid, ann_ids in enumerate(communities):
            cluster_anns = [a for a in announcements if a["ann_id"] in ann_ids]
            if len(cluster_anns) < 2:
                continue

            # 生成时间线
            timeline = sorted(cluster_anns, key=lambda a: a["date"])

            # LLM 摘要
            if self.use_llm and self.llm:
                summary = self._llm_summarize(cluster_anns)
            else:
                summary = self._heuristic_summarize(cluster_anns)

            clusters.append(EventCluster(
                cluster_id=cid,
                name=summary.get("name", f"事件簇{cid}"),
                summary=summary.get("summary", ""),
                events=cluster_anns,
                timeline=timeline,
                key_entities=summary.get("key_entities", []),
                start_date=timeline[0]["date"] if timeline else "",
                end_date=timeline[-1]["date"] if timeline else "",
                event_count=len(cluster_anns),
                category=summary.get("category", ""),
            ))

        # 两层去重（启发 3）
        clusters = self._deterministic_dedup(clusters)
        if self.use_llm and self.llm and len(clusters) > 1:
            clusters = self._semantic_dedup(clusters)

        # 按事件数排序
        clusters.sort(key=lambda c: c.event_count, reverse=True)
        return clusters

    def _deterministic_dedup(self, clusters: List[EventCluster]) -> List[EventCluster]:
        """
        确定性去重: 同名 + 同类别 + 日期重叠 → 合并。
        """
        if len(clusters) <= 1:
            return clusters

        merged = []
        used = set()

        for i, ci in enumerate(clusters):
            if i in used:
                continue
            for j in range(i + 1, len(clusters)):
                if j in used:
                    continue
                cj = clusters[j]
                # 同名 + 同类别 → 合并
                if ci.name == cj.name and ci.category == cj.category:
                    ci.events.extend(cj.events)
                    ci.event_count = len(ci.events)
                    ci.key_entities = list(set(ci.key_entities + cj.key_entities))
                    ci.start_date = min(ci.start_date, cj.start_date)
                    ci.end_date = max(ci.end_date, cj.end_date)
                    used.add(j)
            merged.append(ci)
            used.add(i)

        return merged

    def _semantic_dedup(self, clusters: List[EventCluster]) -> List[EventCluster]:
        """
        LLM 语义去重: 名称不同但内容高度相似 → LLM 判断是否合并。
        """
        if len(clusters) <= 1:
            return clusters

        # 只比较相邻的（按事件数排序后）
        to_merge = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                ci, cj = clusters[i], clusters[j]
                # 关键词重叠 > 50% → 可能是重复
                common_entities = set(ci.key_entities) & set(cj.key_entities)
                if len(common_entities) >= 2:
                    prompt = (
                        f"判断两个事件簇是否为同一主题，应合并:\n"
                        f"簇A [{ci.name}]: {ci.summary[:100]}\n"
                        f"簇B [{cj.name}]: {cj.summary[:100]}\n"
                        f'输出 JSON: {{"same": true/false, "reason": "..."}}'
                    )
                    try:
                        result = self.llm.chat_with_json_output(prompt, temperature=0.0)
                        if result.get("same"):
                            to_merge.append((i, j))
                    except Exception:
                        pass

        # 执行合并
        used = set()
        deduped = []
        for i, c in enumerate(clusters):
            if i in used:
                continue
            for mi, mj in to_merge:
                if i == mj:  # j 被合并到 i
                    c.events.extend(clusters[mi].events)
                    c.event_count = len(c.events)
                    c.key_entities = list(set(c.key_entities + clusters[mi].key_entities))
                    used.add(mi)
            deduped.append(c)

        return deduped

    def _simple_grouping(self, announcements: List[Dict]) -> List[EventCluster]:
        """无共现时的简易分组（按股票代码）"""
        from collections import defaultdict
        groups = defaultdict(list)
        for a in announcements:
            groups[a.get("stock_code", "unknown")].append(a)

        clusters = []
        for cid, (stock, anns) in enumerate(groups.items()):
            timeline = sorted(anns, key=lambda a: a["date"])
            clusters.append(EventCluster(
                cluster_id=cid,
                name=f"{stock} 公告集合",
                summary=f"{stock} 的 {len(anns)} 条公告",
                events=anns,
                timeline=timeline,
                key_entities=[stock],
                start_date=timeline[0]["date"] if timeline else "",
                end_date=timeline[-1]["date"] if timeline else "",
                event_count=len(anns),
            ))
        return clusters

    # =========================================================================
    # LLM 摘要生成
    # =========================================================================

    def _llm_summarize(self, announcements: List[Dict]) -> Dict[str, Any]:
        """使用 LLM 为事件簇生成摘要"""
        # 收集信息
        titles = [a["title"][:100] for a in announcements[:10]]
        dates = [a["date"] for a in announcements if a.get("date")]
        fcode_names = set()
        for a in announcements:
            fcode_names.update(a.get("fcode_names", []))

        prompt = f"""请分析以下公告集合，生成事件簇的摘要。

公告列表 ({len(announcements)} 条，展示前10条):
{chr(10).join(f'{i+1}. [{a.get("date", "")}] {a.get("title", "")}' for i, a in enumerate(announcements[:10]))}

涉及的类型: {', '.join(fcode_names) if fcode_names else '多类型'}
时间跨度: {dates[0] if dates else '?'} ~ {dates[-1] if dates else '?'}

请完成:
1. 为该事件簇命名（如"实控人变更事件"、"股权质押风波"）
2. 用2-3句话描述事件的核心脉络
3. 列出3-5个关键参与实体（公司/人物）
4. 分类: 监管处罚/股权变动/并购重组/风险事件/其他

输出JSON:
{{
  "name": "事件簇名称",
  "summary": "2-3句话摘要",
  "key_entities": ["实体1", "实体2"],
  "category": "类别"
}}"""

        try:
            result = self.llm.chat_with_json_output(prompt, temperature=0.1)
            return result
        except Exception as e:
            print(f"[EventClusterer] LLM summary failed: {e}")
            return self._heuristic_summarize(announcements)

    def _heuristic_summarize(self, announcements: List[Dict]) -> Dict[str, Any]:
        """启发式摘要（无LLM fallback）"""
        fcode_names = set()
        for a in announcements:
            fcode_names.update(a.get("fcode_names", []))

        categories = set()
        for fc in fcode_names:
            categories.add(self.FCODE_CATEGORY.get(fc, fc))

        stock_codes = set(a.get("stock_code", "") for a in announcements)

        return {
            "name": f"{'、'.join(list(categories)[:2])}事件",
            "summary": f"涉及 {len(stock_codes)} 只股票，共 {len(announcements)} 条公告。类型：{'、'.join(list(categories)[:3])}。",
            "key_entities": list(stock_codes)[:5],
            "category": "、'.join(list(categories)[:1]) if categories else '其他",
        }

    # =========================================================================
    # 时间线构建
    # =========================================================================

    def build_timeline(
        self,
        stock_code: str,
        include_announcements: bool = True,
        include_equity_events: bool = True,
    ) -> Timeline:
        """
        构建股票的统一时间线：股权变动 + 公告舆情

        Args:
            stock_code: 股票代码
            include_announcements: 是否包含公告
            include_equity_events: 是否包含股权变动
        """
        events = []

        # 从图谱中提取股权变动（HOLDS 边中的日期信息）
        if include_equity_events and stock_code in self.G:
            for pred in self.G.predecessors(stock_code):
                edge = self.G.get_edge_data(pred, stock_code)
                if edge and edge.get("type") == "HOLDS":
                    holder_name = self.G.nodes.get(pred, {}).get("name", pred)
                    pct = edge.get("pct", 0)
                    end_date = edge.get("end_date", "")
                    ann_date = edge.get("ann_date", "")
                    events.append({
                        "type": "股权变动",
                        "date": end_date or ann_date or "",
                        "description": f"{holder_name} 持股 {pct:.2f}%",
                        "entity": holder_name,
                        "pct": pct,
                    })

        # 从图谱中提取公告
        if include_announcements:
            for node, data in self.G.nodes(data=True):
                if data.get("type") != "Announcement":
                    continue
                if data.get("stock_code") != stock_code:
                    continue
                events.append({
                    "type": "公告",
                    "date": data.get("date", ""),
                    "description": data.get("title", ""),
                    "fcode_names": data.get("fcode_names", []),
                })

        # 时间排序
        events.sort(key=lambda e: e.get("date", "9999"))

        # 聚类
        all_anns = [
            {"ann_id": data.get("title", ""), "title": data.get("title", ""),
             "date": data.get("date", ""), "stock_code": stock_code,
             "fcodes": data.get("fcodes", []), "fcode_names": data.get("fcode_names", [])}
            for node, data in self.G.nodes(data=True)
            if data.get("type") == "Announcement" and data.get("stock_code") == stock_code
        ]
        clusters = self.cluster(all_anns, stock_code) if all_anns else []

        return Timeline(
            stock_code=stock_code,
            events=events,
            clusters=clusters,
        )

    # =========================================================================
    # 股-舆对齐
    # =========================================================================

    def align_equity_with_news(
        self, stock_code: str, days_window: int = 90
    ) -> List[Dict[str, Any]]:
        """
        股-舆对齐：检测股权变动与舆情事件之间的时间关联。

        如果公告紧跟在股权变动后 (≤90天) → 标记为"疑似关联"
        """
        timeline = self.build_timeline(stock_code)

        equity_events = [e for e in timeline.events if e.get("type") == "股权变动"]
        news_events = [e for e in timeline.events if e.get("type") == "公告"]

        alignments = []
        try:
            from datetime import datetime, timedelta
        except ImportError:
            return []

        for ne in news_events:
            ne_date_str = ne.get("date", "")
            if not ne_date_str:
                continue
            try:
                ne_date = datetime.strptime(ne_date_str, "%Y-%m-%d")
            except ValueError:
                continue

            # 查找之前90天内的股权变动
            prior_equity = []
            for ee in equity_events:
                ee_date_str = ee.get("date", "")
                if not ee_date_str:
                    continue
                try:
                    ee_date = datetime.strptime(ee_date_str, "%Y-%m-%d")
                except ValueError:
                    continue

                days_diff = (ne_date - ee_date).days
                if 0 <= days_diff <= days_window:
                    prior_equity.append({**ee, "days_before": days_diff})

            if prior_equity:
                alignments.append({
                    "news": ne,
                    "prior_equity_events": prior_equity,
                    "suspected_causal": True,
                })

        return alignments
