"""
股权穿透引擎 — 任务二最核心的算法模块
BFS 多跳遍历 + 控股权累积计算 + 环检测 + 链路渲染
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx


@dataclass
class Hop:
    """股权穿透链路中的一跳"""
    from_entity: str       # 来源实体 ID
    from_name: str          # 来源实体名称
    to_entity: str          # 目标实体 ID
    to_name: str            # 目标实体名称
    ratio: float            # 持股比例 (%)
    shares: int             # 持股数量
    level: int              # 层级深度（从1开始）
    entity_type: str        # 来源实体类型 (个人/企业/机构)

    def render(self) -> str:
        return f"{self.from_name}({self.ratio:.2f}%)"


@dataclass
class PenetrationChain:
    """一条完整的穿透链路"""
    path: List[Hop] = field(default_factory=list)
    cumulative_control: float = 0.0    # 累积控股权 (%)
    depth: int = 0                      # 穿透深度
    is_complete: bool = False           # 是否穿透到底（个人/国有法人）
    termination_reason: str = ""        # 终止原因
    has_cross_holding: bool = False     # 是否包含交叉持股
    source: str = "graph"              # "graph" | "llm" | "hybrid"

    def render_for_llm(self) -> str:
        """渲染为 LLM 可读文本"""
        if not self.path:
            return "（无穿透链路）"

        segments = []
        for hop in self.path:
            segments.append(f"{hop.from_name}({hop.ratio:.2f}%)")
        segments.append(f"{self.path[-1].to_name}")

        arrow = " → "
        result = arrow.join(segments)
        result += f"  [累积控股权: {self.cumulative_control:.4f}%"
        if self.is_complete:
            result += ", 已穿透到底"
        if self.termination_reason:
            result += f", {self.termination_reason}"
        result += "]"
        return result

    def render_tree(self, indent: int = 0) -> str:
        """渲染为树形结构"""
        prefix = "  " * indent
        if not self.path:
            return f"{prefix}（无链路）"
        lines = [f"{prefix}├─ 累积控股权: {self.cumulative_control:.4f}%"]
        for hop in self.path:
            lines.append(f"{prefix}│  L{hop.level}: {hop.from_name} → {hop.to_name} ({hop.ratio:.2f}%)")
        return "\n".join(lines)


class EquityPenetrationEngine:
    """
    股权穿透引擎。

    核心算法：BFS 多跳遍历，从目标实体出发，沿 HOLDS 边反向（向上溯源）
    或正向（向下穿透）遍历，累计计算控股权。

    控股权计算：
    - 直接持股 = Σ(所有直接股东持股比例)
    - 间接持股 = Σ(路径上所有边的比例连乘)
    - 合并 = 直接 + 间接（多路径汇聚到同一最终控制人时求和）
    """

    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self._cache: Dict[str, List[PenetrationChain]] = {}  # 查询缓存

    # =========================================================================
    # 主入口
    # =========================================================================

    def penetrate(
        self,
        target: str,
        direction: str = "upstream",
        max_depth: int = 5,
        min_ratio: float = 0.5,
        use_cache: bool = True,
    ) -> List[PenetrationChain]:
        """
        股权穿透主入口。

        Args:
            target: 目标实体 ID（股票代码或股东ID）
            direction: "upstream"(向上溯源) | "downstream"(向下穿透) | "both"
            max_depth: 最大穿透深度（层数）
            min_ratio: 最小持股比例阈值 (%)
            use_cache: 是否使用缓存

        Returns:
            排序后的穿透链路列表（按累积控股权降序）
        """
        cache_key = f"{target}|{direction}|{max_depth}|{min_ratio}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # 如果 target 是股票代码，先尝试找到对应的节点
        if target not in self.G:
            # 尝试模糊搜索
            candidates = self._find_by_name(target)
            if candidates:
                target = candidates[0]
            else:
                return []

        chains = []

        if direction in ("upstream", "both"):
            upstream_chains = self._bfs_upstream(target, max_depth, min_ratio)
            chains.extend(upstream_chains)

        if direction in ("downstream", "both"):
            downstream_chains = self._bfs_downstream(target, max_depth, min_ratio)
            chains.extend(downstream_chains)

        # 按累积控股权排序
        chains.sort(key=lambda c: c.cumulative_control, reverse=True)

        if use_cache:
            self._cache[cache_key] = chains

        return chains

    # =========================================================================
    # BFS 向上溯源（谁持有 target？）
    # =========================================================================

    def _bfs_upstream(
        self, target: str, max_depth: int, min_ratio: float
    ) -> List[PenetrationChain]:
        """
        BFS 向上溯源：找到 target 的所有股东，递归向上追到最终控制人。

        每层：遍历所有 predecessors（持有者），检查他们是否还被其他人持有。
        """
        chains = []
        # visited 用于防止环（交叉持股）
        visited: Set[str] = {target}
        # BFS queue: (current_entity, depth, path_so_far, cumulative_ratio)
        queue = deque()

        # 第 1 层：直接股东
        for pred in self.G.predecessors(target):
            edge = self.G.get_edge_data(pred, target)
            if not edge or edge.get("type") != "HOLDS":
                continue

            pct = edge.get("pct", 0)
            if pct < min_ratio:
                continue

            pred_data = self.G.nodes.get(pred, {})
            hop = Hop(
                from_entity=pred,
                from_name=pred_data.get("name", pred),
                to_entity=target,
                to_name=self.G.nodes.get(target, {}).get("name", target),
                ratio=pct,
                shares=edge.get("shares", 0),
                level=1,
                entity_type=pred_data.get("shareholder_type", ""),
            )
            chain = PenetrationChain(
                path=[hop],
                cumulative_control=pct,
                depth=1,
            )
            chains.append(chain)

            # 如果这个股东本身也是公司（可能被穿透），继续 BFS
            if self._is_penetratable(pred):
                visited.add(pred)
                queue.append((pred, 1, [hop], pct))

        # 多层 BFS
        while queue:
            entity, depth, path_so_far, cumulative = queue.popleft()

            if depth >= max_depth:
                continue

            for pred in self.G.predecessors(entity):
                if pred in visited:
                    # 检测到环 → 交叉持股
                    continue

                edge = self.G.get_edge_data(pred, entity)
                if not edge or edge.get("type") != "HOLDS":
                    continue

                pct = edge.get("pct", 0)
                if pct < min_ratio:
                    continue

                visited.add(pred)
                pred_data = self.G.nodes.get(pred, {})
                entity_data = self.G.nodes.get(entity, {})
                hop = Hop(
                    from_entity=pred,
                    from_name=pred_data.get("name", pred),
                    to_entity=entity,
                    to_name=entity_data.get("name", entity),
                    ratio=pct,
                    shares=edge.get("shares", 0),
                    level=depth + 1,
                    entity_type=pred_data.get("shareholder_type", ""),
                )

                new_path = path_so_far + [hop]
                # 累积控股权 = 路径上所有比例连乘
                new_cumulative = cumulative * (pct / 100.0)

                chain = PenetrationChain(
                    path=new_path,
                    cumulative_control=round(new_cumulative * 100, 4),
                    depth=depth + 1,
                )

                # 检查穿透终止条件
                if self._is_penetration_endpoint(pred):
                    chain.is_complete = True
                    chain.termination_reason = self._get_termination_reason(pred)
                else:
                    queue.append((pred, depth + 1, new_path, new_cumulative))

                chains.append(chain)

        return chains

    # =========================================================================
    # BFS 向下穿透（target 持有谁？）
    # =========================================================================

    def _bfs_downstream(
        self, target: str, max_depth: int, min_ratio: float
    ) -> List[PenetrationChain]:
        """
        BFS 向下穿透：找到 target 持股的所有公司，递归向下。
        镜像 bfs_upstream 逻辑，方向改为 successors。
        """
        chains = []
        visited: Set[str] = {target}
        queue = deque()

        # 第 1 层
        for succ in self.G.successors(target):
            edge = self.G.get_edge_data(target, succ)
            if not edge or edge.get("type") != "HOLDS":
                continue

            pct = edge.get("pct", 0)
            if pct < min_ratio:
                continue

            succ_data = self.G.nodes.get(succ, {})
            target_data = self.G.nodes.get(target, {})
            hop = Hop(
                from_entity=target,
                from_name=target_data.get("name", target),
                to_entity=succ,
                to_name=succ_data.get("name", succ),
                ratio=pct,
                shares=edge.get("shares", 0),
                level=1,
                entity_type=target_data.get("shareholder_type", ""),
            )
            chain = PenetrationChain(
                path=[hop],
                cumulative_control=pct,
                depth=1,
            )
            chains.append(chain)

            if self._is_penetratable(succ):
                visited.add(succ)
                queue.append((succ, 1, [hop]))

        while queue:
            entity, depth, path_so_far = queue.popleft()
            if depth >= max_depth:
                continue

            for succ in self.G.successors(entity):
                if succ in visited:
                    continue

                edge = self.G.get_edge_data(entity, succ)
                if not edge or edge.get("type") != "HOLDS":
                    continue

                pct = edge.get("pct", 0)
                if pct < min_ratio:
                    continue

                visited.add(succ)
                entity_data = self.G.nodes.get(entity, {})
                succ_data = self.G.nodes.get(succ, {})
                hop = Hop(
                    from_entity=entity,
                    from_name=entity_data.get("name", entity),
                    to_entity=succ,
                    to_name=succ_data.get("name", succ),
                    ratio=pct,
                    shares=edge.get("shares", 0),
                    level=depth + 1,
                    entity_type=entity_data.get("shareholder_type", "企业"),
                )

                last_ratio = path_so_far[-1].ratio / 100.0 if path_so_far else 1.0
                new_path = path_so_far + [hop]
                chain = PenetrationChain(
                    path=new_path,
                    cumulative_control=round(last_ratio * pct / 100.0 * 100, 4),
                    depth=depth + 1,
                )

                if self._is_penetration_endpoint(succ):
                    chain.is_complete = True
                    chain.termination_reason = self._get_termination_reason(succ)

                chains.append(chain)
                queue.append((succ, depth + 1, new_path))

        return chains

    # =========================================================================
    # 高级查询接口
    # =========================================================================

    def find_ultimate_controller(self, stock_code: str) -> Optional[PenetrationChain]:
        """
        找到实际控制人：向上穿透到底（到达个人或国有法人），
        返回累积控股权最高的完整链路。
        """
        chains = self.penetrate(stock_code, "upstream", max_depth=10, min_ratio=0.5)
        complete = [c for c in chains if c.is_complete]
        if complete:
            return complete[0]  # 已按控股权排序
        # 如果没有完整链路，返回最长的那条
        if chains:
            return max(chains, key=lambda c: c.depth)
        return None

    def compare_holders(self, stock_a: str, stock_b: str) -> Dict[str, Any]:
        """对比两只股票的股东结构"""
        holders_a = {h["holder_name"]: h for h in self._get_graph_builder().get_direct_holders(stock_a)}
        holders_b = {h["holder_name"]: h for h in self._get_graph_builder().get_direct_holders(stock_b)}

        common = set(holders_a.keys()) & set(holders_b.keys())
        return {
            "stock_a": stock_a,
            "stock_b": stock_b,
            "holders_a_count": len(holders_a),
            "holders_b_count": len(holders_b),
            "common_holders": [
                {"name": name, "a_pct": holders_a[name]["pct"], "b_pct": holders_b[name]["pct"]}
                for name in common
            ],
            "common_count": len(common),
        }

    def get_control_summary(self, stock_code: str) -> Dict[str, Any]:
        """
        获取某只股票的控股权摘要。
        包括：实际控制人、Top 5 股东、控股集中度。
        """
        holders = self._get_graph_builder().get_direct_holders(stock_code)
        ultimate = self.find_ultimate_controller(stock_code)

        top5_pct = sum(h["pct"] for h in holders[:5])

        return {
            "stock_code": stock_code,
            "total_holders": len(holders),
            "top5_concentration": round(top5_pct, 2),
            "ultimate_controller": ultimate.render_for_llm() if ultimate else "未找到",
            "top_holders": [
                {"name": h["holder_name"], "pct": round(h["pct"], 2), "type": h["holder_type"]}
                for h in holders[:5]
            ],
        }

    # =========================================================================
    # 内部方法
    # =========================================================================

    def _is_penetratable(self, entity_id: str) -> bool:
        """判断该实体是否可以被继续穿透（即：它本身是否持有其他公司的股份）"""
        node = self.G.nodes.get(entity_id, {})
        # 如果是个人或国有法人 → 穿透终止
        ntype = node.get("shareholder_type", node.get("type", ""))
        nat = node.get("nat", "")
        if ntype in ("个人",) or nat in ("国有法人", "国家"):
            return False
        # 检查是否作为持股方出现在其他 HOLDS 边中
        for succ in self.G.successors(entity_id):
            edge = self.G.get_edge_data(entity_id, succ)
            if edge and edge.get("type") == "HOLDS":
                return True
        return False

    def _is_penetration_endpoint(self, entity_id: str) -> bool:
        """判断是否到达穿透终点"""
        node = self.G.nodes.get(entity_id, {})
        ntype = node.get("shareholder_type", "")

        # 个人 → 终点
        if ntype == "个人":
            return True

        # 国有法人/国家 → 终点
        nat = node.get("nat", "")
        if nat in ("国有法人", "国家"):
            return True

        # 不再持有其他公司股份 → 终点
        has_holdings = False
        for succ in self.G.successors(entity_id):
            edge = self.G.get_edge_data(entity_id, succ)
            if edge and edge.get("type") == "HOLDS":
                has_holdings = True
                break
        if not has_holdings:
            return True

        return False

    def _get_termination_reason(self, entity_id: str) -> str:
        """获取穿透终止原因"""
        node = self.G.nodes.get(entity_id, {})
        ntype = node.get("shareholder_type", "")
        nat = node.get("nat", "")
        if ntype == "个人":
            return "到达自然人"
        if nat in ("国有法人", "国家"):
            return "到达国有法人/国家"
        return "无进一步持股关系"

    def _find_by_name(self, name: str, top_k: int = 5, min_score: float = 0.3) -> List[str]:
        """
        多信号实体匹配 (Priority 1 升级)。
        原仅用子字符串 in 查找，现升级为:
          - 子字符串包含 → +0.3
          - BM25 token overlap → +0.4
          - 模糊匹配 (token_sort_ratio if available) → +0.3
        返回按综合得分降序的候选实体列表。
        """
        candidates: List[Tuple[str, float]] = []
        name_lower = name.lower().strip()

        # 尝试导入模糊匹配库
        try:
            from rapidfuzz import fuzz as fuzz_lib
        except ImportError:
            try:
                from fuzzywuzzy import fuzz as fuzz_lib
            except ImportError:
                fuzz_lib = None

        for node, data in self.G.nodes(data=True):
            node_name = str(data.get("name", "")).lower()
            raw_name = str(data.get("raw_name", "")).lower()

            score = 0.0

            # 信号1: 子字符串包含 (精确匹配优先)
            if name_lower == node_name or name_lower == raw_name:
                score += 0.5  # 精确匹配高分
            elif name_lower in node_name or name_lower in raw_name:
                score += 0.3
            elif node_name in name_lower or raw_name in name_lower:
                score += 0.25

            # 信号2: 词级别的 token overlap (简化 BM25-like)
            query_tokens = set(name_lower.split())
            node_tokens = set(node_name.split())
            if query_tokens and node_tokens:
                overlap = len(query_tokens & node_tokens)
                token_score = overlap / max(len(query_tokens), 1)
                score += token_score * 0.4

            # 信号3: 模糊匹配 (fuzzywuzzy/rapidfuzz)
            if fuzz_lib and score < 0.5:  # 仅当精确匹配不够时
                fuzzy_score = fuzz_lib.token_sort_ratio(name_lower, node_name) / 100.0
                if fuzzy_score >= 0.85:
                    score += fuzzy_score * 0.3
                elif fuzzy_score >= 0.70:
                    score += fuzzy_score * 0.2

            if score >= min_score:
                candidates.append((node, round(score, 3)))

        # 按得分降序
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in candidates[:top_k]]

    def _get_graph_builder(self):
        """延迟导入避免循环依赖"""
        from .graph_builder import StockGraphBuilder
        # 这里需要一个已构建的 builder 实例来用查询方法
        # 简化：直接从 G 复制逻辑
        return _GraphQueryHelper(self.G)

    def clear_cache(self):
        """清除查询缓存"""
        self._cache.clear()


class _GraphQueryHelper:
    """内部辅助类：轻量图查询"""
    def __init__(self, G):
        self.G = G

    def get_direct_holders(self, stock_code: str) -> List[Dict]:
        holders = []
        for pred in self.G.predecessors(stock_code):
            edge = self.G.get_edge_data(pred, stock_code)
            if edge and edge.get("type") == "HOLDS":
                node = self.G.nodes.get(pred, {})
                holders.append({
                    "holder_name": node.get("name", pred),
                    "holder_type": node.get("shareholder_type", ""),
                    "pct": edge.get("pct", 0),
                    "shares": edge.get("shares", 0),
                })
        holders.sort(key=lambda x: x["pct"], reverse=True)
        return holders
