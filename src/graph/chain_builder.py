"""
多跳链路构建器
解决纯数据只有1-hop关系的问题，通过以下策略实现多跳穿透：
1. 名称匹配：当"企业"股东名称匹配某上市公司时，建立跨层链接
2. 模糊匹配：用 fuzzywuzzy 找到名称高度相似的实体对
3. LLM辅助：利用LLM的预训练知识补全非上市公司的股权结构
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

from ..llm import get_llm_client


class ChainBuilder:
    """
    多跳股权链路构建器。

    核心策略：
    1. 名称精确/模糊匹配 → 将作为"企业股东"出现的实体关联到同名上市公司
    2. LLM知识补全 → 对于无法通过数据连接的链路，用LLM补全非上市公司股权结构
    """

    def __init__(self, graph: nx.DiGraph, use_llm: bool = True):
        self.G = graph
        self.llm = get_llm_client() if use_llm else None
        self.use_llm = use_llm

        # 跨层连接记录
        self._cross_links: List[Tuple[str, str, float]] = []  # (holder, stock, confidence)

    # =========================================================================
    # 策略 1: 名称匹配建跨层连接
    # =========================================================================

    def build_cross_links(self, min_confidence: float = 0.85) -> int:
        """
        在股东实体和上市公司之间建立跨层连接。

        逻辑：如果一个"企业"类型股东的名称与某上市公司高度匹配，
        则创建一条 (holder) → (stock) 的 HOLDS 边（权重=历史持股加权平均）。

        Returns:
            新建连接数
        """
        stock_nodes = {n for n, d in self.G.nodes(data=True) if d.get("type") == "Stock"}
        holder_nodes = {
            n: d for n, d in self.G.nodes(data=True)
            if d.get("type") == "Shareholder" and d.get("shareholder_type") == "企业"
        }

        # 构建股票名称索引
        stock_names: Dict[str, str] = {}
        for sn in stock_nodes:
            name = self.G.nodes[sn].get("name", sn)
            stock_names[name.lower()] = sn

        new_links = 0
        matched_pairs = []

        for hid, hdata in holder_nodes.items():
            hname = hdata.get("name", "")
            hname_lower = hname.lower()

            # 尝试精确匹配
            if hname_lower in stock_names:
                matched_pairs.append((hid, stock_names[hname_lower], 1.0))
                continue

            # 尝试模糊匹配
            try:
                from rapidfuzz import fuzz as fuzz_lib
            except ImportError:
                try:
                    from fuzzywuzzy import fuzz as fuzz_lib
                except ImportError:
                    fuzz_lib = None

            if fuzz_lib:
                best_sn = None
                best_score = 0
                for sn in stock_nodes:
                    sname = self.G.nodes[sn].get("name", sn)
                    score = fuzz_lib.token_sort_ratio(hname_lower, sname.lower())
                    if score > best_score:
                        best_score = score
                        best_sn = sn

                if best_score >= min_confidence * 100 and best_sn:
                    matched_pairs.append((hid, best_sn, best_score / 100.0))

        # 建边：holder → matched_stock
        for hid, sn, conf in matched_pairs:
            if conf >= min_confidence:
                # 计算该 holder 的历史平均持股比例
                avg_pct = self._get_avg_holding_pct(hid)
                if avg_pct > 0:
                    self.G.add_edge(
                        hid, sn,
                        type="HOLDS",
                        pct=avg_pct,
                        shares=0,
                        end_date="",
                        ann_date="",
                        share_category="",
                        cross_link=True,
                        cross_confidence=conf,
                    )
                    self._cross_links.append((hid, sn, conf))
                    new_links += 1

        print(f"[ChainBuilder] Cross-links: {len(matched_pairs)} candidates → {new_links} created (min_conf={min_confidence})")
        return new_links

    def _get_avg_holding_pct(self, holder_id: str) -> float:
        """计算某个股东的历史平均持股比例"""
        pcts = []
        for succ in self.G.successors(holder_id):
            edge = self.G.get_edge_data(holder_id, succ)
            if edge and edge.get("type") == "HOLDS":
                pct = edge.get("pct", 0)
                if pct:
                    pcts.append(pct)
        return sum(pcts) / len(pcts) if pcts else 0.0

    # =========================================================================
    # 策略 2: LLM 知识补全
    # =========================================================================

    def llm_enhance_chain(
        self,
        chain_text: str,
        target_entity: str,
    ) -> Optional[Dict[str, Any]]:
        """
        当图谱穿透在中途断开（非上市公司）时，使用 LLM 预训练知识补全。
        仅对已知知名企业有效。

        Args:
            chain_text: 当前已穿透的链路文本
            target_entity: 目标实体

        Returns:
            LLM 推断的剩余链路
        """
        if not self.llm:
            return None

        prompt = f"""你是一个金融股权结构分析专家。以下是一段股权穿透链路，但在中途断开（因为中间实体是非上市公司，数据库中没有其股东数据）。

当前已知链路:
{chain_text}

目标实体: {target_entity}

请利用你的金融知识，推断链路中**非上市公司**的股权结构。

关注以下类型的实体：
- 知名投资控股公司
- 大型国有企业集团
- 在财经新闻中经常出现的控股方

## 输出格式
{{
  "inferred_chains": [
    {{
      "entity": "断点实体名称",
      "known_holders": [
        {{"name": "股东名称", "pct": 估计持股比例, "type": "个人/企业/国有", "source": "你的知识来源"}}
      ],
      "confidence": 0.7  // 你对这个推断的信心程度
    }}
  ],
  "overall_note": "整体说明"
}}

如果无法推断任何信息，返回 {{"inferred_chains": []}}。

IMPORTANT: 只输出你确定知道的信息。不确定的不要编造。"""

        try:
            result = self.llm.chat_with_json_output(prompt, temperature=0.0)
            return result
        except Exception as e:
            print(f"[ChainBuilder] LLM enhancement failed: {e}")
            return None

    # =========================================================================
    # 策略 3: 从公告中提取股权变更线索
    # =========================================================================

    def extract_equity_from_announcements(
        self, stock_code: str
    ) -> List[Dict[str, Any]]:
        """
        从公告(3/)中提取股权变更线索。
        筛选类型码为"权益变动(5230000000)"、"股份增减持(5507200000)"、"收购兼并(5507230000)"的公告。
        """
        fcodes_of_interest = ["5230000000", "5507200000", "5507230000", "5203000000"]

        events = []
        for node, data in self.G.nodes(data=True):
            if data.get("type") != "Announcement":
                continue

            # 检查是否与该股票相关
            if data.get("stock_code") != stock_code:
                continue

            # 检查类型码
            ann_fcodes = data.get("fcodes", [])
            if not any(fc in fcodes_of_interest for fc in ann_fcodes):
                continue

            events.append({
                "ann_id": node,
                "title": data.get("title", ""),
                "date": data.get("date", ""),
                "fcode_names": data.get("fcode_names", []),
            })

        events.sort(key=lambda e: e["date"], reverse=True)
        return events

    # =========================================================================
    # 综合分析
    # =========================================================================

    def comprehensive_penetration(
        self,
        stock_code: str,
        max_depth: int = 5,
    ) -> Dict[str, Any]:
        """
        综合穿透分析：数据链路 + LLM补全 + 公告线索。

        Returns:
            {
                "data_chains": [...],      # 从数据中穿透的链路
                "llm_enhanced": {...},      # LLM 补全的信息
                "announcement_clues": [...], # 公告中的股权线索
                "summary": "..."
            }
        """
        from .equity_engine import EquityPenetrationEngine
        engine = EquityPenetrationEngine(self.G)

        # 1. 数据穿透
        data_chains = engine.penetrate(stock_code, "upstream", max_depth=max_depth, min_ratio=0.5)

        # 2. 公告线索
        ann_clues = self.extract_equity_from_announcements(stock_code)

        # 3. LLM 补全（仅当数据链路不完整时）
        llm_result = None
        incomplete = [c for c in data_chains if not c.is_complete]
        if incomplete and self.use_llm:
            chain_text = "\n".join(c.render_for_llm() for c in incomplete[:3])
            llm_result = self.llm_enhance_chain(chain_text, stock_code)

        # 4. 生成摘要
        summary_parts = []
        if data_chains:
            deepest = max(c.depth for c in data_chains)
            complete = sum(1 for c in data_chains if c.is_complete)
            summary_parts.append(f"数据穿透: {len(data_chains)}条链路, 最深{deepest}层, {complete}条完整")
        if ann_clues:
            summary_parts.append(f"公告线索: {len(ann_clues)}条")
        if llm_result and llm_result.get("inferred_chains"):
            summary_parts.append(f"LLM补全: {len(llm_result['inferred_chains'])}个断点")

        return {
            "stock_code": stock_code,
            "data_chains": [c.render_for_llm() for c in data_chains[:10]],
            "complete_chains": [c.render_for_llm() for c in data_chains if c.is_complete],
            "llm_enhanced": llm_result,
            "announcement_clues": [c["title"] for c in ann_clues[:5]],
            "summary": " | ".join(summary_parts) if summary_parts else "无穿透信息",
        }
