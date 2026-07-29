"""
图谱构建器
从 2/ 股东数据、3/ 公告数据、5/ 研报数据构建 NetworkX 知识图谱。

分层建图策略（来自 GraphRAG 启发）：
- 结构化数据（股东 2/）：直接批量建图，零 API 成本
- 半结构化数据（公告 3/）：元数据入图 + 按需 LLM 实体抽取
- 非结构化数据（研报 5/）：按需 LLM 抽取（Lazy Evaluation）
"""

import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

from ..utils.data_loader import DataLoader
from .name_matcher import NameMatcher


class StockGraphBuilder:
    """
    从 DataFrame 批量构建金融知识图谱。

    节点类型: Stock, Shareholder (个人/企业/机构/政府)
    边类型: HOLDS (含持股比例、持股数量、截止日期)
    """

    def __init__(self, loader: Optional[DataLoader] = None, use_llm: bool = False):
        self.loader = loader or DataLoader()
        self.name_matcher = NameMatcher()
        self.G = nx.DiGraph()

        # 统计
        self.stats = {
            "stock_nodes": 0,
            "shareholder_nodes": 0,
            "hold_edges": 0,
            "announcement_nodes": 0,
            "report_nodes": 0,
            "build_time_sec": 0.0,
        }

    # =========================================================================
    # 阶段 0.2：从股东数据建图
    # =========================================================================

    def build_from_shareholders(
        self,
        nrows: Optional[int] = None,
        batch_size: int = 50000,
    ) -> nx.DiGraph:
        """
        从 2/clean.xlsx 批量构建股权图谱。

        节点: Stock (上市公司) + Shareholder (股东)
        边: (Shareholder)-[:HOLDS]->(Stock) 含 pct/shares/end_date

        Args:
            nrows: 限制读取行数（None=全部 64.6万行）
            batch_size: 每批处理的行数

        Returns:
            NetworkX 有向图
        """
        t0 = time.time()
        print(f"[GraphBuilder] Loading shareholder data...")
        df = self.loader.load_shareholder_data(nrows=nrows)
        total_rows = len(df)
        print(f"[GraphBuilder] {total_rows:,} rows, batch_size={batch_size:,}")

        # 预索引：已创建的节点
        stock_nodes_created: Set[str] = set()
        holder_nodes_created: Set[str] = set()

        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch = df.iloc[batch_start:batch_end]

            # === 创建 Stock 节点 ===
            for _, row in batch.iterrows():
                stock_code = str(row.get("stock_code", ""))
                wind_code = str(row.get("s_info_windcode", ""))

                if stock_code and stock_code not in stock_nodes_created:
                    stock_nodes_created.add(stock_code)
                    self.stats["stock_nodes"] += 1
                    self.G.add_node(
                        stock_code,
                        type="Stock",
                        name=stock_code,
                        wind_code=wind_code,
                        comp_id=str(row.get("s_info_compcode", "")),
                    )

            # === 创建 Shareholder 节点 + HOLDS 边 ===
            for _, row in batch.iterrows():
                holder_raw_name = str(row.get("s_holder_name", ""))
                if not holder_raw_name or holder_raw_name == "nan":
                    continue

                # 名称匹配（别名去重）
                holder_type = row.get("holder_type", "未知")
                holder_id, match_method = self.name_matcher.match(holder_raw_name, entity_type=holder_type)

                # 创建股东节点（如果不存在）
                if holder_id not in holder_nodes_created:
                    holder_nodes_created.add(holder_id)
                    self.stats["shareholder_nodes"] += 1
                    self.G.add_node(
                        holder_id,
                        type="Shareholder",
                        name=self.name_matcher._entities.get(holder_id, {}).get("name", holder_raw_name),
                        raw_name=holder_raw_name,
                        shareholder_type=holder_type,
                        nat=str(row.get("s_holder_nat", "")),
                    )

                # 创建 HOLDS 边
                stock_code = str(row.get("stock_code", ""))
                if stock_code and stock_code in stock_nodes_created:
                    self.stats["hold_edges"] += 1
                    pct = float(row.get("s_holder_pct", 0) or 0)
                    qty = float(row.get("s_holder_quantity", 0) or 0)
                    end_date = row.get("s_holder_enddate", None)

                    self.G.add_edge(
                        holder_id, stock_code,
                        type="HOLDS",
                        pct=round(pct, 4),
                        shares=int(qty) if pd.notna(qty) else 0,
                        end_date=str(end_date)[:10] if pd.notna(end_date) else "",
                        ann_date=str(row.get("ann_dt", ""))[:10] if pd.notna(row.get("ann_dt", "")) else "",
                        share_category=str(row.get("s_holder_sharecategoryname", "")),
                    )

            if (batch_start // batch_size) % 5 == 0:
                pct_done = batch_end / total_rows * 100
                print(f"  [{pct_done:.0f}%] {batch_end:,}/{total_rows:,} rows | "
                      f"{self.stats['stock_nodes']:,} stocks, "
                      f"{self.stats['shareholder_nodes']:,} holders, "
                      f"{self.stats['hold_edges']:,} edges")

        self.stats["build_time_sec"] = round(time.time() - t0, 1)
        print(f"[GraphBuilder] DONE in {self.stats['build_time_sec']}s: "
              f"{self.stats['stock_nodes']} stocks, "
              f"{self.stats['shareholder_nodes']} holders, "
              f"{self.stats['hold_edges']} edges")
        print(f"[GraphBuilder] NameMatcher stats: {self.name_matcher.stats}")

        return self.G

    # =========================================================================
    # 阶段 2.1：从公告数据建图
    # =========================================================================

    def build_from_announcements(
        self,
        nrows: Optional[int] = None,
    ) -> int:
        """
        从 3/clean.xlsx 将公告作为节点添加到图谱。

        公告节点包含：标题、日期、类型码、类型名。
        边: (Announcement)-[:ABOUT]->(Stock)

        Returns:
            新增节点数
        """
        t0 = time.time()
        print(f"[GraphBuilder] Loading announcement data...")
        df = self.loader.load_announcements(nrows=nrows)

        fcode_map = {
            '5107000000': '利润分配', '5203000000': '质押冻结', '5219000000': '回购股权',
            '5230000000': '权益变动', '5404000000': '补充更正', '5406000000': '业绩预告',
            '5502010000': '特别处理', '5502040000': '终止上市', '5506010000': '股东大会',
            '5506040000': '风险提示', '5506050000': '重大合同', '5506100000': '澄清公告',
            '5506140000': '停牌提示', '5506160000': '中介公告', '5506170000': '法律纠纷',
            '5506180000': '公司资料变更', '5506190000': '个股其他公告',
            '5506220000': '员工持股', '5507040000': '关联交易', '5507060000': '违纪违规',
            '5507200000': '股份增减持', '5507210000': '资金投向', '5507220000': '资产重组',
            '5507230000': '收购兼并', '5507240000': '借贷担保', '5507260000': '政策影响',
            '5507270000': '人事变动', '5508000000': '函件',
        }

        ann_count = 0
        for _, row in df.iterrows():
            ann_id = str(row.get("object_id", f"ann_{ann_count}"))
            stock_code = str(row.get("stock_code", ""))

            # 解析公告类型
            fcodes_raw = str(row.get("n_info_fcode", ""))
            fcodes = fcodes_raw.split("|") if fcodes_raw else []
            fcode_names = [fcode_map.get(fc, fc) for fc in fcodes if fc]

            self.G.add_node(
                ann_id,
                type="Announcement",
                title=str(row.get("n_info_title", "")),
                date=str(row.get("ann_dt", ""))[:10] if pd.notna(row.get("ann_dt", "")) else "",
                fcodes=fcodes,
                fcode_names=fcode_names,
                stock_code=stock_code,
            )
            self.stats["announcement_nodes"] += 1
            ann_count += 1

            # 连接到股票
            if stock_code and stock_code in self.G:
                self.G.add_edge(ann_id, stock_code, type="ABOUT")

        print(f"[GraphBuilder] {ann_count} announcements added in {time.time()-t0:.1f}s")
        return ann_count

    # =========================================================================
    # 阶段 3（可选）：从研报数据建图
    # =========================================================================

    def build_from_reports(
        self,
        nrows: Optional[int] = None,
        stock_codes: Optional[List[str]] = None,
    ) -> int:
        """
        从 5/rr_main_*.csv 按需建图（Lazy Evaluation）。
        仅对指定的股票列表抽取研报实体。

        Args:
            nrows: 限制读取行数
            stock_codes: 关注的股票代码列表（None=全部）

        Returns:
            新增节点数
        """
        t0 = time.time()
        print(f"[GraphBuilder] Loading research report data...")
        df = self.loader.load_research_reports(nrows=nrows)

        if stock_codes:
            # 仅保留关注股票的研报
            if "sec_code" in df.columns:
                df = df[df["sec_code"].apply(
                    lambda x: any(c in str(x) for c in stock_codes)
                )]

        report_count = 0
        for _, row in df.iterrows():
            report_id = str(row.get("report_id", f"report_{report_count}"))
            title = str(row.get("title", ""))
            org = str(row.get("org_name", ""))
            stock_code = str(row.get("sec_code", ""))

            self.G.add_node(
                report_id,
                type="Report",
                title=title,
                org_name=org,
                author=str(row.get("author", "")),
                date=str(row.get("write_date", ""))[:10] if pd.notna(row.get("write_date", "")) else "",
                rating=str(row.get("rating_org", "")),
                abstract=str(row.get("abstract", ""))[:500],
                industry=str(row.get("industry_l1", "")),
            )
            self.stats["report_nodes"] += 1
            report_count += 1

            # 连接研报到股票
            if stock_code and stock_code in self.G:
                self.G.add_edge(report_id, stock_code, type="COVERS")

        print(f"[GraphBuilder] {report_count} reports added in {time.time()-t0:.1f}s")
        return report_count

    # =========================================================================
    # 图查询接口
    # =========================================================================

    def get_direct_holders(self, stock_code: str) -> List[Dict[str, Any]]:
        """获取某只股票的直接股东列表（按持股比例排序）"""
        if stock_code not in self.G:
            return []

        holders = []
        for pred in self.G.predecessors(stock_code):
            edge = self.G.get_edge_data(pred, stock_code)
            if edge and edge.get("type") == "HOLDS":
                node = self.G.nodes.get(pred, {})
                holders.append({
                    "holder_id": pred,
                    "holder_name": node.get("name", pred),
                    "holder_type": node.get("shareholder_type", ""),
                    "pct": edge.get("pct", 0),
                    "shares": edge.get("shares", 0),
                    "end_date": edge.get("end_date", ""),
                })

        holders.sort(key=lambda x: x["pct"], reverse=True)
        return holders

    def get_direct_subsidiaries(self, entity_id: str) -> List[Dict[str, Any]]:
        """获取某个实体直接持股的公司列表"""
        if entity_id not in self.G:
            return []

        subsidiaries = []
        for succ in self.G.successors(entity_id):
            edge = self.G.get_edge_data(entity_id, succ)
            if edge and edge.get("type") == "HOLDS":
                node = self.G.nodes.get(succ, {})
                subsidiaries.append({
                    "stock_code": succ,
                    "stock_name": node.get("name", succ),
                    "pct": edge.get("pct", 0),
                    "shares": edge.get("shares", 0),
                })

        subsidiaries.sort(key=lambda x: x["pct"], reverse=True)
        return subsidiaries

    def search_entity(self, name: str) -> List[str]:
        """按名称搜索实体节点"""
        matches = []
        name_lower = name.lower()
        for node, data in self.G.nodes(data=True):
            node_name = str(data.get("name", "")).lower()
            raw_name = str(data.get("raw_name", "")).lower()
            if name_lower in node_name or name_lower in raw_name:
                matches.append(node)
        return matches

    def get_stock_announcements(self, stock_code: str) -> List[Dict]:
        """获取某只股票的所有公告"""
        if stock_code not in self.G:
            return []

        anns = []
        for pred in self.G.predecessors(stock_code):
            node = self.G.nodes.get(pred, {})
            if node.get("type") == "Announcement":
                anns.append({
                    "ann_id": pred,
                    "title": node.get("title", ""),
                    "date": node.get("date", ""),
                    "fcode_names": node.get("fcode_names", []),
                })

        anns.sort(key=lambda x: x.get("date", ""), reverse=True)
        return anns

    def summary(self) -> str:
        """图谱摘要"""
        type_counts = {}
        for _, data in self.G.nodes(data=True):
            t = data.get("type", "Unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        type_str = " | ".join(f"{k}:{v}" for k, v in sorted(type_counts.items()))
        return (
            f"StockGraph: {self.G.number_of_nodes():,} nodes, "
            f"{self.G.number_of_edges():,} edges | {type_str} | "
            f"build in {self.stats['build_time_sec']}s"
        )

    def to_graphml(self, path: str):
        """导出为 GraphML 格式"""
        nx.write_graphml(self.G, path)
        print(f"[GraphBuilder] Exported to {path}")
