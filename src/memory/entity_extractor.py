"""
实体抽取 Pipeline
从对话文本中提取金融实体和关系，包含去重和 ID 归一化。
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..llm import get_llm_client
from ..llm.prompts import ENTITY_EXTRACTION_PROMPT


# 已知的 A 股股票名称 → 代码映射（小字典，后续从数据集自动构建）
STOCK_ALIASES: Dict[str, str] = {
    "贵州茅台": "600519",
    "茅台": "600519",
    "五粮液": "000858",
    "宁德时代": "300750",
    "比亚迪": "002594",
    "隆基绿能": "601012",
    "招商银行": "600036",
    "中国平安": "601318",
}

# 金融指标归一化映射
INDICATOR_ALIASES: Dict[str, str] = {
    "净资产收益率": "ROE",
    "市盈率": "PE",
    "市净率": "PB",
    "存货周转率": "存货周转率",
    "存货周转天数": "存货周转天数",
    "经营性现金流": "经营性现金流",
    "经营现金流": "经营性现金流",
    "营业收入": "营业收入",
    "营收": "营业收入",
    "净利润": "净利润",
    "归母净利润": "归母净利润",
    "毛利率": "毛利率",
    "净利率": "净利率",
    "资产负债率": "资产负债率",
    "ROE": "ROE",
    "ROA": "ROA",
    "PE": "PE",
    "PB": "PB",
}


class EntityExtractor:
    """从金融对话中提取实体和关系，支持 LLM 提取 + 规则兜底"""

    def __init__(self, use_llm: bool = True):
        self.llm = get_llm_client() if use_llm else None
        self.use_llm = use_llm

        # 动态股票字典（从数据集中学习）
        self.stock_dict: Dict[str, str] = dict(STOCK_ALIASES)

        # 实体去重集合
        self._seen_entities: Dict[str, str] = {}  # name → entity_id

    def build_stock_dict_from_data(self, shareholder_df, announcements_df=None):
        """从数据集中自动构建股票名称→代码映射"""
        if "s_holder_aname" in shareholder_df.columns:
            # 从股东数据中无法直接获取股票名称，跳过
            pass
        if "stock_code" in shareholder_df.columns:
            # 记录所有出现的股票代码
            for code in shareholder_df["stock_code"].dropna().unique():
                self.stock_dict[str(code)] = str(code)
        print(f"[EntityExtractor] Stock dict size: {len(self.stock_dict)}")

    def extract(self, user_query: str, agent_response: str = "") -> Tuple[List[Dict], List[Dict]]:
        """
        从一轮对话中提取实体和关系。

        Args:
            user_query: 用户输入
            agent_response: 助手回复（可选）

        Returns:
            (entities, relations): 实体列表和关系列表
        """
        turn_text = f"用户: {user_query}"
        if agent_response:
            turn_text += f"\n助手: {agent_response}"

        if self.use_llm and self.llm:
            entities, relations = self._llm_extract(turn_text)
        else:
            entities, relations = self._rule_extract(user_query)

        # 后处理：去重、归一化
        entities = self._normalize_entities(entities)
        entities = self._deduplicate_entities(entities)
        relations = self._validate_relations(relations, entities)

        return entities, relations

    def _llm_extract(self, turn_text: str) -> Tuple[List[Dict], List[Dict]]:
        """使用 LLM 进行实体抽取"""
        try:
            prompt = ENTITY_EXTRACTION_PROMPT.format(conversation_turn=turn_text)
            result = self.llm.chat_with_json_output(
                user_prompt=prompt,
                temperature=0.0,
                max_retries=1,
            )
            entities = result.get("entities", [])
            relations = result.get("relations", [])
            return entities, relations
        except Exception as e:
            print(f"[EntityExtractor] LLM extraction failed: {e}, falling back to rules")
            return self._rule_extract(turn_text)

    def _rule_extract(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """基于规则的实体抽取（兜底方案）"""
        entities = []
        relations = []

        # 1. 股票代码匹配: 6位数字
        code_pattern = re.compile(r'\b(\d{6})\b')
        for match in code_pattern.finditer(text):
            code = match.group(1)
            entity_id = f"stock_{code}"
            entities.append({
                "id": entity_id, "type": "Stock",
                "name": code, "code": code
            })

        # 2. 已知股票名称匹配
        for name, code in self.stock_dict.items():
            if name in text and len(name) >= 2:
                entity_id = f"stock_{code}"
                if not any(e["id"] == entity_id for e in entities):
                    entities.append({
                        "id": entity_id, "type": "Stock",
                        "name": name, "code": code
                    })

        # 3. 金融指标匹配
        for alias, canonical in INDICATOR_ALIASES.items():
            if alias in text:
                entity_id = f"indicator_{self._safe_id(canonical)}"
                if not any(e["id"] == entity_id for e in entities):
                    entities.append({
                        "id": entity_id, "type": "Indicator",
                        "name": canonical, "category": self._infer_indicator_category(canonical)
                    })

        # 4. 关系构建（实体→当前轮次）
        for entity in entities:
            relations.append({
                "source": "current_turn",
                "target": entity["id"],
                "type": "MENTIONS"
            })

        return entities, relations

    def _normalize_entities(self, entities: List[Dict]) -> List[Dict]:
        """实体归一化：统一名称、补全信息"""
        for entity in entities:
            entity_type = entity.get("type", "")

            if entity_type == "Stock":
                # 补全股票代码
                name = entity.get("name", "")
                if name in self.stock_dict:
                    entity["code"] = self.stock_dict[name]
                if "code" in entity and entity["code"]:
                    entity["id"] = f"stock_{self._normalize_code(entity['code'])}"

            elif entity_type == "Indicator":
                name = entity.get("name", "")
                if name in INDICATOR_ALIASES:
                    entity["name"] = INDICATOR_ALIASES[name]
                entity["id"] = f"indicator_{self._safe_id(entity.get('name', ''))}"
                if "category" not in entity or not entity.get("category"):
                    entity["category"] = self._infer_indicator_category(entity.get("name", ""))

            elif entity_type == "Person":
                name = entity.get("name", "")
                entity["id"] = f"person_{self._safe_id(name)}"

            elif entity_type == "Event":
                name = entity.get("name", "")
                entity["id"] = f"event_{self._safe_id(name)}"

            elif entity_type == "Organization":
                name = entity.get("name", "")
                entity["id"] = f"org_{self._safe_id(name)}"

            elif entity_type == "Report":
                name = entity.get("name", "")
                entity["id"] = f"report_{self._safe_id(name)}"

            # 确保有 id 字段
            if "id" not in entity:
                entity["id"] = f"{entity_type.lower()}_{self._safe_id(entity.get('name', 'unknown'))}"

        return entities

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """实体去重：相同 ID 合并，保留信息更丰富的版本"""
        deduped: Dict[str, Dict] = {}
        for entity in entities:
            eid = entity["id"]
            if eid in deduped:
                # 合并：保留包含更多信息的版本
                existing = deduped[eid]
                for key, val in entity.items():
                    if val and (key not in existing or not existing.get(key)):
                        existing[key] = val
            else:
                deduped[eid] = dict(entity)
        return list(deduped.values())

    def _validate_relations(self, relations: List[Dict], entities: List[Dict]) -> List[Dict]:
        """验证关系的有效性"""
        valid_ids = {e["id"] for e in entities}
        valid_ids.add("current_turn")
        valid = []
        for rel in relations:
            if rel.get("source") in valid_ids and rel.get("target") in valid_ids:
                valid.append(rel)
        return valid

    # ---- 工具方法 ----

    @staticmethod
    def _normalize_code(code: str) -> str:
        """统一股票代码为6位"""
        code = str(code).strip().upper()
        for suffix in [".SH", ".SZ", ".BJ", ".N", ".HK"]:
            if code.endswith(suffix):
                code = code[:-len(suffix)]
        return code.zfill(6)

    @staticmethod
    def _safe_id(text: str) -> str:
        """将文本转换为安全的 ID 片段（英文字母、数字、下划线）"""
        # 保留中文字符的拼音首字母？这里简单处理
        safe = re.sub(r'[^a-zA-Z0-9_一-鿿]', '_', str(text))
        return safe[:50] if safe else "unknown"

    @staticmethod
    def _infer_indicator_category(name: str) -> str:
        """推断金融指标的类别"""
        profitability = ["ROE", "ROA", "毛利率", "净利率", "净利润", "归母净利润", "营业收入", "营收"]
        solvency = ["资产负债率", "流动比率", "速动比率"]
        efficiency = ["存货周转率", "存货周转天数", "应收账款周转率", "总资产周转率"]
        cashflow = ["经营性现金流", "自由现金流", "现金流"]
        valuation = ["PE", "PB", "市净率", "市盈率", "市值"]

        for cat_name, keywords in [
            ("盈利能力", profitability),
            ("偿债能力", solvency),
            ("运营效率", efficiency),
            ("现金流质量", cashflow),
            ("估值水平", valuation),
        ]:
            if any(kw in name for kw in keywords):
                return cat_name
        return "其他"


def test_entity_extraction():
    """测试实体抽取效果"""
    extractor = EntityExtractor(use_llm=True)

    test_queries = [
        "帮我看看贵州茅台的ROE最近五年变化趋势",
        "宁德时代的存货周转率异常高，有没有财务造假风险？",
        "王旭宁通过哪些公司控股九阳股份？给我穿透链路",
        "600519和000858的现金流对比",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        entities, relations = extractor.extract(query)
        print(f"Entities ({len(entities)}):")
        for e in entities:
            print(f"  {e['type']:15s} | {e['name']:20s} | id={e['id']}")
        print(f"Relations ({len(relations)}):")
        for r in relations:
            print(f"  {r['source']} --[{r['type']}]--> {r['target']}")


if __name__ == "__main__":
    test_entity_extraction()
