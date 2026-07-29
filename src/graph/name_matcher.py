"""
股东名称匹配与去重引擎
处理 71,328 个股东的名称歧义问题：精确匹配 → 模糊匹配 → LLM 对齐
"""

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


class NameMatcher:
    """
    股东名称匹配器：三级匹配策略
    Level 1: 精确匹配 (O(1) 哈希查找)
    Level 2: 模糊匹配 (token sort ratio > 90%)
    Level 3: LLM 对齐 (兜底，判断两个名称是否为同一实体)
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client
        # 精确匹配索引: cleaned_name → canonical_id
        self._exact_index: Dict[str, str] = {}
        # 别名字典: alias → canonical_name (从历史匹配中学习)
        self._alias_dict: Dict[str, str] = {}
        # 已注册的实体
        self._entities: Dict[str, Dict] = {}  # canonical_id → entity_info
        # 统计
        self.stats = {"exact": 0, "fuzzy": 0, "llm": 0, "new": 0}

    def normalize(self, name: str) -> str:
        """名称清洗：去空格、去括号、统一全角半角"""
        if pd.isna(name):
            return ""
        name = str(name).strip()
        # 全角→半角
        name = name.replace("（", "(").replace("）", ")")
        name = name.replace("，", ",").replace("；", ";")
        # 去多余空格
        name = re.sub(r'\s+', '', name)
        return name

    def match(self, name: str, entity_type: str = "unknown") -> Tuple[str, str]:
        """
        三级匹配：返回 (canonical_id, match_method)

        Args:
            name: 原始股东名称
            entity_type: 实体类型 (个人/企业/机构)

        Returns:
            (canonical_id, "exact" | "fuzzy" | "llm" | "new")
        """
        clean = self.normalize(name)
        if not clean:
            return ("", "new")

        # Level 1: 精确匹配
        if clean in self._exact_index:
            self.stats["exact"] += 1
            return (self._exact_index[clean], "exact")

        # Level 2: 模糊匹配
        fuzzy_id = self._fuzzy_match(clean)
        if fuzzy_id:
            self.stats["fuzzy"] += 1
            # 学习这个别名
            self._alias_dict[clean] = fuzzy_id
            self._exact_index[clean] = fuzzy_id
            return (fuzzy_id, "fuzzy")

        # Level 3: LLM 对齐（可选）
        if self.llm:
            llm_id = self._llm_align(clean, entity_type)
            if llm_id:
                self.stats["llm"] += 1
                self._alias_dict[clean] = llm_id
                self._exact_index[clean] = llm_id
                return (llm_id, "llm")

        # 新实体
        self.stats["new"] += 1
        canonical_id = self._generate_id(clean, entity_type)
        self._exact_index[clean] = canonical_id
        self._entities[canonical_id] = {
            "name": clean,
            "type": entity_type,
            "aliases": [clean],
        }
        return (canonical_id, "new")

    def _fuzzy_match(self, name: str) -> Optional[str]:
        """
        模糊匹配：使用 token sort ratio 比较。
        处理情况：
        - "无锡市国联发展(集团)有限公司" vs "无锡国联发展集团有限公司"
        - "中国平安保险(集团)股份有限公司" vs "中国平安保险集团股份有限公司"
        """
        # 尝试导入 fuzzywuzzy/rapidfuzz，没有则跳过模糊匹配
        try:
            from rapidfuzz import fuzz
        except ImportError:
            try:
                from fuzzywuzzy import fuzz
            except ImportError:
                return None  # 无法进行模糊匹配

        best_id = None
        best_score = 0

        for canonical_id, entity in self._entities.items():
            canonical_name = entity["name"]
            score = fuzz.token_sort_ratio(name, canonical_name)
            if score > best_score:
                best_score = score
                best_id = canonical_id

        # 阈值：90% 以上视为同一实体
        if best_score >= 90:
            return best_id

        return None

    def _llm_align(self, name: str, entity_type: str) -> Optional[str]:
        """LLM 实体对齐（兜底）"""
        if not self.llm:
            return None

        # 只对"疑似匹配"（70-90分）的候选做 LLM 判断，节省 API 成本
        try:
            from rapidfuzz import fuzz
        except ImportError:
            try:
                from fuzzywuzzy import fuzz
            except ImportError:
                return None  # 无法进行模糊匹配，直接跳过 LLM 对齐

        candidates = []
        for canonical_id, entity in self._entities.items():
            score = fuzz.token_sort_ratio(name, entity["name"])
            if 70 <= score < 90:  # 模糊区间
                candidates.append((canonical_id, entity["name"], score))

        if not candidates:
            return None

        # 取 top 3 候选让 LLM 判断
        candidates.sort(key=lambda x: x[2], reverse=True)
        top3 = candidates[:3]

        prompt = f"""判断以下两个实体名称是否指向同一实体。

名称A: {name}
候选名称:
{chr(10).join(f'{i+1}. {c[1]}' for i, c in enumerate(top3))}

如果名称A与任一候选是同一实体，输出该候选的编号(1/2/3)。
如果都不是同一实体，输出0。

输出格式: {{"match": 1, "reason": "简称与全称关系"}}"""

        try:
            result = self.llm.chat_with_json_output(prompt, temperature=0.0)
            match_idx = int(result.get("match", 0))
            if 1 <= match_idx <= len(top3):
                return top3[match_idx - 1][0]
        except Exception:
            pass

        return None

    @staticmethod
    def _generate_id(name: str, entity_type: str) -> str:
        """生成唯一 ID"""
        prefix = {"个人": "person", "企业": "company", "机构": "org", "政府": "gov"}
        p = prefix.get(entity_type, "entity")
        # 简单哈希：取名称前4个字符 + 长度
        short = re.sub(r'[^a-zA-Z一-鿿0-9]', '', name)[:6]
        return f"{p}_{short}_{abs(hash(name)) % 10000:04d}"

    def learn_alias(self, alias: str, canonical_id: str):
        """手动添加别名映射"""
        clean_alias = self.normalize(alias)
        self._alias_dict[clean_alias] = canonical_id
        self._exact_index[clean_alias] = canonical_id
        if canonical_id in self._entities:
            if clean_alias not in self._entities[canonical_id]["aliases"]:
                self._entities[canonical_id]["aliases"].append(clean_alias)

    @property
    def entity_count(self) -> int:
        return len(self._entities)
