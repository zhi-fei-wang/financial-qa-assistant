"""
意图识别器
分析用户输入，输出意图分类、置信度、实体及建议的工具。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm import get_llm_client
from ..llm.prompts import INTENT_CLASSIFICATION_PROMPT


@dataclass
class IntentResult:
    """意图识别结果"""
    intent: str                          # 主意图
    confidence: float                    # 置信度 0-1
    entities: List[str] = field(default_factory=list)
    sub_intent: str = ""
    suggested_tool: str = ""
    params_hint: Dict[str, str] = field(default_factory=dict)
    reasoning: str = ""

    @property
    def needs_clarification(self) -> bool:
        """置信度过低，需要反问用户澄清"""
        return self.confidence < 0.7

    @property
    def is_chitchat(self) -> bool:
        return self.intent == "CHITCHAT"

    @property
    def requires_tool(self) -> bool:
        """是否需要调用外部工具"""
        return self.intent not in ("CHITCHAT",)


class IntentClassifier:
    """
    金融对话意图分类器。
    支持 LLM 分类 + Few-shot 增强。
    """

    # 意图与工具的映射（含子意图）
    INTENT_TOOL_MAP = {
        "MARKET_DATA": "get_stock_price",
        "FINANCIAL_ANALYSIS": "query_financial_statement",
        "EQUITY_PENETRATION": "equity_penetration",
        "NEWS_EVENT": "search_news",
        "CALCULATION": "financial_calculator",
        "CHITCHAT": None,
    }

    # 子意图 → 工具覆盖
    SUB_INTENT_TOOL_MAP = {
        "ANOMALY_CHECK": "financial_anomaly_check",
        "STATEMENT_QUERY": "query_financial_statement",
        "COMPARISON": "multi_period_analysis",
        "VIOLATION_CHECK": "search_news",
        "EVENT_TRACE": "event_trace",
        "PENETRATION": "equity_penetration",
        "SHAREHOLDER_QUERY": "control_summary",
    }

    # statement_type 关键词推断
    INCOME_KEYWORDS = [
        "营收", "营业收入", "营业总收入", "利润", "净利润", "毛利", "净利",
        "ROE", "ROA", "EPS", "每股收益", "收入", "成本", "费用",
        "销售毛利率", "净利率", "毛利率", "营业利润",
    ]
    BALANCE_KEYWORDS = [
        "资产", "负债", "货币资金", "存货", "应收", "应付", "商誉",
        "总资产", "净资产", "资产负债", "股东权益", "固定资产",
        "流动负债", "长期借款", "短期借款",
    ]
    CASHFLOW_KEYWORDS = [
        "现金流", "经营活动", "投资活动", "筹资活动",
        "经营现金流", "自由现金流",
    ]

    def __init__(self, use_llm: bool = True):
        self.llm = get_llm_client() if use_llm else None
        self.use_llm = use_llm

    def classify(
        self,
        user_query: str,
        context: Optional[str] = None,
        community_summaries: Optional[str] = None,
    ) -> IntentResult:
        """
        分类用户意图。

        Args:
            user_query: 用户输入
            context: 记忆检索结果文本
            community_summaries: 社区摘要文本

        Returns:
            IntentResult: 意图分类结果
        """
        if self.use_llm and self.llm:
            result = self._llm_classify(user_query, context, community_summaries)
            # 叠加关键词增强（修正 LLM 可能的错误）
            result = self._enhance_with_keywords(result, user_query)
            return result
        else:
            return self._rule_classify(user_query)

    def _llm_classify(
        self,
        user_query: str,
        context: Optional[str] = None,
        community_summaries: Optional[str] = None,
    ) -> IntentResult:
        """LLM 驱动的意图分类"""
        try:
            prompt = INTENT_CLASSIFICATION_PROMPT.format(
                retrieved_context=context or "（无历史上下文）",
                community_summaries=community_summaries or "（无话题摘要）",
                user_query=user_query,
            )
            result = self.llm.chat_with_json_output(
                user_prompt=prompt,
                temperature=0.0,
                max_retries=1,
            )
            return IntentResult(
                intent=result.get("intent", "CHITCHAT"),
                confidence=float(result.get("confidence", 0.5)),
                entities=result.get("entities", []),
                sub_intent=result.get("sub_intent", ""),
                suggested_tool=result.get("suggested_tool", ""),
                params_hint=result.get("params_hint", {}),
                reasoning=result.get("reasoning", ""),
            )
        except Exception as e:
            print(f"[IntentClassifier] LLM classification failed: {e}")
            return self._rule_classify(user_query)

    def _enhance_with_keywords(self, result: IntentResult, user_query: str) -> IntentResult:
        """关键词预扫描增强：在 LLM 分类基础上叠加规则修正"""
        # 1. 推断 statement_type（财报报表类型）
        if result.intent == "FINANCIAL_ANALYSIS":
            has_income = any(kw in user_query for kw in self.INCOME_KEYWORDS)
            has_balance = any(kw in user_query for kw in self.BALANCE_KEYWORDS)
            has_cf = any(kw in user_query for kw in self.CASHFLOW_KEYWORDS)

            inferred_st = result.params_hint.get("statement_type", "")
            if not inferred_st:
                if has_income:
                    inferred_st = "income"
                elif has_cf:
                    inferred_st = "cashflow"
                elif has_balance:
                    inferred_st = "balance_sheet"

            if inferred_st:
                result.params_hint["statement_type"] = inferred_st

        # 2. 关键词强信号覆盖 intent（当 LLM 明显分错时）
        violation_kw = ["违规", "处罚", "监管", "监管措施", "被查", "被罚", "违纪"]
        announcement_kw = ["公告", "公报"]
        equity_kw = ["股权穿透", "股权结构", "控股链", "实控人", "实际控制人", "穿透", "股东穿透"]

        if any(kw in user_query for kw in violation_kw + announcement_kw):
            if result.intent not in ("NEWS_EVENT", "EQUITY_PENETRATION"):
                result.intent = "NEWS_EVENT"
                result.sub_intent = "VIOLATION_CHECK"
                result.suggested_tool = "search_news"
                result.confidence = max(result.confidence, 0.85)
                result.reasoning += " [关键词覆盖: 违规/公告]"

        if any(kw in user_query for kw in equity_kw):
            if result.intent != "EQUITY_PENETRATION":
                result.intent = "EQUITY_PENETRATION"
                result.suggested_tool = "equity_penetration"
                result.confidence = max(result.confidence, 0.85)
                result.reasoning += " [关键词覆盖: 股权穿透]"

        # 3. 覆盖 suggested_tool（子意图优先）
        if result.sub_intent and result.sub_intent in self.SUB_INTENT_TOOL_MAP:
            result.suggested_tool = self.SUB_INTENT_TOOL_MAP[result.sub_intent]

        return result

    def _rule_classify(self, user_query: str) -> IntentResult:
        """基于规则的意图分类（兜底）"""
        query = user_query.lower()

        # 关键词 → (intent, suggested_tool)
        keyword_intents = [
            ("EQUITY_PENETRATION", "equity_penetration", [
                "股权穿透", "控股链", "控股结构", "实控人", "股东穿透",
                "穿透", "持股链", "实际控制人", "控制链", "控股股东",
            ]),
            ("NEWS_EVENT", "search_news", [
                "违规", "处罚", "公告", "监管", "监管措施", "风险提示",
                "舆情", "事件", "研报", "利好", "利空",
            ]),
            ("MARKET_DATA", "get_stock_price", [
                "股价", "涨跌", "行情", "收盘", "市值", "盘口", "量比",
                "涨幅", "跌幅", "换手率", "主力资金", "资金流向", "涨停",
                "龙虎榜", "融资融券", "开盘价", "最高价",
            ]),
            ("FINANCIAL_ANALYSIS", "query_financial_statement", [
                "财报", "财务", "ROE", "ROA", "利润率", "存货周转",
                "现金流", "净利润", "营收", "造假", "排雷", "疑点",
                "资产负债", "勾稽", "粉饰", "虚增", "毛利率", "净利率",
                "每股收益", "市盈率", "eps", "pe", "pb", "总资产",
            ]),
            ("CALCULATION", "financial_calculator", [
                "计算", "算一下", "均值", "平均", "求和",
            ]),
        ]

        for intent, tool, keywords in keyword_intents:
            matched_kws = [kw for kw in keywords if kw in query]
            if matched_kws:
                entities = self._extract_entities_heuristic(user_query)
                params_hint = {}

                # 推断 statement_type
                if intent == "FINANCIAL_ANALYSIS":
                    if any(k in user_query for k in self.INCOME_KEYWORDS):
                        params_hint["statement_type"] = "income"
                    elif any(k in user_query for k in self.CASHFLOW_KEYWORDS):
                        params_hint["statement_type"] = "cashflow"
                    elif any(k in user_query for k in self.BALANCE_KEYWORDS):
                        params_hint["statement_type"] = "balance_sheet"

                return IntentResult(
                    intent=intent,
                    confidence=0.75,
                    entities=entities,
                    suggested_tool=tool,
                    params_hint=params_hint,
                    reasoning=f"关键词匹配: {matched_kws}",
                )

        return IntentResult(
            intent="CHITCHAT",
            confidence=0.6,
            entities=[],
            reasoning="无明确金融意图关键词",
        )

    @staticmethod
    def _extract_entities_heuristic(text: str) -> List[str]:
        """启发式实体提取（股票代码 + 常见金融实体）"""
        import re
        entities = []
        # 6位股票代码
        codes = re.findall(r'\b(\d{6})\b', text)
        entities.extend(codes)
        # 常见股票名称
        known_stocks = ["贵州茅台", "茅台", "五粮液", "宁德时代", "比亚迪", "招商银行",
                       "中国平安", "隆基绿能", "九阳股份"]
        for stock in known_stocks:
            if stock in text:
                entities.append(stock)
        return entities


def test_intent_classifier():
    """测试意图分类"""
    classifier = IntentClassifier(use_llm=True)

    test_queries = [
        ("帮我查一下茅台最新股价", "MARKET_DATA"),
        ("茅台ROE最近5年变化趋势", "FINANCIAL_ANALYSIS"),
        ("帮我穿透九阳股份的股权结构", "EQUITY_PENETRATION"),
        ("贵州茅台有没有最近的违规公告", "NEWS_EVENT"),
        ("你好，你是谁", "CHITCHAT"),
        ("计算茅台和五粮液近三年平均毛利率", "CALCULATION"),
        ("存货周转率这么高，有没有虚增利润的可能", "FINANCIAL_ANALYSIS"),
    ]

    for query, expected in test_queries:
        result = classifier.classify(query)
        status = "✅" if result.intent == expected else "❌"
        print(f"{status} [{result.intent:20s}] conf={result.confidence:.2f} | {query[:50]}...")
        if result.intent != expected:
            print(f"   Expected: {expected}, Got: {result.intent}, Reason: {result.reasoning}")


if __name__ == "__main__":
    test_intent_classifier()
