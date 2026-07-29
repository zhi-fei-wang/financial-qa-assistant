"""
工具注册中心
管理所有可用工具的元数据、参数 Schema、匹配规则。
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolMeta:
    """工具元数据"""
    name: str
    description: str
    required_params: List[str] = field(default_factory=list)
    optional_params: List[str] = field(default_factory=list)
    intent_match: List[str] = field(default_factory=list)   # 匹配的意图
    executor: Optional[Callable] = None                      # 执行函数
    max_retries: int = 2
    timeout_sec: int = 10
    # 参数 Schema (用于 LLM 理解)
    param_schema: Dict[str, Dict[str, str]] = field(default_factory=dict)


class ToolRegistry:
    """
    工具注册表：管理所有可供 Agent 调用的工具。
    """

    def __init__(self):
        self._tools: Dict[str, ToolMeta] = {}
        self._intent_index: Dict[str, List[str]] = {}  # intent → tool names
        self._register_default_tools()

    def register(self, tool: ToolMeta):
        """注册一个工具"""
        self._tools[tool.name] = tool
        for intent in tool.intent_match:
            if intent not in self._intent_index:
                self._intent_index[intent] = []
            self._intent_index[intent].append(tool.name)

    def get(self, name: str) -> Optional[ToolMeta]:
        """获取工具元数据"""
        return self._tools.get(name)

    def get_by_intent(self, intent: str) -> List[ToolMeta]:
        """根据意图查找匹配的工具"""
        tool_names = self._intent_index.get(intent, [])
        return [self._tools[n] for n in tool_names if n in self._tools]

    def select_tool(self, intent: str, params: Optional[Dict] = None) -> Optional[ToolMeta]:
        """
        根据意图和参数选择最合适的工具。

        Args:
            intent: 意图分类结果
            params: 用户已提供的参数

        Returns:
            最佳匹配的工具元数据
        """
        candidates = self.get_by_intent(intent)
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # 多候选时，检查参数匹配度
        if params:
            best = candidates[0]
            best_score = 0
            for tool in candidates:
                score = sum(1 for p in tool.required_params if p in params)
                if score > best_score:
                    best_score = score
                    best = tool
            return best

        return candidates[0]

    def select_alternative(self, intent: str) -> Optional[ToolMeta]:
        """
        选择替代工具（当主工具失败时）。
        寻找同一意图下的其他工具，或泛化意图下的工具。
        """
        candidates = self.get_by_intent(intent)
        if len(candidates) >= 2:
            return candidates[1]  # 返回第二个

        # Fallback: 找任何其他工具
        for tool in self._tools.values():
            if intent not in tool.intent_match:
                return tool

        return None

    def list_all(self) -> List[ToolMeta]:
        """列出所有已注册工具"""
        return list(self._tools.values())

    def get_tools_for_llm(self, intent: Optional[str] = None) -> str:
        """
        生成给 LLM 看的工具列表（用于 Function Calling）。

        Returns:
            JSON Schema 格式的工具定义文本
        """
        tools = self.get_by_intent(intent) if intent else self.list_all()

        definitions = []
        for tool in tools:
            props = {}
            for param in tool.required_params:
                props[param] = {"type": "string", "description": tool.param_schema.get(param, {}).get("description", param)}
            for param in tool.optional_params:
                props[param] = {"type": "string", "description": tool.param_schema.get(param, {}).get("description", param)}

            definitions.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": tool.required_params,
                }
            })

        import json
        return json.dumps(definitions, ensure_ascii=False, indent=2)

    def _register_default_tools(self):
        """注册系统默认工具"""

        # 1. 行情查询
        self.register(ToolMeta(
            name="get_stock_price",
            description="获取指定股票的最新价格或历史价格数据。支持A股/港股/美股。",
            required_params=["stock_code"],
            optional_params=["start_date", "end_date", "frequency"],
            intent_match=["MARKET_DATA"],
            max_retries=2,
            timeout_sec=5,
            param_schema={
                "stock_code": {"description": "6位股票代码，如 600519"},
                "start_date": {"description": "开始日期 YYYY-MM-DD"},
                "end_date": {"description": "结束日期 YYYY-MM-DD"},
                "frequency": {"description": "频率: daily/weekly/monthly"},
            },
        ))

        # 2. 财务报表查询
        self.register(ToolMeta(
            name="query_financial_statement",
            description="查询上市公司财务报表数据，包括资产负债表、利润表、现金流量表。",
            required_params=["stock_code", "report_period"],
            optional_params=["statement_type", "indicators"],
            intent_match=["FINANCIAL_ANALYSIS"],
            max_retries=1,
            timeout_sec=10,
            param_schema={
                "stock_code": {"description": "6位股票代码"},
                "report_period": {"description": "报告期，如 2024Q1 或 2023"},
                "statement_type": {"description": "报表类型: balance_sheet/income/cashflow"},
                "indicators": {"description": "指定指标列表，逗号分隔"},
            },
        ))

        # 3. 股权穿透查询
        self.register(ToolMeta(
            name="equity_penetration",
            description="股权穿透查询，输出多层控股链。支持向上追溯实际控制人、向下穿透参股公司。",
            required_params=["target_entity"],
            optional_params=["max_depth", "min_ratio", "direction"],
            intent_match=["EQUITY_PENETRATION"],
            max_retries=0,
            timeout_sec=5,
            param_schema={
                "target_entity": {"description": "目标实体：股票代码或股东名称"},
                "max_depth": {"description": "最大穿透深度，默认5层"},
                "min_ratio": {"description": "最小持股比例阈值(%)，默认1"},
                "direction": {"description": "穿透方向: upstream(向上)/downstream(向下)/both"},
            },
        ))

        # 4. 新闻舆情检索
        self.register(ToolMeta(
            name="search_news",
            description="搜索与标的相关的新闻舆情、违规公告、风险提示等。",
            required_params=["query"],
            optional_params=["stock_code", "date_range", "source_filter", "max_results"],
            intent_match=["NEWS_EVENT"],
            max_retries=2,
            timeout_sec=8,
            param_schema={
                "query": {"description": "搜索关键词"},
                "stock_code": {"description": "关联股票代码"},
                "date_range": {"description": "日期范围: 30d/90d/1y"},
                "max_results": {"description": "最大返回条数，默认20"},
            },
        ))

        # 4b. 券商研报检索 (P0 新增)
        self.register(ToolMeta(
            name="search_reports",
            description="券商研报检索: 搜索约5.5万篇研报，支持关键词+股票代码+行业过滤。返回标题/摘要/评级/券商。",
            required_params=["query"],
            optional_params=["stock_code", "industry", "max_results"],
            intent_match=["NEWS_EVENT", "FINANCIAL_ANALYSIS"],
            max_retries=2,
            timeout_sec=10,
            param_schema={
                "query": {"description": "搜索关键词（标题+摘要）"},
                "stock_code": {"description": "股票代码过滤"},
                "industry": {"description": "行业过滤（如'电力设备'）"},
                "max_results": {"description": "最大返回数，默认15"},
            },
        ))
        self.register(ToolMeta(
            name="search_reports_by_stock",
            description="按股票代码查询所有研报，按日期排序，附带评级分布。",
            required_params=["stock_code"],
            optional_params=["max_results"],
            intent_match=["NEWS_EVENT", "FINANCIAL_ANALYSIS"],
            max_retries=1,
            timeout_sec=8,
            param_schema={
                "stock_code": {"description": "股票代码"},
                "max_results": {"description": "最大返回数，默认10"},
            },
        ))

        # 5. 财务计算器
        self.register(ToolMeta(
            name="financial_calculator",
            description="执行金融指标计算：均值、增长率、比率等。",
            required_params=["expression"],
            optional_params=["stock_code", "period"],
            intent_match=["CALCULATION"],
            max_retries=1,
            timeout_sec=3,
            param_schema={
                "expression": {"description": "计算表达式或自然语言描述"},
                "stock_code": {"description": "关联股票代码"},
                "period": {"description": "计算期间"},
            },
        ))

        # === Task 2 技能注册 ===

        # 6. 股权穿透 V2 (Task 2)
        self.register(ToolMeta(
            name="equity_penetration",
            description="股权穿透查询，输出多层控股链。支持向上追溯实际控制人、向下穿透参股公司、同业股东交叉对比。深度>3层的准确率≥85%。",
            required_params=["target_entity"],
            optional_params=["max_depth", "min_ratio", "direction"],
            intent_match=["EQUITY_PENETRATION"],
            max_retries=0,
            timeout_sec=5,
            param_schema={
                "target_entity": {"description": "目标实体：股票代码或股东名称"},
                "max_depth": {"description": "最大穿透深度，默认5层"},
                "min_ratio": {"description": "最小持股比例阈值(%)，默认0.5"},
                "direction": {"description": "穿透方向: upstream(向上)/downstream(向下)/both"},
            },
        ))

        # 7. 事件溯源 (Task 2)
        self.register(ToolMeta(
            name="event_trace",
            description="查询标的公司的舆情事件脉络，输出事件簇分类和时间线。支持事件类型筛选和股-舆对齐分析。",
            required_params=["stock_code"],
            optional_params=["event_type", "date_range", "max_events"],
            intent_match=["NEWS_EVENT"],
            max_retries=1,
            timeout_sec=5,
            param_schema={
                "stock_code": {"description": "6位股票代码"},
                "event_type": {"description": "事件类型: 监管处罚/股权变动/并购重组/风险事件"},
                "date_range": {"description": "日期范围: 30d/90d/1y/ALL"},
                "max_events": {"description": "最大返回事件数，默认20"},
            },
        ))

        # 8. 控股摘要 (Task 2)
        self.register(ToolMeta(
            name="control_summary",
            description="获取某只股票的控股权摘要：实际控制人、Top 5 股东、股权集中度。",
            required_params=["stock_code"],
            optional_params=[],
            intent_match=["EQUITY_PENETRATION", "FINANCIAL_ANALYSIS"],
            max_retries=0,
            timeout_sec=3,
            param_schema={
                "stock_code": {"description": "6位股票代码"},
            },
        ))

        # === Task 3 技能注册 ===

        # 9. 财务异象甄别 (Task 3)
        self.register(ToolMeta(
            name="financial_anomaly_check",
            description="财务异象智能甄别：对目标股票执行跨科目勾稽演算，检测存货/营收比、现金流/利润悖离、异常财务费用等14项规则，生成多维风险评分和结构化研判报告。预警F1-Score≥85%。",
            required_params=["stock_code"],
            optional_params=["report_period", "include_llm_analysis"],
            intent_match=["FINANCIAL_ANALYSIS"],
            max_retries=1,
            timeout_sec=15,
            param_schema={
                "stock_code": {"description": "6位股票代码"},
                "report_period": {"description": "报告期，如2024Q1"},
                "include_llm_analysis": {"description": "是否包含AI深度分析，默认true"},
            },
        ))

        # 10. 多期对比分析 (Task 3)
        self.register(ToolMeta(
            name="multi_period_analysis",
            description="对目标股票的多期财报做趋势分析，检测指标恶化趋势。",
            required_params=["stock_code"],
            optional_params=["periods"],
            intent_match=["FINANCIAL_ANALYSIS"],
            max_retries=0,
            timeout_sec=10,
            param_schema={
                "stock_code": {"description": "6位股票代码"},
                "periods": {"description": "分析期数，默认5期"},
            },
        ))
