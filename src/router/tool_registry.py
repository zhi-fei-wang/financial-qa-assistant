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

    支持两种注册方式：
      1. register(ToolMeta) — 传统方式（向后兼容）
      2. register_from_class(BaseTool子类) — 插件方式（自动提取元数据）
    """

    def __init__(self, auto_register: bool = True):
        self._tools: Dict[str, ToolMeta] = {}
        self._intent_index: Dict[str, List[str]] = {}  # intent → tool names
        self._tool_classes: Dict[str, Any] = {}  # name → BaseTool 子类
        if auto_register:
            self._register_default_tools()

    def register(self, tool: ToolMeta):
        """注册一个工具（通过 ToolMeta）"""
        self._tools[tool.name] = tool
        for intent in tool.intent_match:
            if intent not in self._intent_index:
                self._intent_index[intent] = []
            if tool.name not in self._intent_index[intent]:
                self._intent_index[intent].append(tool.name)

    def register_from_class(self, tool_cls) -> ToolMeta:
        """
        从 BaseTool 子类注册工具（插件方式）。

        自动提取类属性创建 ToolMeta，并关联 BaseTool 子类以便后续
        执行、验证、Prompt 生成等操作。

        Args:
            tool_cls: BaseTool 子类

        Returns:
            创建的 ToolMeta
        """
        meta = ToolMeta(
            name=tool_cls.name,
            description=tool_cls.description,
            required_params=list(tool_cls.required_params),
            optional_params=list(tool_cls.optional_params),
            intent_match=list(tool_cls.intent_match),
            executor=tool_cls._make_executor(),  # 闭包调用 execute()
            max_retries=getattr(tool_cls, 'max_retries', 2),
            timeout_sec=getattr(tool_cls, 'timeout_sec', 10),
            param_schema=dict(getattr(tool_cls, 'param_schema', {})),
        )
        self.register(meta)
        self._tool_classes[meta.name] = tool_cls
        return meta

    def get_tool_class(self, name: str):
        """获取工具对应的 BaseTool 子类（如果存在）。"""
        return self._tool_classes.get(name)

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

    # =========================================================================
    # 自动生成方法（供 intent_classifier / agent_loop / prompts 使用）
    # =========================================================================

    def get_all_routing_hints(self) -> str:
        """
        汇总所有工具的 routing_hint，用于自动生成 REACT_SYSTEM_PROMPT。
        """
        hints = []
        for name, tool_cls in self._tool_classes.items():
            hint = getattr(tool_cls, 'routing_hint', '')
            if hint:
                hints.append(hint)
        return "\n- ".join(hints) if hints else ""

    def get_intent_tool_map(self) -> Dict[str, str]:
        """
        自动生成 INTENT → 默认工具 映射。
        用于替代 intent_classifier 中的 INTENT_TOOL_MAP 硬编码。
        """
        intent_map = {}
        for intent in self._intent_index:
            tool_names = self._intent_index[intent]
            if tool_names:
                intent_map[intent] = tool_names[0]  # 第一个工具为默认
        return intent_map

    def get_sub_intent_tool_map(self) -> Dict[str, str]:
        """
        自动生成 子意图 → 工具 映射。
        从 _tool_classes 的 sub_intent 属性提取。
        """
        sub_map = {}
        for name, tool_cls in self._tool_classes.items():
            sub = getattr(tool_cls, 'sub_intent', '')
            if sub:
                sub_map[sub] = name
        return sub_map

    def get_keywords_by_intent(self) -> Dict[str, List[str]]:
        """
        自动生成 intent → trigger_keywords 映射。
        用于替代 intent_classifier 中的关键词硬编码。
        """
        kw_map: Dict[str, List[str]] = {}
        for name, tool_cls in self._tool_classes.items():
            keywords = getattr(tool_cls, 'trigger_keywords', []) or []
            for intent in getattr(tool_cls, 'intent_match', []):
                if intent not in kw_map:
                    kw_map[intent] = []
                kw_map[intent].extend(keywords)
        # 去重
        for intent in kw_map:
            kw_map[intent] = list(dict.fromkeys(kw_map[intent]))
        return kw_map

    def get_tool_catalog_for_prompt(self) -> str:
        """
        生成 INTENT_CLASSIFICATION_PROMPT 中的工具目录文本。
        动态生成，替代 prompts.py 中的硬编码意图→工具列表。
        """
        lines = []
        for intent, tools in sorted(self._intent_index.items()):
            if intent == "CHITCHAT":
                continue
            lines.append(f"### {intent}")
            for tname in tools:
                tool = self._tools.get(tname)
                if not tool:
                    continue
                lines.append(f"- {tool.description}")
                # 附加子意图信息
                tc = self._tool_classes.get(tname)
                if tc:
                    sub = getattr(tc, 'sub_intent', '') or ''
                    keywords = getattr(tc, 'trigger_keywords', []) or []
                    if sub or keywords:
                        parts = []
                        if sub:
                            parts.append(f"sub_intent: {sub}")
                        if keywords:
                            parts.append(f"关键词: {', '.join(keywords[:8])}")
                        lines.append(f"  ({'; '.join(parts)})")
                lines.append(f"  → suggested_tool: \"{tool.name}\"")
            lines.append("")
        return "\n".join(lines)

    def _register_default_tools(self):
        """注册系统默认工具"""

        # 1. 行情查询 — BaseTool 插件
        from ..tools.market_data import MarketDataTool
        self.register_from_class(MarketDataTool)

        # 2. 财务报表查询 — BaseTool 插件
        from ..tools.query_financial import QueryFinancialTool
        self.register_from_class(QueryFinancialTool)

        # 3. 股权穿透查询 — BaseTool 插件
        from ..tools.equity_graph import EquityPenetrationTool
        self.register_from_class(EquityPenetrationTool)

        # 4. 新闻舆情检索 — BaseTool 插件
        from ..tools.news_search import NewsSearchTool
        self.register_from_class(NewsSearchTool)

        # 4b. 券商研报检索 — BaseTool 插件
        from ..tools.research_reports import SearchReportsTool, SearchReportsByStockTool
        self.register_from_class(SearchReportsTool)
        self.register_from_class(SearchReportsByStockTool)

        # 5. 财务计算器 — BaseTool 插件
        from ..tools.financial_calculator import FinancialCalculatorTool
        self.register_from_class(FinancialCalculatorTool)

        # === Task 2 技能注册 ===

        # 6. 股权穿透 — 已在 #3 注册 (EquityPenetrationTool)，此处跳过重复

        # 7. 事件溯源 — BaseTool 插件
        from ..tools.equity_graph import EventTraceTool
        self.register_from_class(EventTraceTool)

        # 8. 控股摘要 — BaseTool 插件
        from ..tools.equity_graph import ControlSummaryTool
        self.register_from_class(ControlSummaryTool)

        # === Task 3 技能注册 ===

        # 9. 财务异象甄别 — BaseTool 插件
        from ..tools.financial_anomaly import FinancialAnomalyTool
        self.register_from_class(FinancialAnomalyTool)

        # 10. 多期对比分析 — BaseTool 插件
        from ..tools.financial_anomaly import MultiPeriodAnalysisTool
        self.register_from_class(MultiPeriodAnalysisTool)

        # 11. 联网搜索 — BaseTool 插件 (v2.5.0)
        from ..tools.web_search import WebSearchTool
        self.register_from_class(WebSearchTool)

        # 12. 用户上传数据查询 — BaseTool 插件 (v2.6.0)
        from ..tools.uploaded_data import UploadedDataTool
        self.register_from_class(UploadedDataTool)
