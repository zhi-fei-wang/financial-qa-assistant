"""
BaseTool — 工具插件基类

所有工具继承此类，封装元数据、执行、路由、Prompt、验证为一个自包含单元。
新增工具只需：继承 BaseTool → 实现 execute() → 在 __init__.py 中注册。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type


class BaseTool(ABC):
    """
    自描述工具基类。

    子类必须定义：
      - name / description / required_params / optional_params / param_schema
      - intent_match (意图绑定)
      - execute() 方法

    子类可选定义：
      - sub_intent / routing_hint / trigger_keywords
      - validate_result() / max_retries / timeout_sec
    """

    # =========================================================================
    # 元数据（子类必须覆盖）
    # =========================================================================

    name: str = ""
    """工具唯一标识，如 'web_search'"""

    description: str = ""
    """工具功能描述（给 LLM 看）"""

    required_params: List[str] = []
    """必填参数列表"""

    optional_params: List[str] = []
    """可选参数列表"""

    param_schema: Dict[str, Dict[str, str]] = {}
    """参数 Schema，key 为参数名，value 含 description 等"""

    # =========================================================================
    # 路由绑定（子类必须覆盖）
    # =========================================================================

    intent_match: List[str] = []
    """匹配的意图列表，如 ['MARKET_DATA', 'NEWS_EVENT']"""

    sub_intent: str = ""
    """子意图标签，如 'PENETRATION' / 'ANOMALY_CHECK'"""

    # =========================================================================
    # 执行参数
    # =========================================================================

    max_retries: int = 2
    """最大重试次数"""

    timeout_sec: int = 10
    """超时时间（秒）"""

    # =========================================================================
    # Prompt 提示（子类可选覆盖）
    # =========================================================================

    routing_hint: str = ""
    """
    ReAct 工具选择提示。
    例如: "用户问实时行情/主力资金 → get_stock_price"
    会自动注入 REACT_SYSTEM_PROMPT。
    """

    trigger_keywords: List[str] = []
    """
    触发关键词列表。
    用于规则兜底意图分类（_rule_classify + _enhance_with_keywords）。
    例如: ['股价', '涨跌', '行情', '换手率', '主力资金']
    """

    # =========================================================================
    # 核心接口
    # =========================================================================

    @abstractmethod
    def execute(self, params: Dict[str, Any], data_loader: Any = None) -> Dict[str, Any]:
        """
        执行工具。

        Args:
            params: 调用参数
            data_loader: DataLoader 实例（可选）

        Returns:
            执行结果字典，必须包含:
              - source: 数据来源标识 ('dataset' / 'mock' / 'web' 等)
              - rendered: Markdown 格式的输出文本（给 LLM 直接消费）
            可选:
              - data: 原始结构化数据
              - error: 错误信息
        """
        ...

    # =========================================================================
    # 可选覆盖
    # =========================================================================

    def validate_result(self, data: Dict[str, Any]) -> Optional[str]:
        """
        验证执行结果。返回 None 表示通过，返回字符串表示错误信息。

        Args:
            data: execute() 返回的数据

        Returns:
            None 表示验证通过，否则返回错误描述
        """
        return None  # 默认不验证

    def get_routing_hint(self) -> str:
        """获取工具选择提示（子类可覆盖以动态生成）。"""
        return self.routing_hint

    def get_trigger_keywords(self) -> List[str]:
        """获取触发关键词（子类可覆盖以动态生成）。"""
        return self.trigger_keywords

    # =========================================================================
    # 类方法：注册 + LLM 定义生成
    # =========================================================================

    @classmethod
    def register_to(cls, registry) -> None:
        """
        将工具注册到 ToolRegistry。

        自动从类属性提取元数据，创建 ToolMeta 并注册。
        调用方式: MyTool.register_to(registry)
        """
        from ..router.tool_registry import ToolMeta

        # 构建 ToolMeta
        meta = ToolMeta(
            name=cls.name,
            description=cls.description,
            required_params=list(cls.required_params),
            optional_params=list(cls.optional_params),
            intent_match=list(cls.intent_match),
            executor=cls._make_executor(),
            max_retries=cls.max_retries,
            timeout_sec=cls.timeout_sec,
            param_schema=dict(cls.param_schema),
        )

        # 校验
        if not meta.name:
            raise ValueError(f"{cls.__name__}.name 不能为空")
        if not meta.intent_match:
            raise ValueError(f"{cls.__name__}.intent_match 不能为空")

        registry.register(meta)

    @classmethod
    def _make_executor(cls):
        """
        创建执行函数闭包。

        ToolMeta.executor 签名: (params, data_loader=None) -> Dict
        tool_executor.py 调用时会传入 self.data_loader。
        """
        def executor_fn(params: Dict[str, Any], data_loader: Any = None) -> Dict[str, Any]:
            instance = cls()
            return instance.execute(params, data_loader=data_loader)
        return executor_fn

    @classmethod
    def create_tool_meta(cls) -> Any:
        """
        创建 ToolMeta（不注册到 registry，供手动使用）。
        """
        from ..router.tool_registry import ToolMeta
        return ToolMeta(
            name=cls.name,
            description=cls.description,
            required_params=list(cls.required_params),
            optional_params=list(cls.optional_params),
            intent_match=list(cls.intent_match),
            executor=cls._make_executor(),
            max_retries=getattr(cls, 'max_retries', 2),
            timeout_sec=getattr(cls, 'timeout_sec', 10),
            param_schema=dict(getattr(cls, 'param_schema', {})),
        )

    @classmethod
    def to_llm_definition(cls) -> Dict[str, Any]:
        """
        生成给 LLM 看的工具定义（OpenAI Function Calling 兼容格式）。
        """
        props = {}
        for param in cls.required_params:
            props[param] = {
                "type": "string",
                "description": cls.param_schema.get(param, {}).get("description", param),
            }
        for param in cls.optional_params:
            props[param] = {
                "type": "string",
                "description": cls.param_schema.get(param, {}).get("description", param),
            }

        return {
            "name": cls.name,
            "description": cls.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": list(cls.required_params),
            },
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"


# =============================================================================
# 工具注册表（全局单例，用于自动发现）
# =============================================================================

_tool_registry: List[Type[BaseTool]] = []


def get_all_tool_classes() -> List[Type[BaseTool]]:
    """返回所有已发现的 BaseTool 子类。"""
    return list(_tool_registry)


def register_tool_class(cls: Type[BaseTool]) -> Type[BaseTool]:
    """
    装饰器：将工具类加入全局注册表。

    用法:
        @register_tool_class
        class MyTool(BaseTool):
            ...
    """
    if cls not in _tool_registry:
        _tool_registry.append(cls)
    return cls


def discover_and_register(registry) -> int:
    """
    自动发现所有 BaseTool 子类并注册到 ToolRegistry。

    Returns:
        注册的工具数量
    """
    count = 0
    for cls in _tool_registry:
        try:
            cls.register_to(registry)
            count += 1
        except Exception as e:
            print(f"[BaseTool] 注册 {cls.__name__} 失败: {e}")
    return count
