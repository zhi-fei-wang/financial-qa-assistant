"""
结果验证器
对工具执行结果进行分层验证：语法层 → 语义层 → 逻辑层
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .tool_executor import ToolResult


class ErrorType(Enum):
    """工具调用错误类型"""
    NONE = "none"              # 无错误
    PARAM_ERROR = "param_error"  # 参数错误
    EMPTY_RESULT = "empty_result"  # 结果为空
    FORMAT_ERROR = "format_error"  # 格式异常
    SEMANTIC_ERROR = "semantic_error"  # 语义异常（如价格为负）
    TIMEOUT = "timeout"         # 超时
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    """验证结果"""
    passed: bool
    error_type: ErrorType = ErrorType.NONE
    error_message: str = ""
    suggestions: List[str] = field(default_factory=list)


class ResultValidator:
    """
    分层验证器。
    每个工具注册一组验证规则，对执行结果做语法、语义、逻辑三层检查。
    """

    # 各工具的验证规则
    VALIDATION_RULES: Dict[str, List[Callable[[Dict], ValidationResult]]] = {
        "get_stock_price": [],
        "query_financial_statement": [],
        "equity_penetration": [],
        "search_news": [],
        "financial_calculator": [],
    }

    def __init__(self):
        self._register_default_rules()

    def validate(self, tool_name: str, result: ToolResult) -> ValidationResult:
        """
        验证工具执行结果。

        Args:
            tool_name: 工具名称
            result: 工具执行结果

        Returns:
            ValidationResult
        """
        # 第一层：执行是否成功
        if not result.success:
            error_msg = result.error.lower()
            if "param" in error_msg or "缺少" in error_msg:
                return ValidationResult(False, ErrorType.PARAM_ERROR, result.error)
            if "timeout" in error_msg or "超时" in error_msg:
                return ValidationResult(False, ErrorType.TIMEOUT, result.error)
            if "empty" in error_msg or "为空" in error_msg:
                return ValidationResult(False, ErrorType.EMPTY_RESULT, result.error)
            return ValidationResult(False, ErrorType.UNKNOWN, result.error)

        # 第二层：数据验证
        data = result.data or {}
        rules = self.VALIDATION_RULES.get(tool_name, [])
        for rule in rules:
            validation = rule(data)
            if not validation.passed:
                return validation

        return ValidationResult(True)

    def _register_default_rules(self):
        """注册默认验证规则"""

        # 行情数据验证
        def market_data_rules(data: Dict) -> ValidationResult:
            # 价格必须 > 0
            price = data.get("price", 0)
            if price is not None and float(price) <= 0:
                return ValidationResult(
                    False, ErrorType.SEMANTIC_ERROR,
                    f"价格异常: {price} <= 0",
                    ["检查股票代码是否正确", "尝试使用历史价格"]
                )
            # 必须有时间戳
            if "timestamp" not in data:
                return ValidationResult(
                    False, ErrorType.FORMAT_ERROR,
                    "缺少时间戳字段",
                    ["确认 API 返回格式正确"]
                )
            return ValidationResult(True)

        self.VALIDATION_RULES["get_stock_price"].append(market_data_rules)

        # 财务报表验证
        def financial_statement_rules(data: Dict) -> ValidationResult:
            if "data" in data:
                inner = data["data"]
                if isinstance(inner, dict) and len(inner) < 3:
                    return ValidationResult(
                        False, ErrorType.EMPTY_RESULT,
                        "财务数据字段过少",
                        ["检查 report_period 是否正确", "尝试扩大查询范围"]
                    )
            if "report_period" not in data:
                return ValidationResult(
                    False, ErrorType.FORMAT_ERROR,
                    "缺少 report_period 字段",
                    []
                )
            return ValidationResult(True)

        self.VALIDATION_RULES["query_financial_statement"].append(financial_statement_rules)

        # 股权穿透验证
        def equity_penetration_rules(data: Dict) -> ValidationResult:
            chains = data.get("chains", [])
            if not chains:
                return ValidationResult(
                    False, ErrorType.EMPTY_RESULT,
                    "未找到股权穿透链路",
                    ["尝试扩大 max_depth", "检查目标实体名称是否正确"]
                )
            # 每条链路的比例应在 0-100 之间
            for chain in chains:
                ratio = chain.get("ratio", 50)
                if ratio is not None and (ratio <= 0 or ratio > 100):
                    return ValidationResult(
                        False, ErrorType.SEMANTIC_ERROR,
                        f"持股比例异常: {ratio}%",
                        ["确认数据源准确性"]
                    )
            return ValidationResult(True)

        self.VALIDATION_RULES["equity_penetration"].append(equity_penetration_rules)

        # 新闻检索验证
        def news_search_rules(data: Dict) -> ValidationResult:
            articles = data.get("articles", [])
            total = data.get("total", 0)
            if total == 0 and not articles:
                return ValidationResult(
                    False, ErrorType.EMPTY_RESULT,
                    "未找到相关新闻",
                    ["尝试使用更宽泛的关键词", "扩大日期范围", "不指定 stock_code 搜索"]
                )
            return ValidationResult(True)

        self.VALIDATION_RULES["search_news"].append(news_search_rules)


class ErrorDiagnoser:
    """错误诊断器：分析失败原因并给出修复建议"""

    @staticmethod
    def diagnose(result: ToolResult, validation: ValidationResult) -> Dict[str, Any]:
        """
        综合诊断工具调用错误。

        Returns:
            {
                "error_type": str,
                "root_cause": str,
                "fix_strategy": str,       # "correct_params" | "switch_tool" | "fallback" | "ask_user"
                "suggested_action": str,
            }
        """
        error_type = validation.error_type

        if error_type == ErrorType.PARAM_ERROR:
            return {
                "error_type": "PARAM_ERROR",
                "root_cause": "参数缺失或格式错误",
                "fix_strategy": "correct_params",
                "suggested_action": "使用 LLM 重新解析用户意图，修正参数",
            }
        elif error_type == ErrorType.EMPTY_RESULT:
            return {
                "error_type": "EMPTY_RESULT",
                "root_cause": "查询无结果：可能参数过严或数据不存在",
                "fix_strategy": "switch_tool",
                "suggested_action": "尝试使用替代工具或放宽查询条件",
            }
        elif error_type == ErrorType.SEMANTIC_ERROR:
            return {
                "error_type": "SEMANTIC_ERROR",
                "root_cause": f"结果数据语义异常: {validation.error_message}",
                "fix_strategy": "correct_params",
                "suggested_action": "修正参数 (可能选错股票代码)",
            }
        elif error_type == ErrorType.FORMAT_ERROR:
            return {
                "error_type": "FORMAT_ERROR",
                "root_cause": "返回格式不符合预期",
                "fix_strategy": "switch_tool",
                "suggested_action": "尝试备用数据源",
            }
        elif error_type == ErrorType.TIMEOUT:
            return {
                "error_type": "TIMEOUT",
                "root_cause": "工具调用超时",
                "fix_strategy": "fallback",
                "suggested_action": "使用缓存数据或告知用户稍后重试",
            }
        else:
            return {
                "error_type": "UNKNOWN",
                "root_cause": str(result.error),
                "fix_strategy": "ask_user",
                "suggested_action": "向用户说明问题，请求手动介入",
            }
