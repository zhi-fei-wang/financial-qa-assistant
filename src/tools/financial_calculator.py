"""财务计算器工具 — BaseTool 插件"""
from typing import Any, Dict

from .base import BaseTool, register_tool_class


@register_tool_class
class FinancialCalculatorTool(BaseTool):
    """执行金融指标计算：均值、增长率、比率等。"""

    name = "financial_calculator"
    description = "执行金融指标计算：均值、增长率、比率等。"
    required_params = ["expression"]
    optional_params = ["stock_code", "period"]
    intent_match = ["CALCULATION"]
    param_schema = {
        "expression": {"description": "计算表达式或自然语言描述"},
        "stock_code": {"description": "关联股票代码"},
        "period": {"description": "计算期间"},
    }
    routing_hint = "用户要求计算均值/增长率/求和 → financial_calculator"
    trigger_keywords = ["计算", "算一下", "均值", "平均", "求和", "增长率"]
    max_retries = 1
    timeout_sec = 3

    def execute(self, params: Dict[str, Any], data_loader: Any = None) -> Dict[str, Any]:
        expression = params.get("expression", "")
        return {
            "expression": expression,
            "result": "N/A (Mock)",
            "rendered": f"## 财务计算\n表达式: {expression}\n\n结果: 计算功能需集成具体数值，暂不可用。",
            "source": "mock",
            "note": "财务计算器需集成具体数值",
        }
