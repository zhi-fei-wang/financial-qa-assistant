"""行情查询工具 — BaseTool 插件"""
from typing import Any, Dict

from .base import BaseTool, register_tool_class


@register_tool_class
class MarketDataTool(BaseTool):
    """查询股票行情——系统无实时行情数据，诚实告知并引导使用替代数据。"""

    name = "get_stock_price"
    description = "获取指定股票的最新价格或历史价格数据。支持A股/港股/美股。注：系统无实时行情API。"
    required_params = ["stock_code"]
    optional_params = ["start_date", "end_date", "frequency"]
    intent_match = ["MARKET_DATA"]
    param_schema = {
        "stock_code": {"description": "6位股票代码，如 600519"},
        "start_date": {"description": "开始日期 YYYY-MM-DD"},
        "end_date": {"description": "结束日期 YYYY-MM-DD"},
        "frequency": {"description": "频率: daily/weekly/monthly"},
    }
    routing_hint = (
        "用户问股价/涨跌幅/换手率/主力资金/龙虎榜/量比等行情类问题 → 先调 get_stock_price；"
        "若返回NO_REALTIME_DATA → 换 query_financial_statement / control_summary / search_news"
    )
    trigger_keywords = [
        "股价", "涨跌", "行情", "收盘", "市值", "盘口", "量比",
        "涨幅", "跌幅", "换手率", "主力资金", "资金流向", "涨停",
        "龙虎榜", "融资融券", "开盘价", "最高价", "实时行情",
        "最新价", "市盈率", "pe", "pb", "eps",
    ]
    max_retries = 2
    timeout_sec = 5

    def execute(self, params: Dict[str, Any], data_loader: Any = None) -> Dict[str, Any]:
        stock_code = params.get("stock_code", "")
        return {
            "stock_code": stock_code,
            "error": "NO_REALTIME_DATA",
            "rendered": (
                f"## 行情查询: {stock_code}\n\n"
                "**无法提供实时行情数据**。\n\n"
                "当前系统仅包含以下历史数据：\n"
                "- 财务报表（2023Q4~2026Q1）\n"
                "- 股东持股明细\n"
                "- 公司公告\n"
                "- 券商研报\n\n"
                "如需查询股价/涨跌幅/换手率/主力资金等实时行情，请使用东方财富、同花顺等行情软件。\n"
                "如需查询财务数据或股东信息，我可以帮您查询。"
            ),
            "source": "system",
            "note": "系统无实时行情API接入",
        }
