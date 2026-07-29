"""行情查询工具（Mock/Real）"""

from typing import Any, Dict, Optional


def get_stock_price(
    stock_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    frequency: str = "daily",
) -> Dict[str, Any]:
    """
    获取股票行情数据。

    当前为 Mock 实现，后续可替换为真实 API（如 Tushare / WindPy）。
    """
    # Mock 数据（模拟行情查询）
    mock_prices = {
        "600519": {"name": "贵州茅台", "price": 1850.00, "change_pct": 2.35},
        "000858": {"name": "五粮液", "price": 152.50, "change_pct": -0.82},
        "300750": {"name": "宁德时代", "price": 210.30, "change_pct": 1.15},
        "002594": {"name": "比亚迪", "price": 268.00, "change_pct": 3.20},
        "601318": {"name": "中国平安", "price": 48.60, "change_pct": 0.45},
        "600036": {"name": "招商银行", "price": 38.20, "change_pct": -1.30},
    }

    normalized_code = stock_code.strip().zfill(6)
    info = mock_prices.get(normalized_code, {"name": f"股票{normalized_code}", "price": 0, "change_pct": 0})

    return {
        "stock_code": normalized_code,
        "stock_name": info["name"],
        "price": info["price"],
        "change_pct": info["change_pct"],
        "volume": 3245600,
        "timestamp": "2026-07-27",
        "frequency": frequency,
        "source": "mock",
        "note": "Mock 数据，实际使用时替换为真实行情 API",
    }
