"""
预构建脚本 — 一次性生成股权图谱 pickle 缓存。

运行一次后，后续启动 app.py 即可秒级加载。
只需运行一次：
    python prebuild.py

缓存位置: .cache/stock_graph.pkl (~19MB)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.data_loader import DataLoader
from src.tools.equity_graph import warmup


def main():
    print("=" * 60)
    print("  东吴证券金融AI助手 — 数据预构建")
    print("=" * 60)
    print()
    print("此脚本只需运行一次，生成图谱缓存后")
    print("后续启动 app.py 将秒级加载。")
    print()

    t0 = time.time()

    print("[1/2] 加载数据集...")
    loader = DataLoader()
    loader.load_shareholder_data()
    loader.load_announcements()
    loader.load_balance_sheet()
    loader.load_income()
    loader.load_cashflow()
    print(f"      数据集加载完成 ({time.time()-t0:.1f}s)")

    print("[2/2] 构建股权图谱并缓存...")
    warmup(data_loader=loader)
    print(f"      图谱构建完成 ({time.time()-t0:.1f}s)")

    cache_path = Path(__file__).parent / ".cache" / "stock_graph.pkl"
    size_mb = cache_path.stat().st_size / 1e6 if cache_path.exists() else 0
    print()
    print(f"✅ 缓存已生成: {cache_path} ({size_mb:.1f} MB)")
    print(f"   总耗时: {time.time()-t0:.1f}s")
    print()
    print("现在可以运行: streamlit run app.py")


if __name__ == "__main__":
    main()
