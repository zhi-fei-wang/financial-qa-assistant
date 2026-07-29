"""
数据加载与预处理模块
负责加载 5 个数据集并做基础的清洗、规范化。
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import get_config


class DataLoader:
    """统一的数据加载器，负责所有数据集的读取与基础预处理"""

    def __init__(self):
        self.cfg = get_config().data

    def _load_data_flex(self, folder: str, stem: str, nrows=None, has_excel=False):
        """弹性加载: .csv.gz → .csv → .xlsx（本地 + Streamlit Cloud 兼容）"""
        # 尝试多个可能的项目根目录
        candidates = []
        # 1. 从 __file__ 推算 (开发环境)
        candidates.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        # 2. 从当前工作目录 (Streamlit Cloud: /mount/src/financial-qa-assistant)
        candidates.append(os.getcwd())
        # 3. 常见 Streamlit Cloud 挂载点
        for mount in ['/mount/src/financial-qa-assistant', '/app']:
            if os.path.isdir(mount):
                candidates.append(mount)

        for base in candidates:
            base = os.path.join(base, folder)
            for name in [f"{stem}.csv.gz", f"{stem}.csv"]:
                path = os.path.join(base, name)
                if os.path.exists(path):
                    return pd.read_csv(path, nrows=nrows)
            if has_excel:
                xlsx_path = os.path.join(base, f"{stem}.xlsx")
                if os.path.exists(xlsx_path):
                    return pd.read_excel(xlsx_path, nrows=nrows)

        raise FileNotFoundError(f"No {stem}.[csv.gz|csv|xlsx] in any of: {candidates}")

    @staticmethod
    def _read_csv_smart(path: str, **kwargs) -> pd.DataFrame:
        """
        智能 CSV 读取: .csv.gz > .csv > 原始路径。
        支持 gzip 压缩（GitHub-friendly 部署）。
        """
        gz_path = path + '.gz'
        if os.path.exists(gz_path):
            return pd.read_csv(gz_path, compression='gzip', **kwargs)
        elif os.path.exists(path):
            return pd.read_csv(path, **kwargs)
        else:
            raise FileNotFoundError(f"Neither {gz_path} nor {path} exists")

    # ---- 数据集 1: 评测问答集 ----

    def load_qa_test(self) -> pd.DataFrame:
        """加载评测问答集（优先CSV，降级Excel）
        Returns:
            DataFrame with columns: session_id, question, think_flag
        """
        csv_path = self.cfg.qa_test_path.replace('.xlsx', '.csv')
        if os.path.exists(csv_path):
            df = self._read_csv_smart(csv_path)
        else:
            df = pd.read_excel(self.cfg.qa_test_path)
        df["question"] = df["question"].astype(str).str.strip()
        df["session_id"] = df["session_id"].astype(int)
        return df

    def get_sessions(self) -> Dict[int, pd.DataFrame]:
        """按 session_id 分组返回问答集"""
        df = self.load_qa_test()
        return {sid: group.drop(columns=["session_id"])
                for sid, group in df.groupby("session_id")}

    # ---- 数据集 2: 股东持股数据 ----

    def load_shareholder_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """加载股东持股数据（优先 .csv.gz → .csv → .xlsx）"""
        df = self._load_data_flex("2", "clean", nrows, has_excel=True)
        # 后处理
        if "s_info_windcode" in df.columns:
            df["stock_code"] = df["s_info_windcode"].apply(self._normalize_stock_code)
        for col in ["ann_dt", "s_holder_enddate"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col].astype(str), format='%Y%m%d', errors="coerce")
        if "report_period" in df.columns:
            df["report_period"] = df["report_period"].astype(str)
        return df

        # 统一股票代码格式: 补足6位数字
        if "s_info_windcode" in df.columns:
            df["stock_code"] = df["s_info_windcode"].apply(self._normalize_stock_code)

        # 日期列标准化
        for col in ["ann_dt", "s_holder_enddate"]:
            if col in df.columns:
                # 先转字符串再解析，避免整数被当成纳秒
                df[col] = pd.to_datetime(df[col].astype(str), format='%Y%m%d', errors="coerce")
        # report_period 是报告期标识（如 20251231=2025年报），保留为字符串
        if "report_period" in df.columns:
            df["report_period"] = df["report_period"].astype(str)

        # 持有人类型映射
        if "s_holder_holdercategory" in df.columns:
            df["holder_type"] = df["s_holder_holdercategory"].map({
                1: "个人", 2: "企业"
            })

        return df

    def get_shareholder_by_stock(self, stock_code: str) -> pd.DataFrame:
        """获取某只股票的所有历史股东记录"""
        df = self.load_shareholder_data()
        normalized = self._normalize_stock_code(stock_code)
        return df[df["stock_code"] == normalized]

    # ---- 数据集 3: 公司公告 ----

    def load_announcements(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """加载公司公告数据（优先 .csv.gz → .csv → .xlsx）"""
        return self._load_data_flex("3", "clean", nrows, has_excel=True)

        if "s_info_windcode" in df.columns:
            df["stock_code"] = df["s_info_windcode"].apply(self._normalize_stock_code)

        if "ann_dt" in df.columns:
            df["ann_dt"] = pd.to_datetime(df["ann_dt"], errors="coerce")

        # 解析公告类型码（可能是管道分隔的多类型）
        if "n_info_fcode" in df.columns:
            df["fcode_list"] = df["n_info_fcode"].apply(self._parse_fcode)

        return df

    @staticmethod
    def _parse_fcode(fcode_val) -> List[str]:
        """解析公告类型码为列表（支持 5507060000|5506190000 格式）"""
        if pd.isna(fcode_val):
            return []
        return str(fcode_val).split("|")

    # ---- 数据集 4: 三大财务报表 ----

    def load_balance_sheet(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """加载资产负债表"""
        df = pd.read_csv(self.cfg.balance_sheet_path, nrows=nrows)
        return self._clean_financial_statement(df)

    def load_cashflow(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """加载现金流量表"""
        df = pd.read_csv(self.cfg.cashflow_path, nrows=nrows)
        return self._clean_financial_statement(df)

    def load_income(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """加载利润表"""
        df = pd.read_csv(self.cfg.income_path, nrows=nrows)
        return self._clean_financial_statement(df)

    def load_all_financials(self) -> Dict[str, pd.DataFrame]:
        """加载全部三大财务报表"""
        return {
            "balance_sheet": self.load_balance_sheet(),
            "cashflow": self.load_cashflow(),
            "income": self.load_income(),
        }

    @staticmethod
    def _clean_financial_statement(df: pd.DataFrame) -> pd.DataFrame:
        """财务报表通用清洗"""
        df = df.copy()
        if "s_info_windcode" in df.columns:
            df["stock_code"] = df["s_info_windcode"].apply(
                DataLoader._normalize_stock_code
            )
        for col in ["ann_dt"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        if "report_period" in df.columns:
            df["report_period"] = df["report_period"].astype(str)
        return df

    # ---- 数据集 5: 券商研报 ----

    def load_research_reports(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """加载券商研报数据（5/rr_main_*.csv）
        Returns:
            DataFrame with ~17万 rows, 29 columns
        """
        df = pd.read_csv(self.cfg.research_report_path, nrows=nrows)

        # 解析 sec_code|sec_name 组合字段
        if "sec_code" in df.columns and "sec_name" in df.columns:
            pass  # 已分开
        elif "sec_code" in df.columns:
            # 检查是否是 code|name 合并格式
            sample = df["sec_code"].dropna().iloc[0] if len(df) > 0 else ""
            if isinstance(sample, str) and "|" in sample:
                parts = df["sec_code"].str.split("|", n=1, regex=False)
                df["sec_code_clean"] = parts.str[0]
                df["sec_name_from_code"] = parts.str[1]

        for col in ["write_date", "publish_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        return df

    # ---- 工具方法 ----

    @staticmethod
    def _normalize_stock_code(code: str) -> str:
        """统一股票代码格式：去掉后缀(.SH/.SZ等)，补足6位"""
        if pd.isna(code):
            return ""
        code = str(code).strip().upper()
        # 去掉市场后缀
        for suffix in [".SH", ".SZ", ".BJ", ".N", ".HK"]:
            if code.endswith(suffix):
                code = code[:-len(suffix)]
                break
        # 补足6位
        code = code.zfill(6)
        return code


# ---- 便捷函数 ----

def load_all_datasets(nrows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """一次性加载所有数据集（适合探索阶段）"""
    loader = DataLoader()
    return {
        "qa_test": loader.load_qa_test(),
        "shareholder": loader.load_shareholder_data(nrows=nrows),
        "announcements": loader.load_announcements(nrows=nrows),
        "balance_sheet": loader.load_balance_sheet(nrows=nrows),
        "cashflow": loader.load_cashflow(nrows=nrows),
        "income": loader.load_income(nrows=nrows),
        "research_reports": loader.load_research_reports(nrows=nrows),
    }


def print_dataset_summary():
    """打印所有数据集的概览信息"""
    loader = DataLoader()

    datasets = [
        ("评测问答集 (1/)", loader.load_qa_test()),
        ("股东持股数据 (2/)", loader.load_shareholder_data(nrows=5)),
        ("公司公告 (3/)", loader.load_announcements(nrows=5)),
        ("资产负债表 (4/)", loader.load_balance_sheet(nrows=5)),
        ("现金流量表 (4/)", loader.load_cashflow(nrows=5)),
        ("利润表 (4/)", loader.load_income(nrows=5)),
        ("券商研报 (5/)", loader.load_research_reports(nrows=5)),
    ]

    for name, df in datasets:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")
        print(f"  Columns: {list(df.columns)[:10]}...")
        if len(df) > 0:
            print(f"  Sample row: {df.iloc[0].to_dict() if hasattr(df.iloc[0], 'to_dict') else df.iloc[0]}")


if __name__ == "__main__":
    print_dataset_summary()
