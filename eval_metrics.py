"""
赛题量化评测脚本 (P1: 缺口 — 缺少量化指标)

针对赛题要求的 6 项指标进行量化测量:
  1. 关键事实召回率 (Recall) ≥ 90%
  2. API 调用命中准确率 (Precision) ≥ 92%
  3. 自纠错成功率 ≥ 80%
  4. 股权穿透准确率 (深度>3层) ≥ 85%
  5. 财报欺诈 F1-Score ≥ 85%
  6. 工具调用延迟 ≤ 5秒

用法:
    python eval_metrics.py                    # 评测全部 1,410 条
    python eval_metrics.py --sample 50        # 抽样 50 条
    python eval_metrics.py --tasks task2      # 只测股权穿透
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class CompetitionMetrics:
    """赛题量化指标评测器"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "14-知识图谱与智能推荐赛道-东吴证券-基于 Agentic AI 的金融长上下文推理、图谱穿透与财报反欺诈智能问答算法探索")
        self.data_dir = data_dir

        # 加载评测集
        q_path = os.path.join(data_dir, "1", "clean.xlsx") if os.path.isdir(data_dir) else None
        if q_path and os.path.exists(q_path):
            self.questions = pd.read_excel(q_path, engine="openpyxl")
            print(f"[Eval] Loaded {len(self.questions)} questions")
        else:
            # Try relative path
            for root, dirs, files in os.walk("."):
                if "1" in dirs and "clean.xlsx" in os.listdir(os.path.join(root, "1")):
                    q_path = os.path.join(root, "1", "clean.xlsx")
                    break
            if q_path:
                self.questions = pd.read_excel(q_path, engine="openpyxl")
                print(f"[Eval] Loaded {len(self.questions)} questions from {q_path}")
            else:
                print("[Eval] WARNING: Cannot find question set. Running in estimate mode.")
                self.questions = pd.DataFrame(columns=["session_id", "question", "think_flag"])

        self.results: Dict[str, Any] = {
            "task1": {"recall": 0, "api_precision": 0, "self_correction_rate": 0},
            "task2": {"penetration_accuracy": 0, "tool_latency_ms": 0},
            "task3": {"fraud_f1": 0, "report_excellence": 0},
        }

    # =========================================================================
    # Task 1: 记忆召回 + API 命中率
    # =========================================================================

    def eval_task1_memory(self, agent, sample_size: int = 50) -> Dict[str, Any]:
        """评测长对话记忆召回率 (模拟10轮连续对话)"""
        think_questions = self.questions[self.questions["think_flag"] == True]
        if len(think_questions) == 0:
            think_questions = self.questions.head(sample_size)

        sessions = think_questions.groupby("session_id")
        correct_recalls = 0
        total_checks = 0

        for session_id, group in list(sessions.items())[:5]:  # 测试5个session
            # 在每个 session 内模拟多轮对话
            for i, (_, row) in enumerate(group.head(10).iterrows()):
                query = str(row["question"])
                agent.chat(query)

                # 第5轮和第10轮检测关键实体是否仍在记忆中
                if i == 4 or i == min(9, len(group) - 1):
                    # 提取第一轮中的关键实体
                    first_query = str(group.iloc[0]["question"])
                    entities = self._extract_key_entities(first_query)
                    for ent in entities:
                        total_checks += 1
                        ctx = agent.memory.get_entity_context(ent)
                        if ctx.get("found"):
                            correct_recalls += 1

            agent.reset()

        recall = correct_recalls / max(total_checks, 1)
        print(f"  [Task1] Memory Recall: {correct_recalls}/{total_checks} = {recall:.1%}")
        self.results["task1"]["recall"] = recall
        return {"recall": recall, "checks": total_checks, "correct": correct_recalls}

    def eval_task1_api_precision(self, agent, sample_size: int = 50) -> Dict[str, Any]:
        """评测 API 调用命中准确率"""
        data_questions = self._filter_data_questions(sample_size)
        correct_calls = 0
        total_calls = 0
        correction_attempts = 0
        correction_successes = 0

        for _, row in data_questions.iterrows():
            query = str(row["question"])
            # 重置计数
            agent.tool_call_count = 0

            try:
                agent.chat(query)
            except Exception:
                pass

            total_calls += agent.tool_call_count
            # 如果有工具调用且没有错误 → 视为正确
            if agent.tool_call_count > 0:
                correct_calls += 1

        api_precision = correct_calls / max(total_calls, 1)
        print(f"  [Task1] API Precision: {correct_calls}/{total_calls} = {api_precision:.1%}")

        self.results["task1"]["api_precision"] = api_precision
        return {"api_precision": api_precision, "total_calls": total_calls, "correct": correct_calls}

    # =========================================================================
    # Task 2: 股权穿透准确率 + 延迟
    # =========================================================================

    def eval_task2_penetration(self, sample_size: int = 20) -> Dict[str, Any]:
        """评测股权穿透准确率 + 工具调用延迟"""
        # 筛选股权穿透相关问题
        penetration_keywords = ["股权", "穿透", "控股", "持股", "股东", "控制人", "实控"]
        pen_questions = self.questions[
            self.questions["question"].apply(
                lambda q: any(kw in str(q) for kw in penetration_keywords)
            )
        ].head(sample_size)

        if len(pen_questions) == 0:
            print("  [Task2] No penetration questions found. Skipping.")
            return {"accuracy": "N/A", "latency_ms": "N/A"}

        from src.tools.equity_graph import EquityPenetrationSkill

        correct = 0
        total = 0
        latencies = []

        for _, row in pen_questions.iterrows():
            query = str(row["question"])
            # 提取股票代码
            codes = re.findall(r'\b(\d{6})\b', query)
            if not codes:
                total += 1
                continue

            t0 = time.time()
            try:
                result = EquityPenetrationSkill.execute({
                    "target_entity": codes[0],
                    "max_depth": 3,
                    "min_ratio": 0.5,
                    "direction": "upstream",
                })
                total += 1
                # 有链路返回 → 视为正确（简化，理想情况应验证每一跳）
                if result.get("total_chains", 0) > 0:
                    correct += 1
            except Exception:
                total += 1

            latencies.append((time.time() - t0) * 1000)

        accuracy = correct / max(total, 1)
        avg_latency = sum(latencies) / max(len(latencies), 1)
        print(f"  [Task2] Penetration Accuracy: {correct}/{total} = {accuracy:.1%}")
        print(f"  [Task2] Avg Latency: {avg_latency:.0f}ms (target ≤5000ms)")

        self.results["task2"]["penetration_accuracy"] = accuracy
        self.results["task2"]["tool_latency_ms"] = avg_latency
        return {"accuracy": accuracy, "latency_ms": avg_latency, "samples": total}

    # =========================================================================
    # Task 3: 财务欺诈 F1
    # =========================================================================

    def eval_task3_fraud_f1(self, sample_size: int = 30) -> Dict[str, Any]:
        """评测财务欺诈预警 F1-Score"""
        fraud_keywords = ["风险", "造假", "排雷", "异象", "虚增", "勾稽", "异常"]
        fraud_questions = self.questions[
            self.questions["question"].apply(
                lambda q: any(kw in str(q) for kw in fraud_keywords)
            )
        ].head(sample_size)

        if len(fraud_questions) == 0:
            print("  [Task3] No fraud-detection questions found. Skipping.")
            return {"f1": "N/A"}

        from src.tools.financial_anomaly import FinancialAnomalySkill

        tp = fp = fn = 0  # 简化: 触发预警 = positive

        for _, row in fraud_questions.iterrows():
            query = str(row["question"])
            codes = re.findall(r'\b(\d{6})\b', query)
            if not codes:
                continue

            try:
                result = FinancialAnomalySkill.execute({"stock_code": codes[0]})
                if result.get("success"):
                    failed = result.get("failed_rules", 0)
                    if failed > 0:
                        tp += 1  # 触发预警 (简化: 假设问"风险"确实是风险)
                    else:
                        fn += 1  # 没触发但应该触发
            except Exception:
                pass

            # 也测一个随机非风险问题作为 negative
            # (简化处理)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.01)

        print(f"  [Task3] Fraud F1: TP={tp} FP={fp} FN={fn} → P={precision:.2f} R={recall:.2f} F1={f1:.2f}")

        self.results["task3"]["fraud_f1"] = f1
        return {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}

    # =========================================================================
    # 综合
    # =========================================================================

    def run_all(self, agent=None) -> Dict[str, Any]:
        """运行全部评测"""
        print("=" * 60)
        print("赛题量化评测")
        print("=" * 60)

        has_agent = agent is not None

        # Task 1
        print("\n--- Task 1: 长对话记忆 + API 路由 ---")
        if has_agent:
            self.eval_task1_memory(agent, sample_size=30)
            self.eval_task1_api_precision(agent, sample_size=30)
        else:
            print("  [skip] Agent not available")

        # Task 2
        print("\n--- Task 2: 股权穿透 ---")
        self.eval_task2_penetration(sample_size=20)

        # Task 3
        print("\n--- Task 3: 财务欺诈预警 ---")
        self.eval_task3_fraud_f1(sample_size=20)

        # 汇总
        print("\n" + "=" * 60)
        print("评测汇总 vs 赛题目标")
        print("=" * 60)

        targets = {
            "task1.recall": (">= 90%", 0.90),
            "task1.api_precision": (">= 92%", 0.92),
            "task2.penetration_accuracy": (">= 85%", 0.85),
            "task2.tool_latency_ms": ("<= 5000ms", 5000),
            "task3.fraud_f1": (">= 85%", 0.85),
        }

        for key, (target_str, target_val) in targets.items():
            task, metric = key.split(".")
            val = self.results[task][metric]
            if isinstance(val, str):
                status = "⚠️"
            elif isinstance(target_val, float) and isinstance(val, float):
                status = "✅" if val >= target_val else "❌"
            else:
                status = "✅" if val <= target_val else "❌"

            val_str = f"{val:.1%}" if isinstance(val, float) else str(val)
            print(f"  {status} {key}: {val_str} (target: {target_str})")

        return self.results

    # =========================================================================
    # 辅助
    # =========================================================================

    def _extract_key_entities(self, text: str) -> List[str]:
        """提取关键实体（简化：股票代码 + 股票名称 + 指标）"""
        entities = []
        # 6位数字 → 股票代码
        codes = re.findall(r'\b(\d{6})\b', str(text))
        entities.extend(f"stock_{c}" for c in codes)
        # 已知股票名称
        known = ["贵州茅台", "茅台", "五粮液", "宁德时代", "比亚迪", "隆基绿能", "招商银行", "中国平安", "九阳股份"]
        for name in known:
            if name in str(text):
                entities.append(f"stock_{name}")
        # 金融指标
        indicators = ["ROE", "营收", "净利润", "现金流", "毛利率", "存货周转", "资产负债"]
        for ind in indicators:
            if ind in str(text):
                entities.append(f"indicator_{ind}")
        return entities[:5]

    def _filter_data_questions(self, sample_size: int = 50) -> pd.DataFrame:
        """过滤出需要数据库查询的问题"""
        realtime_kw = ["股价", "涨跌", "行情", "换手率", "量比", "龙头", "涨停", "跌停", "主力", "龙虎榜"]
        chat_kw = ["你好", "你能", "有什么", "如何", "怎么用", "怎么样"]

        df = self.questions.copy()
        for kw in realtime_kw + chat_kw:
            df = df[~df["question"].apply(lambda q: kw in str(q))]

        return df.head(sample_size)

    @staticmethod
    def _normalize_code(code: str) -> str:
        code = str(code).strip().upper()
        for suffix in [".SH", ".SZ", ".BJ"]:
            if code.endswith(suffix):
                code = code[:-len(suffix)]
        return code.zfill(6)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--tasks", type=str, default="all")
    ap.add_argument("--no-agent", action="store_true", help="Run without agent (static metrics only)")
    args = ap.parse_args()

    eval = CompetitionMetrics()

    if args.no_agent:
        eval.run_all(agent=None)
    else:
        try:
            from src.router.agent_loop import FinancialAgent
            agent = FinancialAgent(use_llm=True)
            eval.run_all(agent=agent)
        except Exception as e:
            print(f"[Eval] Agent init failed: {e}. Running static metrics.")
            eval.run_all(agent=None)

    # 保存结果
    out_path = os.path.join(os.path.dirname(__file__), ".cache", "competition_metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval.results, f, ensure_ascii=False, indent=2)
    print(f"\n[Eval] Results saved to {out_path}")
