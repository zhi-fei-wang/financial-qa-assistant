"""
评测脚本
从 1/clean.xlsx 构建评测集，模拟多轮对话，计算关键指标。
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .data_loader import DataLoader


@dataclass
class EvalMetrics:
    """评测指标"""
    # 关键事实召回率
    fact_recall: float = 0.0
    fact_count: int = 0

    # 回答准确率（LLM 裁判）
    answer_accuracy: float = 0.0
    total_answers: int = 0
    correct_answers: int = 0

    # API 命中率
    api_precision: float = 0.0
    api_calls: int = 0
    api_correct_calls: int = 0

    # 自纠错成功率
    correction_success_rate: float = 0.0
    correction_attempts: int = 0
    correction_successes: int = 0

    # 延迟
    avg_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "fact_recall": round(self.fact_recall, 3),
            "answer_accuracy": round(self.answer_accuracy, 3),
            "api_precision": round(self.api_precision, 3),
            "correction_success_rate": round(self.correction_success_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


class EvaluationRunner:
    """
    评测运行器。
    从评测问答集中采样，模拟多轮对话，计算所有赛题指标。
    """

    def __init__(self, agent=None):
        self.agent = agent
        self.loader = DataLoader()
        self.qa_df = self.loader.load_qa_test()

    def run(
        self,
        sessions: Optional[List[int]] = None,
        max_turns_per_session: int = 10,
    ) -> EvalMetrics:
        """
        运行评测。

        Args:
            sessions: 要评测的 session ID 列表 (None = 全部)
            max_turns_per_session: 每个 session 最多评测几轮

        Returns:
            EvalMetrics
        """
        metrics = EvalMetrics()

        if sessions:
            target_df = self.qa_df[self.qa_df["session_id"].isin(sessions)]
        else:
            target_df = self.qa_df

        session_groups = target_df.groupby("session_id")
        total_sessions = len(session_groups)

        for session_id, group in session_groups:
            if self.agent:
                self.agent.reset()  # 新 session 重置状态

            # 只取最多 max_turns_per_session 轮
            turns = group.head(max_turns_per_session)
            prev_entities = set()

            for _, row in turns.iterrows():
                question = row["question"]
                think_flag = row.get("think_flag", False)

                # 1. 运行 Agent
                start_time = time.time()
                response = self._get_response(question, think_flag)
                latency_ms = (time.time() - start_time) * 1000
                metrics.total_latency_ms += latency_ms
                metrics.total_answers += 1

                # 2. 评估事实召回率（简化为实体提及率）
                entities_mentioned = self._count_entities_in_response(response)
                if think_flag and self.agent:
                    # 从记忆中检查是否有相关实体
                    memory_entities = self._get_memory_entities()
                    recalled = len(entities_mentioned & memory_entities)
                    if memory_entities:
                        metrics.fact_recall += recalled / len(memory_entities)
                        metrics.fact_count += 1

                # 3. API 命中率（从 router 统计获取）
                if self.agent:
                    stats = self.agent.get_router_stats()
                    metrics.api_calls = stats.get("total_calls", 0)
                    metrics.api_correct_calls = stats.get("successful", 0)
                    metrics.correction_attempts = stats.get("corrected", 0) + stats.get("failed", 0)
                    metrics.correction_successes = stats.get("corrected", 0)

            # 记录进度
            if session_id % 5 == 0:
                print(f"  Evaluated {session_id}/{total_sessions} sessions...")

        # 计算最终指标
        if metrics.total_answers > 0:
            metrics.avg_latency_ms = metrics.total_latency_ms / metrics.total_answers

        if metrics.fact_count > 0:
            metrics.fact_recall = metrics.fact_recall / metrics.fact_count

        if metrics.api_calls > 0:
            metrics.api_precision = metrics.api_correct_calls / metrics.api_calls

        if metrics.correction_attempts > 0:
            metrics.correction_success_rate = metrics.correction_successes / metrics.correction_attempts

        return metrics

    def _get_response(self, question: str, think_flag: bool) -> str:
        """获取回答（Agent 或 Mock）"""
        if self.agent:
            return self.agent.chat(question)
        return f"[Mock] 关于 {question[:50]}..."

    def _count_entities_in_response(self, response: str) -> set:
        """统计回复中提及的实体"""
        # 简化：统计被提及的已知股票
        known_stocks = ["茅台", "五粮液", "宁德时代", "比亚迪", "九阳股份",
                       "600519", "000858", "300750", "002594", "002242"]
        return {s for s in known_stocks if s.lower() in response.lower()}

    def _get_memory_entities(self) -> set:
        """获取记忆中存储的实体"""
        if not self.agent:
            return set()
        try:
            graph = self.agent.memory.graph
            return {
                data.get("name", node)
                for node, data in graph.G.nodes(data=True)
                if data.get("type") in ("Stock", "Person", "Indicator")
            }
        except Exception:
            return set()

    def print_report(self, metrics: EvalMetrics):
        """打印评测报告"""
        d = metrics.to_dict()
        target = {
            "fact_recall": 0.90,
            "answer_accuracy": 0.90,
            "api_precision": 0.92,
            "correction_success_rate": 0.80,
            "avg_latency_ms": 5000,
        }

        print("\n" + "=" * 70)
        print("  📊 任务一 评测报告")
        print("=" * 70)
        for key, val in d.items():
            tgt = target.get(key, 0)
            status = "✅" if val >= tgt else "⚠️"
            label = {
                "fact_recall": "关键事实召回率",
                "answer_accuracy": "回答准确率",
                "api_precision": "API调用命中率",
                "correction_success_rate": "自纠错成功率",
                "avg_latency_ms": "平均延迟(ms)",
            }.get(key, key)
            print(f"  {status} {label:16s}: {val:.3f}  (目标: {tgt})")
        print("=" * 70)


def quick_eval():
    """快速评测：在评测集上运行 2 个 session"""
    from src.router.agent_loop import FinancialAgent

    print("初始化 Agent...")
    agent = FinancialAgent(use_llm=True)

    runner = EvaluationRunner(agent=agent)
    print(f"评测集: {len(runner.qa_df)} 条问答, {runner.qa_df['session_id'].nunique()} 个 session")

    # 只评估前 2 个 session，每个最多 5 轮
    metrics = runner.run(sessions=[1, 2], max_turns_per_session=5)
    runner.print_report(metrics)
    return metrics


if __name__ == "__main__":
    quick_eval()
