"""
自纠错闭环
工具调用失败 → 验证 → 诊断 → 修正 → 重试 → 降级
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from ..llm import get_llm_client
from ..llm.prompts import PARAM_CORRECTION_PROMPT
from .tool_executor import ToolExecutor, ToolResult
from .tool_registry import ToolMeta, ToolRegistry
from .validators import ErrorDiagnoser, ErrorType, ResultValidator, ValidationResult


class SelfCorrectingRouter:
    """
    自适应路由 + 自纠错闭环。

    执行流程:
    1. 选择工具 → 执行
    2. 验证结果
    3. 如果失败 → 诊断 → 修正参数或切换工具 → 重试
    4. 最多重试 max_retries 次
    5. 全部失败 → 降级策略（缓存/替代方式/告知用户）
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        use_llm: bool = True,
    ):
        self.registry = tool_registry
        self.executor = tool_executor
        self.validator = ResultValidator()
        self.diagnoser = ErrorDiagnoser()
        self.llm = get_llm_client() if use_llm else None

        # 统计
        self.stats = {
            "total_calls": 0,
            "successful": 0,
            "corrected": 0,
            "fallback_used": 0,
            "failed": 0,
        }

    def execute_with_correction(
        self,
        intent: str,
        params: Dict[str, Any],
        user_query: str = "",
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        带自纠错的工具执行主入口。

        Args:
            intent: 意图分类结果
            params: 本次调用的参数
            user_query: 用户原始输入（用于参数修正）
            max_retries: 最大重试次数

        Returns:
            {"success": bool, "data": ..., "error": ..., "attempts": int, "corrections": [...]}
        """
        self.stats["total_calls"] += 1
        tool = self.registry.select_tool(intent, params)
        if not tool:
            self.stats["failed"] += 1
            return {"success": False, "error": f"未找到匹配意图 {intent} 的工具"}

        corrections_log = []

        for attempt in range(max_retries + 1):
            # Step 1: 执行
            result = self.executor.execute(tool, params)
            result.attempts = attempt + 1

            # Step 2: 验证
            validation = self.validator.validate(tool.name, result)

            if validation.passed:
                self.stats["successful"] += 1
                if attempt > 0:
                    self.stats["corrected"] += 1
                return {
                    "success": True,
                    "data": result.data,
                    "attempts": attempt + 1,
                    "corrections": corrections_log,
                    "execution_time_ms": result.execution_time_ms,
                }

            # Step 3: 诊断
            diagnosis = self.diagnoser.diagnose(result, validation)
            corrections_log.append({
                "attempt": attempt + 1,
                "error": validation.error_message,
                "diagnosis": diagnosis,
            })

            # Step 4: 修复策略
            if attempt >= max_retries:
                break  # 最后一次尝试也失败了

            fix_strategy = diagnosis.get("fix_strategy", "ask_user")

            if fix_strategy == "correct_params":
                # LLM 修正参数
                corrected = self._correct_params(tool, params, result.error, user_query)
                if corrected:
                    params = corrected
                    continue

            elif fix_strategy == "switch_tool":
                # 切换替代工具
                alt_tool = self.registry.select_alternative(intent)
                if alt_tool and alt_tool.name != tool.name:
                    tool = alt_tool
                    continue

            # 无法修复 → 退出循环
            break

        # 全部重试失败
        self.stats["failed"] += 1
        return {
            "success": False,
            "error": f"工具 {tool.name} 调用失败，已尝试 {max_retries + 1} 次",
            "corrections": corrections_log,
            "last_error": result.error,
        }

    def _correct_params(
        self,
        tool: ToolMeta,
        current_params: Dict,
        error_msg: str,
        user_query: str,
    ) -> Optional[Dict[str, Any]]:
        """
        使用 LLM 修正调用参数。

        Args:
            tool: 当前工具
            current_params: 当前（失败的）参数
            error_msg: 错误信息
            user_query: 用户原始输入

        Returns:
            修正后的参数，或 None（无法修正）
        """
        if not self.llm or not user_query:
            # 无 LLM 时的简单修正：尝试放宽参数
            return self._heuristic_correct_params(tool, current_params, error_msg)

        try:
            prompt = PARAM_CORRECTION_PROMPT.format(
                user_query=user_query,
                tool_name=tool.name,
                current_params=str(current_params),
                error_message=error_msg,
            )
            result = self.llm.chat_with_json_output(
                user_prompt=prompt, temperature=0.0, max_retries=1
            )
            corrected = result.get("corrected_params", {})
            if corrected and corrected != current_params:
                return corrected
        except Exception as e:
            print(f"[SelfCorrecting] LLM param correction failed: {e}")

        return self._heuristic_correct_params(tool, current_params, error_msg)

    @staticmethod
    def _heuristic_correct_params(
        tool: ToolMeta, params: Dict, error_msg: str
    ) -> Optional[Dict[str, Any]]:
        """启发式参数修正（无 LLM fallback）"""
        corrected = dict(params)

        # 常见错误模式修正
        error_lower = error_msg.lower()

        if "stock_code" in corrected and ("代码" in error_lower or "code" in error_lower):
            # 尝试规范化股票代码
            code = str(corrected["stock_code"]).strip()
            code = code.zfill(6)  # 补足6位
            if code != corrected["stock_code"]:
                corrected["stock_code"] = code
                return corrected

        if "report_period" in corrected and ("period" in error_lower or "日期" in error_lower):
            # 尝试修正报告期格式
            period = str(corrected["report_period"]).strip()
            if not any(q in period for q in ["Q1", "Q2", "Q3", "Q4"]):
                corrected["report_period"] = period + "Q4"
                return corrected

        # 尝试删除可选参数
        for opt in tool.optional_params:
            if opt in corrected:
                del corrected[opt]
                return corrected

        return None

    def get_stats(self) -> Dict:
        """获取路由统计"""
        return dict(self.stats)
