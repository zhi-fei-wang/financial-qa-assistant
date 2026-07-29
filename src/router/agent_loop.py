"""
金融问答 Agent 主循环 (ReAct Loop)
统筹记忆检索 → 意图识别 → 工具调用 → 回复生成 → 记忆更新
"""

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..llm import get_llm_client
from ..llm.prompts import RESPONSE_GENERATION_PROMPT
from ..memory.memory_manager import MemoryManager
from ..utils.data_loader import DataLoader
from ..utils.temp_store import get_temp_store
from .intent_classifier import IntentClassifier, IntentResult
from .self_correction import SelfCorrectingRouter
from .tool_executor import ToolExecutor, ToolResult
from .tool_registry import ToolRegistry


class FinancialAgent:
    """
    金融智能问答 Agent 主循环。

    对外暴露的唯一接口: chat(user_query) → response
    """

    warmup_status: Dict[str, Any] = {}  # 类级别共享预热状态

    def __init__(
        self,
        memory: Optional[MemoryManager] = None,
        tool_registry: Optional[ToolRegistry] = None,
        data_loader: Optional[DataLoader] = None,
        use_llm: bool = True,
    ):
        self.llm = get_llm_client() if use_llm else None
        self.use_llm = use_llm

        # 核心模块
        self.memory = memory or MemoryManager(use_llm=use_llm)
        self.tool_registry = tool_registry or ToolRegistry()
        self.data_loader = data_loader or DataLoader()
        self.tool_executor = ToolExecutor(data_loader=self.data_loader)
        self.router = SelfCorrectingRouter(
            self.tool_registry, self.tool_executor, use_llm=use_llm
        )
        self.intent_classifier = IntentClassifier(use_llm=use_llm)

        # 对话历史
        self.conversation_history: List[Dict[str, str]] = []

        # 统计
        self.turn_count = 0
        self.tool_call_count = 0

        # === 预热：DataFrame 秒级加载（不改图，图用 pickle） ===
        if not FinancialAgent.warmup_status:
            self._warmup_dataframes()

    def _warmup_dataframes(self):
        """快速预热：仅预加载 DataFrame（秒级），图由 pickle 懒加载"""
        import time
        t0 = time.time()
        try:
            self.data_loader.load_shareholder_data()
            self.data_loader.load_announcements()
            self.data_loader.load_balance_sheet()
            self.data_loader.load_income()
            self.data_loader.load_cashflow()
            FinancialAgent.warmup_status["dataframes_loaded"] = True
            FinancialAgent.warmup_status["dataframe_time"] = time.time() - t0
        except Exception as e:
            FinancialAgent.warmup_status["dataframe_error"] = str(e)

    def chat(self, user_query: str) -> str:
        """
        单轮对话入口（ReAct 多工具调用循环）。

        Args:
            user_query: 用户输入
        Returns:
            Agent 回复文本
        """
        self.turn_count += 1

        # === Step 1: 记忆检索 ===
        memory_context = self.memory.retrieve(user_query, top_k=8)
        context_text = self.memory.get_context_for_llm(user_query)

        # === Step 2: 意图识别（快速路径判断） ===
        community_text = self.memory.community.get_context_text(max_communities=3)
        intent: IntentResult = self.intent_classifier.classify(
            user_query=user_query,
            context=self.memory.retriever.format_as_text(memory_context),
            community_summaries=community_text,
        )

        # 闲聊 → 无需工具
        if not intent.requires_tool:
            response = self._generate_response_chat(user_query, context_text)
            response = self._sanitize_response(response)
            self._finalize_turn(user_query, response, [], intent.intent)
            return response

        # 低置信度 → 反问
        if intent.needs_clarification:
            response = self._ask_clarification(intent, user_query)
            response = self._sanitize_response(response)
            self._finalize_turn(user_query, response, [], intent.intent)
            return response

        # === Step 2.5: 行动规划（启发 1: 轻量 Plan-before-ReAct） ===
        action_plan = None
        complex_intents = {"EQUITY_PENETRATION", "FINANCIAL_ANALYSIS"}
        if intent.intent in complex_intents and self.llm:
            action_plan = self._plan_actions(user_query, intent, context_text)
            if len(action_plan) <= 1:
                action_plan = None  # 单工具不需要计划

        # === Step 3: ReAct 多工具调用循环 ===
        if self.llm:
            response, tool_results = self._react_loop(
                user_query, context_text, intent, max_iterations=5,
                action_plan=action_plan,
            )
        else:
            response, tool_results = self._single_tool_fallback(user_query, intent)

        # === Step 3.5: 如果回复没有实际数据，自动补充查询股票基本信息 ===
        response = self._sanitize_response(response)
        if self._is_empty_response(response) and (intent.entities or intent.params_hint.get("stock_code")):
            stock_code = intent.params_hint.get("stock_code", "")
            if not stock_code:
                stock_code = intent.entities[0] if intent.entities else ""
            supplement = self._supplement_query(stock_code)
            if supplement:
                response = response.rstrip() + "\n\n---\n\n" + supplement

        # === Step 4: 记忆更新 ===
        self._finalize_turn(user_query, response, tool_results, intent.intent)
        return response

    def _finalize_turn(self, user_query, response, tool_results, intent):
        """收尾：记录对话历史 + 记忆更新"""
        self.memory.add_turn(
            user_query=user_query,
            agent_response=response,
            tool_results=tool_results,
            intent=intent,
        )
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": response})

    def _is_empty_response(self, text: str) -> bool:
        """检测回复是否没有给用户任何实际数据（有数字+单位才算有数据）"""
        import re
        empty_markers = [
            "无法提供", "无法查询", "无法获取", "不具备实时",
            "没有实时", "我无法", "系统没有", "暂无数据",
        ]
        # 必须有实际数字才算有数据（避免"我可以帮您查营收"这种空头支票）
        has_real_data = bool(re.search(r'\d[\d,.]*(?:\s*[亿万元%股])', text))
        has_empty = any(m in text for m in empty_markers)
        return has_empty and not has_real_data

    def _supplement_query(self, entity) -> str:
        """自动补充查询：获取该股票在数据库中实际有的数据"""
        import re
        stock_code = None
        # 尝试从 entity 提取代码（支持纯数字、float、6位、不足6位）
        entity_str = str(entity).strip()
        # 提取所有数字
        digits = re.sub(r'\D', '', entity_str)
        if digits:
            stock_code = digits.zfill(6)  # 补足前导零
        if not stock_code or len(stock_code) != 6:
            return ""

        parts = ["**📊 该股票在系统中的可用数据：**\n"]

        # 财报
        try:
            r = self.tool_executor._mock_financial_statement({
                "stock_code": stock_code, "statement_type": "income"
            })
            if r.get("source") == "dataset":
                s = r.get("summary", {})
                rp = s.get("report_period", "?")
                rev = s.get("tot_oper_rev", "")
                profit = s.get("net_profit_incl_min_int_inc", "") or s.get("net_profit_excl_min_int_inc", "")
                if rev:
                    parts.append(f"- 最新财报({rp}): 营收 {rev:,.0f} 元")
                if profit:
                    parts.append(f"- 净利润: {profit:,.0f} 元")
        except Exception:
            pass

        # 股东
        try:
            r = self.tool_executor._exec_control_summary({"stock_code": stock_code})
            if r.get("source") == "dataset":
                holders = r.get("top_holders", [])
                if holders:
                    parts.append(f"- 前{min(3,len(holders))}大股东: " + ", ".join(
                        f"{h['name'][:20]}({h['pct']:.1f}%)" for h in holders[:3]
                    ))
        except Exception:
            pass

        # 公告
        try:
            r = self.tool_executor._mock_news_search({"stock_code": stock_code})
            if r.get("total", 0) > 0:
                parts.append(f"- 相关公告: {r['total']}条")
        except Exception:
            pass

        if len(parts) > 1:
            parts.append("\n如需详细分析，请告诉我您关注的具体方面。")
            return "\n".join(parts)
        return ""

    def _single_tool_fallback(self, user_query, intent):
        """无 LLM 时的单工具降级路径"""
        tool_result = self.router.execute_with_correction(
            intent=intent.intent,
            params=intent.params_hint,
            user_query=user_query,
            max_retries=2,
        )
        self.tool_call_count += 1
        response = self._format_tool_result_simple(user_query, tool_result) if tool_result else f"收到：{user_query[:50]}..."
        return response, [tool_result] if tool_result else []

    # =========================================================================
    # ReAct 多工具调用循环
    # =========================================================================

    REACT_SYSTEM_PROMPT = """你是一个金融智能问答助手，可以调用工具来获取数据。

## 可用工具
{tool_definitions}

## 工具选择策略
- 用户问"主力资金/资金流向/龙虎榜/换手率/涨停"等行情类问题 → 先调 get_stock_price
- 如果行情工具返回"无实时数据" → **不要放弃**，尝试查财报(query_financial_statement)、股东(control_summary)、公告(search_news)
- 用户问财务/营收/利润/现金流 → query_financial_statement（务必传 statement_type；问"最新/最近"时不要指定report_period，工具会自动取最新）
- 用户问股东/持股/股权 → control_summary 或 equity_penetration
- 用户问违规/处罚/公告 → search_news
- 用户问风险/造假/排雷 → financial_anomaly_check
- **对比类问题（A vs B）**：工具返回的 overview 一览表包含该股票所有报告期。必须：
  - 先查 A（不指定 report_period）→ 看 overview → 记下有哪些年报(1231)
  - 再查 B（不指定 report_period）→ 看 overview → 与 A 的 overview 对照
  - **从 overview 中找两只股票都有的年报(1231)** 作为对比基准（如都有 20251231）
  - 明确传该 report_period 分别再查一次 → 确保同口径
  - 如果 overview 显示没有共同年报 → 选共同季报 → 都没有 → 诚实告知
  - 年报用 ÷365，季报用 ÷90

## 规则（严格遵守）
1. 格式: {{"action":"tool","tool":"工具名","params":{{...}},"reason":"..."}} 或 {{"action":"answer","content":"..."}}
2. **工具返回"无数据"或"NO_REALTIME_DATA"时，你必须立即换一个工具！** 不要在第1个工具失败后就回答
3. 对于任何涉及具体股票的问题，至少调用 2 个不同工具（如先查行情→行情不可用→立即查财报或股东）
4. 回答中要包含你实际查到的数据，不要说"我可以帮你查"却不查
5. 最多 {max_iterations} 次调用，之后必须给出回答
6. **【关键】当输出 action:answer 时，content 必须用纯文本格式！不要在 content 中再嵌套 JSON！**

## 关键原则
- **绝不编造数据**，工具返回什么就说什么，没查到就是没有
- **严禁使用训练数据/预训练知识回答**。你的训练数据不是系统数据源，只有工具返回的结果才是真实数据。如果工具返回58亿，就答58亿；如果工具返回13万，就答13万并标注异常
- **禁止使用"公开信息""根据公开资料""据了解""根据公开财报"等措辞补充数据**——工具没返回的一律不能说
- 系统没有实时行情（股价/涨跌幅/换手率/主力资金/龙虎榜），诚实告知
- 系统有真实财报、股东、公告、研报数据，**主动提供这些替代信息**
- "自选股""我的持仓"需要用户给具体股票代码

## 当前上下文
{memory_context}

## 对话历史
{conversation_history}

Respond ONLY with valid JSON."""

    def _plan_actions(self, user_query: str, intent: IntentResult, context_text: str) -> List[str]:
        """
        启发 1: 轻量 Plan-before-ReAct。
        LLM 生成 2-4 步的工具调用计划，作为 ReAct 循环的建议路径。
        """
        tools_def = self.tool_registry.get_tools_for_llm(intent=intent.intent)
        plan_prompt = (
            f"你需要回答用户问题: {user_query}\n\n"
            f"可用工具:\n{tools_def}\n\n"
            f"请规划需要的工具调用步骤（2-4步），每步一个工具名。\n"
            f"只输出 JSON: {{\"plan\": [\"tool_a\", \"tool_b\"], \"reasoning\": \"...\"}}"
        )
        try:
            raw = self.llm.chat(
                messages=[{"role": "user", "content": plan_prompt}],
                temperature=0.0, max_tokens=256,
            )
            parsed = self._parse_react_json(raw)
            plan = parsed.get("plan", [])
            if plan and len(plan) > 0:
                print(f"[Plan] {user_query[:40]}... -> {' → '.join(plan)}")
            return plan
        except Exception:
            return []

    def _react_loop(
        self,
        user_query: str,
        context_text: str,
        intent: IntentResult,
        max_iterations: int = 5,
        action_plan: Optional[List[str]] = None,
    ) -> tuple:
        """
        ReAct 循环：LLM 可以多次调用不同工具，直到信息充足。

        Returns:
            (final_response: str, tool_results: list)
        """
        tools_def = self.tool_registry.get_tools_for_llm(intent=intent.intent)

        history_str = "\n".join(
            f"{m['role']}: {m['content'][:300]}"
            for m in self.conversation_history[-6:]
        )

        # 注入规划建议
        plan_hint = ""
        if action_plan and len(action_plan) > 1:
            plan_hint = f"\n## 推荐执行计划\n建议按顺序调用: {' → '.join(action_plan)}\n你可以根据实际情况调整计划。\n"

        system_msg = self.REACT_SYSTEM_PROMPT.format(
            tool_definitions=tools_def,
            max_iterations=max_iterations,
            memory_context=context_text[:2000],
            conversation_history=history_str or "（新对话开始）",
        ) + plan_hint

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"用户问题: {user_query}\n\n请逐步分析，需要数据时调用工具。"},
        ]

        tool_results = []

        for iteration in range(max_iterations):
            try:
                # 使用 chat() + 手动 JSON 解析（chat_with_json_output 不支持 messages 参数）
                raw = self.llm.chat(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2048,
                )
                llm_output = self._parse_react_json(raw)
            except Exception as e:
                print(f"[ReAct] LLM call failed at iteration {iteration}: {e}")
                break

            action = llm_output.get("action", "")

            if action == "answer":
                # LLM 认为信息够了，直接返回
                return llm_output.get("content", "抱歉，无法生成回答。"), tool_results

            elif action == "tool":
                tool_name = llm_output.get("tool", "")
                params = llm_output.get("params", {})
                reason = llm_output.get("reason", "")

                if not tool_name:
                    messages.append({
                        "role": "user",
                        "content": "错误: 未指定工具名，请重新输出。"
                    })
                    continue

                # 执行工具
                print(f"[ReAct] Iter {iteration+1}: calling {tool_name}({params}), reason: {reason}")
                result = self._execute_tool_by_name(tool_name, params)
                tool_results.append(result)
                self.tool_call_count += 1

                # 格式化工具结果给 LLM
                if result.get("success"):
                    data = result.get("data", {})
                    rendered = data.get("rendered", "")
                    result_text = rendered if rendered else json.dumps(
                        data, ensure_ascii=False, indent=2,
                        default=self._json_serializer
                    )[:2000]
                else:
                    result_text = f"工具调用失败: {result.get('error', '未知错误')}"

                # 将工具结果注入对话
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(llm_output, ensure_ascii=False)
                })
                messages.append({
                    "role": "user",
                    "content": f"## {tool_name} 执行结果:\n{result_text}\n\n请继续分析。如果信息充足请输出 answer。"
                })

            else:
                # 格式错误，提醒 LLM
                messages.append({
                    "role": "user",
                    "content": "输出格式错误，请输出 JSON: {\"action\":\"tool\",...} 或 {\"action\":\"answer\",...}"
                })

        # 达到最大迭代次数，强制要求回答
        try:
            messages.append({
                "role": "user",
                "content": (
                    "已达到最大工具调用次数。请基于以上所有数据，给出最佳回答。\n"
                    "重要：请直接用纯文本回答（action: answer），不要在 content 中输出 JSON 格式。"
                )
            })
            raw = self.llm.chat(messages=messages, temperature=0.3, max_tokens=2048)
            final = self._parse_react_json(raw)
            if final.get("action") == "answer":
                return final.get("content", "抱歉，无法生成完整回答。"), tool_results
        except Exception:
            pass

        # 最终兜底
        return self._synthesize_from_results(user_query, tool_results), tool_results

    def _execute_tool_by_name(self, tool_name: str, params: Dict) -> Dict:
        """根据工具名直接执行（绕过意图路由）"""
        tool_meta = self.tool_registry.get(tool_name)
        if not tool_meta:
            return {"success": False, "error": f"未知工具: {tool_name}"}

        # 尝试参数补齐
        for req in tool_meta.required_params:
            if req not in params or not params[req]:
                params[req] = "unknown"

        result = self.tool_executor.execute(tool_meta, params)
        return {
            "success": result.success,
            "tool_name": tool_name,
            "data": result.data,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms,
        }

    @staticmethod
    def _parse_react_json(raw: str) -> Dict:
        """解析 ReAct 循环中的 LLM JSON 输出，处理各种畸形格式。

        核心原则：宁可多花解析功夫，也不能让 JSON 原样泄露给用户。
        """
        import re
        raw = raw.strip()

        # Step 0: 尝试从 markdown 代码块中提取 JSON
        code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        if code_block_match:
            raw = code_block_match.group(1).strip()

        # Step 1: 去掉首尾的 markdown 代码块标记（宽松匹配）
        raw = re.sub(r'^```(?:json)?\s*\n?', '', raw)
        raw = re.sub(r'\n?```\s*$', '', raw)
        raw = raw.strip()

        # Step 2: 尝试直接解析
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Step 3: 修复常见的 JSON 语法错误后重试
        fixes = [
            # 修复未转义的换行符在字符串值中
            lambda s: re.sub(r'(?<=[^\\])\\(?!["\\/bfnrtu])', r'\\\\', s),
            # 修复中文引号在 JSON 值中
            lambda s: s.replace('“', '"').replace('”', '"'),
        ]
        for fix in fixes:
            try:
                fixed = fix(raw)
                return json.loads(fixed)
            except (json.JSONDecodeError, ValueError):
                continue

        # Step 4: 括号平衡提取最外层 JSON 对象
        start = raw.find('{')
        if start >= 0:
            depth = 0
            in_string = False
            escape_next = False
            for i in range(start, len(raw)):
                ch = raw[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\':
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(raw[start:i+1])
                        except json.JSONDecodeError:
                            # 尝试修复后再解析
                            for fix in fixes:
                                try:
                                    return json.loads(fix(raw[start:i+1]))
                                except (json.JSONDecodeError, ValueError):
                                    continue
                            break

        # Step 5: 正则直接提取 action 和 content 字段（最后的兜底）
        action_match = re.search(r'"action"\s*:\s*"(\w+)"', raw)
        content_match = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
        if action_match:
            action = action_match.group(1)
            content = ""
            if content_match:
                content = content_match.group(1)
                # 还原转义字符
                content = content.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
            if action in ("answer", "tool"):
                return {"action": action, "content": content}
            if action == "tool":
                tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', raw)
                params_match = re.search(r'"params"\s*:\s*(\{[^}]+\})', raw)
                result = {"action": "tool", "tool": tool_match.group(1) if tool_match else ""}
                if params_match:
                    try:
                        result["params"] = json.loads(params_match.group(1))
                    except json.JSONDecodeError:
                        result["params"] = {}
                return result

        # Step 6: 完全无法解析 → 当作纯文本 answer（但必须确保不包含 JSON 结构）
        # 移除所有看起来像 JSON 结构的内容
        cleaned = re.sub(r'\{"action"\s*:\s*"[^"]*"\s*,\s*"content"\s*:\s*"', '', raw)
        cleaned = re.sub(r'"\s*\}\s*$', '', cleaned)
        return {"action": "answer", "content": cleaned[:2000]}

    @staticmethod
    def _sanitize_response(text: str) -> str:
        """清理回复文本，去掉可能泄露的 JSON 结构。

        防御策略（多层）：
        1) 整个文本是合法 JSON → 提取 content
        2) 文本某处包含合法 JSON 对象 → 提取 content
        3) 文本包含 markdown JSON 代码块 → 提取并重试
        4) 文本以 "{"action" 开头但有破损 → 正则提取 content 字段
        5) 残余的 JSON 语法结构 → 用正则暴力剥离
        """
        import re
        import json as _json

        if not text or not text.strip():
            return text

        original = text
        text = text.strip()

        # ---- 策略 1: 整个文本就是合法 JSON ----
        if text.startswith('{'):
            for parser in [
                lambda t: _json.loads(t),
                lambda t: _json.loads(t.replace('\n', '\\n').replace('\r', '')),
            ]:
                try:
                    obj = parser(text)
                    if isinstance(obj, dict):
                        action = obj.get("action", "")
                        if action == "answer":
                            content = obj.get("content", "")
                            if content and len(content) >= 4:  # 中文回答4字以上即有效
                                return content
                        elif action == "tool":
                            # tool 格式不应该是最终回复，直接移除整个 JSON
                            return "[系统提示：工具调用格式未正确解析，请重新提问]"
                except (_json.JSONDecodeError, KeyError, TypeError, AttributeError, UnicodeDecodeError):
                    continue

        # ---- 策略 2: 从 markdown 代码块中提取 JSON ----
        code_block = re.search(r'```(?:json)?\s*\n?(\{.*?"action".*?\})\s*\n?```', text, re.DOTALL)
        if code_block:
            for parser in [
                lambda t: _json.loads(t),
                lambda t: _json.loads(t.replace('\n', '\\n').replace('\r', '')),
            ]:
                try:
                    obj = parser(code_block.group(1))
                    if isinstance(obj, dict) and obj.get("action") == "answer":
                        content = obj.get("content", "")
                        if content:
                            # 替换整个代码块为提取的 content
                            return text.replace(code_block.group(0), content)
                except (_json.JSONDecodeError, KeyError, TypeError):
                    continue

        # ---- 策略 3: 在文本任意位置找到完整的 JSON 对象 ----
        json_objects = re.finditer(r'\{[^{}]*"action"\s*:\s*"(?:answer|tool)"[^{}]*\}', text)
        for match in json_objects:
            json_str = match.group(0)
            for parser in [
                lambda s: _json.loads(s),
                lambda s: _json.loads(s.replace('\n', '\\n')),
            ]:
                try:
                    obj = parser(json_str)
                    if isinstance(obj, dict) and obj.get("action") == "answer":
                        content = obj.get("content", "")
                        if content and len(content) >= 3:
                            return content
                except (_json.JSONDecodeError, KeyError, TypeError):
                    continue

        # ---- 策略 4: 正则暴力提取 content 字段 ----
        if text.startswith('{"action"') or '{"action"' in text:
            # 尝试提取 "content":"..." 中的值
            content_match = re.search(
                r'"content"\s*:\s*"((?:[^"\\]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*)"',
                text, re.DOTALL
            )
            if content_match:
                content = content_match.group(1)
                # 还原 JSON 转义
                content = content.replace('\\n', '\n').replace('\\t', '\t')
                content = content.replace('\\"', '"').replace('\\\\', '\\')
                if len(content) > 20:
                    return content

        # ---- 策略 5: 移除残留的 JSON 结构碎片 ----
        # 如果文本看起来以 JSON 结构开头但前面策略都没匹配到
        if text.startswith('{"'):
            # 尝试移除开头的 JSON-like 结构
            cleaned = re.sub(
                r'^\{"action"\s*:\s*"(?:answer|tool)"\s*,\s*"content"\s*:\s*"',
                '', text
            )
            # 移除尾部可能残留的 "}
            cleaned = re.sub(r'"\s*\}\s*$', '', cleaned)
            if cleaned != text and len(cleaned) >= 4:
                return cleaned

        # ---- 策略 6: 移除 markdown 代码块残留 ----
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)

        return text.strip()

    def _synthesize_from_results(self, user_query: str, tool_results: list) -> str:
        """当 ReAct loop 耗尽时，基于所有工具结果合成回答"""
        if not tool_results:
            return self._generate_response_chat(user_query, "")

        # 拼合所有工具的 rendered 文本
        parts = []
        for r in tool_results:
            if r.get("success") and r.get("data"):
                rendered = r["data"].get("rendered", "")
                if rendered:
                    parts.append(rendered[:1500])

        combined = "\n\n---\n\n".join(parts) if parts else "无有效工具结果"

        if self.llm:
            try:
                prompt = (
                    f"基于以下工具执行结果，回答用户问题。\n\n"
                    f"用户问题: {user_query}\n\n"
                    f"工具结果:\n{combined}\n\n"
                    f"请用纯文本给出专业分析（不要输出JSON格式）："
                )
                response = self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2048,
                ).strip()
                # 确保合成结果也经过 sanitize
                return self._sanitize_response(response)
            except Exception:
                pass

        return self._sanitize_response(f"查询结果汇总:\n\n{combined}")

    def chat_multi_turn(self, queries: List[str]) -> List[str]:
        """多轮对话快捷接口"""
        responses = []
        for query in queries:
            resp = self.chat(query)
            responses.append(resp)
        return responses

    # ---- 回复生成 ----

    @staticmethod
    def _json_serializer(obj):
        """处理 pandas/numpy 类型 → JSON"""
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return str(obj)

    def _generate_response_with_tools(
        self, user_query: str, tool_result: Dict, context_text: str
    ) -> str:
        """融合工具结果生成回复"""
        if not self.llm:
            return self._format_tool_result_simple(user_query, tool_result)

        try:
            # 优先使用 rendered 文本（Skill 已格式化好的LLM友好文本）
            data = tool_result.get("data", {})
            rendered = data.get("rendered", "")

            if rendered:
                # rendered 已经是易读文本，直接作为工具结果注入
                tools_str = rendered[:3000]
            else:
                # fallback: JSON 序列化
                tools_str = json.dumps(
                    data,
                    ensure_ascii=False, indent=2, default=self._json_serializer
                )[:2000]

            # 提取关键元信息
            source = data.get("source", "unknown")
            total = data.get("total_chains", data.get("total_announcements", data.get("total_rules", "")))

            history_str = "\n".join(
                f"{m['role']}: {m['content'][:200]}" for m in self.conversation_history[-6:]
            )

            prompt = RESPONSE_GENERATION_PROMPT.format(
                memory_context=context_text[:2000],
                tool_results=tools_str,
                conversation_history=history_str or "（新对话开始）",
                user_query=user_query,
            )

            # 如果是真实数据源，添加强化指令
            if source and source != "mock":
                prompt += (
                    f"\n\n**重要**: 以上工具执行结果来自真实数据集（来源: {source}）。"
                    "请基于这些真实数据进行分析和回答，不要将其描述为模拟数据或示例数据。"
                )

            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048,
            )
            return response.strip()
        except Exception as e:
            print(f"[Agent] LLM response generation failed: {e}")
            return self._format_tool_result_simple(user_query, tool_result)

    def _generate_response_chat(self, user_query: str, context_text: str) -> str:
        """纯对话（无工具调用）"""
        if not self.llm:
            return f"收到您的消息：{user_query[:50]}...（LLM 未连接）"

        try:
            messages = [
                {"role": "system", "content": f"你是一个专业的金融智能问答助手。当前记忆上下文:\n{context_text[:2000]}"},
                *self.conversation_history[-10:],
                {"role": "user", "content": user_query},
            ]
            response = self.llm.chat(messages=messages, temperature=0.3, max_tokens=1024)
            return response.strip()
        except Exception as e:
            return f"抱歉，生成回复时出现错误：{e}"

    def _format_tool_result_simple(self, user_query: str, tool_result: Dict) -> str:
        """简单格式化工具结果（无 LLM 模式）"""
        if not tool_result.get("success"):
            return f"抱歉，查询「{user_query[:50]}...」时遇到问题：{tool_result.get('error', '未知错误')}"

        data = tool_result.get("data", {})
        return f"查询结果：\n```json\n{json.dumps(data, ensure_ascii=False, indent=2, default=str)}\n```"

    def _ask_clarification(self, intent: IntentResult, user_query: str) -> str:
        """反问用户以澄清意图"""
        return (
            f"抱歉，我不太确定您的具体需求（置信度: {intent.confidence:.0%}）。\n"
            f"我的理解是：{intent.reasoning}\n"
            f"请确认您想要做什么？例如：\n"
            f"- 查询行情数据\n"
            f"- 分析财务报表\n"
            f"- 穿透股权结构\n"
            f"- 检索新闻事件"
        )

    # ---- 查询接口 ----

    def get_memory_summary(self) -> str:
        """获取记忆系统摘要"""
        return self.memory.summary()

    def get_router_stats(self) -> Dict:
        """获取路由统计"""
        return self.router.get_stats()

    def reset(self):
        """重置 Agent 状态"""
        self.memory = MemoryManager(use_llm=self.use_llm)
        self.conversation_history.clear()
        self.turn_count = 0
        self.tool_call_count = 0
        # 启发 4: 清理临时文件存储
        try:
            get_temp_store().clear()
        except Exception:
            pass
