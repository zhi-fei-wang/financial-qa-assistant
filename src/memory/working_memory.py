"""
工作记忆模块
管理当前对话窗口内的消息，支持溢出降级到短期记忆。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TurnRecord:
    """单轮对话记录"""
    turn_id: str
    timestamp: float
    user_query: str
    agent_response: str
    tool_results: List[Dict[str, Any]]
    entities: List[str]       # 提取的实体 ID 列表
    intent: str               # 意图分类结果
    summary: str              # 轮次摘要
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkingMemory:
    """
    工作记忆：维护最近 N 轮对话的完整内容。
    超出容量的旧轮次会被标记为"待压缩"并降级到短期记忆。
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.turns: List[TurnRecord] = []
        self._turn_counter = 0
        self._overflow_queue: List[TurnRecord] = []  # 等待降级的旧轮次

    def add(
        self,
        user_query: str,
        agent_response: str,
        tool_results: Optional[List[Dict]] = None,
        entities: Optional[List[str]] = None,
        intent: str = "",
        summary: str = "",
        metadata: Optional[Dict] = None,
    ) -> TurnRecord:
        """添加一轮对话到工作记忆"""
        self._turn_counter += 1
        turn = TurnRecord(
            turn_id=f"turn_{self._turn_counter}",
            timestamp=time.time(),
            user_query=user_query,
            agent_response=agent_response,
            tool_results=tool_results or [],
            entities=entities or [],
            intent=intent,
            summary=summary,
            metadata=metadata or {},
        )
        self.turns.append(turn)

        # 溢出管理
        if len(self.turns) > self.max_turns:
            overflow = self.turns.pop(0)  # 最旧的轮次
            self._overflow_queue.append(overflow)

        return turn

    def get_recent(self, n: int = 10) -> List[TurnRecord]:
        """获取最近 n 轮对话"""
        return self.turns[-n:] if n > 0 else []

    def get_all(self) -> List[TurnRecord]:
        """获取所有工作记忆中的轮次"""
        return list(self.turns)

    def get_by_entity(self, entity_id: str) -> List[TurnRecord]:
        """获取提到特定实体的所有轮次"""
        return [t for t in self.turns if entity_id in t.entities]

    def get_overflow(self) -> List[TurnRecord]:
        """获取待降级的溢出轮次，并清空队列"""
        overflow = list(self._overflow_queue)
        self._overflow_queue.clear()
        return overflow

    def clear(self):
        """清空工作记忆"""
        self.turns.clear()
        self._overflow_queue.clear()
        self._turn_counter = 0

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def is_full(self) -> bool:
        return len(self.turns) >= self.max_turns

    def to_context_text(self, max_turns: int = 10) -> str:
        """将工作记忆格式化为 LLM 上下文文本"""
        recent = self.get_recent(max_turns)
        lines = []
        for t in recent:
            lines.append(f"[{t.turn_id}] 用户: {t.user_query}")
            if t.agent_response:
                # 截断过长的回复
                resp = t.agent_response[:500] + "..." if len(t.agent_response) > 500 else t.agent_response
                lines.append(f"[{t.turn_id}] 助手: {resp}")
            if t.tool_results:
                lines.append(f"[{t.turn_id}] 工具调用: {len(t.tool_results)} 次")
        return "\n".join(lines)

    def to_messages(self, max_turns: int = 10) -> List[Dict[str, str]]:
        """将工作记忆转换为 OpenAI 消息格式"""
        recent = self.get_recent(max_turns)
        messages = []
        for t in recent:
            messages.append({"role": "user", "content": t.user_query})
            if t.agent_response:
                messages.append({"role": "assistant", "content": t.agent_response})
        return messages
