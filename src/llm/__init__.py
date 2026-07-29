"""
LLM 统一调用封装
支持 DeepSeek API（OpenAI 兼容格式 / Anthropic 兼容格式）
"""

import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..utils.config import get_config, LLMConfig


class LLMClient:
    """统一的 LLM 调用接口，封装重试、JSON Mode、流式输出"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.cfg = config or get_config().llm
        self._client = OpenAI(
            api_key=self.cfg.api_key,
            base_url=self.cfg.base_url,
            timeout=self.cfg.timeout,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        标准对话接口。

        Args:
            messages: [{"role": "user", "content": "..."}, ...]
            system: 系统提示词
            temperature: 温度参数
            max_tokens: 最大输出 token 数
            stream: 是否流式输出

        Returns:
            LLM 回复文本
        """
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        response = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=full_messages,
            temperature=temperature or self.cfg.temperature,
            max_tokens=max_tokens or self.cfg.max_tokens,
            stream=stream,
        )

        if stream:
            return self._collect_stream(response)

        return response.choices[0].message.content or ""

    def chat_with_json_output(
        self,
        user_prompt: str,
        system: Optional[str] = None,
        messages_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        要求 LLM 严格输出 JSON 的对话接口。

        Args:
            user_prompt: 用户提示
            system: 系统提示
            messages_history: 历史消息
            temperature: 温度(默认0以增强确定性)
            max_retries: JSON 解析失败时的最大重试次数

        Returns:
            解析后的 dict
        """
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        if messages_history:
            full_messages.extend(messages_history)

        # 强化 JSON 输出指令
        json_instruction = (
            f"{user_prompt}\n\n"
            "IMPORTANT: Respond ONLY with valid JSON. No markdown code blocks, "
            "no explanations outside the JSON. The response must be parseable by json.loads()."
        )
        full_messages.append({"role": "user", "content": json_instruction})

        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=full_messages,
                    temperature=temperature if temperature is not None else 0.0,
                    max_tokens=self.cfg.max_tokens,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or "{}"
                return self._parse_json(raw)
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    # 在重试时添加更强的 JSON 要求
                    full_messages.append({
                        "role": "assistant",
                        "content": response.choices[0].message.content or ""
                    })
                    full_messages.append({
                        "role": "user",
                        "content": f"Your response was not valid JSON (error: {e}). "
                                   f"Please output ONLY valid JSON. No markdown, no extra text."
                    })
                    continue
                raise
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                raise

        return {}

    def embed(self, texts: List[str]) -> List[List[float]]:
        """文本向量化（暂用 DeepSeek 普通接口替代，后续切换到专用 Embedding 模型）"""
        # DeepSeek 不直接支持 embeddings，这里返回占位
        # 实际使用时切换到 BGE-M3 / text2vec-large-chinese
        raise NotImplementedError(
            "Embedding not available via DeepSeek API. "
            "Use BGE-M3 or text2vec-large-chinese locally."
        )

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        """鲁棒的 JSON 解析：处理 markdown 代码块包裹的情况"""
        raw = raw.strip()
        # 去掉 markdown 代码块标记
        if raw.startswith("```"):
            lines = raw.split("\n")
            # 去掉第一行（```json 或 ```）和最后一行（```）
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)
        return json.loads(raw)

    @staticmethod
    def _collect_stream(response) -> str:
        """收集流式响应为完整字符串"""
        chunks = []
        for chunk in response:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        return "".join(chunks)


# 全局单例
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """获取全局 LLM 客户端单例"""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def test_connection() -> bool:
    """测试 LLM 连接是否正常"""
    try:
        client = get_llm_client()
        response = client.chat(
            messages=[{"role": "user", "content": "你好，请用一句话介绍自己。"}],
            max_tokens=50,
        )
        print(f"[LLM Test] Response: {response[:200]}")
        return True
    except Exception as e:
        print(f"[LLM Test] Connection failed: {e}")
        return False
