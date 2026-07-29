"""
TempFileStore — 工具间大数据传递 (启发 4: 元数据物化)

在 ReAct 多工具调用中，工具 A 查出大量数据后不是全部塞进 rendered 文本
（会被 LLM 上下文截断），而是写入临时 JSON 文件，只返回 file_id + 摘要。
工具 B 通过 file_id 读取完整数据。

用法:
    store = TempFileStore()
    file_id = store.put({"rows": [...], "summary": "..."})
    data = store.get(file_id)
    store.clear()  # 会话结束时清理
"""

import json
import os
import time
import uuid
from typing import Any, Dict, Optional


class TempFileStore:
    """
    临时文件存储，用于工具间传递大数据。

    文件存储在 .cache/temp_store/ 下，会话结束时清理。
    """

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", ".cache", "temp_store"
            )
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self._files: Dict[str, str] = {}  # file_id → file_path

    def put(self, data: Dict[str, Any], prefix: str = "tool_data") -> str:
        """
        写入临时文件。

        Args:
            data: 任意可 JSON 序列化的字典
            prefix: 文件名前缀

        Returns:
            file_id (用于后续 get)
        """
        file_id = f"{prefix}_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        file_path = os.path.join(self.base_dir, f"{file_id}.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)

        self._files[file_id] = file_path
        return file_id

    def get(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        读取临时文件。

        Args:
            file_id: put() 返回的文件 ID

        Returns:
            数据字典，或 None (文件不存在时)
        """
        file_path = self._files.get(file_id)
        if not file_path:
            file_path = os.path.join(self.base_dir, f"{file_id}.json")

        if not os.path.exists(file_path):
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_summary(self, file_id: str, max_keys: int = 5) -> str:
        """
        获取数据的摘要文本（供 LLM 消费，不完整加载）。

        Args:
            file_id: 文件 ID
            max_keys: 摘要中包含的最多 key 数

        Returns:
            摘要文本
        """
        data = self.get(file_id)
        if not data:
            return f"[文件 {file_id} 不存在]"

        summary_parts = [f"[临时数据: {file_id}]"]

        for key in list(data.keys())[:max_keys]:
            val = data[key]
            if isinstance(val, list):
                summary_parts.append(f"- {key}: {len(val)} 条记录")
                if val and isinstance(val[0], dict):
                    # 展示第一条记录的 key
                    first_keys = list(val[0].keys())[:5]
                    summary_parts.append(f"  字段: {', '.join(first_keys)}")
            elif isinstance(val, dict):
                summary_parts.append(f"- {key}: {len(val)} 个键")
            elif isinstance(val, (int, float)):
                summary_parts.append(f"- {key}: {val}")
            else:
                summary_parts.append(f"- {key}: {str(val)[:100]}")

        return "\n".join(summary_parts)

    def clear(self):
        """清理所有临时文件"""
        for file_path in self._files.values():
            try:
                os.remove(file_path)
            except OSError:
                pass
        self._files.clear()

        # 清理残余文件
        try:
            for fname in os.listdir(self.base_dir):
                if fname.startswith("tool_data_"):
                    os.remove(os.path.join(self.base_dir, fname))
        except OSError:
            pass

    @property
    def file_count(self) -> int:
        return len(self._files)


# 全局单例
_store: Optional[TempFileStore] = None


def get_temp_store() -> TempFileStore:
    """获取全局 TempFileStore 单例"""
    global _store
    if _store is None:
        _store = TempFileStore()
    return _store
