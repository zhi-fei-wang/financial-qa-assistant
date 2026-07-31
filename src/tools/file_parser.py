"""用户上传文件解析器

支持 CSV / Excel / TXT / MD，自动建图 + BM25 索引。
结构化数据 → NetworkX 图（列→节点类型，行→节点实例）
非结构化数据 → chunk 切分 → BM25 关键词索引
"""

import os
import re
import pickle
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import networkx as nx

from ..memory.signal_fusion import BM25Scorer


# =========================================================================
# 数据结构
# =========================================================================

@dataclass
class FileRecord:
    """上传文件记录"""
    name: str
    path: str
    file_type: str           # csv / xlsx / txt / md
    size: int                # bytes
    row_count: int = 0       # 结构化行数 或 chunk 数
    columns: List[str] = field(default_factory=list)
    is_structured: bool = False


class UploadIndex:
    """
    上传文件统一索引：图（结构化） + BM25（非结构化） + 元数据。

    使用方式:
        index = UploadIndex()
        index.add_file(file_path, file_type)
        results = index.search("茅台营收")
        index.remove_file(file_name)
        index.clear()
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.graph = nx.DiGraph()
        self.bm25 = BM25Scorer()
        self.files: Dict[str, FileRecord] = {}  # name → record
        self._chunks: List[str] = []             # BM25 文本 chunk
        self._chunk_meta: List[Dict] = []        # chunk → 来源文件
        self._cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), '..', '..', '.cache', 'uploads'
        )
        os.makedirs(self._cache_dir, exist_ok=True)

    # =====================================================================
    # 文件管理
    # =====================================================================

    def add_file(self, file_path: str, file_type: str) -> FileRecord:
        """添加文件，自动解析并建索引。"""
        # 解析
        parsed = FileParser.parse(file_path, file_type)

        # 记录元数据
        name = os.path.basename(file_path)
        size = os.path.getsize(file_path)
        record = FileRecord(
            name=name, path=file_path, file_type=file_type,
            size=size, row_count=parsed.get("row_count", 0),
            columns=parsed.get("columns", []),
            is_structured=parsed.get("is_structured", False),
        )

        # 建图（结构化）
        if parsed.get("is_structured") and parsed.get("dataframe") is not None:
            self._build_graph(parsed["dataframe"], name)
            record.columns = list(parsed["dataframe"].columns)

        # 建 BM25 索引（非结构化）
        chunks = parsed.get("chunks", [])
        for ch in chunks:
            idx = len(self._chunks)
            self._chunks.append(ch)
            self._chunk_meta.append({"file": name, "type": file_type})
            if ch.strip():
                self.bm25.add_document(str(idx), ch)

        self.files[name] = record
        self._save_cache()
        return record

    def remove_file(self, name: str):
        """删除文件及其索引。"""
        if name not in self.files:
            return
        record = self.files[name]

        # 从图中删除该文件的所有节点
        prefix = f"upload:{_safe_id(name)}:"
        nodes_to_remove = [n for n in self.graph.nodes if n.startswith(prefix)]
        self.graph.remove_nodes_from(nodes_to_remove)

        # 从 BM25 中移除对应 chunk（标记为 None）
        for i, meta in enumerate(self._chunk_meta):
            if meta.get("file") == name:
                self._chunks[i] = ""
                self._chunk_meta[i] = {}

        # 清理
        del self.files[name]
        self._save_cache()

    def clear(self):
        """清空所有索引。"""
        self.graph.clear()
        self.bm25 = BM25Scorer()
        self.files.clear()
        self._chunks.clear()
        self._chunk_meta.clear()
        self._save_cache()

    def list_files(self) -> List[FileRecord]:
        return list(self.files.values())

    @property
    def is_empty(self) -> bool:
        return len(self.files) == 0

    # =====================================================================
    # 搜索
    # =====================================================================

    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        统一搜索：图遍历 + BM25 融合。
        结构化匹配优先，非结构化 BM25 补充。
        """
        results = []
        sources = set()

        # 1. 图搜索：查找匹配的节点
        for node, attrs in self.graph.nodes(data=True):
            node_name = attrs.get("name", node)
            node_text = f"{node_name} {str(attrs)}"
            if self._fuzzy_match(query, node_text):
                results.append({
                    "type": "structured",
                    "source_file": attrs.get("source_file", ""),
                    "node": node_name,
                    "data": {k: v for k, v in attrs.items()
                             if k not in ("type", "source_file")},
                    "score": 0.9,
                })
                sources.add(attrs.get("source_file", ""))

        # 2. BM25 搜索：非结构化文本
        if self._chunks and self.bm25._N > 0:
            bm25_results = self._bm25_search(query, top_k)
            for doc_id, score in bm25_results:
                idx = int(doc_id)
                if 0 <= idx < len(self._chunks):
                    chunk = self._chunks[idx]
                    if chunk and len(chunk.strip()) > 5:
                        meta = self._chunk_meta[idx] if idx < len(self._chunk_meta) else {}
                        results.append({
                            "type": "text",
                            "source_file": meta.get("file", ""),
                            "text": chunk[:500],
                            "score": float(score),
                        })
                        sources.add(meta.get("file", ""))

        # 排序
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]

        # 渲染 Markdown
        rendered = self._render_results(query, results, sources)

        return {
            "query": query,
            "total": len(results),
            "results": results,
            "sources": list(sources),
            "rendered": rendered,
            "source": "uploaded_file",
        }

    # =====================================================================
    # 内部方法
    # =====================================================================

    def _build_graph(self, df: pd.DataFrame, file_name: str):
        """从 DataFrame 构建 NetworkX 子图。"""
        prefix = f"upload:{_safe_id(file_name)}"
        columns = list(df.columns)
        row_count = len(df)

        # 为每行创建节点 + 关联
        for i, (_, row) in enumerate(df.iterrows()):
            node_id = f"{prefix}:row_{i}"
            attrs = {"type": "UploadRow", "source_file": file_name, "row_index": i}

            for col in columns:
                val = row[col]
                if pd.isna(val):
                    continue
                val_str = str(val)
                attrs[col] = val_str

                # 尝试解析为数值
                try:
                    num_val = float(str(val).replace(",", "").replace("亿", "e8").replace("万", "e4"))
                    attrs[f"{col}_numeric"] = num_val
                except (ValueError, TypeError):
                    pass

            self.graph.add_node(node_id, **attrs)

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """用 BM25Scorer 搜索所有 chunk，返回 top-k (doc_id, score)。"""
        scores = []
        for i in range(len(self._chunks)):
            doc_id = str(i)
            try:
                s = self.bm25.score(query, doc_id)
                if s > 0:
                    scores.append((doc_id, s))
            except Exception:
                continue
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    @staticmethod
    def _fuzzy_match(query: str, text: str) -> bool:
        """模糊匹配：查询中的关键词是否出现在文本中。"""
        query_lower = query.lower()
        text_lower = text.lower()
        # 查询词 2-gram 匹配
        keywords = query_lower.split()
        matches = sum(1 for kw in keywords if kw in text_lower)
        return matches >= max(1, len(keywords) * 0.5)

    def _render_results(self, query: str, results: List[Dict],
                        sources: set) -> str:
        """渲染搜索结果为 Markdown。"""
        if not results:
            return (
                f"## 📁 上传数据检索: {query}\n\n"
                "上传文件中未找到相关信息。\n"
            )

        lines = [
            f"## 📁 上传数据检索: {query}",
            f"来源文件: {', '.join(sources) if sources else 'N/A'}",
            f"匹配: {len(results)} 条\n",
        ]

        struct_results = [r for r in results if r["type"] == "structured"]
        text_results = [r for r in results if r["type"] == "text"]

        if struct_results:
            lines.append("### 📊 结构化数据匹配\n")
            seen = set()
            for r in struct_results[:10]:
                key = f"{r['source_file']}:{r['node']}"
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"- **{r['node']}** (来自: {r['source_file']})")
                data_items = {k: v for k, v in r.get("data", {}).items()
                              if not k.endswith("_numeric")}
                if data_items:
                    for k, v in data_items.items():
                        lines.append(f"  - {k}: {v}")
                lines.append("")

        if text_results:
            lines.append("### 📝 文本内容匹配\n")
            for r in text_results[:5]:
                lines.append(f"- **{r['source_file']}** (相关度: {r['score']:.2f})")
                lines.append(f"  {r['text'][:300]}")
                lines.append("")

        lines.append("*数据来源: 用户上传文件*")
        return "\n".join(lines)

    def _save_cache(self):
        """保存索引到磁盘（pickle）。"""
        try:
            cache_path = os.path.join(self._cache_dir, "upload_index.pkl")
            state = {
                "files": self.files,
                "chunks": self._chunks,
                "chunk_meta": self._chunk_meta,
            }
            with open(cache_path, "wb") as f:
                pickle.dump(state, f)
        except Exception as e:
            print(f"[UploadIndex] Cache save failed: {e}")

    def load_cache(self) -> bool:
        """从磁盘恢复索引。"""
        try:
            cache_path = os.path.join(self._cache_dir, "upload_index.pkl")
            if not os.path.exists(cache_path):
                return False
            with open(cache_path, "rb") as f:
                state = pickle.load(f)
            self.files = state.get("files", {})
            self._chunks = state.get("chunks", [])
            self._chunk_meta = state.get("chunk_meta", [])
            # 重建 BM25
            valid_chunks = [
                (i, ch) for i, ch in enumerate(self._chunks) if ch and len(ch.strip()) > 5
            ]
            if valid_chunks:
                for idx, text in valid_chunks:
                    self.bm25.add_document(str(idx), text)
            return True
        except Exception as e:
            print(f"[UploadIndex] Cache load failed: {e}")
            return False


# =========================================================================
# 文件解析器
# =========================================================================

class FileParser:
    """多格式文件解析器。"""

    @staticmethod
    def parse(file_path: str, file_type: str) -> Dict[str, Any]:
        """
        解析文件，返回统一数据结构。

        Returns:
            {
                "is_structured": bool,
                "columns": [str],
                "row_count": int,
                "dataframe": pd.DataFrame | None,   # 结构化数据
                "chunks": [str],                     # 非结构化文本
                "file_type": str,
            }
        """
        file_type = file_type.lower().lstrip(".")
        if file_type in ("csv",):
            return FileParser._parse_csv(file_path)
        elif file_type in ("xlsx", "xls"):
            return FileParser._parse_excel(file_path)
        elif file_type in ("txt",):
            return FileParser._parse_text(file_path)
        elif file_type in ("md", "markdown"):
            return FileParser._parse_markdown(file_path)
        else:
            return {
                "is_structured": False, "columns": [], "row_count": 0,
                "dataframe": None, "chunks": [f"不支持的文件格式: {file_type}"],
                "file_type": file_type,
            }

    @staticmethod
    def _parse_csv(file_path: str) -> Dict[str, Any]:
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="gbk")
        columns = list(df.columns)
        chunks = [
            f"列: {', '.join(columns)}",
            *[f"第{i+1}行: " + " | ".join(
                f"{col}={val}" for col, val in row.items() if pd.notna(val)
            ) for i, (_, row) in enumerate(df.head(500).iterrows())]
        ]
        return {
            "is_structured": True, "columns": columns,
            "row_count": len(df), "dataframe": df,
            "chunks": chunks, "file_type": "csv",
        }

    @staticmethod
    def _parse_excel(file_path: str) -> Dict[str, Any]:
        try:
            df = pd.read_excel(file_path, engine="openpyxl")
        except Exception:
            df = pd.read_excel(file_path)
        columns = list(df.columns)
        chunks = [
            f"列: {', '.join(columns)}",
            *[f"第{i+1}行: " + " | ".join(
                f"{col}={val}" for col, val in row.items() if pd.notna(val)
            ) for i, (_, row) in enumerate(df.head(500).iterrows())]
        ]
        return {
            "is_structured": True, "columns": columns,
            "row_count": len(df), "dataframe": df,
            "chunks": chunks, "file_type": "xlsx",
        }

    @staticmethod
    def _parse_text(file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="gbk") as f:
                text = f.read()
        chunks = _chunk_text(text)
        return {
            "is_structured": False, "columns": [],
            "row_count": len(chunks), "dataframe": None,
            "chunks": chunks, "file_type": "txt",
        }

    @staticmethod
    def _parse_markdown(file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="gbk") as f:
                text = f.read()
        chunks = _chunk_markdown(text)
        return {
            "is_structured": False, "columns": [],
            "row_count": len(chunks), "dataframe": None,
            "chunks": chunks, "file_type": "md",
        }


# =========================================================================
# 辅助函数
# =========================================================================

def _chunk_text(text: str, max_chars: int = 500) -> List[str]:
    """将纯文本按段落+长度切分成 chunk。"""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) < max_chars:
            current += para + "\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n"
    if current:
        chunks.append(current.strip())
    return chunks if chunks else [text[:max_chars]]


def _chunk_markdown(text: str, max_chars: int = 500) -> List[str]:
    """将 Markdown 按 ## 标题切分，超过长度的再细分。"""
    # 按 ## 分割
    sections = re.split(r'\n(?=## )', text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            # 按段落分
            paragraphs = section.split("\n\n")
            sub = ""
            for para in paragraphs:
                if len(sub) + len(para) < max_chars:
                    sub += para + "\n\n"
                else:
                    if sub:
                        chunks.append(sub.strip())
                    sub = para + "\n\n"
            if sub:
                chunks.append(sub.strip())
    return chunks if chunks else [text[:max_chars]]


def _safe_id(name: str) -> str:
    """文件名 → 安全标识符。"""
    return re.sub(r'[^a-zA-Z0-9_]', '_', os.path.splitext(name)[0])
