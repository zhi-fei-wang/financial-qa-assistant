# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**东吴证券 · 金融智能问答助手** — 面向 ToC 金融信息服务的 Agentic AI 系统。
第五届中国研究生金融科技创新大赛「揭榜挂帅」赛题。

- **版本**: v2.6.0
- **架构**: 三层 Agentic AI（记忆层 + 路由层 + 执行层）
- **工具系统**: BaseTool 插件化（13 个 Tool，新增工具只需 1 个文件）
- **UI**: Streamlit (app.py)
- **LLM**: DeepSeek V3 (deepseek-chat)，OpenAI 兼容接口

## Key Architecture Decisions

### Tool Plugin System (v2.5+)
- All tools inherit from `src/tools/base.py:BaseTool`
- New tool = 1 file with `@register_tool_class` decorator
- Auto-registration via `ToolRegistry.register_from_class()`
- Tool metadata, routing, execution, prompts all self-contained in one class

### Data Priority (v2.6+)
DB → Uploaded Files → Web Search (three-tier fallback)
Conflict detection when DB and upload disagree

### Memory System
- Working: last 20 turns
- Short-term: NetworkX graph (9 node types, 6 edge types)
- Long-term: Leiden community detection + LLM summaries
- Source labeling: turns tagged with `source_type` (database/uploaded_file/web_search)

## Project Structure

```
src/
├── memory/     # Memory system (8 modules)
├── router/     # Router + ReAct loop (7 modules)
├── graph/      # Equity graph system (8 modules)
├── finance/    # Financial analysis (4 modules)
├── tools/      # Tool plugins (12 modules, BaseTool pattern)
│   ├── base.py             # BaseTool ABC
│   ├── equity_graph.py     # Equity penetration + events + control
│   ├── financial_anomaly.py # Anomaly detection + multi-period
│   ├── research_reports.py  # Research report search
│   ├── query_financial.py   # Financial statement query
│   ├── financial_calculator.py # Financial calculator
│   ├── market_data.py       # Market data (honest degradation)
│   ├── news_search.py       # News/announcement search
│   ├── web_search.py        # DuckDuckGo web search
│   ├── uploaded_data.py     # User-uploaded file search
│   └── file_parser.py       # Multi-format parser + Graph + BM25
├── llm/        # LLM client + prompt templates
└── utils/      # Config, data loader, evaluation

app.py          # Streamlit web UI
```

## Adding a New Tool

1. Create `src/tools/my_tool.py`:
```python
from .base import BaseTool, register_tool_class

@register_tool_class
class MyTool(BaseTool):
    name = "my_tool"
    description = "..."
    required_params = ["query"]
    optional_params = []
    intent_match = ["NEWS_EVENT"]
    param_schema = {"query": {"description": "..."}}
    routing_hint = "用户问X → my_tool"
    trigger_keywords = ["关键词1", "关键词2"]

    def execute(self, params, data_loader=None):
        return {"source": "...", "rendered": "..."}
```

2. Register in `src/router/tool_registry.py:_register_default_tools()`:
```python
from ..tools.my_tool import MyTool
self.register_from_class(MyTool)
```

## Key Files for Common Tasks

| Task | Files to modify |
|------|----------------|
| Add tool | `src/tools/xxx.py` + `src/router/tool_registry.py` (1 line) |
| Add intent | `src/tools/base.py` (BaseTool.intent_match) |
| Fix tool behavior | `src/tools/xxx.py:execute()` |
| Change UI | `app.py` |
| Update prompts | `src/llm/prompts.py` |
| Agent loop logic | `src/router/agent_loop.py` |
