# 东吴证券 · 金融智能问答助手

> **第五届中国研究生金融科技创新大赛 —「揭榜挂帅」赛题**
>
> 基于 Agentic AI 的金融长上下文推理、图谱穿透与财报反欺诈智能问答算法探索

---

## 一、项目概述

面向 ToC 金融信息服务场景的 **Plan-before-ReAct 多工具调用 Agentic AI 系统**。

| 能力 | 描述 |
|------|------|
| **ReAct 多工具调用** | LLM 自主决策调用 11 个工具，单轮最多 5 次，Plan 计划→执行→验证 |
| **长对话记忆** | 三级记忆（工作/短期/长期）+ 四路多信号融合检索（图+BM25+实体+Fact）|
| **股权穿透** | BFS 多跳 + 多信号实体匹配 + LLM 知识补全 + 带权重逻辑链条 |
| **财务反欺诈** | 14 条 A/B 级勾稽规则 + 五维风险评分 + ResultEnvelope 证据驱动输出 |
| **券商研报** | BM25 检索 55K 篇研报，支持关键词/股票/行业过滤 + 评级分布 |
| **事件脉络** | Louvain 聚类 + 两层去重 + 因果推理 + 股权-舆情时间线对齐 |
| **诚实降级** | 无实时行情数据时诚实告知 + 自动补充财报/股东/公告替代数据 |

### 数据集（5 个 A 股脱敏数据，2020~2026）

| 数据集 | 内容 | 规模 | 利用率 |
|--------|------|------|:--:|
| `1/` | 评测问答集 | 1,410 条 / 35 session | 100% |
| `2/` | 股东持股明细 | 64.6 万行 / 6,161 股票 | 90% |
| `3/` | 公司公告 | 7,311 条 | 60% |
| `4/` | 三大财务报表（10 季度） | ~3.9 万行/表 | 85% |
| `5/` | 券商研报摘要 | ~5.5 万篇 / ~17 万行 | 100% (v2.2) |

---

## 二、评测结果

### 全量评测（1,410 条）

| 指标 | 实际值 | 赛题目标 | 判定 |
|------|--------|----------|:--:|
| 编造率 | **0.0%** | ≤5% | ✅ |
| 遗漏率 | **0.0%** | ≤10% | ✅ |
| 综合质量 | **94.6%** | ≥80% | ✅ |

### 赛题量化指标（`eval_metrics.py`）

| 指标 | 目标 |
|------|:--:|
| 关键事实召回率 | ≥ 90% |
| API 调用命中率 | ≥ 92% |
| 股权穿透准确率 (>3层) | ≥ 85% |
| 财报欺诈 F1-Score | ≥ 85% |
| 工具调用延迟 | ≤ 5s |

---

## 三、快速开始

### 前置条件
- **Python 3.10+**
- **任一 LLM API Key**：

| 提供商 | 环境变量 | 默认模型 | 费用 |
|--------|---------|---------|:--:|
| **DeepSeek** (推荐) | `DEEPSEEK_API_KEY` | deepseek-chat | 极低 |
| **OpenAI** | `OPENAI_API_KEY` + `LLM_MODEL=gpt-4o` | gpt-4o | 中 |
| **自定义兼容 API** | `LLM_API_KEY` + `LLM_BASE_URL` + `LLM_MODEL` | 自定义 | — |

> 使用统一的 OpenAI 兼容接口。`LLM_API_KEY` > `DEEPSEEK_API_KEY` > `OPENAI_API_KEY` 优先级递减。

### Windows — 一键启动

```bash
set DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
start.bat
```

### Mac / Linux

```bash
export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
chmod +x start.sh && ./start.sh
```

### 手动启动

```bash
pip install -r requirements.txt
python prebuild.py              # 首次 ~90s（构建图谱缓存）
streamlit run app.py            # → http://localhost:8501
```

```bash
# 评测
python eval_metrics.py --sample 50   # 赛题 6 项指标
python smart_eval.py                 # 全量 1,410 条
```

---

## 四、架构

```
用户输入
  → 记忆检索（四路多信号融合）
     Final = 0.35×Graph + 0.30×BM25 + 0.25×Entity + 0.10×Vector
  → Plan-before-ReAct（复杂查询先规划 2-4 步）
  → ReAct 循环（LLM 自主调用工具，最多 5 次）
     ├─ control_summary            → 股东/控股摘要（DataFrame ~3s）
     ├─ query_financial_statement  → 财报查询（自动推断报表类型）
     ├─ search_news                → 公告检索
     ├─ search_reports             → 券商研报 BM25 检索 ⭐ v2.2
     ├─ search_reports_by_stock    → 按股票查研报 + 评级 ⭐ v2.2
     ├─ financial_anomaly_check    → 14 条规则风险评分
     ├─ multi_period_analysis      → 多期趋势
     ├─ equity_penetration         → 股权穿透（多信号匹配）
     ├─ event_trace                → 事件脉络 + 因果推理
     └─ get_stock_price            → 行情（诚实降级）
  → ResultEnvelope 包装（结论 + 证据 + 置信度 + 局限）
  → JSON 泄露防御（6 层 sanitize）
  → Fact 提取 + 记忆更新（一次 LLM 调用）
```

---

## 五、项目结构

```
├── app.py                      # Streamlit Web UI
├── prebuild.py                 # 图谱预构建（一次性）
├── smart_eval.py               # 全量评测（1,410 条）
├── eval_metrics.py             # 赛题 6 项指标评测 ⭐ v2.2
├── start.bat / start.sh        # 一键启动脚本
├── requirements.txt
├── README.md / CLAUDE.md
│
├── src/
│   ├── memory/                 # 记忆层（8 模块）
│   │   ├── memory_manager.py       # 三级记忆统筹
│   │   ├── knowledge_graph.py      # NetworkX + Fact 节点
│   │   ├── hybrid_retrieval.py     # 四路多信号融合检索 ⭐
│   │   ├── signal_fusion.py        # BM25 + 实体 + 图融合 ⭐
│   │   ├── fact_extractor.py       # 结构化 Fact 提取 ⭐
│   │   ├── entity_extractor.py     # LLM+规则双通道实体抽取
│   │   ├── community.py            # Leiden 社区发现 + 摘要
│   │   └── working_memory.py       # 20 轮滑动窗口
│   │
│   ├── router/                 # 路由层（7 模块）
│   │   ├── agent_loop.py           # Plan-before-ReAct 主循环 ⭐
│   │   ├── result_envelope.py      # 证据驱动 Skill 输出 ⭐
│   │   ├── intent_classifier.py    # 7 类意图 + 关键词修正
│   │   ├── tool_registry.py        # 11 个 Tool 注册中心
│   │   ├── tool_executor.py        # 统一执行 + 快速路径
│   │   ├── self_correction.py      # 自纠错闭环
│   │   └── validators.py           # 三层验证
│   │
│   ├── graph/                  # 图谱系统（8 模块）
│   │   ├── graph_builder.py        # 批量构建 NetworkX 图
│   │   ├── equity_engine.py        # BFS 多跳 + 多信号匹配
│   │   ├── chain_builder.py        # 跨层连接 + LLM 补全
│   │   ├── name_matcher.py         # 精确→模糊→LLM 三级匹配
│   │   ├── event_clusterer.py      # Louvain 聚类 + 两层去重
│   │   ├── openie_extractor.py     # 公告/研报三元组抽取 ⭐
│   │   └── causal_reasoner.py      # 四层因果推理 ⭐
│   │
│   ├── tools/                  # 可调用 Skill（7 模块）
│   │   ├── equity_graph.py         # 股权穿透 + 事件溯源 + 控股
│   │   ├── financial_anomaly.py    # 异象甄别 + 多期分析
│   │   ├── research_reports.py     # 研报检索 ⭐
│   │   ├── financial_db.py         # 财报查询
│   │   ├── market_data.py          # 行情（诚实降级）
│   │   └── news_search.py          # 公告舆情检索
│   │
│   ├── finance/                # 财务系统（4 模块）
│   │   ├── data_extractor.py       # 18 科目标准化
│   │   ├── rule_engine.py          # 14 条 A/B 勾稽规则
│   │   ├── risk_scorer.py          # 五维加权评分
│   │   └── report_generator.py     # Markdown 报告
│   │
│   ├── llm/                    # LLM 封装 + 9 类 Prompt
│   └── utils/                  # 配置 + 数据加载 + TempStore + Neo4j/Vector 适配
│
├── 1/ ~ 5/                     # 赛题 5 个数据集
├── .cache/                     # 图谱 pickle 缓存 (~19MB)
└── tests/                      # 集成测试
```

---

## 六、已注册 Tool（11 个）

| # | 工具名 | 用途 | 数据源 | 版本 |
|---|--------|------|--------|:--:|
| 1 | `get_stock_price` | 行情查询（诚实降级） | — | v1.0 |
| 2 | `query_financial_statement` | 财报查询（自动推断类型+报告期） | 4/ | v1.0 |
| 3 | `equity_penetration` | 多层股权穿透 + 多信号匹配 | 2/ + 图谱 | v1.0 |
| 4 | `search_news` | 公告舆情检索 | 3/ | v1.0 |
| 5 | `financial_calculator` | 财务指标计算 | — | v1.0 |
| 6 | `event_trace` | 事件脉络/时间线 + 因果推理 | 3/ | v1.0 |
| 7 | `control_summary` | 控股摘要/Top10 股东（~3s） | 2/ | v1.0 |
| 8 | `financial_anomaly_check` | 14 规则 + 五维风险 + 证据报告 | 4/ | v1.0 |
| 9 | `multi_period_analysis` | 多期趋势分析 | 4/ | v1.0 |
| 10 | `search_reports` | 券商研报 BM25 检索 | 5/ | v2.2 |
| 11 | `search_reports_by_stock` | 按股票查研报 + 评级分布 | 5/ | v2.2 |

---

## 七、版本演进

| 版本 | 日期 | 内容 |
|:--:|------|------|
| v1.0 | 2026-07-27 | ReAct 循环、三级记忆、14 条规则、诚实降级、6 层 JSON 防御 |
| v2.0 | 2026-07-29 | 多信号融合检索 (SignalFusion)、结构化 Fact 记忆、多信号实体匹配 |
| v2.1 | 2026-07-29 | Plan-before-ReAct、ResultEnvelope、两层去重、TempFileStore |
| v2.2 | 2026-07-29 | 研报检索 (P0)、量化评测 (P1)、OpenIE 抽取 (P2)、因果推理 (P3)、Neo4j+Vector 适配 (P4) |

---

## 八、命令行

```bash
# 预构建图谱
python prebuild.py

# 启动 Web
streamlit run app.py

# 评测
python eval_metrics.py --sample 50        # 赛题 6 项指标
python smart_eval.py                      # 全量 1,410 条

# 查看所有工具
python -c "from src.router.tool_registry import ToolRegistry; [print(f'{t.name}') for t in ToolRegistry().list_all()]"
```

---

> 📅 最后更新：2026-07-29 | 🤖 LLM: 多提供商兼容 | 🏦 命题单位：东吴证券 | 🏫 第五届中国研究生金融科技创新大赛
