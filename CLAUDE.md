# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a competition dataset for the **"第五届中国研究生金融科技创新大赛" — 东吴证券赛题**. The challenge is to build an Agentic AI system for financial long-context reasoning, equity penetration graph analysis, and financial statement fraud detection. No code has been written yet — this is a data-only project.

## Dataset Structure

```
14-知识图谱与智能推荐赛道-东吴证券-基于 Agentic AI 的金融长上下文推理、图谱穿透与财报反欺诈智能问答算法探索/
├── 1/clean.xlsx         # 测试问答集 (1410条, 35个session)
├── 2/clean.xlsx         # 股东持股数据 (64.6万行, 6161只股票)
│   └── dict.txt         # 字段字典
├── 3/clean.xlsx         # 公司公告数据 (7311条, 2585只股票)
│   └── ditct.txt        # 字段字典+公告类型码表
├── 4/                   # 上市公司三大财务报表CSV (~3.9万行/表)
│   ├── asharebalancesheet.csv   # 资产负债表 (182字段)
│   ├── asharecashflow.csv       # 现金流量表 (126字段)
│   ├── ashareincome.csv         # 利润表 (114字段)
│   └── *dict.txt                # 各表中文数据字典
├── 5/                   # 券商研报数据
│   ├── rr_main_*.csv            # ~17万行研报(含摘要文本)
│   └── rr_main_dict.txt         # 字段字典
└── 赛题说明.docx        # 完整赛题文档 (命题说明、技术指标、攻关任务)
```

## Data Directory Details

| Dir | Purpose | Key Columns |
|-----|---------|-------------|
| **1** | 评测问答集 — Agent对话测试基准 | `session_id`, `question`, `think_flag` |
| **2** | 股权穿透数据 — 十大股东明细 | `s_holder_name`, `s_holder_pct`, `s_holder_holdercategory`(1=个人,2=企业) |
| **3** | 公司公告 — 违规处罚/风险提示等 | `n_info_fcode`(公告类型码), `ann_dt`, `n_info_title` |
| **4** | 三大财务报表 — 全科目财务数据 | `S_INFO_WINDCODE`, `REPORT_PERIOD`, 100+财务科目 |
| **5** | 券商研报摘要 — 含详细文本 | `title`, `abstract`, `rating_org`, `industry_l1/l2/l3` |

## Competition Task Requirements (3攻关任务)

1. **长对话记忆增强与自适应路由Agent** — 0.5M+ Tokens窗口、10轮+对话、自纠错API路由
2. **股权穿透与事件脉络推理** — 多层隐性控股链路、图谱+Agent工具化、舆情事件簇聚合
3. **财务异象甄别与评分引擎** — 跨科目勾稽演算(存货/营收比、现金流/利润悖离等)、风险评分生成

## Technical Targets

- 长文本问答: 关键事实召回率≥90%, API调用命中率≥92%
- 股权穿透: 深度>3层准确率≥85%, 工具调用延迟≤5秒
- 财报欺诈: 预警F1-Score≥85%, 报告盲评优秀率≥80%

## Recommended Tools

- Python (pandas, openpyxl for Excel, python-docx for DOCX)
- Use `uv` or `pip` for dependency management
- Use `python3 <script>` for data exploration and model building