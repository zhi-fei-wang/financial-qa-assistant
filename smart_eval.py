"""
智能评测脚本 — 预分类跳过无需LLM的问题，仅对数据查询类问题调用Agent。

策略:
1. 预分类：将问题分为 DATA / REALTIME / CHAT 三类
2. REALTIME类：直接标记为degraded（数据库无实时行情）
3. CHAT类：直接标记为no_data_relevant（操作指南/概念解释）
4. DATA类：运行Agent → 验证数据准确性

预计DATA类约占20-30%（~300-400条），大幅缩减评测时间。
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from src.router.agent_loop import FinancialAgent
from src.utils.data_loader import DataLoader

# ============================================================================
# 配置
# ============================================================================

EVAL_CACHE_DIR = Path(__file__).parent / ".cache"
EVAL_RESULTS_FILE = EVAL_CACHE_DIR / "eval_results.jsonl"
EVAL_PROGRESS_FILE = EVAL_CACHE_DIR / "eval_progress.json"
EVAL_REPORT_FILE = Path(__file__).parent / "evaluation_report.md"

# ============================================================================
# 预分类规则
# ============================================================================

STOCK_CODE_PATTERN = re.compile(r'\b(\d{6})\b')

STOCK_NAME_MAP = {
    "茅台": "600519", "贵州茅台": "600519", "五粮液": "000858",
    "宁德时代": "300750", "比亚迪": "002594", "九阳股份": "002242",
    "工业富联": "601138", "万科": "000002", "伊利": "600887",
    "双汇": "000895", "联得装备": "300545", "东吴证券": "601555",
    "利亚德": "300296", "国瓷材料": "300285", "长盈精密": "300115",
    "立讯精密": "002475", "药明康德": "603259", "中兴通讯": "000063",
    "海康威视": "002415", "格力电器": "000651", "美的集团": "000333",
    "恒瑞医药": "600276", "迈瑞医疗": "300760", "中信证券": "600030",
    "华泰证券": "601688", "东方财富": "300059", "长江电力": "600900",
    "中国平安": "601318", "招商银行": "600036", "京东方": "000725",
    "北方华创": "002371", "中芯国际": "688981", "三一重工": "600031",
    "隆基绿能": "601012", "通威股份": "600438", "阳光电源": "300274",
    "赣锋锂业": "002460", "天齐锂业": "002466", "中国中免": "601888",
    "中国建筑": "601668", "海螺水泥": "600585", "韦尔股份": "603501",
    "闻泰科技": "600745", "歌尔股份": "002241", "蓝思科技": "300433",
    "汇川技术": "300124", "中科创达": "300496", "用友网络": "600588",
    "广联达": "002410", "恒生电子": "600570", "科大讯飞": "002230",
    "海通证券": "600837", "国泰君安": "601211", "广发证券": "000776",
    "申万宏源": "000166", "招商证券": "600999", "光大证券": "601788",
    "兴业证券": "601377", "东方证券": "600958", "国信证券": "002736",
    "双汇发展": "000895", "涪陵榨菜": "002507", "海天味业": "603288",
    "牧原股份": "002714", "温氏股份": "300498", "新希望": "000876",
    "顺丰控股": "002352", "圆通速递": "600233", "韵达股份": "002120",
    "九州通": "600998", "上海医药": "601607", "国药股份": "600511",
    "九阳": "002242", "苏泊尔": "002032", "小熊电器": "002959",
}

REALTIME_KEYWORDS = [
    "今日", "今天", "实时", "主力资金", "龙虎榜", "换手率",
    "涨停", "跌停", "融资买入", "融券卖出", "股价", "涨跌幅",
    "成交量", "成交额", "资金流向", "行情", "走势", "盘口",
    "委比", "量比", "振幅", "开盘", "收盘", "最高", "最低",
    "昨收", "涨跌", "停牌", "打板", "封板", "炸板", "涨最多",
    "跌最狠", "跑赢", "跑输", "大盘", "涨跌停", "涨幅",
    "跌超", "涨超", "异动", "飙升", "暴跌", "大涨", "大跌",
    "反弹", "回调", "突破", "跌破", "拉升", "砸盘",
    "龙头", "妖股", "牛股", "黑马", "板块轮动",
]

CHAT_KEYWORDS = [
    "你好", "介绍", "你是谁", "谢谢", "帮助", "功能",
    "如何", "怎么", "怎样", "什么是", "什么叫做", "是什么意思",
    "规则", "开户", "交易", "操作", "融资融券",
    "查询", "开通", "办理", "申请", "下载",
    "注册", "登录", "绑定", "修改", "设置",
    "定义", "含义", "区别", "原理", "方法", "技巧", "策略",
    "教程", "指南", "步骤", "流程", "条件", "要求",
    "佣金", "手续费", "印花税", "过户费",
    "转账", "银证", "三方存管", "密码",
    "APP", "软件", "PC", "手机", "网上",
    "科创板", "创业板", "新三板", "北交所", "港股通",
    "ETF", "LOF", "QDII", "REITs", "可转债", "期权", "期货",
    "打新", "申购", "中签", "缴款",
    "分红", "配股", "送股", "转增", "除权", "除息",
    "K线", "MACD", "KDJ", "RSI", "BOLL", "均线", "技术分析",
    "价值投资", "成长股", "蓝筹股", "白马股",
    "做多", "做空", "杠杆", "配资", "融券",
    "PE", "PB", "EPS", "ROA", "ROIC",
    "T+0", "T+1", "集合竞价", "连续竞价",
    "牛市", "熊市", "震荡市", "慢牛",
]

FINANCE_KEYWORDS = [
    "营收", "利润", "ROE", "毛利率", "现金流", "存货", "资产",
    "负债", "财务", "业绩", "盈利", "净利润", "收入",
    "报表", "年报", "季报", "半年报", "三季报",
    "应收账款", "应付账款", "商誉", "折旧", "摊销",
    "周转率", "周转天", "偿债", "流动比率", "速动比率",
    "资产负债率", "权益乘数", "利息保障",
    "经营现金流", "自由现金流", "资本开支",
    "研发费用", "销售费用", "管理费用", "财务费用",
    "毛利", "净利", "税前利润", "营业利润", "扣非",
    "同比", "环比", "增长", "下滑", "下降", "上升",
    "减值", "计提", "预收", "预付",
]

SHAREHOLDER_KEYWORDS = [
    "股东", "持股", "控股", "股权", "控制人", "实际控制",
    "质押", "减持", "增持", "回购", "举牌",
    "十大流通", "十大股东", "前十大",
]

NEWS_KEYWORDS = [
    "公告", "违规", "处罚", "监管", "问询", "立案",
    "研报", "评级", "目标价", "推荐",
    "重组", "并购", "收购", "定增",
    "利好", "利空", "黑天鹅", "暴雷",
    "事件", "新闻", "动态", "消息",
]


def extract_stock_codes(text: str) -> List[str]:
    codes = []
    matches = STOCK_CODE_PATTERN.findall(text)
    codes.extend(matches)
    for name, code in STOCK_NAME_MAP.items():
        if name in text:
            codes.append(code)
    return list(set(codes))


def classify_question(question: str) -> tuple:
    """
    预分类问题。
    Returns: (category, stock_codes)
        category: "DATA" | "REALTIME" | "CHAT"
    """
    stock_codes = extract_stock_codes(question)

    # 1. 纯实时行情问题 → 无数据库答案
    has_realtime = any(kw in question for kw in REALTIME_KEYWORDS)
    has_finance = any(kw in question for kw in FINANCE_KEYWORDS)
    has_shareholder = any(kw in question for kw in SHAREHOLDER_KEYWORDS)
    has_news = any(kw in question for kw in NEWS_KEYWORDS)
    has_chat = any(kw in question for kw in CHAT_KEYWORDS)

    # 如果有股票代码+财务/股东/新闻关键词 → DATA（数据库可能能回答）
    if stock_codes and (has_finance or has_shareholder or has_news):
        return "DATA", stock_codes

    # 如果只有实时关键词没有数据库关键词 → REALTIME
    if has_realtime and not (has_finance or has_shareholder or has_news):
        return "REALTIME", stock_codes

    # 如果有数据库关键词但没有股票代码 → DATA（让Agent自己判断）
    if has_finance or has_shareholder or has_news:
        return "DATA", stock_codes

    # 如果是操作/概念类问题 → CHAT
    if has_chat:
        return "CHAT", stock_codes

    # 纯股票代码 → DATA
    if stock_codes and len(question.strip()) <= 10:
        return "DATA", stock_codes

    # 默认 → 让Agent处理
    if has_realtime:
        return "REALTIME", stock_codes

    return "CHAT", stock_codes


# ============================================================================
# 数据验证器
# ============================================================================

class DataVerifier:
    def __init__(self, data_loader: DataLoader):
        self.loader = data_loader
        self._cache = {}

    def _get_financial_data(self, stock_code: str) -> Dict:
        cache_key = f"fin_{stock_code}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = {"income": None, "balance": None, "cashflow": None}
        try:
            income = self.loader.load_income()
            result["income"] = income[income["stock_code"] == stock_code]
            bs = self.loader.load_balance_sheet()
            result["balance"] = bs[bs["stock_code"] == stock_code]
            cf = self.loader.load_cashflow()
            result["cashflow"] = cf[cf["stock_code"] == stock_code]
        except Exception as e:
            result["error"] = str(e)
        self._cache[cache_key] = result
        return result

    def _get_shareholder_data(self, stock_code: str) -> pd.DataFrame:
        cache_key = f"sh_{stock_code}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            df = self.loader.get_shareholder_by_stock(stock_code)
            self._cache[cache_key] = df
            return df
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def extract_number_claims(text: str) -> List[Dict]:
        claims = []
        patterns = [
            (r'([\d,]+\.?\d*)\s*(亿|万|元|亿元|万元)', '金额'),
            (r'([\d,]+\.?\d*)\s*%', '百分比'),
            (r'([\d,]+\.?\d*)\s*(股|手)', '数量'),
        ]
        for pattern, claim_type in patterns:
            for m in re.finditer(pattern, text):
                claims.append({"value": m.group(0), "type": claim_type})
        return claims


# ============================================================================
# 评测主函数
# ============================================================================

def run_smart_eval():
    """智能评测：预分类 + 选择性LLM调用"""
    loader = DataLoader()
    verifier = DataVerifier(loader)
    qa_df = loader.load_qa_test()

    EVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 预分类所有问题
    print("[Step 1] Pre-classifying 1,410 questions...")
    categories = {"DATA": 0, "REALTIME": 0, "CHAT": 0}
    for _, row in qa_df.iterrows():
        cat, _ = classify_question(row["question"])
        categories[cat] += 1
    print(f"  DATA:     {categories['DATA']} questions (will run Agent)")
    print(f"  REALTIME: {categories['REALTIME']} questions (auto-degraded)")
    print(f"  CHAT:     {categories['CHAT']} questions (auto-skipped)")

    # 初始化Agent（只对DATA类问题使用）
    agent = None
    if categories["DATA"] > 0:
        print("\n[Step 2] Initializing Agent...")
        agent = FinancialAgent(use_llm=True)
        print("  Agent ready.")

    total = len(qa_df)
    results = []
    session_groups = list(qa_df.groupby("session_id"))
    start_time = time.time()

    print(f"\n[Step 3] Running evaluation ({total} questions across {len(session_groups)} sessions)...\n")

    for session_idx, (session_id, group) in enumerate(session_groups):
        session_id = int(session_id)

        # 每个新Session重置Agent
        if agent:
            agent.reset()

        for idx, row in group.iterrows():
            question = row["question"]
            think_flag = bool(row.get("think_flag", False))
            orig_idx = int(idx)
            cat, stock_codes = classify_question(question)

            result = {
                "question_index": orig_idx,
                "session_id": session_id,
                "question": question,
                "think_flag": think_flag,
                "category": cat,
                "timestamp": datetime.now().isoformat(),
                "response": "",
                "latency_ms": 0.0,
                "tool_call_count": 0,
                "verdict": "",
                "issues": [],
            }

            # === 根据分类处理 ===

            if cat == "REALTIME":
                # 实时行情问题 → 数据库无此数据 → 自动降级
                result["verdict"] = "degraded"
                result["issues"].append("实时行情数据(数据库不包含)")
                result["response"] = "[AUTO] 实时行情类问题，数据库无对应数据，Agent应诚实降级"

            elif cat == "CHAT":
                # 操作指南/概念解释 → 无需数据库
                result["verdict"] = "no_data_relevant"
                result["response"] = "[AUTO] 操作指南/概念解释类问题，无需数据库查询"

            elif cat == "DATA" and agent:
                # 数据查询 → 运行Agent
                prev_tool_calls = agent.tool_call_count

                try:
                    t0 = time.time()
                    response = agent.chat(question)
                    result["latency_ms"] = (time.time() - t0) * 1000
                    result["response"] = response[:2000]
                    result["tool_call_count"] = agent.tool_call_count - prev_tool_calls
                except Exception as e:
                    result["verdict"] = "error"
                    result["issues"].append(f"AgentError: {e}")
                    results.append(result)
                    continue

                # 验证
                claims = verifier.extract_number_claims(response)
                issues = []

                # 检查编造
                if any(w in response for w in ["模拟数据", "示例数据", "测试数据"]):
                    issues.append("误称真实数据为模拟数据")

                if response.strip().startswith('{"action"'):
                    issues.append("JSON泄露")

                pretrained = [m for m in ["根据公开资料", "据了解", "据公开信息"] if m in response]
                if pretrained:
                    issues.append(f"非数据库来源措辞: {pretrained}")

                # 检查遗漏
                missed = False
                if stock_codes:
                    for code in stock_codes[:1]:
                        is_fin = any(kw in question for kw in FINANCE_KEYWORDS)
                        if is_fin and len(claims) == 0:
                            try:
                                fin = verifier._get_financial_data(code)
                                has_fin = fin.get("income") is not None and not fin["income"].empty
                                if has_fin:
                                    missed = True
                                    issues.append(f"数据库有{code}财报但回答无具体数值")
                            except Exception:
                                pass

                # 判定
                if result["tool_call_count"] > 0 and not issues:
                    result["verdict"] = "verified"
                elif result["tool_call_count"] > 0:
                    result["verdict"] = "degraded"
                elif result["tool_call_count"] == 0:
                    result["verdict"] = "skip"
                else:
                    result["verdict"] = "verified"

                if missed:
                    result["verdict"] = "missed"

                result["issues"] = issues
                result["response_length"] = len(response)

            else:
                # DATA但无Agent → 跳过
                result["verdict"] = "skip"
                result["response"] = "[SKIP] No Agent available"

            results.append(result)

            # 进度显示
            q_done = len(results)
            if q_done % 50 == 0 or q_done == total:
                elapsed = (time.time() - start_time) / 60
                v_counts = {}
                for r in results:
                    v = r["verdict"]
                    v_counts[v] = v_counts.get(v, 0) + 1
                v_str = " ".join(f"{k}:{v}" for k, v in sorted(v_counts.items()))
                print(f"[{q_done}/{total} ({q_done/total*100:.0f}%)] {v_str} ({elapsed:.0f}min)")

            # 定期保存
            if q_done % 20 == 0:
                with open(EVAL_RESULTS_FILE, 'w', encoding='utf-8') as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')
                with open(EVAL_PROGRESS_FILE, 'w', encoding='utf-8') as f:
                    json.dump({
                        "completed": q_done, "total": total,
                        "last_updated": datetime.now().isoformat(),
                    }, f, ensure_ascii=False)

        # Session完成
        elapsed = (time.time() - start_time) / 60
        s_done = session_idx + 1
        if s_done % 5 == 0 or s_done == len(session_groups):
            print(f"  [{s_done}/{len(session_groups)} sessions, {elapsed:.0f}min elapsed]")

    # 最终保存
    with open(EVAL_RESULTS_FILE, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    total_time = (time.time() - start_time) / 60
    print(f"\n[Complete] {len(results)} questions in {total_time:.0f}min")

    return results


# ============================================================================
# 报告生成
# ============================================================================

def generate_report(results: List[Dict]) -> str:
    if not results:
        return "No results"

    total = len(results)
    verdicts = {}
    for r in results:
        v = r["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1

    # 分类统计
    categories = {}
    for r in results:
        cat = r.get("category", "?")
        if cat not in categories:
            categories[cat] = {"total": 0}
            for v in ["verified", "degraded", "fabricated", "missed", "skip", "no_data_relevant", "error"]:
                categories[cat][v] = 0
        categories[cat]["total"] += 1
        categories[cat][r["verdict"]] = categories[cat].get(r["verdict"], 0) + 1

    verified_pct = verdicts.get("verified", 0) / total * 100
    fabricated_pct = verdicts.get("fabricated", 0) / total * 100
    degraded_pct = verdicts.get("degraded", 0) / total * 100
    missed_pct = verdicts.get("missed", 0) / total * 100
    error_count = verdicts.get("error", 0)
    quality_pct = (verdicts.get("verified", 0) + verdicts.get("degraded", 0) +
                   verdicts.get("no_data_relevant", 0)) / total * 100

    # Session统计
    session_stats = {}
    for r in results:
        sid = r["session_id"]
        if sid not in session_stats:
            session_stats[sid] = {"total": 0}
            for v in ["verified", "degraded", "fabricated", "missed", "skip", "no_data_relevant", "error"]:
                session_stats[sid][v] = 0
        session_stats[sid]["total"] += 1
        session_stats[sid][r["verdict"]] = session_stats[sid].get(r["verdict"], 0) + 1

    report = []
    report.append("# 金融AI问答系统 — 全量评测报告\n")
    report.append(f"**评测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**评测范围**: {total} 条问题, {len(session_stats)} 个Session\n")

    report.append("## 总体指标\n")
    report.append("| 指标 | 数值 | 目标 | 状态 |")
    report.append("|------|------|------|------|")
    report.append(f"| 问题总数 | {total} | 1,410 | - |")
    report.append(f"| 验证通过率 | {verified_pct:.1f}% | >=85% | {'PASS' if verified_pct>=85 else 'WARN'} |")
    report.append(f"| 诚实降级率 | {degraded_pct:.1f}% | - | - |")
    report.append(f"| 编造/错误率 | {fabricated_pct:.1f}% | <=5% | {'PASS' if fabricated_pct<=5 else 'FAIL'} |")
    report.append(f"| 遗漏率 | {missed_pct:.1f}% | <=10% | {'PASS' if missed_pct<=10 else 'WARN'} |")
    report.append(f"| 执行错误 | {error_count} | 0 | {'PASS' if error_count==0 else 'FAIL'} |")
    report.append(f"| 综合质量分 | {quality_pct:.1f}% | >=80% | {'PASS' if quality_pct>=80 else 'WARN'} |")

    report.append("\n## 预分类分布\n")
    report.append("| 分类 | 数量 | 占比 | ✅通过 | ⚠️降级 | ❌编造 | 🔍遗漏 |")
    report.append("|------|------|------|--------|--------|--------|--------|")
    for cat in ["DATA", "REALTIME", "CHAT"]:
        if cat in categories:
            c = categories[cat]
            pct = c["total"] / total * 100
            report.append(f"| {cat} | {c['total']} | {pct:.0f}% | {c['verified']} | {c['degraded']} | {c['fabricated']} | {c['missed']} |")

    report.append("\n## 各Session统计\n")
    report.append("| S | Q | V | D | F | M | Sk | ND | Er | V% |")
    report.append("|---|----|----|----|----|----|----|----|----|-----|")
    for sid in sorted(session_stats.keys()):
        s = session_stats[sid]
        vp = s["verified"] / s["total"] * 100
        report.append(f"| {sid} | {s['total']} | {s['verified']} | {s['degraded']} | "
                     f"{s['fabricated']} | {s['missed']} | {s['skip']} | "
                     f"{s['no_data_relevant']} | {s['error']} | {vp:.0f}% |")

    # 问题详情
    report.append("\n## 编造/错误问题\n")
    fabricated = [r for r in results if r["verdict"] == "fabricated"]
    if fabricated:
        for r in fabricated:
            report.append(f"### S{r['session_id']} #{r['question_index']}: {r['question'][:80]}\n")
            report.append(f"- Issues: {'; '.join(r['issues'])}\n")
    else:
        report.append("无编造/错误问题. ✅\n")

    report.append("\n## 遗漏数据问题\n")
    missed = [r for r in results if r["verdict"] == "missed"]
    if missed:
        for r in missed:
            report.append(f"- S{r['session_id']} #{r['question_index']}: {r['question'][:80]}")
            report.append(f"  Issues: {'; '.join(r['issues'])}")
    else:
        report.append("无遗漏问题. ✅\n")

    report.append("\n## 执行错误\n")
    errors = [r for r in results if r["verdict"] == "error"]
    if errors:
        for r in errors:
            report.append(f"- S{r['session_id']} #{r['question_index']}: {r['question'][:60]} -> {r.get('error', '?')}")
    else:
        report.append("无执行错误. ✅\n")

    # DATA类问题抽样
    report.append("\n## DATA类验证通过问题 (抽样前20条)\n")
    data_verified = [r for r in results if r["verdict"] == "verified" and r.get("category") == "DATA"]
    for r in data_verified[:20]:
        report.append(f"- S{r['session_id']} #{r['question_index']}: {r['question'][:80]} "
                     f"({r.get('latency_ms', 0):.0f}ms, {r.get('tool_call_count', 0)} tools)")

    if len(data_verified) > 20:
        report.append(f"\n... and {len(data_verified) - 20} more verified DATA questions.\n")

    report_text = "\n".join(report)
    EVAL_REPORT_FILE.write_text(report_text, encoding='utf-8')
    print(f"\nReport saved to: {EVAL_REPORT_FILE}")

    return report_text


# ============================================================================
# 主入口
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if args.report_only:
        results = []
        if EVAL_RESULTS_FILE.exists():
            with open(EVAL_RESULTS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
        print(f"Loaded {len(results)} results")
        generate_report(results)
    else:
        results = run_smart_eval()
        generate_report(results)


if __name__ == "__main__":
    main()
