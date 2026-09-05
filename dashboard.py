from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from research_status import build_status
from stockpilot.audit import verify_audit_chain, verify_protocol_addendum
from stockpilot.config import Settings
from stockpilot.daily_prediction.freshness import evaluate_daily_prediction_freshness
from stockpilot.future_test import future_test_status

st.set_page_config(page_title="StockPilot CN", page_icon="📈", layout="wide")
st.title("StockPilot CN · A股股票预测")
st.caption("正式 DAILY Gen2 20日横截面相对强弱排名｜严格 PIT｜研究用途｜禁止自动交易")

settings = Settings.from_env()
fallback_stock_names = {
    "601985": "中国核电",
    "600030": "中信证券",
    "601901": "方正证券",
    "601117": "中国化学",
    "000538": "云南白药",
    "601868": "中国能建",
    "601169": "北京银行",
    "601688": "华泰证券",
    "600011": "华能国际",
    "601838": "成都银行",
    "601668": "中国建筑",
    "600926": "杭州银行",
    "601111": "中国国航",
    "600016": "民生银行",
    "600795": "国电电力",
    "601818": "光大银行",
    "601658": "邮储银行",
    "601390": "中国中铁",
    "002352": "顺丰控股",
    "600918": "中泰证券",
}
stock_name_path = settings.data_dir / "stock_names.csv"


@st.cache_data(ttl="1h", max_entries=2)
def load_stock_name_map(path: str, modified_at_ns: int) -> dict[str, str]:
    del modified_at_ns
    data = pd.read_csv(path, dtype={"symbol": str})
    data["symbol"] = data["symbol"].str.zfill(6)
    return dict(zip(data["symbol"], data["name"], strict=False))


stock_names = fallback_stock_names.copy()
if stock_name_path.exists():
    stock_names.update(
        load_stock_name_map(str(stock_name_path), stock_name_path.stat().st_mtime_ns)
    )


@st.fragment(run_every="60s")
def render_formal_daily_prediction() -> None:
    freshness = evaluate_daily_prediction_freshness()
    status = freshness["freshness_status"]
    st.subheader("正式 DAILY Gen2 股票预测")
    st.caption(
        "唯一正式首页数据源：artifacts/daily_predictions/gen2/latest.json；"
        "每60秒只读刷新，不会触发下载、特征生成、模型推理或补跑。"
    )
    metrics = st.columns(4)
    metrics[0].metric("时效状态", status, border=True)
    metrics[1].metric(
        "应有最新交易日", freshness.get("expected_latest_session") or "尚无", border=True
    )
    metrics[2].metric(
        "正式预测日期", freshness.get("latest_prediction_date") or "尚无", border=True
    )
    metrics[3].metric(
        "下一次计划运行",
        freshness.get("scheduled_time") or "日历范围外",
        border=True,
    )

    if status == "INVALID":
        st.error(f"正式预测产物完整性验证失败：{freshness['reason']}")
        st.caption("系统已 fail closed；旧研究排名不会被提升为正式预测。")
        return
    if status == "NO_FORMAL_PREDICTION":
        st.error("NO_FORMAL_PREDICTION：当前不存在通过完整性验证的正式 DAILY Gen2 排名。")
        st.info(
            f"下一合法预测日期：{freshness.get('next_verified_prediction_date')}；"
            f"最早数据窗口：{freshness.get('earliest_legal_time')}。"
        )
        st.caption("2026-09-01、2026-09-03、2026-09-04 均保持不补跑、不回填。")
        return
    if status == "STALE":
        st.error(
            f"STALE：最近正式预测落后 {freshness['lag_sessions']} 个已完成交易日。"
            "以下排名保留用于审计，但不标记为当日预测。"
        )
    else:
        st.success("CURRENT：正式预测日期与最新已完成交易日一致，且 immutable manifest 验证通过。")

    prediction = freshness["prediction"]
    ranking = pd.DataFrame(prediction["predictions"]).sort_values("rank").head(20).copy()
    ranking["symbol"] = ranking["symbol"].astype(str).str.zfill(6)
    ranking.insert(2, "name", ranking["symbol"].map(stock_names).fillna("—"))
    ranking.insert(
        0, "list", ranking["rank"].map(lambda value: "Top10" if value <= 10 else "Top20")
    )
    st.dataframe(
        ranking[
            [
                "list",
                "rank",
                "symbol",
                "name",
                "score",
                "percentile",
                "signal_strength",
                "broad_sector",
            ]
        ],
        column_config={
            "list": st.column_config.TextColumn("榜单"),
            "rank": st.column_config.NumberColumn("全截面排名", format="%d"),
            "symbol": st.column_config.TextColumn("股票代码", pinned=True),
            "name": st.column_config.TextColumn("股票名称"),
            "score": st.column_config.NumberColumn("模型分数", format="%.6f"),
            "percentile": st.column_config.NumberColumn("相对百分位", format="%.2f%%"),
            "signal_strength": st.column_config.TextColumn("相对强度"),
            "broad_sector": st.column_config.TextColumn("宽行业"),
        },
        hide_index=True,
        width="stretch",
        key="formal_daily_gen2_top20",
    )
    with st.expander("完整性与预测身份"):
        st.write(f"Prediction ID: `{prediction['prediction_id']}`")
        st.write(f"Model: `{prediction['model_id']}`")
        st.write(f"Model hash: `{prediction['model_hash']}`")
        st.write(f"Feature manifest hash: `{prediction['feature_manifest_hash']}`")
        st.write(f"Input seal hash: `{prediction['input_seal_hash']}`")
        st.write(
            f"Universe / eligible: {prediction['universe_count']} / {prediction['eligible_count']}"
        )
        st.caption("PIT 已通过；research only；execution authorized = false；broker requests = 0。")


render_formal_daily_prediction()

st.divider()
st.subheader("历史研究与回测（非当日正式预测）")
research_state = build_status()
with st.expander("研究升级状态 · 研究模型不等于正式 DAILY 产品"):
    st.write(
        f"独立修正版：{research_state['candidate_model']}；阶段：{research_state['candidate_stage']}"
    )
    process = research_state["candidate_process"]
    if process.get("identity_verified"):
        st.caption(f"已核验研究进程：PID {process['pid']}；尚未据此认定性能通过。")
    elif research_state["candidate_stage"] in {
        "interrupted",
        "process_unverified",
        "process_identity_mismatch",
        "invalid_status",
        "invalid_registry",
        "incomplete_report",
        "candidate_not_frozen",
    }:
        st.error("研究运行状态异常或无法核验，请查看决策日志；不会自动重启或切换正式模型。")
    elif research_state["candidate_stage"] == "failed":
        st.error(research_state["candidate_runtime"].get("error", "研究运行失败，原记录已保留。"))
    st.warning(
        "V17旧回测存在未来数据泄漏，94%胜率不可作为有效成绩。V20完成修复验证前不切换正式模型。"
    )
    historical = research_state["historical_snapshot"]
    st.write(
        f"旧研究清单：选股日期 {historical['prediction_date']}；择时日期 {historical['timing_date']}"
    )
    for reason in historical["reasons"]:
        st.caption(reason)
    st.caption("研究候选不产生交易指令；未通过全部冻结门槛和未来验证，不替换正式 DAILY 产品。")

summary_path = settings.artifact_dir / "summary.json"
if not summary_path.exists():
    st.info("尚无历史研究汇总；正式 DAILY 状态仍已在页面顶部独立显示。")
    st.stop()

summary = json.loads(summary_path.read_text(encoding="utf-8"))
equity = pd.read_csv(settings.artifact_dir / "equity.csv", parse_dates=["date"])
latest = pd.read_csv(settings.artifact_dir / "latest_signals.csv", dtype={"symbol": str})
signals = pd.read_csv(settings.artifact_dir / "signals.csv", dtype={"symbol": str})

v6_live_dir = settings.artifact_dir / "research_v6" / "live"
latest_signal_paths = sorted((v6_live_dir / "signals").glob("*.csv"))
latest_prediction_paths = sorted((v6_live_dir / "predictions").glob("*.csv"))
v6_report_path = settings.artifact_dir / "research_v6" / "report.json"
v6_report = (
    json.loads(v6_report_path.read_text(encoding="utf-8")) if v6_report_path.exists() else None
)

health = summary.get("data_health", {})
for warning in health.get("warnings", []):
    st.error(f"数据审计：{warning}")

if v6_report and v6_report.get("replacement_approved"):
    active_metrics = v6_report["metrics"]
    st.caption("当前全局指标：V6行业均衡集成 · 回顾性走步测试")
    metric_columns = st.columns(5)
    metric_columns[0].metric("V6策略收益", f"{active_metrics['total_return']:.1%}", border=True)
    metric_columns[1].metric("同期基准", f"{active_metrics['benchmark_return']:.1%}", border=True)
    metric_columns[2].metric("累计超额", f"{active_metrics['excess_return']:.1%}", border=True)
    metric_columns[3].metric("V6最大回撤", f"{active_metrics['max_drawdown']:.1%}", border=True)
    metric_columns[4].metric("V6平均Rank IC", f"{active_metrics['mean_rank_ic']:.3f}", border=True)
    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "正超额年份", f"{active_metrics['positive_test_year_ratio']:.1%}", border=True
    )
    metric_columns[1].metric(
        "非负行业IC",
        f"{active_metrics['nonnegative_broad_sector_ic_ratio']:.1%}",
        border=True,
    )
    metric_columns[2].metric(
        "平均现金权重", f"{active_metrics['average_cash_weight']:.1%}", border=True
    )
else:
    st.caption("当前全局指标：历史Ridge回测")
    metric_columns = st.columns(5)
    metric_columns[0].metric("累计收益", f"{summary['total_return']:.1%}", border=True)
    metric_columns[1].metric("年化收益", f"{summary['annual_return']:.1%}", border=True)
    metric_columns[2].metric("最大回撤", f"{summary['max_drawdown']:.1%}", border=True)
    metric_columns[3].metric("夏普比率", f"{summary['sharpe']:.2f}", border=True)
    metric_columns[4].metric("平均Rank IC", f"{summary['mean_rank_ic']:.3f}", border=True)

(
    tab_shadow,
    tab_probability,
    tab_today,
    tab_backtest,
    tab_race,
    tab_validation,
    tab_v4,
    tab_v3,
    tab_model,
    tab_log,
) = st.tabs(
    [
        "V6历史研究快照",
        "V30历史研究快照",
        "历史回测候选",
        "回测表现",
        "模型赛马",
        "严格验证",
        "V4稳定因子",
        "V3研究",
        "模型解释",
        "历史信号",
    ]
)
with tab_shadow:
    st.subheader("V6历史研究快照")
    st.error(
        "HISTORICAL_RESEARCH_SNAPSHOT：此区域不是 DAILY 正式预测源，也没有持续日更承诺。"
        "其日期只代表最后一次 V6 研究快照。"
    )
    if not latest_signal_paths or not latest_prediction_paths:
        st.info("尚无 V6 历史研究快照。")
    else:
        shadow_latest = pd.read_csv(latest_signal_paths[-1], dtype={"symbol": str})
        prediction_latest = pd.read_csv(latest_prediction_paths[-1], dtype={"symbol": str})
        shadow_latest["symbol"] = shadow_latest["symbol"].str.zfill(6)
        prediction_latest["symbol"] = prediction_latest["symbol"].str.zfill(6)
        shadow_latest.insert(3, "name", shadow_latest["symbol"].map(stock_names).fillna("—"))
        signal_date = latest_signal_paths[-1].stem
        metric_columns = st.columns(5)
        metric_columns[0].metric("信号日期", signal_date, border=True)
        metric_columns[1].metric("可交易股票", len(prediction_latest), border=True)
        metric_columns[2].metric("入选候选", len(shadow_latest), border=True)
        metric_columns[3].metric("持有周期", "5个交易日", border=True)
        metric_columns[4].metric("正式日更", "否", border=True)

        st.warning(
            "V6仅为保留的研究快照：相对V4改进门槛已通过，但严格超额收益门槛未通过。"
            "结果仅供研究，禁止自动交易；不得将它解释为今日正式榜单。"
        )
        display_columns = ["rank", "symbol", "name", "close", "score", "weight"]
        st.dataframe(
            shadow_latest[display_columns],
            column_config={
                "rank": st.column_config.NumberColumn("全截面排名", format="%d"),
                "symbol": st.column_config.TextColumn("股票代码", pinned=True),
                "name": st.column_config.TextColumn("股票名称"),
                "close": st.column_config.NumberColumn("快照收盘价", format="¥%.2f"),
                "score": st.column_config.ProgressColumn(
                    "模型分数", min_value=-0.5, max_value=0.5, format="%.4f"
                ),
                "weight": st.column_config.NumberColumn("模型权重", format="percent"),
            },
            hide_index=True,
            width="stretch",
        )

        st.subheader("查询单只股票的当日模型排名")
        selected_symbol = st.selectbox(
            "股票代码",
            prediction_latest["symbol"].tolist(),
            index=None,
            placeholder="输入或选择股票代码",
            format_func=lambda symbol: f"{symbol}  {stock_names.get(symbol, '')}".strip(),
            key="v6_symbol_lookup",
            bind="query-params",
        )
        if selected_symbol:
            row = prediction_latest.loc[prediction_latest["symbol"] == selected_symbol].iloc[0]
            top_label = f"Top-{len(shadow_latest)}"
            selected_text = f"已进入{top_label}" if bool(row["selected"]) else f"未进入{top_label}"
            metric_columns = st.columns(4)
            metric_columns[0].metric(
                "股票",
                f"{stock_names.get(selected_symbol, '未知')} {selected_symbol}",
                border=True,
            )
            metric_columns[1].metric(
                "横截面排名", f"{int(row['pred_rank'])}/{len(prediction_latest)}", border=True
            )
            metric_columns[2].metric("模型分数", f"{row['score']:.4f}", border=True)
            metric_columns[3].metric("本期状态", selected_text, border=True)
            st.caption("未出现在下拉框中的股票，表示它不在本期可交易预测截面内。")

with tab_probability:
    st.subheader("V30r1 历史研究快照 · 冻结模型前向跟踪")
    st.warning(
        "此区域不承担 DAILY 正式预测日更；production_prediction_ready 未通过时仅作研究观察。"
    )
    v30r1_dir = settings.artifact_dir / "prediction_v30r1"
    v30r1_latest_path = v30r1_dir / "live" / "latest.json"
    v30r1_status_path = v30r1_dir / "certification" / "status.json"
    forward_latest_path = settings.artifact_dir / "prediction_forward" / "v30r1" / "latest.json"
    display_latest_path = forward_latest_path if forward_latest_path.exists() else v30r1_latest_path
    if not display_latest_path.exists() or not v30r1_status_path.exists():
        st.info("尚无V30r1真实预测快照。请先完成冻结验证并运行 predict-v30r1-latest。")
    else:
        probability_metadata = json.loads(display_latest_path.read_text(encoding="utf-8"))
        probability_status = json.loads(v30r1_status_path.read_text(encoding="utf-8"))
        probability_path = Path(probability_metadata["snapshot_path"])
        if not probability_path.exists():
            st.error("V30r1元数据存在，但不可变预测快照缺失。")
        else:
            probability_frame = pd.read_csv(probability_path, dtype={"symbol": str})
            probability_frame["symbol"] = probability_frame["symbol"].str.zfill(6)
            metric_columns = st.columns(5)
            metric_columns[0].metric(
                "预测日期", probability_metadata["prediction_date"], border=True
            )
            metric_columns[1].metric(
                "预测股票数", probability_metadata["prediction_count"], border=True
            )
            metric_columns[2].metric(
                "立即预测认证",
                "通过" if probability_status["production_prediction_ready"] else "未通过",
                border=True,
            )
            metric_columns[3].metric(
                "126日长期确认",
                "已确认" if probability_status["future_126d_confirmed"] else "收集中",
                border=True,
            )
            metric_columns[4].metric("实盘授权", "否", border=True)
            if not probability_status["production_prediction_ready"]:
                st.error(
                    "当前仅展示真实数据上的研究预测，不能视为已认证概率或买卖建议。"
                    "V30r1虽修复了校准反转，但20日校准、基线与稳定性门禁仍未通过。"
                )
            if probability_metadata.get("drift_status") == "SEVERE":
                st.warning("最新特征分布漂移为 SEVERE，系统已把置信度降级为 LOW。")
            display_probability = probability_frame.sort_values("rank_5d").head(20)
            st.dataframe(
                display_probability[
                    [
                        "rank_5d",
                        "symbol",
                        "name",
                        "close",
                        "p_up_1d",
                        "p_up_5d",
                        "p_up_20d",
                        "expected_return_5d",
                        "expected_return_20d",
                        "confidence_level",
                        "risk_level",
                        "prediction_ready",
                    ]
                ],
                column_config={
                    "rank_5d": st.column_config.NumberColumn("5日排名", format="%d"),
                    "symbol": st.column_config.TextColumn("股票代码", pinned=True),
                    "name": st.column_config.TextColumn("股票名称"),
                    "close": st.column_config.NumberColumn("收盘价", format="¥%.2f"),
                    "p_up_1d": st.column_config.NumberColumn("上涨概率 1D", format="percent"),
                    "p_up_5d": st.column_config.NumberColumn("上涨概率 5D", format="percent"),
                    "p_up_20d": st.column_config.NumberColumn("上涨概率 20D", format="percent"),
                    "expected_return_5d": st.column_config.NumberColumn(
                        "预期收益 5D", format="percent"
                    ),
                    "expected_return_20d": st.column_config.NumberColumn(
                        "预期收益 20D", format="percent"
                    ),
                    "confidence_level": st.column_config.TextColumn("置信度"),
                    "risk_level": st.column_config.TextColumn("风险等级"),
                    "prediction_ready": st.column_config.CheckboxColumn("认证可用"),
                },
                hide_index=True,
                width="stretch",
            )
            st.caption(
                f"模型 {probability_metadata['model_version']}；不可变快照 SHA-256："
                f"{probability_metadata['sha256']}"
            )
            if display_latest_path == forward_latest_path:
                st.caption("当前展示为冻结V30r1模型的最新HFQ/PIT前向推理，未重训、未调参。")

with tab_today:
    latest["symbol"] = latest["symbol"].str.zfill(6)
    latest["score"] = latest["score"].map(lambda x: f"{x:.4%}")
    latest["weight"] = latest["weight"].map(lambda x: f"{x:.1%}")
    latest["limit_rate"] = latest["limit_rate"].map(lambda x: f"{x:.0%}")
    latest["close"] = latest["close"].map(lambda x: f"{x:.2f}")
    st.subheader(f"历史回测最近信号日：{latest['date'].iloc[0]}")
    st.dataframe(latest, width="stretch", hide_index=True)
    st.caption("这一页来自历史回测产物；当前正式 DAILY 状态与榜单仅显示在页面顶部。")
    st.warning(
        "候选仅表示模型相对排序靠前，不构成买卖建议。实盘前需人工确认停牌、涨跌停和公告风险。"
    )

with tab_backtest:
    curve = equity.melt(
        id_vars="date", value_vars=["equity", "benchmark"], var_name="组合", value_name="净值"
    )
    curve["组合"] = curve["组合"].map({"equity": "模型组合", "benchmark": "股票池等权"})
    fig = px.line(curve, x="date", y="净值", color="组合", title="样本外净值曲线")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"区间 {summary['start_date']} 至 {summary['end_date']}；"
        f"每轮成本假设 {summary['transaction_cost_roundtrip']:.2%}。"
    )
    if health:
        st.subheader("数据健康")
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("股票数量", health["symbols"])
        h2.metric("成分股中位涨幅", f"{health['median_constituent_return']:.1%}")
        h3.metric("最大成分股涨幅", f"{health['best_constituent_return']:.1%}")
        h4.metric("快照日期", health.get("snapshot_date") or "未提供")
        if (
            health.get("membership_mode") == "point_in_time"
            and health.get("membership_symbol_coverage", 0) >= 0.9
        ):
            st.success(
                f"已启用 point-in-time 成分过滤："
                f"{health.get('membership_snapshots', 0)} 个变更快照，"
                f"时间覆盖 {health.get('membership_coverage', 0):.1%}，"
                f"成员行情覆盖 {health.get('membership_symbol_coverage', 0):.1%}。"
            )

with tab_race:
    comparison_path = settings.artifact_dir / "comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        display = comparison.copy()
        for column in [
            "total_return",
            "annual_return",
            "benchmark_return",
            "excess_return",
            "max_drawdown",
            "mean_rank_ic",
            "win_rate",
            "execution_rate",
        ]:
            if column in display:
                display[column] = display[column].map(lambda value: f"{value:.2%}")
        if "sharpe" in display:
            display["sharpe"] = display["sharpe"].map(lambda value: f"{value:.2f}")
        st.dataframe(display, width="stretch", hide_index=True)
        race = comparison.melt(
            id_vars="model",
            value_vars=["total_return", "benchmark_return", "excess_return"],
            var_name="metric",
            value_name="value",
        )
        st.plotly_chart(
            px.bar(
                race,
                x="model",
                y="value",
                color="metric",
                barmode="group",
                title="同条件样本外收益与相对基准收益",
            ),
            width="stretch",
        )
        st.plotly_chart(
            px.bar(
                comparison,
                x="model",
                y="mean_rank_ic",
                title="逐日横截面平均 Rank IC",
            ),
            width="stretch",
        )
        st.warning("赛马结果也可能被反复选择所过拟合，应另留未参与选模的最终测试期。")
    else:
        st.info("尚未生成模型比较。运行 stockpilot compare 后可在这里查看。")

with tab_validation:
    validation_dir = settings.artifact_dir / "validation_v2"
    report_path = validation_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        test = report["test_metrics"]
        if report["test_pass"]:
            st.success("最终测试通过；可以进入受控模拟盘观察，但仍不代表可实盘。")
        else:
            st.error("最终测试未通过：当前策略仅限研究，不应接入自动交易。")
        st.caption(report["note"])
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("验证期选中", report["selected"]["name"])
        v2.metric("测试期收益", f"{test['total_return']:.2%}")
        v3.metric("测试期基准", f"{test['benchmark_return']:.2%}")
        v4.metric("测试期超额", f"{report['test_excess_return']:.2%}")
        candidates = pd.read_csv(validation_dir / "validation_candidates.csv")
        st.subheader("仅使用验证期排序的候选")
        st.dataframe(candidates, width="stretch", hide_index=True)
        final_equity = pd.read_csv(validation_dir / "final_test" / "equity.csv")
        final_curve = final_equity.melt(
            id_vars="date",
            value_vars=["equity", "benchmark"],
            var_name="portfolio",
            value_name="value",
        )
        st.plotly_chart(
            px.line(
                final_curve,
                x="date",
                y="value",
                color="portfolio",
                title="锁定配置的最终测试期净值",
            ),
            width="stretch",
        )
    else:
        st.info("运行 stockpilot validate-v2 后可查看隔离的验证期与最终测试期结果。")
    future_manifest = settings.artifact_dir / "future_test" / "manifest.lock.json"
    future_market = settings.data_dir / "market_history.csv"
    if future_manifest.exists() and future_market.exists():
        future = future_test_status(
            future_manifest,
            future_market,
            settings.data_dir / "shadow" / "bars",
            settings.artifact_dir / "future_test" / "signals",
        )
        st.subheader("未来未触碰影子测试")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("已观察交易日", future["observed_trading_days"])
        f2.metric("最低要求", future["minimum_trading_days"])
        f3.metric("尚需交易日", future["remaining_trading_days"])
        f4.metric("信号快照", future["signal_snapshots"])
        evaluation_path = settings.artifact_dir / "future_test" / "evaluation.json"
        ledger_path = settings.artifact_dir / "future_test" / "ledger.csv"
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("已成熟周期", evaluation["matured_periods"])
            e2.metric("待成熟信号", evaluation["pending_signals"])
            e3.metric("影子累计收益", f"{evaluation['total_return']:.2%}")
            e4.metric("相对基准", f"{evaluation['excess_return']:.2%}")
            if ledger_path.exists():
                shadow_ledger = pd.read_csv(ledger_path)
                if not shadow_ledger.empty:
                    curve = shadow_ledger.melt(
                        id_vars="signal_date",
                        value_vars=["equity", "benchmark"],
                        var_name="portfolio",
                        value_name="value",
                    )
                    st.plotly_chart(
                        px.line(
                            curve,
                            x="signal_date",
                            y="value",
                            color="portfolio",
                            title="冻结信号的事前影子净值",
                        ),
                        width="stretch",
                    )
        if not future["frozen_inputs_intact"]:
            st.error("冻结输入完整性校验失败，影子测试必须暂停。")
        addendum_path = settings.artifact_dir / "future_test" / "protocol.addendum.lock.json"
        audit_path = settings.artifact_dir / "future_test" / "audit_chain.jsonl"
        if addendum_path.exists() and audit_path.exists():
            protocol_checks = verify_protocol_addendum(addendum_path, raise_on_error=False)
            audit = verify_audit_chain(audit_path, raise_on_error=False)
            if all(protocol_checks.values()) and audit["intact"]:
                st.success(f"完整协议锁与追加哈希链正常（{audit['records']} 条记录）")
            else:
                st.error("完整协议锁或追加哈希链校验失败，必须暂停影子测试。")
        adjudication_path = settings.artifact_dir / "future_test" / "adjudication_status.json"
        if adjudication_path.exists():
            adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
            phase_labels = {
                "collecting": "收集中",
                "awaiting_label_maturity": "等待末期标签成熟",
                "ready": "已完成裁决",
            }
            st.caption(
                "自动裁决阶段：" + phase_labels.get(adjudication["phase"], adjudication["phase"])
            )
        st.warning(
            f"从 {future['evaluation_start']} 起只收集影子信号；"
            "协议明确禁止自动下单，达到最低天数前不得判断通过。"
        )
        shadow_signal_dir = settings.artifact_dir / "future_test" / "signals"
        shadow_signals = sorted(shadow_signal_dir.glob("*.csv"))
        if shadow_signals:
            shadow_latest = pd.read_csv(shadow_signals[-1], dtype={"symbol": str})
            shadow_latest["symbol"] = shadow_latest["symbol"].str.zfill(6)
            st.caption(f"最新影子候选：{shadow_signals[-1].stem}（仅记录，不授权交易）")
            st.dataframe(shadow_latest, width="stretch", hide_index=True)

with tab_v4:
    v4_dir = settings.artifact_dir / "research_v4"
    v4_report_path = v4_dir / "report.json"
    if not v4_report_path.exists():
        st.info("尚未运行预注册Research V4。")
    else:
        v4_report = json.loads(v4_report_path.read_text(encoding="utf-8"))
        v4 = v4_report["metrics"]
        st.subheader("预注册稳定因子模型")
        if v4_report["passed"]:
            st.success("V4通过回顾性门槛，只能作为新一轮未来影子协议候选。")
        else:
            st.error("V4未通过全部门槛，保留研究结果但不能替换冻结模型。")
        metric_columns = st.columns(5)
        metric_columns[0].metric("策略收益", f"{v4['total_return']:.2%}", border=True)
        metric_columns[1].metric("同期基准", f"{v4['benchmark_return']:.2%}", border=True)
        metric_columns[2].metric("超额收益", f"{v4['excess_return']:.2%}", border=True)
        metric_columns[3].metric("平均Rank IC", f"{v4['mean_rank_ic']:.4f}", border=True)
        metric_columns[4].metric("最大回撤", f"{v4['max_drawdown']:.2%}", border=True)
        st.caption(
            f"规则锁：{v4_report['plan_lock_sha256'][:12]}…；"
            f"正超额年份比例 {v4['positive_test_year_ratio']:.1%}；"
            f"公告日违规 {v4_report['fundamental_pit_violations']} 条。"
        )
        annual = pd.read_csv(v4_dir / "annual_metrics.csv")
        st.subheader("逐年测试结果")
        st.dataframe(
            annual,
            column_config={
                "test_year": st.column_config.NumberColumn("测试年", format="%d"),
                "periods": st.column_config.NumberColumn("调仓期数", format="%d"),
                "total_return": st.column_config.NumberColumn("策略收益", format="percent"),
                "benchmark_return": st.column_config.NumberColumn("基准收益", format="percent"),
                "excess_return": st.column_config.NumberColumn("超额收益", format="percent"),
                "mean_rank_ic": st.column_config.NumberColumn("平均Rank IC", format="%.4f"),
                "max_drawdown": st.column_config.NumberColumn("最大回撤", format="percent"),
            },
            hide_index=True,
            width="stretch",
        )
        factor_specs = pd.read_csv(v4_dir / "factor_specs.csv")
        st.subheader("每年仅由历史训练期确定的因子规格")
        st.dataframe(
            factor_specs,
            column_config={
                "selected": st.column_config.CheckboxColumn("启用"),
                "weight": st.column_config.NumberColumn("权重", format="percent"),
                "mean_rank_ic": st.column_config.NumberColumn("训练期Rank IC", format="%.4f"),
                "direction_consistency": st.column_config.NumberColumn(
                    "方向一致率", format="percent"
                ),
            },
            hide_index=True,
            width="stretch",
        )
        st.warning(v4_report["warning"])

with tab_v3:
    v3_dir = settings.artifact_dir / "research_v3"
    v3_report_path = v3_dir / "report.json"
    if not v3_report_path.exists():
        st.info("尚未运行Research V3。")
    else:
        v3_report = json.loads(v3_report_path.read_text(encoding="utf-8"))
        v3 = v3_report["metrics"]
        st.subheader("多周期基本面集成研究")
        if v3_report["passed"]:
            st.success("V3通过回顾性嵌套验证，可作为下一轮未来协议候选。")
        else:
            st.error("V3未通过嵌套验证，不能替换当前冻结模型。")
        metric_columns = st.columns(5)
        metric_columns[0].metric("嵌套策略收益", f"{v3['total_return']:.2%}", border=True)
        metric_columns[1].metric("同期基准", f"{v3['benchmark_return']:.2%}", border=True)
        metric_columns[2].metric("超额收益", f"{v3['excess_return']:.2%}", border=True)
        metric_columns[3].metric("平均Rank IC", f"{v3['mean_rank_ic']:.4f}", border=True)
        metric_columns[4].metric("最大回撤", f"{v3['max_drawdown']:.2%}", border=True)
        st.caption(
            f"PIT基本面：{v3_report['fundamental_symbols']}只、"
            f"{v3_report['fundamental_rows']:,}条；"
            f"公告日违规 {v3_report['fundamental_pit_violations']} 条。"
        )
        candidate_metrics = pd.read_csv(v3_dir / "candidate_metrics.csv")
        st.subheader("候选模型完整历史对比")
        st.dataframe(
            candidate_metrics,
            column_config={
                "total_return": st.column_config.NumberColumn(format="percent"),
                "benchmark_return": st.column_config.NumberColumn(format="percent"),
                "excess_return": st.column_config.NumberColumn(format="percent"),
                "max_drawdown": st.column_config.NumberColumn(format="percent"),
                "average_cash_weight": st.column_config.NumberColumn(format="percent"),
            },
            hide_index=True,
            width="stretch",
        )
        folds = pd.read_csv(v3_dir / "nested_folds.csv")
        st.subheader("逐年嵌套选择结果")
        st.dataframe(
            folds,
            column_config={
                "test_return": st.column_config.NumberColumn(format="percent"),
                "test_benchmark_return": st.column_config.NumberColumn(format="percent"),
                "test_excess_return": st.column_config.NumberColumn(format="percent"),
            },
            hide_index=True,
            width="stretch",
        )
        st.warning(v3_report["warning"])

with tab_model:
    weights = pd.DataFrame(
        [{"feature": key, "weight": value} for key, value in summary["feature_weights"].items()]
    ).sort_values("weight")
    st.plotly_chart(
        px.bar(weights, x="weight", y="feature", orientation="h", title="最新模型标准化系数"),
        width="stretch",
    )
    st.caption("正系数代表特征值更高时模型倾向给出更高排名；系数不是因果解释。")

with tab_log:
    signals["symbol"] = signals["symbol"].str.zfill(6)
    st.dataframe(signals.sort_values(["date", "rank"], ascending=[False, True]), width="stretch")

st.divider()
st.caption(summary["warning"])
