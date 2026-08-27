from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .features import FEATURE_COLUMNS, build_dataset
from .health import assess_panel
from .model import RankingModel, create_model
from .portfolio import portfolio_weights, select_with_buffer_and_cap, turnover


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    signals: pd.DataFrame
    predictions: pd.DataFrame
    latest_signals: pd.DataFrame
    yearly: pd.DataFrame
    metrics: dict
    feature_weights: dict[str, float]

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.equity.to_csv(target / "equity.csv", index=False, encoding="utf-8-sig")
        self.signals.to_csv(target / "signals.csv", index=False, encoding="utf-8-sig")
        self.predictions.to_csv(target / "predictions.csv", index=False, encoding="utf-8-sig")
        self.latest_signals.to_csv(target / "latest_signals.csv", index=False, encoding="utf-8-sig")
        self.yearly.to_csv(target / "yearly.csv", index=False, encoding="utf-8-sig")
        payload = {**self.metrics, "feature_weights": self.feature_weights}
        (target / "summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def _safe_sharpe(returns: pd.Series, periods_per_year: float) -> float:
    std = returns.std(ddof=1)
    return float(returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0


def _rank_ic(group: pd.DataFrame) -> float:
    score_rank = group["score"].rank()
    label_rank = group["label"].rank()
    if score_rank.nunique() < 2 or label_rank.nunique() < 2:
        return float("nan")
    return float(np.corrcoef(score_rank, label_rank)[0, 1])


def _train_rows(
    dataset: pd.DataFrame, current_date: pd.Timestamp, settings: Settings
) -> pd.DataFrame:
    mature = dataset[
        dataset["eligible"] & dataset["label"].notna() & (dataset["label_end_date"] <= current_date)
    ]
    dates = mature["date"].drop_duplicates().sort_values()
    if len(dates) > settings.train_window_days:
        mature = mature[mature["date"] >= dates.iloc[-settings.train_window_days]]
    return mature


def _latest_predictions(
    dataset: pd.DataFrame,
    settings: Settings,
    previous_symbols: set[str] | None = None,
) -> tuple[pd.DataFrame, RankingModel]:
    eligible = dataset["eligible"]
    if settings.evaluation_end:
        eligible &= dataset["date"] <= pd.Timestamp(settings.evaluation_end)
    latest_date = dataset.loc[eligible, "date"].max()
    train = _train_rows(dataset, latest_date, settings)
    model = _fit_model(train, settings)
    current = dataset[(dataset["date"] == latest_date) & dataset["eligible"]].copy()
    current["score"] = model.predict(current[FEATURE_COLUMNS])
    current["pred_rank"] = current["score"].rank(ascending=False, method="first")
    current = select_with_buffer_and_cap(
        current,
        settings.top_n,
        previous_symbols,
        settings.hold_buffer,
        settings.industry_cap,
    )
    current["rank"] = np.arange(1, len(current) + 1)
    current["weight"] = portfolio_weights(current, settings.weighting)
    columns = [
        "date",
        "rank",
        "symbol",
        "name",
        "board",
        "close",
        "score",
        "weight",
        "limit_rate",
    ]
    return current[[c for c in columns if c in current.columns]], model


def _fit_model(train: pd.DataFrame, settings: Settings) -> RankingModel:
    ordered = train.sort_values(["date", "symbol"])
    groups = ordered.groupby("date", sort=False).size().to_numpy()
    return create_model(settings.model_name, settings.ridge_alpha).fit(
        ordered[FEATURE_COLUMNS], ordered["label"], group_sizes=groups
    )


def generate_latest_prediction_snapshot(
    panel: pd.DataFrame,
    settings: Settings | None = None,
    previous_symbols: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = settings or Settings()
    dataset = build_dataset(panel, settings.horizon, settings.label_mode)
    eligible = dataset["eligible"]
    if settings.evaluation_end:
        eligible &= dataset["date"] <= pd.Timestamp(settings.evaluation_end)
    latest_date = dataset.loc[eligible, "date"].max()
    train = _train_rows(dataset, latest_date, settings)
    model = _fit_model(train, settings)
    current = dataset[(dataset["date"] == latest_date) & dataset["eligible"]].copy()
    current["score"] = model.predict(current[FEATURE_COLUMNS])
    current["pred_rank"] = current["score"].rank(ascending=False, method="first")
    selected = select_with_buffer_and_cap(
        current,
        settings.top_n,
        previous_symbols,
        settings.hold_buffer,
        settings.industry_cap,
    )
    selected["rank"] = np.arange(1, len(selected) + 1)
    selected["weight"] = portfolio_weights(selected, settings.weighting)
    signal_columns = [
        "date",
        "rank",
        "symbol",
        "name",
        "board",
        "close",
        "score",
        "weight",
        "limit_rate",
    ]
    signals = selected[[column for column in signal_columns if column in selected.columns]]
    weights = selected.set_index("symbol")["weight"]
    prediction_columns = ["date", "symbol", "name", "score", "pred_rank", "eligible"]
    prediction = current[
        [column for column in prediction_columns if column in current.columns]
    ].copy()
    prediction["selected"] = prediction["symbol"].isin(set(selected["symbol"]))
    prediction["planned_weight"] = prediction["symbol"].map(weights).fillna(0.0)
    return signals, prediction.sort_values("pred_rank").reset_index(drop=True)


def generate_latest_signals(
    panel: pd.DataFrame,
    settings: Settings | None = None,
    previous_symbols: set[str] | None = None,
) -> pd.DataFrame:
    signals, _ = generate_latest_prediction_snapshot(panel, settings, previous_symbols)
    return signals


def run_walk_forward(panel: pd.DataFrame, settings: Settings | None = None) -> BacktestResult:
    settings = settings or Settings()
    dataset = build_dataset(panel, settings.horizon, settings.label_mode)
    complete = dataset[dataset["eligible"] & dataset["future_return"].notna()]
    all_dates = complete["date"].drop_duplicates().sort_values().reset_index(drop=True)
    if len(all_dates) <= settings.min_train_days + settings.horizon:
        raise ValueError("可用历史不足，无法完成走步回测")
    rebalance_dates = all_dates.iloc[settings.min_train_days :: settings.rebalance_every]
    if settings.evaluation_start:
        rebalance_dates = rebalance_dates[
            rebalance_dates >= pd.Timestamp(settings.evaluation_start)
        ]
    if settings.evaluation_end:
        rebalance_dates = rebalance_dates[rebalance_dates <= pd.Timestamp(settings.evaluation_end)]

    model: RankingModel | None = None
    last_train_position = -10_000
    equity_value = 1.0
    benchmark_value = 1.0
    equity_rows: list[dict] = []
    signal_rows: list[dict] = []
    prediction_rows: list[dict] = []
    previous_weights: dict[str, float] = {}

    date_positions = {date: i for i, date in enumerate(all_dates)}
    buy_cost_rate = settings.fee_rate + settings.slippage
    sell_cost_rate = settings.fee_rate + settings.slippage + settings.stamp_duty
    for date_number, date in enumerate(rebalance_dates):
        position = date_positions[date]
        if model is None or position - last_train_position >= settings.retrain_every:
            train = _train_rows(dataset, date, settings)
            model = _fit_model(train, settings)
            last_train_position = position

        current = complete[complete["date"] == date].copy()
        current["score"] = model.predict(current[FEATURE_COLUMNS])
        current["pred_rank"] = current["score"].rank(ascending=False, method="first")
        for row in current.itertuples():
            prediction_rows.append(
                {
                    "date": date,
                    "symbol": row.symbol,
                    "score": row.score,
                    "future_return": row.future_return,
                    "label": row.label,
                }
            )
        selected = select_with_buffer_and_cap(
            current,
            settings.top_n,
            set(previous_weights),
            settings.hold_buffer,
            settings.industry_cap,
        )
        if selected.empty:
            continue
        selected["planned_weight"] = portfolio_weights(selected, settings.weighting)
        continuing = selected["symbol"].isin(previous_weights)
        selected["holding_return"] = selected["execution_exit_open"] / selected["entry_open"] - 1
        selected["realized_return"] = selected["execution_return"].where(
            ~continuing, selected["holding_return"]
        )
        selected["executed"] = (continuing | selected["entry_tradable"]) & selected[
            "realized_return"
        ].notna()
        selected["weight"] = selected["planned_weight"].where(selected["executed"], 0.0)
        current_weights = {
            symbol: float(weight)
            for symbol, weight in zip(selected["symbol"], selected["weight"])
            if weight > 0
        }
        buy_turnover, sell_turnover = turnover(previous_weights, current_weights)
        if date_number == len(rebalance_dates) - 1:
            sell_turnover += sum(current_weights.values())
        gross_return = float((selected["weight"] * selected["realized_return"].fillna(0)).sum())
        transaction_cost = buy_turnover * buy_cost_rate + sell_turnover * sell_cost_rate
        net_return = gross_return - transaction_cost
        executed_weight = float(selected["weight"].sum())
        benchmark_return = float(current["future_return"].mean())
        equity_value *= 1 + net_return
        benchmark_value *= 1 + benchmark_return
        equity_rows.append(
            {
                "date": date,
                "equity": equity_value,
                "benchmark": benchmark_value,
                "period_return": net_return,
                "benchmark_return": benchmark_return,
                "holdings": int(selected["executed"].sum()),
                "cash_weight": 1 - executed_weight,
                "buy_turnover": buy_turnover,
                "sell_turnover": sell_turnover,
                "transaction_cost": transaction_cost,
            }
        )
        for rank, row in enumerate(selected.sort_values("score", ascending=False).itertuples(), 1):
            signal_rows.append(
                {
                    "date": date,
                    "rank": rank,
                    "symbol": row.symbol,
                    "name": getattr(row, "name", row.symbol),
                    "score": row.score,
                    "weight": row.weight,
                    "executed": bool(row.executed),
                    "entry_limit_up": bool(row.entry_limit_up),
                    "exit_deferred": bool(row.exit_deferred),
                    "future_return": row.future_return,
                    "execution_return": row.realized_return,
                }
            )
        previous_weights = current_weights

    equity = pd.DataFrame(equity_rows)
    signals = pd.DataFrame(signal_rows)
    predictions = pd.DataFrame(prediction_rows)
    if equity.empty:
        raise ValueError("回测没有产生有效交易")
    periods = 252 / settings.horizon
    n_years = len(equity) / periods
    annual_return = equity_value ** (1 / max(n_years, 1 / periods)) - 1
    date_ic = predictions.groupby("date", group_keys=False)[["score", "label"]].apply(_rank_ic)
    data_health = assess_panel(panel, equity["date"].min())
    metrics = {
        "start_date": str(equity["date"].min().date()),
        "end_date": str(equity["date"].max().date()),
        "periods": len(equity),
        "total_return": float(equity_value - 1),
        "annual_return": float(annual_return),
        "benchmark_return": float(benchmark_value - 1),
        "sharpe": _safe_sharpe(equity["period_return"], periods),
        "max_drawdown": _max_drawdown(equity["equity"]),
        "win_rate": float((equity["period_return"] > 0).mean()),
        "mean_rank_ic": float(date_ic.mean()),
        "transaction_cost_roundtrip": float(buy_cost_rate + sell_cost_rate),
        "model": settings.model_name,
        "horizon_days": settings.horizon,
        "top_n": settings.top_n,
        "label_mode": settings.label_mode,
        "weighting": settings.weighting,
        "hold_buffer": settings.hold_buffer,
        "industry_cap": settings.industry_cap,
        "average_one_way_turnover": float(
            (equity["buy_turnover"] + equity["sell_turnover"]).mean() / 2
        ),
        "signal_execution_rate": float(signals["executed"].mean()),
        "blocked_limit_up": int((signals["entry_limit_up"] & ~signals["executed"]).sum()),
        "deferred_exits": int(signals["exit_deferred"].sum()),
        "average_cash_weight": float(equity["cash_weight"].mean()),
        "data_health": data_health,
        "warning": "研究用途；结果来自历史样本外走步回测，不代表未来收益。",
    }
    yearly = (
        equity.assign(year=pd.to_datetime(equity["date"]).dt.year)
        .groupby("year")
        .agg(
            strategy_return=("period_return", lambda values: (1 + values).prod() - 1),
            benchmark_return=("benchmark_return", lambda values: (1 + values).prod() - 1),
            periods=("period_return", "size"),
        )
        .reset_index()
    )
    prediction_year = predictions.assign(year=pd.to_datetime(predictions["date"]).dt.year)
    yearly_ic = prediction_year.groupby("year", group_keys=False)[["score", "label"]].apply(
        _rank_ic
    )
    yearly = yearly.merge(yearly_ic.rename("rank_ic"), on="year", how="left")
    yearly["excess_return"] = yearly["strategy_return"] - yearly["benchmark_return"]
    latest, latest_model = _latest_predictions(dataset, settings)
    return BacktestResult(
        equity=equity,
        signals=signals,
        predictions=predictions,
        latest_signals=latest,
        yearly=yearly,
        metrics=metrics,
        feature_weights=latest_model.feature_weights(FEATURE_COLUMNS),
    )
