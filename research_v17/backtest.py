from __future__ import annotations

import numpy as np
import pandas as pd

from .config import V17Settings


MODES = ("v16_ungated", "v17_timing")


def max_drawdown(returns):
    equity = pd.concat([pd.Series([1.0]), (1 + returns).cumprod()], ignore_index=True)
    return float((equity / equity.cummax() - 1).min())


def run_v17_backtest(
    equity_path: str,
    settings: V17Settings | None = None,
) -> pd.DataFrame:
    """在 V16 无门控线的逐期记账上叠加市场动量择时，产出两线对比。

    择时信号 = 过去 timing_window_periods 期的基准收益，仅用已完整走完的期，
    因此无前视。V16 equity 的 period_return 已是扣成本后的净值收益。
    """
    settings = settings or V17Settings()
    equity = pd.read_csv(equity_path).sort_values("date").reset_index(drop=True)
    equity["date"] = pd.to_datetime(equity["date"])

    bench_cum = (1.0 + equity["benchmark_return"]).cumprod()
    momentum = bench_cum / bench_cum.shift(settings.timing_window_periods) - 1.0
    in_market = momentum > settings.timing_threshold
    in_market = in_market.fillna(True)  # 历史不足时默认持仓

    frames = []
    for mode in MODES:
        frame = equity.copy()
        frame["mode"] = mode
        if mode == "v17_timing":
            frame["in_market"] = in_market.to_numpy()
            frame["timing_momentum"] = momentum.to_numpy()
            frame["period_return"] = np.where(
                in_market.to_numpy(), equity["period_return"].to_numpy(), 0.0
            )
        else:
            frame["in_market"] = True
            frame["timing_momentum"] = np.nan
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    result["excess_period_return"] = result["period_return"] - result["benchmark_return"]
    return result
