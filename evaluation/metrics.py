"""
metrics.py

Performance metrics, transaction-cost-aware backtesting, and equity curves,
designed to plug into sample_split.py.

Pipeline:
    position series (target contracts) + price
        -> backtest()          per-bar gross P&L, costs, net P&L, trades
        -> daily_frame()       aggregate to CME trading days
        -> performance_stats() Sharpe, vol, return, drawdown, turnover
        -> compare_is_oos()    side-by-side IS vs OOS table
        -> plot_equity_curve() equity curve with OOS shaded

Conventions
  * Position decided at bar t is held from t to t+1 (we shift by one bar), so
    there is no look-ahead.
  * Fixed capital (no compounding): equity = capital + cumulative net P&L.
  * Risk-free rate = 0; 252 trading days per year.
  * Defaults are for CME E-mini S&P 500 (ES): $50/point, 0.25 tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from evaluation.split_sample import Split, add_trading_day, SampleSplitter

TRADING_DAYS = 252


# Config
@dataclass
class CostModel:
    multiplier: float = 50.0          # $ per index point (ES)
    tick_size: float = 0.25
    fee_per_contract: float = 2.0     # commission + exchange fees, $ per contract per side
    slippage_ticks: float = 0.5       # assumed slippage per contract per side, in ticks
    capital: float = 1_000_000.0      # fixed capital base for return calculations

    @property
    def cost_per_contract(self) -> float:
        return self.fee_per_contract + self.slippage_ticks * self.tick_size * self.multiplier


# Backtest
def mid_price(df: pd.DataFrame) -> pd.Series:
    """Top-of-book mid from an mbp-N DataFrame."""
    return ((df["bid_px_00"] + df["ask_px_00"]) / 2).dropna()


def backtest(price: pd.Series, position: pd.Series, costs: CostModel) -> pd.DataFrame:
    """
    price    : price per bar (e.g. mid_price(df))
    position : target position in contracts per bar (same index; can be +/-/0)
    Returns a per-bar frame: position, trades, gross_pnl, cost, net_pnl.
    """
    pos = position.reindex(price.index).fillna(0.0)
    held = pos.shift(1).fillna(0.0)                       # position held over (t-1 -> t)
    gross = held * price.diff().fillna(0.0) * costs.multiplier
    trades = held.diff().abs().fillna(held.abs())         # contracts traded at each bar
    cost = trades * costs.cost_per_contract
    return pd.DataFrame(
        {"price": price, "position": held, "trades": trades,
         "gross_pnl": gross, "cost": cost, "net_pnl": gross - cost}
    )


def daily_frame(bt: pd.DataFrame, costs: CostModel) -> pd.DataFrame:
    """Aggregate per-bar results to CME trading days."""
    day = add_trading_day(bt)
    g = bt.groupby(day.values)
    out = pd.DataFrame(
        {
            "gross_pnl": g["gross_pnl"].sum(),
            "cost": g["cost"].sum(),
            "net_pnl": g["net_pnl"].sum(),
            "contracts_traded": g["trades"].sum(),
            "avg_abs_position": g["position"].apply(lambda s: s.abs().mean()),
            "avg_price": g["price"].mean(),
        }
    )
    out.index.name = "trading_day"
    out["gross_ret"] = out["gross_pnl"] / costs.capital
    out["net_ret"] = out["net_pnl"] / costs.capital
    out["notional_traded"] = out["contracts_traded"] * out["avg_price"] * costs.multiplier
    return out


# Metrics - - - - - - - - - - - - - - - - - -
def equity_curve(daily: pd.DataFrame, costs: CostModel, col: str = "net_pnl") -> pd.Series:
    return costs.capital + daily[col].cumsum()


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline as a (negative) fraction of the peak."""
    if equity.empty:
        return np.nan
    return float((equity / equity.cummax() - 1.0).min())


def performance_stats(daily: pd.DataFrame, costs: CostModel) -> dict[str, float]:
    """Core metrics on daily returns, both gross and net of costs."""
    if len(daily) == 0:
        raise ValueError("No daily data to evaluate.")
    n = len(daily)
    stats: dict[str, float] = {"n_days": n}

    for label, col, pnl in (("gross", "gross_ret", "gross_pnl"), ("net", "net_ret", "net_pnl")):
        r = daily[col]
        vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS) if n > 1 else np.nan
        ann_ret = r.mean() * TRADING_DAYS
        stats[f"{label}_ann_return"] = ann_ret
        stats[f"{label}_volatility"] = vol
        stats[f"{label}_sharpe"] = ann_ret / vol if vol and vol > 0 else np.nan
        stats[f"{label}_max_drawdown"] = max_drawdown(equity_curve(daily, costs, pnl))

    stats["total_net_pnl"] = daily["net_pnl"].sum()
    stats["total_costs"] = daily["cost"].sum()
    # Turnover
    stats["avg_daily_contracts_traded"] = daily["contracts_traded"].mean()
    stats["annual_turnover_x_capital"] = (
        daily["notional_traded"].mean() * TRADING_DAYS / costs.capital
    )
    return stats


def compare_is_oos(
    is_daily: pd.DataFrame, oos_daily: pd.DataFrame, costs: CostModel
) -> pd.DataFrame:
    """Side-by-side IS vs OOS table with an OOS/IS Sharpe ratio row."""
    table = pd.DataFrame(
        {
            "in_sample": performance_stats(is_daily, costs),
            "out_of_sample": performance_stats(oos_daily, costs),
        }
    )
    is_s, oos_s = table.loc["net_sharpe"]
    table.loc["oos_over_is_net_sharpe"] = [np.nan, oos_s / is_s if is_s else np.nan]
    return table


# Plot quity curve
def plot_equity_curve(
    is_daily: pd.DataFrame,
    oos_daily: pd.DataFrame,
    costs: CostModel,
    path: str | None = "equity_curve.png",
    title: str = "Equity curve: in-sample vs out-of-sample",
):
    """Gross and net equity curves; OOS period shaded. Saves PNG if `path` given."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    both = pd.concat([is_daily, oos_daily])
    gross = costs.capital + both["gross_pnl"].cumsum()
    net = costs.capital + both["net_pnl"].cumsum()
    x = pd.to_datetime(both.index)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, gross.values, label="Gross", color="#9aa0a6", linestyle="--")
    ax.plot(x, net.values, label="Net of costs", color="#1a73e8", linewidth=2)
    if len(oos_daily):
        ax.axvspan(pd.to_datetime(oos_daily.index[0]), x[-1], color="orange", alpha=0.12,
                   label="Out-of-sample")
    ax.set_title(title)
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


# Glue to sample_split
FitFn = Callable[[pd.DataFrame], Any]
PositionFn = Callable[[pd.DataFrame, Any], pd.Series]  # (data, params) -> target position


def run_split(
    split: Split,
    fit_fn: FitFn,
    position_fn: PositionFn,
    costs: CostModel | None = None,
    plot_path: str | None = "equity_curve.png",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], Any]:
    """
    Fit on IS only, generate positions on IS and OOS with frozen params,
    backtest both with costs, and return (comparison table, daily frames, params).
    """
    costs = costs or CostModel()
    params = fit_fn(split.train)

    daily = {}
    for name, data in (("is", split.train), ("oos", split.test)):
        px = mid_price(data)
        pos = position_fn(data, params).reindex(px.index)
        daily[name] = daily_frame(backtest(px, pos, costs), costs)

    table = compare_is_oos(daily["is"], daily["oos"], costs)
    if plot_path:
        plot_equity_curve(daily["is"], daily["oos"], costs, plot_path)
    return table, daily, params



# Example
if __name__ == "__main__":
    from local_data import load_local, TOP_OF_BOOK
     
    # Reads .dbn.zst files from $DATA_DIR (HiPerGator). 
    df = load_local(
        start="2024-01-08T00:00:00Z",
        end="2024-01-13T00:00:00Z",   # end is exclusive
        columns=TOP_OF_BOOK,
    )
    
    def imbalance(d):
        bid, ask = d["bid_sz_00"], d["ask_sz_00"]
        return (bid - ask) / (bid + ask).replace(0, np.nan)

    def fit_fn(train):                      # threshold chosen on IS only
        return float(imbalance(train).abs().quantile(0.9))

    def position_fn(d, thr):                # +1 / -1 contract when imbalance is extreme
        imb = imbalance(d)
        return np.sign(imb).where(imb.abs() > thr, 0.0)

    split = SampleSplitter(oos_fraction=0.4, embargo="5min").split(df)
    print(split.describe())

    table, daily, params = run_split(split, fit_fn, position_fn, CostModel())
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print(table)
    print("\nSaved equity_curve.png")