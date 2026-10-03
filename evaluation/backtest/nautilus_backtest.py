"""
nautilus_backtest.py

Out-of-sample replay in NautilusTrader, run AFTER the model is trained.

Division of labor:
    pandas / XGBoost / GP  -> all training and tuning, on in-sample days only
    Nautilus (this file)   -> replay the FROZEN model on the OOS window against the
                              recorded order book, then produce the report + graphs

Nothing here fits or tunes anything. The OOS window is run once.

The training script must use the same day split, so call the same splitter there:
    train_days, test_days = SampleSplitter(oos_fraction, embargo).split_day_list(all_days)

Outputs:
    oos_tearsheet.html   Nautilus interactive tearsheet (equity, drawdown, returns, stats)
    equity_curve.png     our gross-vs-net equity curve (metrics.py)
    printed stats table  Sharpe, vol, return, drawdown, turnover, net of fees

Install (Nautilus 2.x is a pre-release on PyPI):
    uv pip install --pre "nautilus_trader[visualization]"
"""

import os

import numpy as np
import pandas as pd

from nautilus_trader.backtest import BacktestNode
from nautilus_trader.config import (
    BacktestDataConfig,
    BacktestEngineConfig,
    BacktestRunConfig,
    BacktestVenueConfig,
)

from nautilus_trader.config import ImportableStrategyConfig, LoggerConfig
from nautilus_trader.model import InstrumentId, Venue
from nautilus_trader.common import LogLevel
from nautilus_trader.model import AccountType, BookType, Currency, OmsType

from metrics import CostModel, compare_is_oos, performance_stats, plot_equity_curve
from split_sample import SampleSplitter, add_trading_day, trading_day_window

CATALOG_DIR = os.environ.get("CATALOG_DIR", "catalog")
INSTRUMENT_ID = os.environ.get("INSTRUMENT_ID", "ESH4.GLBX")
VENUE_NAME = INSTRUMENT_ID.split(".")[-1]  # "GLBX"

# Swap these two lines when the XGBoost/GP strategy replaces the placeholder.
STRATEGY_PATH = "nautilus_strategy:ImbalanceStrategy"
STRATEGY_CONFIG_PATH = "nautilus_strategy:ImbalanceConfig"

# Nautilus simulates fills against recorded depth, so spread/depth slippage is already
# in the P&L. We only add fees afterwards, hence slippage_ticks=0.
COSTS = CostModel(fee_per_contract=2.0, slippage_ticks=0.0, capital=1_000_000.0)


# Config
def make_run_config(strategy_params: dict, start: str, end: str):
    iid = InstrumentId.from_str(INSTRUMENT_ID)

    strategy_config = ImportableStrategyConfig(
        strategy_path=STRATEGY_PATH,
        config_path=STRATEGY_CONFIG_PATH,
        config={"instrument_id": str(iid), **strategy_params},  # str(), as in the docs
    )

    run_config = BacktestRunConfig(
        engine=BacktestEngineConfig(
            logging=LoggerConfig(stdout_level=LogLevel.ERROR),
        ),
        data=[BacktestDataConfig(
                catalog_path=str(CATALOG_DIR),
                data_type="OrderBookDepth10",
                instrument_id=iid,
                start_time=start,
                end_time=end,
            )
        ],
        venues=[BacktestVenueConfig(
                name=VENUE_NAME,
                oms_type=OmsType.NETTING,
                account_type=AccountType.MARGIN,
                base_currency=Currency.from_str("USD"),
                starting_balances=[f"{int(COSTS.capital)} USD"],
                book_type=BookType.L2_MBP,  # depth10 updates an L2 book
            ) 
        ],
        dispose_on_completion=False,   # NEW: keep reports and cache alive after run()
    )
    return run_config, strategy_config


# Engine results -> daily frame expected by metrics.py
def _utc_index(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(idx, utc=True))


def daily_from_nautilus(engine, trading_days, costs: CostModel = COSTS) -> pd.DataFrame:
    """
    Daily frame from the engine's account + fills reports.

    VERIFY on your install: print(account.columns) and print(fills.columns). Assumes
    the account report has a numeric 'total' balance column, and the fills report has
    'filled_qty', 'avg_px' and a timestamp column ('ts_last').

    Balance is realized-only. The strategy closes everything at the end of the window,
    so window totals are fully realized, but intraday drawdowns are not visible here.
    (The Nautilus tearsheet has its own statistics.)
    """
    days = pd.DatetimeIndex(sorted(pd.Timestamp(d).normalize() for d in trading_days))

    acct = engine.trader.generate_account_report(Venue(VENUE_NAME))
    bal = pd.DataFrame(
        {"total": pd.to_numeric(acct["total"], errors="coerce").values},
        index=_utc_index(acct.index),
    ).dropna()
    bal_day = bal.groupby(add_trading_day(bal).values)["total"].last()
    bal_day = bal_day.reindex(days).ffill().fillna(costs.capital)
    gross_pnl = bal_day - bal_day.shift(1).fillna(costs.capital)

    fills = engine.trader.generate_order_fills_report()
    if fills is None or len(fills) == 0:
        qty = pd.Series(0.0, index=days)
        px = pd.Series(np.nan, index=days)
    else:
        ts_col = "ts_last" if "ts_last" in fills.columns else "ts_init"
        f = pd.DataFrame(
            {
                "qty": pd.to_numeric(fills["filled_qty"], errors="coerce").values,
                "px": pd.to_numeric(fills["avg_px"], errors="coerce").values,
            },
            index=_utc_index(fills[ts_col]),
        ).dropna()
        key = add_trading_day(f).values
        qty = f.groupby(key)["qty"].sum().reindex(days).fillna(0.0)
        px = f.groupby(key)["px"].mean().reindex(days)
    px = px.ffill().bfill()

    daily = pd.DataFrame(index=days)
    daily.index.name = "trading_day"
    daily["gross_pnl"] = gross_pnl.values
    daily["contracts_traded"] = qty.values
    daily["avg_price"] = px.values
    daily["avg_abs_position"] = np.nan
    daily["cost"] = daily["contracts_traded"] * costs.fee_per_contract
    daily["net_pnl"] = daily["gross_pnl"] - daily["cost"]
    daily["gross_ret"] = daily["gross_pnl"] / costs.capital
    daily["net_ret"] = daily["net_pnl"] / costs.capital
    daily["notional_traded"] = (
        daily["contracts_traded"] * daily["avg_price"].fillna(0.0) * costs.multiplier
    )
    return daily


# Tearsheet
def save_tearsheet(engine, path: str, title: str, theme: str = "nautilus_dark") -> None:
    """Interactive Plotly HTML report. Needs the 'visualization' extra."""
    try:
        from nautilus_trader.analysis import TearsheetConfig, create_tearsheet
    except ImportError:
        print('Tearsheet skipped. Install: uv pip install --pre "nautilus_trader[visualization]"')
        return
    create_tearsheet(
        engine=engine,
        output_path=path,
        title=title,
        config=TearsheetConfig(theme=theme),  # themes: plotly_white/dark, nautilus, nautilus_dark
    )
    print(f"Saved {path}")


# Workflow
def run_window(strategy_params: dict, days, label: str, start_offset: str = "0s"):
    start, end = trading_day_window(days, start_offset=start_offset)
    cfg, strategy_cfg = make_run_config(strategy_params, start, end)
    node = BacktestNode(configs=[cfg])
    node.build()
    node.add_strategy_from_config(cfg.id, strategy_cfg)
    [result] = node.run()
    daily = daily_from_nautilus(node, cfg.id, days)
    
    return node, result, daily


def run_oos(
    strategy_params: dict,
    first_day: str = "2024-01-02",
    last_day: str = "2024-01-31",
    oos_fraction: float = 0.3,
    embargo: str = "5min",
    also_run_is: bool = False,
):
    """
    Replay the frozen strategy on the OOS window once.

    also_run_is=True additionally replays the IS window (a single run, no tuning) so
    IS vs OOS are measured by the same engine. Without it, compare against the IS
    numbers from your training pipeline, but note those come from a different
    simulator (no depth slippage), so they are not like-for-like.
    """
    all_days = pd.bdate_range(first_day, last_day)
    train_days, test_days = SampleSplitter(oos_fraction, embargo).split_day_list(all_days)
    print(f"OOS: {test_days[0].date()} -> {test_days[-1].date()} "
          f"({len(test_days)}d), embargo {embargo}")

    engine, oos_daily = run_window(strategy_params, test_days, "oos", start_offset=embargo)
    save_tearsheet(engine, "oos_tearsheet.html", "Out-of-sample backtest")

    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    if also_run_is:
        _, is_daily = run_window(strategy_params, train_days, "is")
        table = compare_is_oos(is_daily, oos_daily, COSTS)
        plot_equity_curve(is_daily, oos_daily, COSTS, "equity_curve.png")
    else:
        table = pd.Series(performance_stats(oos_daily, COSTS), name="out_of_sample").to_frame()
        empty_is = oos_daily.iloc[0:0]
        plot_equity_curve(empty_is, oos_daily, COSTS, "equity_curve.png")
    print(table)
    print("Saved equity_curve.png")
    return table, oos_daily


if __name__ == "__main__":
    # Placeholder strategy params; replace with your trained model's settings.
    run_oos({"threshold": 0.8}, also_run_is=False)