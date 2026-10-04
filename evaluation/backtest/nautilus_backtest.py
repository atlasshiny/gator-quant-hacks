"""
nautilus_backtest.py

Out-of-sample replay in NautilusTrader, run AFTER the model and GA are frozen.

Division of labor:
    pandas / XGBoost / GA  -> all training and tuning, on in-sample days only
    Nautilus (this file)   -> replay the FROZEN strategy (nautilus_strategy.py)
                              against the recorded order book, then report

Nothing here fits or tunes anything. The OOS window is run once.

Required before running (all produced from in-sample data only):
    data/ga_inputs/xgboost_final.json   3-class XGBoost (0=sell, 1=hold, 2=buy)
    data/ga_inputs/frozen_bundle.json   GA mask, tau, lookback, mean/std, names
                                        (see nautilus_strategy.build_bundle)

The training script must use the same day split, so call the same splitter there:
    train_days, test_days = SampleSplitter(oos_fraction, embargo).split_day_list(all_days)

Outputs:
    oos_tearsheet.html     Nautilus tearsheet (GROSS of our per-fill fee)
    equity_curve.png       our gross-vs-net equity curve (metrics.py)
    signal_log_oos.csv     every entry: time, side, GA score S, XGBoost score
    printed stats table    return, vol, Sharpe, drawdown, turnover, net of fees

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

from evaluation.backtest.metrics import CostModel, compare_is_oos, performance_stats, plot_equity_curve
from evaluation.backtest.split_sample import SampleSplitter, add_trading_day, trading_day_window

CATALOG_DIR = os.environ.get("CATALOG_DIR", "catalog")
INSTRUMENT_ID = os.environ.get("INSTRUMENT_ID", "ESH4.GLBX")
VENUE_NAME = INSTRUMENT_ID.split(".")[-1]  # "GLBX"

# Frozen artifacts from the in-sample pipeline.
BUNDLE_PATH = os.environ.get("BUNDLE_PATH", "data/ga_inputs/frozen_bundle.json")
MODEL_PATH = os.environ.get("MODEL_PATH", "data/ga_inputs/xgboost_final.json")

# Full dotted path: you run `python -m evaluation.backtest.nautilus_backtest` from the
# repo root, so a bare "nautilus_strategy:..." would not be importable by Nautilus.
STRATEGY_MODULE = os.environ.get("STRATEGY_MODULE", "evaluation.backtest.nautilus_strategy")
STRATEGY_PATH = f"{STRATEGY_MODULE}:XGBBookStateStrategy"
STRATEGY_CONFIG_PATH = f"{STRATEGY_MODULE}:XGBBookStateConfig"

# Assumed order-to-venue latency (ms). This is an ASSUMPTION: report results for
# several values (e.g. 0, 5, 20, 50). Set to 0 to disable the latency model.
LATENCY_MS = float(os.environ.get("LATENCY_MS", "5"))

# Nautilus simulates fills against recorded depth, so spread/depth slippage is already
# in the P&L. We only add fees afterwards, hence slippage_ticks=0.
COSTS = CostModel(fee_per_contract=2.0, slippage_ticks=0.0, capital=1_000_000.0)


def default_params() -> dict:
    """Strategy settings. These must be fixed from in-sample work, not tuned on OOS."""
    return {
        "bundle_path": BUNDLE_PATH,
        "model_path": MODEL_PATH,
        "trade_size": 1,
        "holding_ms": 1500,        # set from the in-sample duration of the label horizon
        "min_interval_ms": 500,
        "cooldown_ms": 500,
        "capital": COSTS.capital,   # risk manager equity base = venue starting balance
        # Set USE_RISK_MANAGER=0 to run the plain rule (ablation). Report both.
        "use_risk_manager": os.environ.get("USE_RISK_MANAGER", "1") == "1",
    }


# Config
def make_latency_config():
    """Fixed latency applied to every order submission. None disables it."""
    if LATENCY_MS <= 0:
        return None
    # In the 2.x API the venue config takes a model OBJECT (see Nautilus
    # examples/backtest/model_configs_example.py), not an Importable config.
    # Market orders are inserts. We put the whole delay in the base latency and leave
    # the per-operation extras at 0 (assuming they are added on top of the base;
    # check with a quick run at two latencies that results actually change).
    from nautilus_trader.execution import StaticLatencyModel

    ns = int(LATENCY_MS * 1_000_000)
    return StaticLatencyModel(
        base_latency_nanos=ns,
        insert_latency_nanos=0,
        update_latency_nanos=0,
        cancel_latency_nanos=0,
    )


def make_run_config(strategy_params: dict, start: str, end: str):
    iid = InstrumentId.from_str(INSTRUMENT_ID)

    strategy_config = ImportableStrategyConfig(
        strategy_path=STRATEGY_PATH,
        config_path=STRATEGY_CONFIG_PATH,
        config={"instrument_id": str(iid), **strategy_params},  # str(), as in the docs
    )

    venue_kwargs = dict(
        name=VENUE_NAME,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=Currency.from_str("USD"),
        starting_balances=[f"{int(COSTS.capital)} USD"],
        book_type=BookType.L2_MBP,  # depth10 updates an L2 book
    )
    latency = make_latency_config()
    if latency is not None:
        venue_kwargs["latency_model"] = latency

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
        venues=[BacktestVenueConfig(**venue_kwargs)],
        dispose_on_completion=False,   # keep reports and cache alive after run()
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

    Balance is realized-only. The strategy is flat at the end of each session (it
    flattens before the maintenance break), so daily totals are realized, but
    intraday drawdowns are not visible here. (The Nautilus tearsheet has its own
    statistics.)
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
def save_tearsheet(result, node, path: str, title: str, theme: str = "nautilus_dark") -> None:
    try:
        from nautilus_trader.analysis import create_tearsheet
        from nautilus_trader.config import TearsheetConfig   # moved: now in config
    except ImportError:
        print('Tearsheet skipped. Install: uv pip install --pre "nautilus_trader[visualization]"')
        return
    create_tearsheet(
        engine=result,          # the BacktestResult from node.run()
        node=node,              # needed for starting balances; requires dispose_on_completion=False
        output_path=path,
        title=title,
        config=TearsheetConfig(theme=theme),
    )
    print(f"Saved {path} (gross of our per-fill fee; use the printed net table for headline numbers)")


# Sanity check on what the strategy actually did
def report_trades(signal_log_path: str, label: str) -> None:
    if not os.path.exists(signal_log_path):
        print(f"[WARN] {label}: no signal log at {signal_log_path}; strategy may not have started.")
        return
    log = pd.read_csv(signal_log_path)
    if log.empty:
        print(f"[WARN] {label}: ZERO entries. Check warm-up, tau, and that the depth handler fires.")
        return
    n_long = int((log["side"] == "BUY").sum())
    n_short = int((log["side"] == "SELL").sum())
    print(f"{label}: {len(log):,} entries ({n_long:,} long / {n_short:,} short); "
          f"median |S| = {log['S'].abs().median():.2f}")


def report_position_stats(engine, label: str, contracts: int = 1) -> None:
    """
    Per-trade view of net P&L. On a short test window (a few days) a daily Sharpe
    is NaN or near-meaningless; on the full sample this complements it with a
    trade-level t-statistic for 'does the edge exist at all'.

    VERIFY on your install: print(engine.trader.generate_positions_report().columns).
    Assumes a 'realized_pnl' column, either numeric or like '12.50 USD'. If Nautilus
    instruments carry non-zero commissions, realized_pnl may already include fees and
    this would double-count them.
    """
    try:
        pos = engine.trader.generate_positions_report()
        if pos is None or len(pos) == 0:
            print(f"[WARN] {label}: no closed positions to summarize.")
            return
        pnl = pd.to_numeric(
            pos["realized_pnl"].astype(str).str.split().str[0], errors="coerce"
        ).dropna()
        net = pnl - 2 * COSTS.fee_per_contract * contracts   # entry fill + exit fill
        n = len(net)
        se = net.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        t = net.mean() / se if se and se > 0 else np.nan
        print(f"{label}: {n:,} trades | mean net ${net.mean():,.2f}/trade | "
              f"win rate {(net > 0).mean():.1%} | t-stat {t:.2f} | total net ${net.sum():,.2f}")
    except Exception as exc:  # reports differ across Nautilus versions
        print(f"[WARN] {label}: could not build per-trade stats ({exc})")


def report_risk(summary_path: str, label: str) -> None:
    """Print what the risk manager blocked and why each trade was exited."""
    import json

    if not os.path.exists(summary_path):
        print(f"[WARN] {label}: no risk summary at {summary_path}.")
        return
    s = json.load(open(summary_path))
    mode = "risk manager ON" if s.get("use_risk_manager") else "plain rule (risk manager OFF)"
    print(f"{label} [{mode}]: {s['entries_submitted']:,} entries, {s['trades_closed']:,} closed, "
          f"strategy net ${s['strategy_net_pnl_usd']:,.2f}")
    print(f"  blocked entries by reason: {s['blocks_by_reason'] or 'none'}")
    print(f"  exits by reason:           {s['exits_by_reason'] or 'none'}")


# Workflow
def run_window(strategy_params: dict, days, label: str, start_offset: str = "0s"):
    start, end = trading_day_window(days, start_offset=start_offset)
    params = {
        **strategy_params,
        "signal_log_path": f"signal_log_{label}.csv",
        "risk_summary_path": f"risk_summary_{label}.json",
    }
    cfg, strategy_cfg = make_run_config(params, start, end)
    node = BacktestNode(configs=[cfg])
    node.build()
    node.add_strategy_from_config(cfg.id, strategy_cfg)
    [result] = node.run()

    engine = node.get_engine(cfg.id)   # engine holds the account and fills reports
    daily = daily_from_nautilus(engine, days)
    report_trades(params["signal_log_path"], label)
    report_position_stats(engine, label, int(params.get("trade_size", 1)))
    report_risk(params["risk_summary_path"], label)

    return node, result, daily


def run_oos(
    strategy_params: dict | None = None,
    first_day: str = "2024-01-02",
    last_day: str = "2024-01-31",
    oos_fraction: float = 0.3,
    embargo: str = "5min",
    also_run_is: bool = True,
):
    """
    Replay the frozen strategy on the OOS window once.

    first_day / last_day must be days that actually exist in the catalog.

    also_run_is=True additionally replays the IS window (a single run, no tuning) so
    IS and OOS are measured by the same engine, with the same latency and fees.
    The IS number is optimistic by construction (the model and GA saw those days);
    it is reported only to show the IS/OOS gap.
    """
    strategy_params = strategy_params or default_params()
    all_days = pd.bdate_range(first_day, last_day)
    train_days, test_days = SampleSplitter(oos_fraction, embargo).split_day_list(all_days)
    print(f"IS : {train_days[0].date()} -> {train_days[-1].date()} ({len(train_days)}d)")
    print(f"OOS: {test_days[0].date()} -> {test_days[-1].date()} "
          f"({len(test_days)}d), embargo {embargo}, latency {LATENCY_MS} ms")

    node, oos_result, oos_daily = run_window(
        strategy_params, test_days, "oos", start_offset=embargo
    )
    save_tearsheet(oos_result, node, "oos_tearsheet.html", "Out-of-sample backtest")

    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    if also_run_is:
        _, _, is_daily = run_window(strategy_params, train_days, "is")
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
    # Settings come from default_params(); change them only from in-sample evidence.
    run_oos(also_run_is=True)