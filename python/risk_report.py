"""
Risk report for the ES burst-reversion backtest: the numbers behind the write-up's
"Risk Decomposition" section.

INPUT: one row per round-trip trade (CSV or Parquet), e.g. logged by the Nautilus strategy:
    entry_ts, exit_ts          timestamps (UTC)
    side                       +1 long / -1 short
    qty                        contracts
    entry_fill, exit_fill      average fill prices
    entry_bid, entry_ask       best bid / ask when the ENTRY was decided
    exit_bid, exit_ask         best bid / ask when the EXIT was decided
  optional numeric tags to group on, e.g. burst_size, vol_ratio

The strategy already has the decision-time book when it calls the risk manager, so logging
these columns is one dict per round trip, written to CSV in on_stop().

OUTPUT
  1. P&L decomposition (ticks per contract): gross edge (mid-to-mid move) - spread - slippage
     beyond the touch - fees = net. Checked against fills to the cent.
  2. Edge by group: burst-size quintile (the paper's testable prediction), hour (CT),
     spread at entry, volatility regime.
  3. Tail risk: 97.5% Expected Shortfall of per-trade and daily P&L, worst trades.
  4. Cost stress: net P&L with costs x1.5 / x2 / x3 and extra latency; breakeven cost.
     (Approximation from the logged trades. The exact version re-runs NautilusTrader on a
     book with doubled spread and halved depth.)
  5. Sharpe significance: Probabilistic Sharpe Ratio for the single out-of-sample run, and
     Deflated Sharpe Ratio for the in-sample selection given the number of GA trials
     (Bailey & Lopez de Prado, 2014).

USAGE (from the repo root)
    python -m python.risk_report --trades oos_trades.csv
    python -m python.risk_report --trades is_trades.csv --n-trials 400000   # in-sample selection
    python -m python.risk_report                                             # synthetic demo
"""
from __future__ import annotations

import argparse
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

TICK, TICK_VALUE, FEE_PER_SIDE = 0.25, 12.50, 2.00   # ES; fees match evaluation/metrics.py
TD = 252
EULER_GAMMA = 0.5772156649
N01 = NormalDist()


# ============================================================================ 1. decomposition
def decompose(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    s = t["side"]
    mid_in = (t["entry_bid"] + t["entry_ask"]) / 2
    mid_out = (t["exit_bid"] + t["exit_ask"]) / 2
    touch_in = np.where(s > 0, t["entry_ask"], t["entry_bid"])     # price we cross to get in
    touch_out = np.where(s > 0, t["exit_bid"], t["exit_ask"])      # price we cross to get out
    t["gross_ticks"] = s * (mid_out - mid_in) / TICK               # what the signal earned
    t["spread_ticks"] = (s * (touch_in - mid_in) + s * (mid_out - touch_out)) / TICK
    t["slippage_ticks"] = (s * (t["entry_fill"] - touch_in) + s * (touch_out - t["exit_fill"])) / TICK
    t["fee_ticks"] = 2 * FEE_PER_SIDE / TICK_VALUE
    t["cost_ticks"] = t["spread_ticks"] + t["slippage_ticks"] + t["fee_ticks"]
    t["net_ticks"] = t["gross_ticks"] - t["cost_ticks"]
    t["net_usd"] = t["net_ticks"] * TICK_VALUE * t["qty"]
    check = s * (t["exit_fill"] - t["entry_fill"]) / TICK - t["fee_ticks"]
    if not np.allclose(check, t["net_ticks"], atol=1e-9):
        raise ValueError("decomposition does not add back to fill-to-fill P&L")
    t["entry_spread_ticks"] = (t["entry_ask"] - t["entry_bid"]) / TICK
    entry_ct = pd.to_datetime(t["entry_ts"], utc=True).dt.tz_convert("America/Chicago")
    t["hour_ct"] = entry_ct.dt.hour
    exit_ct = pd.to_datetime(t["exit_ts"], utc=True).dt.tz_convert("America/Chicago")
    t["trading_day"] = (exit_ct + pd.Timedelta(hours=7)).dt.date      # CME day starts 17:00 CT
    return t


def decomposition_table(t: pd.DataFrame) -> pd.DataFrame:
    w = t["qty"]
    rows = {c: (t[c] * w).sum() / w.sum() for c in
            ["gross_ticks", "spread_ticks", "slippage_ticks", "fee_ticks", "net_ticks"]}
    out = pd.DataFrame({"ticks_per_contract": rows})
    out["usd_total"] = out["ticks_per_contract"] * TICK_VALUE * w.sum()
    out.loc[["spread_ticks", "slippage_ticks", "fee_ticks"], "usd_total"] *= -1
    return out


# ============================================================================ 2. edge by group
def edge_by(t: pd.DataFrame, col: str, quintiles: bool = False) -> pd.DataFrame:
    key = pd.qcut(t[col], 5, labels=[f"Q{i}" for i in range(1, 6)], duplicates="drop") if quintiles else t[col]
    g = t.groupby(key, observed=True)
    out = pd.DataFrame({
        "trades": g.size(),
        "gross_ticks": g["gross_ticks"].mean(),
        "net_ticks": g["net_ticks"].mean(),
        "net_t_stat": g["net_ticks"].mean() / g["net_ticks"].std() * np.sqrt(g.size()),
        "hit_rate": g["net_ticks"].apply(lambda x: (x > 0).mean()),
    })
    return out


# ============================================================================ 3. tail risk
def expected_shortfall(x: pd.Series, alpha: float = 0.025) -> float:
    q = x.quantile(alpha)
    return float(-x[x <= q].mean())


def daily_pnl(t: pd.DataFrame) -> pd.Series:
    return t.groupby("trading_day")["net_usd"].sum()


# ============================================================================ 4. cost stress
def cost_stress(t: pd.DataFrame) -> pd.DataFrame:
    w = t["qty"]
    gross = (t["gross_ticks"] * w).sum() / w.sum()
    cost = (t["cost_ticks"] * w).sum() / w.sum()
    rows = {}
    for k in (1.0, 1.5, 2.0, 3.0):
        net = gross - k * cost
        rows[f"costs x{k:g}"] = (net, net * TICK_VALUE * w.sum())
    for lat in (0.25, 0.5, 1.0):                                    # extra adverse ticks per side
        net = gross - cost - 2 * lat
        rows[f"+{lat:g} tick latency/side"] = (net, net * TICK_VALUE * w.sum())
    out = pd.DataFrame(rows, index=["net_ticks_per_contract", "net_usd_total"]).T
    out.attrs["breakeven_cost_multiple"] = gross / cost if cost > 0 else float("inf")
    out.attrs["breakeven_cost_ticks"] = gross
    return out


# ============================================================================ 5. Sharpe significance
def _sr_moments(r: pd.Series):
    sr = r.mean() / r.std(ddof=1)
    return sr, float(r.skew()), float(r.kurt()) + 3.0, len(r)     # kurt(): excess -> raw


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark) given the sample's length, skew and kurtosis (per-period SR)."""
    sr, g3, g4, n = _sr_moments(r)
    denom = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr * sr, 1e-12))
    return N01.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom)


def deflated_sharpe(r: pd.Series, n_trials: int, var_trial_sr: float | None = None) -> dict:
    """
    Deflated Sharpe Ratio: the PSR against the Sharpe you would expect from the BEST of
    n_trials strategies with no skill. var_trial_sr = variance of the trials' per-period
    Sharpes; if unknown we use 1/T, the sampling variance of a Sharpe estimate whose true
    value is zero. GA trials are highly correlated, so the raw trial count overstates the
    effective number of independent tests: the result is conservative.
    """
    sr, _, _, n = _sr_moments(r)
    v = var_trial_sr if var_trial_sr is not None else 1.0 / n
    z = (1 - EULER_GAMMA) * N01.inv_cdf(1 - 1 / n_trials) + EULER_GAMMA * N01.inv_cdf(1 - 1 / (n_trials * math.e))
    sr0 = math.sqrt(v) * z
    return dict(sharpe_annual=sr * math.sqrt(TD), hurdle_annual=sr0 * math.sqrt(TD),
                dsr=probabilistic_sharpe(r, sr0), n_trials=n_trials, days=n)


# ============================================================================ report
def report(trades: pd.DataFrame, capital: float = 1_000_000.0, n_trials: int | None = None) -> str:
    t = decompose(trades)
    d = daily_pnl(t)
    r = d / capital
    L = [f"# Risk report: {len(t):,} trades, {t['qty'].sum():,} contracts, {len(d)} trading days", ""]

    L += ["## 1. P&L decomposition (ticks per contract)", decomposition_table(t).round(3).to_string(), ""]

    L += ["## 2. Edge by group"]
    if "burst_size" in t:
        g = edge_by(t, "burst_size", quintiles=True)
        rising = bool(np.all(np.diff(g["gross_ticks"].to_numpy()) > 0))
        L += ["Burst-size quintile (hypothesis: gross edge rises with burst size -> "
              f"{'HOLDS' if rising else 'NOT monotonic'}):", g.round(3).to_string(), ""]
    L += ["Hour of entry (CT):", edge_by(t, "hour_ct").round(3).to_string(), ""]
    L += ["Spread at entry (ticks):", edge_by(t, "entry_spread_ticks").round(3).to_string(), ""]
    if "vol_ratio" in t:
        t["vol_regime"] = pd.cut(t["vol_ratio"], [0, 1.5, 2.5, np.inf], labels=["normal", "elevated", "high"])
        L += ["Volatility regime at entry:", edge_by(t, "vol_regime").round(3).to_string(), ""]

    L += ["## 3. Tail risk",
          f"Per-trade ES 97.5%: ${expected_shortfall(t['net_usd']):,.0f}   worst trade ${t['net_usd'].min():,.0f}",
          f"Daily ES 97.5%:     ${expected_shortfall(d):,.0f} ({expected_shortfall(r):.2%} of capital)   "
          f"worst day ${d.min():,.0f}",
          "Worst 5 trades:",
          t.nsmallest(5, "net_usd")[["entry_ts", "side", "qty", "gross_ticks", "cost_ticks", "net_usd"]]
          .assign(entry_ts=lambda x: pd.to_datetime(x["entry_ts"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S"))
          .round(2).to_string(index=False), ""]

    cs = cost_stress(t)
    L += ["## 4. Cost stress (approximation from logged trades)", cs.round(3).to_string(),
          f"Breakeven: costs can rise {cs.attrs['breakeven_cost_multiple']:.2f}x "
          f"(to {cs.attrs['breakeven_cost_ticks']:.2f} ticks per round trip) before net P&L reaches zero.", ""]

    ann = r.mean() / r.std(ddof=1) * math.sqrt(TD) if r.std(ddof=1) > 0 else float("nan")
    L += ["## 5. Sharpe significance",
          f"Daily Sharpe (annualized): {ann:.2f} over {len(r)} days",
          f"Probabilistic Sharpe (P[true Sharpe > 0]), single run: {probabilistic_sharpe(r):.3f}"]
    if n_trials:
        ds = deflated_sharpe(r, n_trials)
        L += [f"Deflated Sharpe with {n_trials:,} trials: hurdle {ds['hurdle_annual']:.2f} annualized, "
              f"DSR = {ds['dsr']:.3f}  (use on the IN-SAMPLE selection; >= 0.95 is the usual bar)"]
    return "\n".join(L)


# ============================================================================ demo
def synthetic_trades(n: int = 3_000, days: int = 30, seed: int = 0) -> pd.DataFrame:
    """Fake trades where reversion (gross edge) grows with burst size, as the paper predicts."""
    rng = np.random.default_rng(seed)
    session_days = pd.bdate_range("2024-01-08", periods=days)          # weekdays only
    day = pd.DatetimeIndex(session_days[rng.integers(0, days, n)]).tz_localize("America/Chicago")
    entry = day + pd.Timedelta(hours=8, minutes=35) + pd.to_timedelta(rng.uniform(0, 6.3 * 3600, n), "s")
    burst = rng.lognormal(3, 0.6, n)
    side = rng.choice([-1, 1], n)
    spread = np.where(rng.random(n) < 0.97, 1, 2) * TICK
    mid_in = 5000 + rng.normal(0, 20, n)
    move = (1.2 + 2.0 * (np.argsort(np.argsort(burst)) / n)) + rng.normal(0, 3.0, n)   # ticks, our favour
    mid_out = mid_in + side * move * TICK
    t = pd.DataFrame({
        "entry_ts": entry.tz_convert("UTC"), "exit_ts": (entry + pd.Timedelta(seconds=1.5)).tz_convert("UTC"),
        "side": side, "qty": rng.choice([1, 2], n),
        "entry_bid": mid_in - spread / 2, "entry_ask": mid_in + spread / 2,
        "exit_bid": mid_out - TICK / 2, "exit_ask": mid_out + TICK / 2,
        "burst_size": burst, "vol_ratio": rng.lognormal(0, 0.3, n),
    })
    slip_in = rng.choice([0, 0, 0, 1], n) * TICK
    slip_out = rng.choice([0, 0, 0, 0, 1], n) * TICK
    t["entry_fill"] = np.where(side > 0, t["entry_ask"] + slip_in, t["entry_bid"] - slip_in)
    t["exit_fill"] = np.where(side > 0, t["exit_bid"] - slip_out, t["exit_ask"] + slip_out)
    return t.sort_values("entry_ts").reset_index(drop=True)


def load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trades", help="CSV/Parquet of round-trip trades (omit for a synthetic demo)")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--n-trials", type=int, default=None,
                    help="GA candidates evaluated (population x generations) for the Deflated Sharpe")
    a = ap.parse_args()
    trades = load(a.trades) if a.trades else synthetic_trades()
    pd.set_option("display.width", 140)
    if not a.trades:
        print("(SYNTHETIC DEMO DATA: numbers illustrate the report, not the strategy)\n")
    print(report(trades, a.capital, a.n_trials if a.trades else (a.n_trials or 400_000)))
