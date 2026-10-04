"""
Risk management for the GA + XGBoost alpha strategy.

Two layers:
  BaseRiskManager   general best practice that any systematic strategy should run
  GAXGBRiskManager  same machinery with tighter, simulation-chosen parameters, plus
                    checks that only make sense for an adaptive ML signal

Daily cycle
  rm.new_day(nav)                    once at the open: drawdown ladder, cooldown, reset daily halt
  report = rm.step(signals, state)   whenever we rebalance: returns safe target weights + report
  rm.check_trade_exit(trade, ...)    per open position, every bar: volatility stop and time stop

Inside step(), in order
  1. De-risking checks   -> one "risk scale" in [0, 1]   (1 = normal, 0.5 = half size, 0 = flat)
  2. Portfolio build     -> signal / volatility sizing, factor neutralization, volatility target,
                            per-asset, liquidity, group, gross and net limits
  3. Post-build limits   -> forward Expected Shortfall and stress-test losses must fit their limits
  4. Report              -> risk decomposition, exposures, stress results, any limit breaches

Parameters marked [sim] were chosen by risk_sim/regime_sim.py (8 regimes, 11,520 settings)
and risk_sim/position_stops.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

TD = 252  # trading days per year


# ============================================================================ #
# Parameters
# ============================================================================ #
@dataclass
class RiskParams:
    """General best-practice defaults for a market-neutral systematic strategy."""
    # --- Volatility targeting ---
    vol_target: float = 0.10            # aim for 10% annual portfolio volatility
    max_leverage: float = 4.0           # cap on vol-target scaling so calm markets can't over-lever
    cov_halflife: int = 60              # days; weight of recent data in the covariance estimate
    cov_shrink: float = 0.30            # blend 30% toward a diagonal matrix to reduce estimation noise

    # --- Position and exposure limits (fractions of NAV) ---
    max_weight: float = 0.05            # any single asset
    adv_cap: float = 0.05               # position $ <= 5% of the asset's average daily $ volume
    max_gross: float = 2.0              # sum of |weights|
    max_net: float = 0.10               # sum of weights
    max_group_share: float = 0.30       # any sector / correlation cluster <= 30% of gross
    max_beta: float = 0.20              # portfolio market beta
    max_factor_exposure: float = 0.25   # |exposure| to any style factor (z-score x NAV)
    max_risk_share: float = 0.10        # flag any asset contributing > 10% of portfolio risk

    # --- Volatility regime (market vol: recent vs long-run) ---
    vol_short_window: int = 20
    vol_long_window: int = 252
    vol_reduce_ratio: float = 2.0
    vol_flat_ratio: float = 3.0
    vol_reduce_scale: float = 0.5

    # --- Drawdown ladder ---
    dd_reduce: float = -0.10
    dd_flat: float = -0.20
    dd_reduce_scale: float = 0.5
    flat_cooldown_days: int = 10        # after going flat: wait, then reset the peak and restart

    # --- Daily loss limit ---
    daily_loss_halt: float = -0.03

    # --- Tail risk ---
    es_alpha: float = 0.025             # Expected Shortfall = average of the worst 2.5% of days
    es_window: int = 504                # 2 years of history to re-price today's portfolio
    es_limit: float = 0.04              # forward ES must be <= 4% of NAV
    stress_loss_limit: float = 0.10     # worst stress scenario must lose <= 10% of NAV

    # --- Operational ---
    max_data_age_sec: float = 300


@dataclass
class GAXGBRiskParams(RiskParams):
    """Tighter limits for a short-horizon, high-turnover ML strategy + strategy-specific knobs."""
    # --- Overrides ---
    max_weight: float = 0.02
    adv_cap: float = 0.01               # minute-level turnover: stay tiny vs. volume
    max_gross: float = 1.5
    max_group_share: float = 0.20
    max_beta: float = 0.10
    max_factor_exposure: float = 0.20
    vol_reduce_ratio: float = 1.5       # [sim]
    vol_flat_ratio: float = 2.5         # [sim]
    dd_reduce: float = -0.05            # [sim]
    dd_flat: float = -0.15              # [sim] -8% / -10% sold the bottom too often
    daily_loss_halt: float = -0.015     # [sim] -1% stopped on noise; -3% was too late
    es_limit: float = 0.03
    stress_loss_limit: float = 0.08
    max_data_age_sec: float = 60        # minute bars: one missed bar is already a problem

    # --- Strategy-specific: model health ---
    ic_window: int = 20                 # days of measured IC to average
    ic_floor: float = -0.01             # [sim] act only when skill is clearly negative
    ic_cut_scale: float = 0.5
    jaccard_trigger: float = 0.0        # [sim] 0 = monitor only; cutting on feature turnover didn't pay
    mask_probation: int = 3
    mask_cut_scale: float = 0.5
    min_signal: float = 0.02            # drop signals too weak to beat 2-5 bps costs
    pred_range_buffer: float = 0.10     # prediction >10% outside training range -> kill switch

    # --- Strategy-specific: position-level stops (execution layer) ---
    horizon_bars: int = 60              # model predicts ~60 one-minute bars ahead
    vol_stop_k: float = 3.0             # [sim] exit if loss > 3x expected move over the horizon
    time_stop_mult: float = 1.5         # [sim] hold at most 1.5x horizon (90 bars)
    time_stop_mult_high_vol: float = 1.0  # [sim] 1x horizon when the vol regime is elevated


# Illustrative stress scenarios. Factor shocks are one-period factor returns; "sigma" adds a
# move of that many (vol_mult x current) portfolio daily sigmas against us for losses the
# factors miss (crowding, liquidity). Calibrate against real factor data before relying on them.
STRESS_SCENARIOS = {
    "crash_day_2020":     dict(factors=dict(mkt=-0.12, size=-0.02, momentum=0.02, value=-0.03), vol_mult=3.0, sigma=3.0),
    "quant_unwind_2007":  dict(factors=dict(mkt=-0.01, size=0.00, momentum=-0.06, value=-0.04), vol_mult=1.5, sigma=4.0),
    "momentum_crash_2009": dict(factors=dict(mkt=0.05, size=0.03, momentum=-0.15, value=0.04), vol_mult=2.0, sigma=2.0),
    "rate_shock_2022":    dict(factors=dict(mkt=-0.04, size=-0.02, momentum=0.01, value=0.03), vol_mult=1.5, sigma=2.0),
}


# ============================================================================ #
# Inputs and outputs
# ============================================================================ #
@dataclass
class MarketState:
    asset_returns: pd.DataFrame          # daily returns, rows = days, columns = assets
    market_returns: pd.Series            # daily index returns (vol regime)
    today_return: float                  # portfolio P&L so far today, fraction of NAV
    adv: pd.Series                       # average daily $ volume per asset
    nav: float
    data_age_sec: float
    exposures: pd.DataFrame | None = None       # assets x factors (beta, size, momentum, value)
    factor_returns: pd.DataFrame | None = None  # daily factor returns, same factor columns
    groups: pd.Series | None = None             # sector or correlation-cluster label per asset


@dataclass
class GAXGBState(MarketState):
    ic_history: pd.Series = field(default_factory=pd.Series)  # daily measured IC
    masks: list = field(default_factory=list)                 # GA feature masks, oldest -> newest
    train_pred_range: tuple = (-1.0, 1.0)                     # min/max prediction seen in training


@dataclass
class CheckResult:
    name: str
    scale: float
    reason: str


@dataclass
class RiskReport:
    weights: pd.Series
    risk_scale: float
    regime: str
    checks: list
    exposures: dict
    decomposition: dict
    stress: dict
    breaches: list

    def summary(self) -> str:
        lines = [f"regime={self.regime}  risk_scale={self.risk_scale:.2f}"]
        lines += [f"  {c.name:18s} {c.scale:.2f}  {c.reason}" for c in self.checks]
        e = self.exposures
        lines.append(f"  gross={e['gross']:.1%} net={e['net']:+.1%} beta={e.get('beta', 0):+.3f} "
                     f"vol={e['vol']:.1%} ES={e['es']:.2%} positions={e['n_positions']}")
        d = self.decomposition
        if "factor_share" in d:
            lines.append(f"  risk: {d['factor_share']:.0%} factor / {d['specific_share']:.0%} specific; "
                         f"top asset {d['top_asset']} = {d['top_asset_share']:.1%} of risk")
        lines.append("  stress: " + ", ".join(f"{k} {v:+.2%}" for k, v in self.stress.items()))
        lines.append("  breaches: " + (", ".join(self.breaches) if self.breaches else "none"))
        return "\n".join(lines)


# ============================================================================ #
# Helpers
# ============================================================================ #
def ewma_cov(returns: pd.DataFrame, halflife: int, shrink: float) -> np.ndarray:
    """Exponentially weighted covariance, shrunk toward its diagonal."""
    x = returns.fillna(0).to_numpy()
    w = 0.5 ** (np.arange(len(x))[::-1] / halflife)
    w /= w.sum()
    xc = x - (w[:, None] * x).sum(0)
    cov = (w[:, None] * xc).T @ xc
    return (1 - shrink) * cov + shrink * np.diag(np.diag(cov))


def neutralize(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Remove the part of w explained by exposures X, so X.T @ w ~ 0."""
    coef, *_ = np.linalg.lstsq(X, w, rcond=None)
    return w - X @ coef


def correlation_clusters(returns: pd.DataFrame, threshold: float = 0.7) -> pd.Series:
    """Group assets whose return correlation exceeds threshold (union-find)."""
    c = returns.corr().to_numpy()
    parent = list(range(len(c)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in zip(*np.where(np.triu(c, 1) > threshold)):
        parent[find(i)] = find(j)
    return pd.Series([f"c{find(i)}" for i in range(len(c))], index=returns.columns)


def expected_shortfall(pnl: np.ndarray, alpha: float) -> float:
    """Average loss on the worst alpha share of days, as a positive number."""
    cutoff = np.quantile(pnl, alpha)
    return float(-pnl[pnl <= cutoff].mean())


def risk_decomposition(w, cov, assets, X=None, F=None, factor_names=None):
    """
    Split portfolio variance into per-asset and factor/specific pieces.
      total variance   s^2 = w' S w
      marginal risk    MCR_i = (S w)_i / s
      contribution     CCR_i = w_i * MCR_i      (sums to s)
      factor variance  e' F e, with e = X' w
    """
    var = float(w @ cov @ w)
    if var <= 0:
        return {"vol": 0.0}
    vol = np.sqrt(var)
    share = w * (cov @ w) / var                       # each asset's share of variance, sums to 1
    out = {"vol": vol * np.sqrt(TD),
           "asset_share": pd.Series(share, index=assets),
           "top_asset": assets[int(np.argmax(share))],
           "top_asset_share": float(share.max())}
    if X is not None and F is not None:
        e = X.T @ w
        fvar = float(e @ F @ e)
        out["factor_share"] = min(fvar / var, 1.0)
        out["specific_share"] = 1 - out["factor_share"]
        out["by_factor"] = dict(zip(factor_names, (e * (F @ e) / var).round(4)))
    return out


# ============================================================================ #
# Base risk manager
# ============================================================================ #
class BaseRiskManager:
    params_cls = RiskParams

    def __init__(self, params: RiskParams | None = None):
        self.p = params or self.params_cls()
        self.peak_nav: float | None = None
        self.cooldown_left = 0
        self.halted_today = False
        self.dd_scale = 1.0
        self.dd_reason = "not started"

    # ------------------------------------------------------------------ daily
    def new_day(self, nav: float) -> None:
        """Call once at the start of each trading day."""
        self.halted_today = False
        if self.peak_nav is None:
            self.peak_nav = nav
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            if self.cooldown_left == 0:
                self.peak_nav = nav                    # restart the ladder from today's level
                self.dd_scale, self.dd_reason = 1.0, "cooldown over, peak reset"
            else:
                self.dd_scale, self.dd_reason = 0.0, f"cooldown, {self.cooldown_left} days left"
            return
        self.peak_nav = max(self.peak_nav, nav)
        dd = nav / self.peak_nav - 1
        if dd <= self.p.dd_flat:
            self.cooldown_left = self.p.flat_cooldown_days
            self.dd_scale, self.dd_reason = 0.0, f"drawdown {dd:.1%} -> flat for {self.cooldown_left} days"
        elif dd <= self.p.dd_reduce:
            self.dd_scale, self.dd_reason = self.p.dd_reduce_scale, f"drawdown {dd:.1%} -> reduce"
        else:
            self.dd_scale, self.dd_reason = 1.0, f"drawdown {dd:.1%}"

    # ------------------------------------------------------------------ checks
    def vol_ratio(self, s: MarketState) -> float:
        r = s.market_returns
        if len(r) < self.p.vol_long_window:
            return 1.0
        return float(r.iloc[-self.p.vol_short_window:].std() / r.iloc[-self.p.vol_long_window:].std())

    def regime(self, s: MarketState) -> str:
        ratio = self.vol_ratio(s)
        if ratio >= self.p.vol_flat_ratio:
            return "crisis"
        if ratio >= self.p.vol_reduce_ratio:
            return "high_vol"
        return "calm" if ratio < 0.8 else "normal"

    def check_vol_regime(self, s):
        ratio = self.vol_ratio(s)
        if ratio >= self.p.vol_flat_ratio:
            return CheckResult("vol_regime", 0.0, f"market vol {ratio:.1f}x normal -> flat")
        if ratio >= self.p.vol_reduce_ratio:
            return CheckResult("vol_regime", self.p.vol_reduce_scale, f"market vol {ratio:.1f}x normal -> reduce")
        return CheckResult("vol_regime", 1.0, f"market vol {ratio:.1f}x normal")

    def check_drawdown(self, s):
        return CheckResult("drawdown", self.dd_scale, self.dd_reason)

    def check_daily_loss(self, s):
        if s.today_return <= self.p.daily_loss_halt:
            self.halted_today = True
        if self.halted_today:
            return CheckResult("daily_loss", 0.0, f"today {s.today_return:.2%} -> halted until tomorrow")
        return CheckResult("daily_loss", 1.0, f"today {s.today_return:+.2%}")

    def check_stale_data(self, s):
        if s.data_age_sec > self.p.max_data_age_sec:
            return CheckResult("stale_data", 0.0, f"data {s.data_age_sec:.0f}s old -> kill switch")
        return CheckResult("stale_data", 1.0, f"data {s.data_age_sec:.0f}s old")

    def checks(self):
        return [self.check_stale_data, self.check_daily_loss, self.check_drawdown,
                self.check_vol_regime]

    # ------------------------------------------------------------------ construction
    def prefilter(self, signals: pd.Series, s) -> pd.Series:
        """Strategy hook: edit signals before sizing."""
        return signals

    def _limits(self, w, cov, X, adv_cap, groups):
        p = self.p
        for _ in range(8):                             # alternate until all limits hold together
            if X is not None:
                w = neutralize(w, X)
            vol = np.sqrt(max(w @ cov @ w, 1e-18) * TD)
            w = w * min(p.vol_target / vol, p.max_leverage)
            w = np.clip(w, -np.minimum(p.max_weight, adv_cap), np.minimum(p.max_weight, adv_cap))
            if groups is not None:
                gross = np.abs(w).sum()
                for g in np.unique(groups):
                    m = groups == g
                    gg = np.abs(w[m]).sum()
                    if gross > 0 and gg > p.max_group_share * gross:
                        w[m] *= p.max_group_share * gross / gg
            gross = np.abs(w).sum()
            if gross > p.max_gross:
                w *= p.max_gross / gross
        net = w.sum()
        if abs(net) > p.max_net:                       # final small shift if clipping left a tilt
            w -= (net - np.sign(net) * p.max_net) / len(w)
        return w

    def step(self, signals: pd.Series, s: MarketState) -> RiskReport:
        p = self.p
        results = [c(s) for c in self.checks()]
        scale = min(r.scale for r in results)

        assets = list(s.asset_returns.columns)
        sig = self.prefilter(signals.reindex(assets).fillna(0.0), s).to_numpy()
        hist = s.asset_returns.iloc[-p.es_window:]
        cov = ewma_cov(hist, p.cov_halflife, p.cov_shrink)
        asset_vol = np.sqrt(np.diag(cov))

        # factor matrix for neutralization: a column of ones (dollar-neutral) + factor exposures
        X = np.ones((len(assets), 1))
        if s.exposures is not None:
            X = np.column_stack([X, s.exposures.reindex(assets).fillna(0).to_numpy()])
        groups = s.groups.reindex(assets).fillna("other").to_numpy() if s.groups is not None else None
        adv_cap = (p.adv_cap * s.adv.reindex(assets).fillna(0) / s.nav).to_numpy()

        w = np.where(asset_vol > 0, sig / asset_vol, 0.0)
        w = self._limits(w, cov, X, adv_cap, groups) if np.abs(w).sum() > 0 else w

        # post-build tail limits: re-price today's book on past days and under stress scenarios
        pnl_hist = hist.fillna(0).to_numpy() @ w
        es = expected_shortfall(pnl_hist, p.es_alpha) if np.abs(w).sum() > 0 else 0.0
        stress = self.stress_test(w, cov, s, assets)
        worst = max([-v for v in stress.values()] + [0.0])
        tail_scale = min(1.0, p.es_limit / es if es > 0 else 1.0,
                         p.stress_loss_limit / worst if worst > 0 else 1.0)
        if tail_scale < 1.0:
            results.append(CheckResult("tail_limits", tail_scale,
                                       f"ES {es:.2%} / worst stress {worst:.2%} -> shrink to fit"))
            scale = min(scale, tail_scale) if scale > 0 else 0.0
        w = w * scale
        stress = {k: v * scale for k, v in stress.items()}
        es *= scale

        # report
        F = s.factor_returns.cov().to_numpy() if s.factor_returns is not None else None
        Xf = s.exposures.reindex(assets).fillna(0).to_numpy() if s.exposures is not None else None
        decomp = risk_decomposition(w, cov, assets, Xf, F,
                                    list(s.exposures.columns) if s.exposures is not None else None)
        exposures = dict(gross=float(np.abs(w).sum()), net=float(w.sum()),
                         vol=float(decomp.get("vol", 0.0)), es=es, n_positions=int((w != 0).sum()))
        if Xf is not None:
            fe = dict(zip(s.exposures.columns, Xf.T @ w))
            exposures.update(factors=fe, beta=float(fe.get("beta", 0.0)))
        report = RiskReport(pd.Series(w, index=assets), scale, self.regime(s), results,
                            exposures, decomp, stress, [])
        report.breaches = self.compliance(report, s, adv_cap, groups)
        return report

    # ------------------------------------------------------------------ tail + compliance
    def stress_test(self, w, cov, s, assets) -> dict:
        daily_sigma = float(np.sqrt(max(w @ cov @ w, 0.0)))
        out = {}
        for name, sc in STRESS_SCENARIOS.items():
            pnl = 0.0
            if s.exposures is not None:
                X = s.exposures.reindex(assets).fillna(0)
                for f, shock in sc["factors"].items():
                    col = "beta" if f == "mkt" else f
                    if col in X:
                        pnl += float(X[col].to_numpy() @ w) * shock
            pnl -= sc["sigma"] * sc["vol_mult"] * daily_sigma
            out[name] = pnl
        return out

    def compliance(self, r: RiskReport, s, adv_cap, groups) -> list:
        """Final check that every hard limit holds; anything listed here should block the trade."""
        p, w, e, tol = self.p, r.weights.to_numpy(), r.exposures, 1e-9
        b = []
        if np.abs(w).max(initial=0) > p.max_weight + tol: b.append("single-asset weight")
        if (np.abs(w) > adv_cap + tol).any(): b.append("liquidity (ADV)")
        if e["gross"] > p.max_gross + tol: b.append("gross exposure")
        if abs(e["net"]) > p.max_net + tol: b.append("net exposure")
        if abs(e.get("beta", 0.0)) > p.max_beta: b.append("market beta")
        for f, v in e.get("factors", {}).items():
            if f != "beta" and abs(v) > p.max_factor_exposure: b.append(f"factor {f}")
        if groups is not None and e["gross"] > 0:
            for g in np.unique(groups):
                if np.abs(w[groups == g]).sum() > p.max_group_share * e["gross"] + 1e-6:
                    b.append(f"group {g}")
        if e["vol"] > p.vol_target * 1.05: b.append("volatility target")
        if r.decomposition.get("top_asset_share", 0) > p.max_risk_share:
            b.append(f"risk concentration ({r.decomposition['top_asset']})")
        return b


# ============================================================================ #
# Strategy-specific risk manager
# ============================================================================ #
class GAXGBRiskManager(BaseRiskManager):
    params_cls = GAXGBRiskParams

    def __init__(self, params=None):
        super().__init__(params)
        self.last_signals = pd.Series(dtype=float)

    def check_signal_decay(self, s):
        ic = s.ic_history.iloc[-self.p.ic_window:]
        if len(ic) < self.p.ic_window:
            return CheckResult("signal_decay", 1.0, "not enough IC history")
        m = float(ic.mean())
        if m <= self.p.ic_floor:
            return CheckResult("signal_decay", self.p.ic_cut_scale, f"20d IC {m:+.3f} <= floor -> reduce")
        return CheckResult("signal_decay", 1.0, f"20d IC {m:+.3f}")

    @staticmethod
    def jaccard(a, b) -> float:
        a, b = np.asarray(a, bool), np.asarray(b, bool)
        u = (a | b).sum()
        return 1.0 if u == 0 else float((a & b).sum() / u)

    def check_mask_instability(self, s):
        m = s.masks
        if len(m) < 2:
            return CheckResult("mask_stability", 1.0, "not enough masks")
        lo = max(1, len(m) - self.p.mask_probation)
        worst = min(self.jaccard(m[t], m[t - 1]) for t in range(lo, len(m)))
        if worst < self.p.jaccard_trigger:
            return CheckResult("mask_stability", self.p.mask_cut_scale, f"feature overlap {worst:.2f} -> reduce")
        return CheckResult("mask_stability", 1.0, f"feature overlap {worst:.2f} (monitor)")

    def check_prediction_range(self, s):
        lo, hi = s.train_pred_range
        buf = self.p.pred_range_buffer * (hi - lo)
        x = self.last_signals
        if len(x) and ((x < lo - buf) | (x > hi + buf)).any():
            return CheckResult("prediction_range", 0.0, "prediction outside training range -> kill switch")
        return CheckResult("prediction_range", 1.0, "predictions in range")

    def checks(self):
        return super().checks() + [self.check_signal_decay, self.check_mask_instability,
                                   self.check_prediction_range]

    def step(self, signals, s):
        self.last_signals = signals
        return super().step(signals, s)

    def prefilter(self, signals, s):
        return signals.where(signals.abs() >= self.p.min_signal, 0.0)

    # ------------------------------------------------------------------ per-trade stops
    def check_trade_exit(self, side: int, entry_price: float, price: float,
                         horizon_sigma: float, bars_held: int, high_vol: bool):
        """
        side          +1 long / -1 short
        horizon_sigma expected size of a move over the prediction horizon, as a return
                      (asset daily vol / sqrt(390 / horizon_bars))
        """
        p = self.p
        ret = side * (price / entry_price - 1)
        if ret <= -p.vol_stop_k * horizon_sigma:
            return True, f"vol stop: {ret:.2%} < -{p.vol_stop_k}x horizon sigma"
        mult = p.time_stop_mult_high_vol if high_vol else p.time_stop_mult
        if bars_held >= mult * p.horizon_bars:
            return True, f"time stop: held {bars_held} bars"
        return False, ""


# ============================================================================ #
# Demo on synthetic data
# ============================================================================ #
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    N, T = 150, 520
    assets = [f"A{i:02d}" for i in range(N)]
    factors = ["beta", "size", "momentum", "value"]

    expo = pd.DataFrame(rng.normal(0, 1, (N, 4)), index=assets, columns=factors)
    expo["beta"] = rng.normal(1.0, 0.3, N)
    fvol = np.array([0.012, 0.004, 0.005, 0.004])
    vol_path = np.r_[np.ones(T - 20), np.full(20, 2.2)]      # market vol spikes in the last month
    fret = pd.DataFrame(rng.normal(0, 1, (T, 4)) * fvol * vol_path[:, None], columns=factors)
    spec = rng.normal(0, 0.015, (T, N)) * vol_path[:, None]
    rets = pd.DataFrame(fret.to_numpy() @ expo.to_numpy().T + spec, columns=assets)
    sectors = pd.Series([f"sec{i % 6}" for i in range(N)], index=assets)

    state = GAXGBState(
        asset_returns=rets, market_returns=fret["beta"], today_return=-0.004,
        adv=pd.Series(rng.uniform(2e7, 2e8, N), index=assets), nav=5e6, data_age_sec=12,
        exposures=expo, factor_returns=fret, groups=sectors,
        ic_history=pd.Series(rng.normal(0.04, 0.1, 60)),
        masks=[rng.integers(0, 2, 40) for _ in range(6)], train_pred_range=(-0.3, 0.3),
    )
    signals = pd.Series(rng.normal(0, 0.08, N), index=assets)

    for mgr in (BaseRiskManager(), GAXGBRiskManager()):
        mgr.new_day(state.nav)
        print(f"\n=== {type(mgr).__name__} ===")
        print(mgr.step(signals, state).summary())

    print("\n=== Drawdown ladder over a losing streak (GAXGB) ===")
    rm, nav = GAXGBRiskManager(), 1.0
    for day, r in enumerate([-0.04] * 9 + [0.004] * 13):
        rm.new_day(nav)
        print(f"day {day:2d} nav {nav:.3f} scale {rm.dd_scale:.1f}  {rm.dd_reason}")
        nav *= 1 + r * max(rm.dd_scale, 0)

    print("\n=== Per-trade stops ===")
    rm = GAXGBRiskManager()
    for args in [(+1, 100, 97.0, 0.008, 20, False), (+1, 100, 100.2, 0.008, 95, False),
                 (+1, 100, 100.2, 0.008, 65, True), (-1, 100, 99.5, 0.008, 30, False)]:
        print(args, "->", rm.check_trade_exit(*args))
