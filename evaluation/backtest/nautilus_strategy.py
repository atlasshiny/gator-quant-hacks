"""
nautilus_strategy.py  --  book-state strategy (frozen GA rule + XGBoost score)
                          with the ES risk manager wired in

What it does
------------
On every OrderBookDepth10 update it:
  1. feeds the top of book to the risk manager (data health, short-horizon vol),
  2. computes the snapshot-computable book features (same formulas as
     prepare_xgb_features.py),
  3. optionally runs the frozen 3-class XGBoost model (0=sell, 1=hold, 2=buy) and
     forms  xgb_score = P(buy) - P(sell),
  4. standardizes features with IN-SAMPLE mean/std and sums the GA-selected ones:
         S = sum_j mask_j * z_j,
  5. if S > tau -> candidate BUY, if S < -tau -> candidate SELL,
  6. asks the risk manager whether the entry is allowed (calendar, spread, touch
     size, vol regime, loss limits, throttles). Size is min(trade_size, rm size),
     so the risk manager can only BLOCK or keep size at trade_size (default 1),
  7. enters with a market order; exits are decided by the risk manager
     (price stop, time stop = holding_ms, forced flat before the maintenance
     break / news / roll windows, halts).

Ablation: `use_risk_manager=False` runs the plain rule (time exit + maintenance
blackout only). Report BOTH. The GA fitness does not include stops, blackouts or
vol-regime blocks, so the risk manager changes realized behavior; never tune its
parameters on OOS.

Frozen "bundle" JSON (written from in-sample data only; see build_bundle below):
    {
      "feature_names_28": [28 GA column names: snapshot feature names, one
                           "xgb_score", and "pad" for unused columns],
      "xgb_input_names":  [names fed to XGBoost, in the order it was trained],
      "mask":   [28 x 0/1]    GA feature flags (bit j <-> feature_names_28[j]),
      "mean":   [28],  "std": [28]   in-sample standardization stats,
      "tau":    float          GA threshold_raw / 1000 (in z-units),
      "lookback": int          GA decoded lookback (warm-up rows)
    }

IMPORTANT (train/serve parity)
------------------------------
* Four training features need raw MBP *event* fields that a depth snapshot does
  not carry: cancel_add_ratio, vpin, cancel_trade_ratio, message_intensity.
  They must not be XGBoost inputs and their GA mask bits must be 0. The
  strategy raises at start-up if the bundle violates this.
* The GA must have been run on columns standardized with the SAME mean/std
  stored in the bundle, otherwise tau is meaningless.
* The OFI formula below copies prepare_xgb_features.py exactly (including its
  "price moved away -> 0" branch) so live and training values match.
* Verify parity: replay one day through both code paths and compare vectors.

Risk manager notes
------------------
* default_2024h1_calendar() only covers Jan 8 - Jul 8, 2024. Outside that range
  the news / roll / holiday blackouts do nothing.
* The edge gate (use_edge_gate) is OFF by default: expected_move_ticks() is bounded
  by the label barrier, so with a barrier near the round-trip cost no trade can
  clear min_edge_to_cost (1.5x). Turn it on only with a larger label barrier.

API note: subscribe_order_book_depth / on_order_book_depth and
OrderBookDepth10.bid_counts/ask_counts are documented in the current Nautilus
docs. Check the other names (frozen StrategyConfig, on_order_filled, etc.)
against your installed pre-release.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, deque
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import xgboost as xgb
import polars as pl

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import InstrumentId, OrderBookDepth10, OrderSide
from nautilus_trader.trading import Strategy

GA_PROBABILITY_FEATURES = ("xgb_p_short", "xgb_p_flat", "xgb_p_long")

try:
    from python.es_risk_manager import (
        Book, ContractSpec, ESRiskManager, Position, RiskConfig,
        cme_trading_day, default_2024h1_calendar, expected_move_ticks,
    )
except ImportError:  # es_risk_manager.py sitting next to this file
    from es_risk_manager import (
        Book, ContractSpec, ESRiskManager, Position, RiskConfig,
        cme_trading_day, default_2024h1_calendar, expected_move_ticks,
    )

_EPS = 1e-8
_ET = ZoneInfo("America/New_York")

SNAPSHOT_FEATURES = (
    *[f"spread_l{i}" for i in range(5)],
    *[f"obi_l{i}" for i in range(5)],
    "depth_imbalance", "bid_slope", "ask_slope",
    *[f"ofi_zscore_l{i}" for i in range(3)],
    "bid_quote_dispersion", "ask_quote_dispersion",
    "buy_vwap_slippage", "sell_vwap_slippage",
    "deep_bid_slope", "deep_ask_slope", "deep_count_imbalance",
)  # 23 features computable from a depth snapshot

EVENT_ONLY_FEATURES = (
    "cancel_add_ratio", "vpin", "cancel_trade_ratio", "message_intensity",
)  # need raw MBP event fields -> cannot be served from snapshots


def _f(x) -> float:
    fn = getattr(x, "as_double", None)
    return fn() if fn is not None else float(x)


class _Rolling:
    """Rolling mean / sample std (ddof=1) over the last n values."""

    def __init__(self, n: int) -> None:
        self.n, self.buf, self.s, self.ss = n, deque(), 0.0, 0.0

    def push(self, x: float) -> None:
        self.buf.append(x)
        self.s += x
        self.ss += x * x
        if len(self.buf) > self.n:
            old = self.buf.popleft()
            self.s -= old
            self.ss -= old * old

    @property
    def full(self) -> bool:
        return len(self.buf) == self.n

    def mean_std(self) -> tuple[float, float]:
        n = len(self.buf)
        mean = self.s / n
        var = max((self.ss - self.s * self.s / n) / (n - 1), 0.0)
        return mean, var ** 0.5


class BookFeatureEngine:
    """Causal feature computation from consecutive OrderBookDepth10 snapshots."""

    def __init__(self, window: int = 1000) -> None:
        self._prev = None
        self._ofi = [_Rolling(window) for _ in range(3)]

    def update(self, depth: OrderBookDepth10) -> dict[str, float] | None:
        bids, asks = list(depth.bids)[:10], list(depth.asks)[:10]
        if len(bids) < 10 or len(asks) < 10:
            return None
        bp = [_f(o.price) for o in bids]
        bs = [_f(o.size) for o in bids]
        ap = [_f(o.price) for o in asks]
        az = [_f(o.size) for o in asks]
        if min(bp) <= 0 or min(ap) <= 0 or ap[0] < bp[0]:
            return None  # empty level or crossed book: skip, keep prev state

        # OFI (top 3 levels) vs previous snapshot -- same rules as training
        if self._prev is not None:
            pbp, pbs, pap, paz = self._prev
            for i in range(3):
                if bp[i] > pbp[i]:
                    bf = bs[i]
                elif bp[i] == pbp[i]:
                    bf = bs[i] - pbs[i]
                else:
                    bf = 0.0
                if ap[i] < pap[i]:
                    af = az[i]
                elif ap[i] == pap[i]:
                    af = az[i] - paz[i]
                else:
                    af = 0.0
                self._ofi[i].push(bf - af)
        self._prev = (bp, bs, ap, az)
        if not all(r.full for r in self._ofi):
            return None  # rolling window not yet filled (training dropped these)

        mid = (bp[0] + ap[0]) / 2.0
        cum_b, cum_a = sum(bs), sum(az)
        feats: dict[str, float] = {}
        for i in range(5):
            feats[f"spread_l{i}"] = (ap[i] - bp[i]) / mid
            feats[f"obi_l{i}"] = (bs[i] - az[i]) / (bs[i] + az[i] + _EPS)
        feats["depth_imbalance"] = (cum_b - cum_a) / (cum_b + cum_a + _EPS)
        feats["bid_slope"] = (bp[0] - bp[9]) / (cum_b + _EPS)
        feats["ask_slope"] = (ap[9] - ap[0]) / (cum_a + _EPS)
        for i, roll in enumerate(self._ofi):
            m, s = roll.mean_std()
            feats[f"ofi_zscore_l{i}"] = (roll.buf[-1] - m) / (s + _EPS)
        feats["bid_quote_dispersion"] = sum(((p - mid) / mid) ** 2 for p in bp) / 10.0
        feats["ask_quote_dispersion"] = sum(((p - mid) / mid) ** 2 for p in ap) / 10.0
        bid_vwap = sum(p * s for p, s in zip(bp, bs)) / (cum_b + _EPS)
        ask_vwap = sum(p * s for p, s in zip(ap, az)) / (cum_a + _EPS)
        feats["buy_vwap_slippage"] = (ask_vwap - ap[0]) / mid
        feats["sell_vwap_slippage"] = (bp[0] - bid_vwap) / mid
        feats["deep_bid_slope"] = (bp[2] - bp[9]) / (sum(bs[2:10]) + _EPS)
        feats["deep_ask_slope"] = (ap[9] - ap[2]) / (sum(az[2:10]) + _EPS)
        bc = list(getattr(depth, "bid_counts", [0] * 10))[:10]
        ac = list(getattr(depth, "ask_counts", [0] * 10))[:10]
        db, da = sum(bc[2:10]), sum(ac[2:10])
        feats["deep_count_imbalance"] = (db - da) / (db + da + _EPS)
        return feats


def _top_of_book(depth: OrderBookDepth10) -> Book | None:
    """Risk-manager view of the book (best bid/ask and their sizes)."""
    bids, asks = list(depth.bids)[:1], list(depth.asks)[:1]
    if not bids or not asks:
        return None
    bp, ap = _f(bids[0].price), _f(asks[0].price)
    if bp <= 0 or ap <= 0 or ap < bp:
        return None
    return Book(int(depth.ts_event), int(depth.ts_init), bp, ap,
                _f(bids[0].size), _f(asks[0].size))


import os
_REPO = os.environ.get("REPO", ".")
class XGBBookStateConfig(StrategyConfig):
    instrument_id: InstrumentId
    bundle_path: str = os.environ.get(
        "BUNDLE_PATH", f"{_REPO}/output/ga_results_january/frozen_bundle.json")
    model_path: str = os.environ.get(
        "MODEL_PATH", f"{_REPO}/output/models/xgboost_january.json")
    event_features_path: str = os.environ.get(
        "EVENT_FEATURES_PATH", f"{_REPO}/output/features/stationary_features_january.parquet")
    trade_size: int = 1              # contracts per entry (risk manager can only block)
    holding_ms: int = 1500           # exit horizon h (match the label horizon!)
    min_interval_ms: int = 500       # min gap between entries
    cooldown_ms: int = 500           # min gap after an exit
    # --- risk manager ---
    use_risk_manager: bool = True    # False = plain rule (ablation baseline)
    capital: float = 1_000_000.0     # must equal the venue starting balance / COSTS.capital
    use_edge_gate: bool = False      # see module docstring before enabling
    edge_barrier_bps: float = 0.7    # label cost hurdle c (only used by the edge gate)
    risk_summary_path: str = "risk_summary.json"
    # --- plain-rule blackout (only used when use_risk_manager=False) ---
    flatten_start_et: str = "16:55"
    resume_after_et: str = "18:05"
    signal_log_path: str = "signal_log.csv"


class XGBBookStateStrategy(Strategy):
    def __init__(self, config: XGBBookStateConfig) -> None:
        super().__init__(config)
        self.instrument = None
        self._engine = BookFeatureEngine()
        self._model: xgb.Booster | None = None
        self._last_probs: tuple[float, float, float] | None = None
        self._event_features: dict[str, np.ndarray] = {}

        # position / order state
        self._pos = 0.0
        self._pending = False
        self._closing = False
        self._entry_ns = 0
        self._entry_side = 0
        self._entry_qty = self._entry_cost = 0.0
        self._exit_qty = self._exit_cost = 0.0
        self._last_order_ns = 0
        self._last_exit_ns = 0
        self._position: Position | None = None
        self._decision_book: Book | None = None
        self._pending_stop = 0.0

        # clocks / counters
        self._et_bucket = -1
        self._et_min = 0
        self._day_bucket = -1
        self._trading_day = None
        self._n_seen = self._n_skipped = self._n_ready = 0
        self._n_long = self._n_short = 0
        self._blocks: Counter = Counter()
        self._exit_reasons: Counter = Counter()
        self._closed_pnl: list[float] = []
        self._log_rows: list[tuple] = []

        self._hold_ns = config.holding_ms * 1_000_000
        self._gap_ns = config.min_interval_ms * 1_000_000
        self._cool_ns = config.cooldown_ms * 1_000_000
        h, m = (int(x) for x in config.flatten_start_et.split(":"))
        self._blk_start = h * 60 + m
        h, m = (int(x) for x in config.resume_after_et.split(":"))
        self._blk_end = h * 60 + m

        self._rm: ESRiskManager | None = None
        if config.use_risk_manager:
            self._rm = ESRiskManager(RiskConfig(
                capital=config.capital,
                holding_seconds=config.holding_ms / 1000.0,   # time stop = strategy horizon
                **default_2024h1_calendar(),
            ))

        self._load_bundle()

    # ------------------------------------------------------------------ setup
    def _load_bundle(self) -> None:
        b = json.loads(Path(self.config.bundle_path).read_text())
        self._names = list(b["feature_names_28"])
        self._xgb_names = list(b["xgb_input_names"])
        self._mask = np.array(b["mask"], dtype=bool)
        self._mean = np.array(b["mean"], dtype=np.float64)
        self._std = np.array(b["std"], dtype=np.float64)
        self._tau = float(b["tau"])
        self._warmup = max(int(b["lookback"]), 1001)

        if not (len(self._names) == len(self._mask) == len(self._mean) == len(self._std) == 28):
            raise ValueError("Bundle must describe exactly 28 GA columns.")
        if self._names.count("xgb_score") > 1:
            raise ValueError("At most one 'xgb_score' column is supported.")
        self._score_idx = (self._names.index("xgb_score")
                           if "xgb_score" in self._names else None)
        if not self._mask.any():
            raise ValueError("GA mask selects no features.")
        bad = [n for n in self._xgb_names
               if n not in SNAPSHOT_FEATURES and n not in EVENT_ONLY_FEATURES]
        if bad:
            raise ValueError(f"Unsupported XGBoost inputs: {bad}")
        allowed = set(SNAPSHOT_FEATURES) | set(EVENT_ONLY_FEATURES) | {
            "xgb_score", *GA_PROBABILITY_FEATURES
        }
        bad = [n for j, n in enumerate(self._names)
               if self._mask[j] and n not in allowed]
        if bad:
            raise ValueError(f"GA selected columns not servable (pad or event-only): {bad}")

        needs_event = (
            any(n in EVENT_ONLY_FEATURES for n in self._xgb_names)
            or any(self._mask[j] and self._names[j] in EVENT_ONLY_FEATURES
                   for j in range(28))
        )
        if needs_event:
            path = Path(self.config.event_features_path)
            if not path.is_file():
                raise FileNotFoundError(f"Event feature Parquet not found: {path}")
            columns = ["ts_event", *EVENT_ONLY_FEATURES]
            frame = pl.read_parquet(path, columns=columns)
            self._event_features = {
                name: frame[name].to_numpy().astype(np.float32, copy=False)
                for name in EVENT_ONLY_FEATURES
            }
        if (
            (self._score_idx is not None and self._mask[self._score_idx])
            or any(self._mask[j] and self._names[j] in GA_PROBABILITY_FEATURES
                   for j in range(28))
        ):
            self._model = xgb.Booster()
            self._model.load_model(self.config.model_path)
            self._model.set_param({"device": "cpu"})

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not in cache")
            self.stop()
            return
        self.subscribe_order_book_depth(self.config.instrument_id)

    # ----------------------------------------------------------------- helpers
    def _in_blackout(self, now_ns: int) -> bool:
        """Plain-rule maintenance blackout (ET). Not used when the risk manager is on."""
        bucket = now_ns // 60_000_000_000
        if bucket != self._et_bucket:
            dt = datetime.fromtimestamp(now_ns / 1e9, tz=_ET)
            self._et_min, self._et_bucket = dt.hour * 60 + dt.minute, bucket
        return self._blk_start <= self._et_min < self._blk_end

    def _maybe_new_session(self, now_ns: int) -> None:
        """Tell the risk manager when a new CME trading day (17:00 CT) begins."""
        bucket = now_ns // 60_000_000_000
        if bucket == self._day_bucket:
            return
        self._day_bucket = bucket
        day = cme_trading_day(datetime.fromtimestamp(now_ns / 1e9, tz=timezone.utc))
        if day != self._trading_day:
            self._trading_day = day
            self._rm.on_session_start(day, self._rm.equity)

    def _score(self, feats: dict[str, float]) -> tuple[float, float]:
        """Return (S, xgb_score). S = sum of selected standardized features."""
        z = np.zeros(28)
        xgb_score = float("nan")
        self._last_probs = None
        ready_index = self._n_ready - 1
        if self._event_features:
            if ready_index >= len(next(iter(self._event_features.values()))):
                raise IndexError("Event feature Parquet is shorter than replayed data.")
            feats = {
                **feats,
                **{
                    name: float(values[ready_index])
                    for name, values in self._event_features.items()
                },
            }
        probs: tuple[float, float, float] | None = None
        for j in range(28):
            if not self._mask[j]:
                continue
            if j == self._score_idx or self._names[j] in GA_PROBABILITY_FEATURES:
                x = np.array([[feats[n] for n in self._xgb_names]], dtype=np.float32)
                if probs is None:
                    p = self._model.predict(xgb.DMatrix(x))[0]
                    probs = (float(p[0]), float(p[1]), float(p[2]))
                    self._last_probs = probs
                    xgb_score = probs[2] - probs[0]
                raw = xgb_score if self._names[j] == "xgb_score" else {
                    "xgb_p_short": probs[0],
                    "xgb_p_flat": probs[1],
                    "xgb_p_long": probs[2],
                }[self._names[j]]
            else:
                raw = feats[self._names[j]]
            z[j] = (raw - self._mean[j]) / (self._std[j] + _EPS)
        return float(z[self._mask].sum()), xgb_score

    def _exit_decision(self, now_ns: int) -> tuple[bool, str]:
        if self._rm is not None and self._position is not None:
            return self._rm.check_exit(now_ns, self._position)
        if self._entry_ns and now_ns - self._entry_ns >= self._hold_ns:
            return True, "time stop"
        if self._in_blackout(now_ns):
            return True, "maintenance blackout"
        return False, ""

    # ------------------------------------------------------------------- logic
    def on_order_book_depth(self, depth: OrderBookDepth10) -> None:
        self._n_seen += 1
        book = _top_of_book(depth)
        if self._rm is not None and book is not None:
            self._rm.on_book(book)                 # data health + short-horizon vol
        feats = self._engine.update(depth)         # always update state (OFI needs it)
        if feats is None:
            self._n_skipped += 1
        else:
            self._n_ready += 1

        now = self.clock.timestamp_ns()
        if self._rm is not None:
            self._maybe_new_session(now)

        # --- exits first: they must run even when features are unavailable
        if self._pos != 0:
            if not self._closing:
                exit_now, why = self._exit_decision(now)
                if exit_now:
                    self._closing = True
                    self._exit_reasons[why.split(" (")[0]] += 1
                    self.close_all_positions(self.config.instrument_id)
            return

        # --- entries: flat only, one position at a time
        if feats is None or book is None:
            return
        if self._pending or self._n_ready < self._warmup:
            return
        if now - self._last_order_ns < self._gap_ns or now - self._last_exit_ns < self._cool_ns:
            return
        if self._rm is None and self._in_blackout(now):
            return

        s, xgb_score = self._score(feats)
        if s > self._tau:
            side = OrderSide.BUY
        elif s < -self._tau:
            side = OrderSide.SELL
        else:
            return

        contracts = self.config.trade_size
        if self._rm is not None:
            side_i = 1 if side == OrderSide.BUY else -1
            predicted = None
            if self.config.use_edge_gate and self._last_probs is not None:
                p_sell, _, p_buy = self._last_probs
                predicted = expected_move_ticks(
                    p_sell, p_buy, side_i, book.mid,
                    barrier_bps=self.config.edge_barrier_bps, spec=self._rm.spec)
            d = self._rm.pre_trade_check(now, side_i, self._position, predicted)
            if not d.allowed:
                reason = d.reasons[0] if d.reasons else "blocked"
                self._blocks[reason.split(" (")[0]] += 1
                return
            contracts = min(self.config.trade_size, d.contracts)
            self._pending_stop = d.stop_ticks
            self._decision_book = book

        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=side,
            quantity=self.instrument.make_qty(Decimal(contracts)),
        )
        self._pending = True
        self._last_order_ns = now
        self.submit_order(order)
        if side == OrderSide.BUY:
            self._n_long += 1
        else:
            self._n_short += 1
        self._log_rows.append((now, side.name, s, xgb_score))

    # ------------------------------------------------------------ order events
    def on_order_filled(self, event) -> None:
        qty = float(event.last_qty)
        px = _f(event.last_px)
        old = self._pos
        self._pos += qty if event.order_side == OrderSide.BUY else -qty
        if abs(self._pos) < 1e-9:
            self._pos = 0.0
        now = self.clock.timestamp_ns()
        self._pending = False

        if abs(self._pos) > abs(old) + 1e-9:                       # entry fill
            self._entry_side = 1 if self._pos > 0 else -1
            self._entry_qty += qty
            self._entry_cost += qty * px
            if self._entry_ns == 0:
                self._entry_ns = now
            if self._rm is not None:
                if self._decision_book is not None:
                    self._rm.on_fill(self._entry_side, px, self._decision_book)
                self._position = Position(
                    self._entry_side, int(round(abs(self._pos))),
                    self._entry_cost / self._entry_qty, self._entry_ns, self._pending_stop)
        else:                                                      # exit fill
            self._exit_qty += qty
            self._exit_cost += qty * px
            if self._pos == 0.0:
                self._finish_trade(now)

    def _finish_trade(self, now_ns: int) -> None:
        spec = self._rm.spec if self._rm is not None else ContractSpec()
        q = self._entry_qty
        if q > 0 and self._exit_qty > 0:
            avg_in = self._entry_cost / self._entry_qty
            avg_out = self._exit_cost / self._exit_qty
            pnl = (self._entry_side * (avg_out - avg_in) * spec.multiplier * q
                   - 2 * spec.fee_per_side * q)                    # entry + exit fee
            self._closed_pnl.append(pnl)
            if self._rm is not None:
                self._rm.on_trade_closed(pnl, int(round(q)), now_ns)
        self._pos, self._entry_ns, self._closing = 0.0, 0, False
        self._entry_side, self._entry_qty, self._entry_cost = 0, 0.0, 0.0
        self._exit_qty = self._exit_cost = 0.0
        self._position, self._last_exit_ns = None, now_ns

    def on_order_rejected(self, event) -> None:
        self._pending = False
        if self._pos != 0:
            self._closing = False          # let the exit logic retry

    def on_order_denied(self, event) -> None:
        self._pending = False
        if self._pos != 0:
            self._closing = False

    # -------------------------------------------------------------------- stop
    def on_stop(self) -> None:
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_order_book_depth(self.config.instrument_id)
        self.log.info(
            f"snapshots={self._n_seen:,} skipped={self._n_skipped:,} "
            f"ready={self._n_ready:,} long={self._n_long} short={self._n_short} "
            f"blocked={sum(self._blocks.values())}"
        )
        with open(self.config.signal_log_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ts_ns", "side", "S", "xgb_score"])
            w.writerows(self._log_rows)
        summary = {
            "use_risk_manager": self._rm is not None,
            "entries_submitted": self._n_long + self._n_short,
            "trades_closed": len(self._closed_pnl),
            "strategy_net_pnl_usd": float(sum(self._closed_pnl)),
            "blocks_by_reason": dict(self._blocks),
            "exits_by_reason": dict(self._exit_reasons),
            "risk_status": self._rm.status() if self._rm is not None else None,
        }
        Path(self.config.risk_summary_path).write_text(json.dumps(summary, indent=2, default=str))


# ---------------------------------------------------------------------------
# Offline helpers: run AFTER in-sample work is finished, BEFORE the OOS replay.
# ---------------------------------------------------------------------------
def compute_stats(features_bin: str, out_json: str, n_features: int = 28) -> None:
    """In-sample mean/std per GA column. Standardize the GA input with these."""
    x = np.fromfile(features_bin, dtype=np.float32).reshape(-1, n_features)
    Path(out_json).write_text(json.dumps(
        {"mean": x.mean(0).tolist(), "std": x.std(0).tolist()}))


def build_bundle(best_strategy_json: str, stats_json: str,
                 column_names_28: list[str], xgb_input_names: list[str],
                 out_path: str) -> None:
    """column_names_28: GA column order, e.g. 25 snapshot features + ['xgb_score', 'pad', 'pad']."""
    best = json.loads(Path(best_strategy_json).read_text())["strategy"]
    stats = json.loads(Path(stats_json).read_text())
    bundle = {
        "feature_names_28": list(column_names_28),
        "xgb_input_names": list(xgb_input_names),
        "mask": [int(v) for v in best["feature_flags"]],
        "mean": stats["mean"],
        "std": stats["std"],
        "tau": best["threshold_param"] / 1000.0,
        "lookback": int(best["lookback_window"]),
    }
    Path(out_path).write_text(json.dumps(bundle, indent=2))