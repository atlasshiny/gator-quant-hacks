"""
nautilus_strategy.py  --  book-state strategy (frozen GA rule + XGBoost score)

What it does
------------
On every OrderBookDepth10 update it:
  1. computes the snapshot-computable book features (same formulas as
     prepare_xgb_features.py),
  2. optionally runs the frozen 3-class XGBoost model and forms
         xgb_score = P(up) - P(down),
  3. standardizes features with IN-SAMPLE mean/std and sums the GA-selected
     ones:  S = sum_j mask_j * z_j,
  4. goes long if S > tau, short if S < -tau (market order, fixed size),
  5. exits after `holding_ms` or before the CME maintenance break.

Nothing here fits or tunes anything. Everything it needs comes from a frozen
"bundle" JSON written after the in-sample work is finished:

    {
      "feature_names_28": [27 feature names in training order..., "xgb_score"],
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

API note: subscribe_order_book_depth / on_order_book_depth and
OrderBookDepth10.bid_counts/ask_counts are documented in the current Nautilus
docs. Check the other names (frozen StrategyConfig, on_order_filled, etc.)
against your installed pre-release.
"""

from __future__ import annotations

import csv
import json
from collections import deque
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import xgboost as xgb

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import InstrumentId, OrderBookDepth10, OrderSide
from nautilus_trader.trading import Strategy

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


class XGBBookStateConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bundle_path: str = "data/ga_inputs/frozen_bundle.json"
    model_path: str = "data/ga_inputs/xgboost_final.json"
    trade_size: int = 1              # fixed contracts per entry
    holding_ms: int = 1500           # exit horizon h (match the label horizon!)
    min_interval_ms: int = 500       # min gap between entries
    cooldown_ms: int = 500           # min gap after an exit
    flatten_start_et: str = "16:55"  # flatten + block entries from here...
    resume_after_et: str = "18:05"   # ...until here (CME maintenance break)
    signal_log_path: str = "signal_log.csv"


class XGBBookStateStrategy(Strategy):
    def __init__(self, config: XGBBookStateConfig) -> None:
        super().__init__(config)
        self.instrument = None
        self._engine = BookFeatureEngine()
        self._model: xgb.Booster | None = None

        self._pos = 0.0
        self._pending = False
        self._closing = False
        self._entry_ns = 0
        self._last_order_ns = 0
        self._last_exit_ns = 0

        self._et_bucket = -1
        self._et_min = 0
        self._n_seen = self._n_skipped = self._n_ready = 0
        self._n_long = self._n_short = 0
        self._log_rows: list[tuple] = []

        self._hold_ns = config.holding_ms * 1_000_000
        self._gap_ns = config.min_interval_ms * 1_000_000
        self._cool_ns = config.cooldown_ms * 1_000_000
        h, m = (int(x) for x in config.flatten_start_et.split(":"))
        self._blk_start = h * 60 + m
        h, m = (int(x) for x in config.resume_after_et.split(":"))
        self._blk_end = h * 60 + m

        self._load_bundle()

    # setup
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
            raise ValueError("Bundle must describe exactly 28 columns (27 features + xgb_score).")
        if self._names[27] != "xgb_score":
            raise ValueError("Column 27 must be 'xgb_score'.")
        if not self._mask.any():
            raise ValueError("GA mask selects no features.")
        bad = [n for n in self._xgb_names if n not in SNAPSHOT_FEATURES]
        if bad:
            raise ValueError(f"XGBoost inputs not computable from snapshots: {bad}")
        bad = [n for j, n in enumerate(self._names[:27])
               if self._mask[j] and n not in SNAPSHOT_FEATURES]
        if bad:
            raise ValueError(f"GA selected features not computable from snapshots: {bad}")

        if self._mask[27]:
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

    def _in_blackout(self, now_ns: int) -> bool:
        bucket = now_ns // 60_000_000_000
        if bucket != self._et_bucket:
            dt = datetime.fromtimestamp(now_ns / 1e9, tz=_ET)
            self._et_min, self._et_bucket = dt.hour * 60 + dt.minute, bucket
        return self._blk_start <= self._et_min < self._blk_end

    def _score(self, feats: dict[str, float]) -> tuple[float, float]:
        """Return (S, xgb_score). S = sum of selected standardized features."""
        z = np.zeros(28)
        xgb_score = float("nan")
        for j in range(27):
            if self._mask[j]:
                z[j] = (feats[self._names[j]] - self._mean[j]) / (self._std[j] + _EPS)
        if self._mask[27]:
            x = np.array([[feats[n] for n in self._xgb_names]], dtype=np.float32)
            p = self._model.predict(xgb.DMatrix(x))[0]      # [P(down), P(flat), P(up)]
            xgb_score = float(p[2] - p[0])
            z[27] = (xgb_score - self._mean[27]) / (self._std[27] + _EPS)
        return float(z[self._mask].sum()), xgb_score

    # logic
    def on_order_book_depth(self, depth: OrderBookDepth10) -> None:
        self._n_seen += 1
        feats = self._engine.update(depth)   # always update state (OFI needs it)
        if feats is None:
            self._n_skipped += 1
            return
        self._n_ready += 1

        now = self.clock.timestamp_ns()
        blackout = self._in_blackout(now)

        # exits: timed, or flatten for the maintenance break
        if self._pos != 0:
            timed_out = self._entry_ns and (now - self._entry_ns >= self._hold_ns)
            if not self._closing and (blackout or timed_out):
                self._closing = True
                self.close_all_positions(self.config.instrument_id)
            return

        # entries: flat only, one position at a time
        if self._pending or blackout or self._n_ready < self._warmup:
            return
        if now - self._last_order_ns < self._gap_ns or now - self._last_exit_ns < self._cool_ns:
            return

        s, xgb_score = self._score(feats)
        if s > self._tau:
            side = OrderSide.BUY
        elif s < -self._tau:
            side = OrderSide.SELL
        else:
            return

        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=side,
            quantity=self.instrument.make_qty(Decimal(self.config.trade_size)),
        )
        self._pending = True
        self._last_order_ns = now
        self.submit_order(order)
        if side == OrderSide.BUY:
            self._n_long += 1
        else:
            self._n_short += 1
        self._log_rows.append((now, side.name, s, xgb_score))

    # order events
    def on_order_filled(self, event) -> None:
        qty = float(event.last_qty)
        self._pos += qty if event.order_side == OrderSide.BUY else -qty
        now = self.clock.timestamp_ns()
        self._pending = False
        if abs(self._pos) < 1e-9:
            self._pos, self._entry_ns, self._closing = 0.0, 0, False
            self._last_exit_ns = now
        elif self._entry_ns == 0:
            self._entry_ns = now

    def on_order_rejected(self, event) -> None:
        self._pending = False

    def on_order_denied(self, event) -> None:
        self._pending = False

    def on_stop(self) -> None:
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_order_book_depth(self.config.instrument_id)
        self.log.info(
            f"snapshots={self._n_seen:,} skipped={self._n_skipped:,} "
            f"ready={self._n_ready:,} long={self._n_long} short={self._n_short}"
        )
        with open(self.config.signal_log_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ts_ns", "side", "S", "xgb_score"])
            w.writerows(self._log_rows)


# Offline helpers: run AFTER in-sample work is finished, BEFORE the OOS replay.
def compute_stats(features_bin: str, out_json: str, n_features: int = 28) -> None:
    """In-sample mean/std per GA column. Standardize the GA input with these."""
    x = np.fromfile(features_bin, dtype=np.float32).reshape(-1, n_features)
    Path(out_json).write_text(json.dumps(
        {"mean": x.mean(0).tolist(), "std": x.std(0).tolist()}))


def build_bundle(best_strategy_json: str, stats_json: str,
                 feature_names_27: list[str], xgb_input_names: list[str],
                 out_path: str) -> None:
    best = json.loads(Path(best_strategy_json).read_text())["strategy"]
    stats = json.loads(Path(stats_json).read_text())
    bundle = {
        "feature_names_28": list(feature_names_27) + ["xgb_score"],
        "xgb_input_names": list(xgb_input_names),
        "mask": [int(v) for v in best["feature_flags"]],
        "mean": stats["mean"],
        "std": stats["std"],
        "tau": best["threshold_param"] / 1000.0,
        "lookback": int(best["lookback_window"]),
    }
    Path(out_path).write_text(json.dumps(bundle, indent=2))