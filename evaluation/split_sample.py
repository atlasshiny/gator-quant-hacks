"""
sample_split.py

In-sample (IS) / out-of-sample (OOS) splitting and evaluation for Databento
time-series data (e.g. GLBX.MDP3 mbp-10).

Design rules that matter for market data:
  * Splits are strictly chronological (never shuffled).
  * Splits fall on TRADING-DAY boundaries, so a session is never cut in half.
    CME Globex trading day rolls at 17:00 America/Chicago.
  * An optional embargo (purge gap) drops data right after the IS window so
    rolling features / labels that look back or forward can't leak across the split.
  * Parameters are fit on IS only; OOS is scored once with those frozen params.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

SESSION_TZ = "America/Chicago"
SESSION_ROLL_HOUR = 17  # CME Globex trading day starts 17:00 CT the prior evening


def _timestamps(df: pd.DataFrame, ts_col: str | None = None) -> pd.DatetimeIndex:
    """Return tz-aware UTC timestamps from the index or a column."""
    if ts_col is not None:
        ts = pd.DatetimeIndex(df[ts_col])
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
    elif "ts_event" in df.columns:
        ts = pd.DatetimeIndex(df["ts_event"])
    else:
        raise ValueError("DataFrame needs a DatetimeIndex, a 'ts_event' column, or ts_col=...")
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def add_trading_day(df: pd.DataFrame, ts_col: str | None = None) -> pd.Series:
    """Map each row to its CME trading day (rolls at 17:00 CT)."""
    local = _timestamps(df, ts_col).tz_convert(SESSION_TZ)
    shift = pd.Timedelta(hours=24 - SESSION_ROLL_HOUR)
    days = (local + shift).normalize().tz_localize(None)
    return pd.Series(days, index=df.index, name="trading_day")


def trading_day_window(
    days, start_offset: str | pd.Timedelta = "0s"
) -> tuple[str, str]:
    """
    UTC [start, end) ISO timestamps covering the given CME trading days.
    Trading day D runs from 17:00 CT on the previous calendar day to 17:00 CT on D.
    `start_offset` pushes the start later (use it as an embargo for OOS windows).
    Used to feed date windows to engines that read data directly (e.g. Nautilus).
    """
    first = pd.Timestamp(min(days)).normalize()
    last = pd.Timestamp(max(days)).normalize()
    roll = pd.Timedelta(hours=SESSION_ROLL_HOUR)
    start = (first - pd.Timedelta(days=1) + roll).tz_localize(SESSION_TZ)
    end = (last + roll).tz_localize(SESSION_TZ)
    start = start + pd.Timedelta(start_offset)
    return start.tz_convert("UTC").isoformat(), end.tz_convert("UTC").isoformat()


# Split container
@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    train_days: list[pd.Timestamp]
    test_days: list[pd.Timestamp]
    fold: int = 0

    def describe(self) -> str:
        def rng(d):
            return f"{d[0].date()} -> {d[-1].date()} ({len(d)}d)" if d else "empty"
        return (
            f"fold {self.fold}: IS {rng(self.train_days)} [{len(self.train):,} rows] | "
            f"OOS {rng(self.test_days)} [{len(self.test):,} rows]"
        )


# Splitter
class SampleSplitter:
    """
    Chronological IS/OOS splitter operating on trading days.

    Parameters
    ----------
    oos_fraction : fraction of trading days held out as OOS (used by `split`).
    embargo      : time dropped from the START of OOS (purge gap) to prevent leakage
                   from lookback windows / forward-looking labels. e.g. "30min".
    ts_col       : timestamp column; defaults to the DatetimeIndex or 'ts_event'.
    """

    def __init__(
        self,
        oos_fraction: float = 0.3,
        embargo: str | pd.Timedelta = "0s",
        ts_col: str | None = None,
    ):
        if not 0 < oos_fraction < 1:
            raise ValueError("oos_fraction must be in (0, 1).")
        self.oos_fraction = oos_fraction
        self.embargo = pd.Timedelta(embargo)
        self.ts_col = ts_col

    def _prepare(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.Series]:
        if df.empty:
            raise ValueError("Cannot split an empty DataFrame.")
        ts = _timestamps(df, self.ts_col)
        if not ts.is_monotonic_increasing:
            order = np.argsort(ts.values, kind="stable")
            df = df.iloc[order]
            ts = ts[order]
        return df, ts, add_trading_day(df, self.ts_col)

    def _slice(self, df, ts, day, train_days, test_days, fold=0) -> Split:
        train = df[day.isin(train_days).values]
        test = df[day.isin(test_days).values]
        if self.embargo > pd.Timedelta(0) and len(test) and len(train):
            cutoff = ts[day.isin(train_days).values].max() + self.embargo
            test = test[_timestamps(test, self.ts_col) > cutoff]
        return Split(train, test, list(train_days), list(test_days), fold)

    def split(self, df: pd.DataFrame) -> Split:
        """Single chronological IS/OOS split by `oos_fraction` of trading days."""
        df, ts, day = self._prepare(df)
        days = sorted(day.unique())
        if len(days) < 2:
            raise ValueError(f"Need >= 2 trading days to split, found {len(days)}.")
        n_oos = max(1, int(round(len(days) * self.oos_fraction)))
        n_oos = min(n_oos, len(days) - 1)
        return self._slice(df, ts, day, days[:-n_oos], days[-n_oos:])

    def split_day_list(self, days) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
        """
        Same chronological IS/OOS day split as `split`, but on a plain list of
        trading days (no DataFrame needed). Returns (train_days, test_days).
        """
        days = sorted(pd.Timestamp(d).normalize() for d in days)
        if len(days) < 2:
            raise ValueError(f"Need >= 2 trading days to split, found {len(days)}.")
        n_oos = min(max(1, int(round(len(days) * self.oos_fraction))), len(days) - 1)
        return days[:-n_oos], days[-n_oos:]

    def split_at(self, df: pd.DataFrame, oos_start: str | pd.Timestamp) -> Split:
        """Split so OOS begins on the trading day containing/after `oos_start`."""
        df, ts, day = self._prepare(df)
        cut = pd.Timestamp(oos_start).tz_localize(None).normalize()
        days = sorted(day.unique())
        train_days = [d for d in days if d < cut]
        test_days = [d for d in days if d >= cut]
        if not train_days or not test_days:
            raise ValueError("oos_start leaves IS or OOS empty.")
        return self._slice(df, ts, day, train_days, test_days)

    def walk_forward(
        self,
        df: pd.DataFrame,
        train_days: int,
        test_days: int = 1,
        step_days: int | None = None,
        expanding: bool = False,
    ) -> Iterator[Split]:
        """
        Walk-forward folds over trading days.
        rolling   (expanding=False): fixed-length IS window slides forward.
        expanding (expanding=True) : IS window grows from the first day.
        """
        df, ts, day = self._prepare(df)
        days = sorted(day.unique())
        step = step_days or test_days
        if train_days + test_days > len(days):
            raise ValueError(
                f"train_days+test_days ({train_days + test_days}) exceeds available days ({len(days)})."
            )
        fold, start = 0, 0
        while start + train_days + test_days <= len(days):
            tr_start = 0 if expanding else start
            tr = days[tr_start : start + train_days]
            te = days[start + train_days : start + train_days + test_days]
            yield self._slice(df, ts, day, tr, te, fold)
            fold += 1
            start += step


# Evaluation
FitFn = Callable[[pd.DataFrame], Any]            # IS data -> params / fitted model
ScoreFn = Callable[[pd.DataFrame, Any], float]   # (data, params) -> metric


@dataclass
class FoldResult:
    fold: int
    is_score: float
    oos_score: float
    params: Any
    n_is: int
    n_oos: int

    @property
    def degradation(self) -> float:
        """OOS/IS ratio. ~1 = holds up, <<1 = overfit, <0 = sign flip."""
        return np.nan if self.is_score == 0 else self.oos_score / self.is_score


@dataclass
class EvaluationReport:
    folds: list[FoldResult] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                dict(fold=f.fold, is_score=f.is_score, oos_score=f.oos_score,
                     degradation=f.degradation, n_is=f.n_is, n_oos=f.n_oos)
                for f in self.folds
            ]
        ).set_index("fold")

    def summary(self) -> str:
        t = self.to_frame()
        return (
            f"folds={len(t)} | mean IS={t.is_score.mean():.4f} | "
            f"mean OOS={t.oos_score.mean():.4f} | "
            f"mean OOS/IS={t.degradation.mean():.2f} | "
            f"OOS>0 in {(t.oos_score > 0).mean():.0%} of folds"
        )


def evaluate(split: Split, fit_fn: FitFn, score_fn: ScoreFn) -> FoldResult:
    """
    Fit on IS only, then score IS and OOS with the frozen params.
    OOS data never touches `fit_fn`.
    """
    if split.train.empty or split.test.empty:
        raise ValueError(f"Fold {split.fold} has an empty IS or OOS set.")
    params = fit_fn(split.train)
    return FoldResult(
        fold=split.fold,
        is_score=float(score_fn(split.train, params)),
        oos_score=float(score_fn(split.test, params)),
        params=params,
        n_is=len(split.train),
        n_oos=len(split.test),
    )


def evaluate_walk_forward(
    df: pd.DataFrame,
    fit_fn: FitFn,
    score_fn: ScoreFn,
    splitter: SampleSplitter,
    **wf_kwargs,
) -> EvaluationReport:
    report = EvaluationReport()
    for split in splitter.walk_forward(df, **wf_kwargs):
        print(split.describe())
        report.folds.append(evaluate(split, fit_fn, score_fn))
    return report


# Example usage
if __name__ == "__main__":
    from local_data import load_local, TOP_OF_BOOK

    # Reads .dbn.zst files from $DATA_DIR (HiPerGator). No API calls.
    df = load_local(
        start="2024-01-08T00:00:00Z",
        end="2024-01-13T00:00:00Z",   # end is exclusive
        columns=TOP_OF_BOOK,
    )

    # Toy example: order-book imbalance signal with a fitted threshold.
    def _imbalance(d: pd.DataFrame) -> pd.Series:
        bid, ask = d["bid_sz_00"], d["ask_sz_00"]
        return (bid - ask) / (bid + ask).replace(0, np.nan)

    def _mid(d: pd.DataFrame) -> pd.Series:
        return (d["bid_px_00"] + d["ask_px_00"]) / 2

    def fit_fn(train: pd.DataFrame) -> float:
        # "Fit" = pick threshold from IS only
        return float(_imbalance(train).abs().quantile(0.9))

    def score_fn(d: pd.DataFrame, thr: float, horizon: int = 100) -> float:
        sig = np.sign(_imbalance(d)).where(_imbalance(d).abs() > thr, 0.0)
        fwd = _mid(d).shift(-horizon) - _mid(d)
        pnl = (sig * fwd).dropna()
        return float(pnl.mean())  # mean fwd move in price points per signal

    splitter = SampleSplitter(oos_fraction=0.4, embargo="5min")

    # 1) Single IS/OOS split
    s = splitter.split(df)
    print(s.describe())
    r = evaluate(s, fit_fn, score_fn)
    print(f"IS={r.is_score:.5f}  OOS={r.oos_score:.5f}  OOS/IS={r.degradation:.2f}")

    # 2) Walk-forward (3 train days -> 1 test day, rolling)
    rep = evaluate_walk_forward(df, fit_fn, score_fn, splitter, train_days=3, test_days=1)
    print(rep.to_frame())
    print(rep.summary())