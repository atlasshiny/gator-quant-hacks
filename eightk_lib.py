"""Library extracted from the Gator Quant Hacks 8-K starter notebook.
Definitions only: nothing here calls the API at import time. Call init_api() first."""


display = print   # notebook helper; plain print outside Jupyter


import os, json, time, hashlib, re

from dataclasses import dataclass, field

from pathlib import Path

from getpass import getpass

import numpy as np

import pandas as pd

import requests

import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter

from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday, nearest_workday, USMartinLutherKingJr,
                                    USPresidentsDay, GoodFriday, USMemorialDay, USLaborDay, USThanksgivingDay)

from pandas.tseries.offsets import CustomBusinessDay

BASE_URL = "https://api.massive.com"

def load_api_key(name: str = "MASSIVE_API_KEY") -> str:
    """Environment first, then a `.env` file beside the notebook, then a prompt. Never paste a key into a cell."""
    key = (os.environ.get(name) or "").strip()
    env_file = Path(".env")
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith(f"{name}="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        key = getpass(f"{name} (https://massive.com/dashboard/keys): ").strip()
    return key

def api_get(path_or_url: str, params: dict | None = None) -> dict:
    """GET one page from the Massive REST API. Cached on disk by the full URL, so reruns are free."""
    url = path_or_url if path_or_url.startswith("http") else BASE_URL + path_or_url
    full_url = requests.Request("GET", url, params=params).prepare().url
    cache_file = CACHE_DIR / (hashlib.sha1(full_url.encode()).hexdigest() + ".json")
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    for attempt in range(10):
        resp = SESSION.get(full_url, timeout=60)
        if resp.status_code in (429, 500, 502, 503, 504):   # rate-limited or a transient server error
            retry_after = resp.headers.get("Retry-After", "")
            time.sleep(float(retry_after) if retry_after.isdigit() else min(2 ** attempt, 20))
            continue
        break
    resp.raise_for_status()
    payload = resp.json()
    cache_file.write_text(json.dumps(payload))
    return payload

def api_get_all(path: str, params: dict | None = None, max_pages: int = 500) -> list[dict]:
    """Follow `next_url` pagination and return every result row."""
    payload = api_get(path, params)
    rows = list(payload.get("results") or [])
    pages = 1
    while payload.get("next_url") and pages < max_pages:
        payload = api_get(payload["next_url"])
        rows.extend(payload.get("results") or [])
        pages += 1
    return rows


CACHE_DIR = Path(".massive_cache")
CACHE_DIR.mkdir(exist_ok=True)
SESSION = requests.Session()
API_KEY = None


def init_api(key: str | None = None) -> str:
    """Load the key (arg, env var, .env, or prompt) and attach it to the HTTP session."""
    global API_KEY
    API_KEY = key or load_api_key()
    SESSION.headers["Authorization"] = f"Bearer {API_KEY}"
    return API_KEY


STUDY_START, STUDY_END = "2024-01-01", "2025-12-31"      # in-sample
OOS_START, OOS_END = "2026-01-01", "2026-08-31"          # out-of-sample
HOLDOUT_START, HOLDOUT_END = "2023-06-01", "2023-08-31"  # sealed window (judges)

TOP_100 = """
AAPL ABBV ABT ACN ADBE AIG AMD AMGN AMT AMZN AVGO AXP BA BAC BK BKNG BLK BMY BRK.B C
CAT CHTR CL CMCSA COF COP COST CRM CSCO CVS CVX DE DHR DIS DUK EMR FDX GD GE GILD
GM GOOGL GS HD HON IBM INTC INTU ISRG JNJ JPM KO LIN LLY LMT LOW MA MCD MDLZ MDT
MET META MMM MO MRK MS MSFT NEE NFLX NKE NOW NVDA ORCL PEP PFE PG PLTR PM PYPL QCOM
RTX SBUX SCHW SO T TGT TMO TMUS TSLA TXN UBER UNH UNP UPS USB V VZ WFC WMT XOM
""".split()

EXPIRY_BUCKETS = {
    "1m":   (21, 45, 30),
    "2m":   (46, 80, 60),
    "3-6m": (90, 180, 120),
}

BASELINE_BUCKET = "3-6m"

HORIZONS = [1, 2, 3, 5, 10, 21, 42, 63]

OTM_PCT = 0.05

OTM_GRID = [0.03, 0.05, 0.10]

ENTRY = "post"

RISK_FREE = 0.04

STRIKE_WINDOW = 0.25

MAX_STALE_SESSIONS = 3

MAX_EVENTS = None

RUN_PLACEBO = True

N_PLACEBO = 120

PLACEBO_GAP_DAYS = 30

FETCH_ACCEPTANCE_TIMES = True

SEC_USER_AGENT = "GatorQuantHacks team-name your@email.edu"

RUN_HOLDOUT = False


class NYSEHolidays(AbstractHolidayCalendar):
    """NYSE full-day closures. Differs from the federal calendar: Good Friday closed, Columbus and
    Veterans Day open, no observance when New Year's Day falls on a Saturday."""
    rules = [
        Holiday("New Year's Day", month=1, day=1,
                observance=lambda d: d + pd.Timedelta(days=1) if d.weekday() == 6 else d),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
        Holiday("National day of mourning, President Carter", year=2025, month=1, day=9),
    ]

def trading_sessions(start, end) -> pd.DatetimeIndex:
    holidays = NYSEHolidays().holidays(pd.Timestamp(start) - pd.Timedelta(days=7), pd.Timestamp(end) + pd.Timedelta(days=7))
    return pd.bdate_range(start, end, freq=CustomBusinessDay(holidays=holidays))

CAL = trading_sessions("2021-06-01", "2027-12-31")

TODAY = pd.Timestamp.today().normalize()

LAST_SESSION = CAL[CAL.searchsorted(TODAY, side="right") - 1]

def session_on_or_after(day) -> pd.Timestamp:
    return CAL[CAL.searchsorted(pd.Timestamp(day), side="left")]

def session_before(day) -> pd.Timestamp:
    return CAL[CAL.searchsorted(pd.Timestamp(day), side="left") - 1]

def sessions_between(a, b) -> int:
    """Number of sessions strictly after `a` up to and including `b`."""
    return int(CAL.searchsorted(pd.Timestamp(b), side="right") - CAL.searchsorted(pd.Timestamp(a), side="right"))


def normalize_ticker(t) -> str | None:
    """Filings write share classes as BRK/B or BRK.B; Massive's options use BRK.B as the underlying."""
    if not isinstance(t, str) or not t.strip():
        return None
    return t.strip().upper().replace("/", ".")

def fetch_disclosures(tag: str, start: str, end: str) -> pd.DataFrame:
    rows = api_get_all("/stocks/filings/8-K/vX/disclosures", {
        "tertiary_category": tag, "filing_date.gte": start, "filing_date.lte": end,
        "limit": 1000, "sort": "filing_date.asc",
    })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df

def build_events(tag: str, start: str, end: str, universe: list[str]) -> pd.DataFrame:
    """One row per (filer, filing date) in the universe, with the filing session and the session before it."""
    raw = fetch_disclosures(tag, start, end)
    print(f"{len(raw):,} '{tag}' disclosures across all filers, {start}..{end}")
    ex = raw.explode("tickers").rename(columns={"tickers": "ticker"})
    ex["ticker"] = ex["ticker"].map(normalize_ticker)
    ex = ex[ex["ticker"].isin(universe)]
    ev = (ex.sort_values(["cik", "filing_date"])
            .groupby(["cik", "filing_date"], as_index=False)
            .agg(ticker=("ticker", "first"), accession_number=("accession_number", "first"),
                 filing_url=("filing_url", "first"), supporting_text=("supporting_text", "first"))
            .sort_values("filing_date").reset_index(drop=True))
    ev["t_0"] = ev["filing_date"].map(session_on_or_after)      # the filing session: first session on or after the filing date
    ev["t_pre"] = ev["t_0"].map(session_before)                 # the session before it: the chain here cannot know the news
    ev["days_since_prev_event"] = ev.groupby("ticker")["filing_date"].diff().dt.days
    return ev


def option_bars(opt_ticker: str, start, end) -> pd.DataFrame:
    """Daily bars for one contract: index = session, columns close and volume. Only sessions where it traded."""
    rows = api_get_all(f"/v2/aggs/ticker/{opt_ticker}/range/1/day/{pd.Timestamp(start):%Y-%m-%d}/{pd.Timestamp(end):%Y-%m-%d}",
                       {"adjusted": "false", "sort": "asc", "limit": 50000})
    if not rows:
        return pd.DataFrame(columns=["close", "volume"], index=pd.DatetimeIndex([], name="session"))
    idx = (pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True)
             .tz_convert("America/New_York").normalize().tz_localize(None))
    return pd.DataFrame({"close": [float(r["c"]) for r in rows], "volume": [float(r.get("v") or 0) for r in rows]},
                        index=pd.DatetimeIndex(idx, name="session"))

def fetch_chain(ticker: str, as_of: pd.Timestamp, dte_lo: int, dte_hi: int) -> pd.DataFrame:
    """Every standard contract that existed on `as_of` with an expiry `dte_lo`..`dte_hi` days out."""
    rows = api_get_all("/v3/reference/options/contracts", {
        "underlying_ticker": ticker, "as_of": as_of.strftime("%Y-%m-%d"),
        "expiration_date.gte": (as_of + pd.Timedelta(days=dte_lo)).strftime("%Y-%m-%d"),
        "expiration_date.lte": (as_of + pd.Timedelta(days=dte_hi)).strftime("%Y-%m-%d"),
        "limit": 1000,
    })
    chain = pd.DataFrame(rows)
    if chain.empty:
        return chain
    if "shares_per_contract" in chain:
        chain = chain[chain["shares_per_contract"].fillna(100) == 100]      # drop post-split non-standard series
    chain = chain[["ticker", "contract_type", "strike_price", "expiration_date"]].copy()
    chain["expiration_date"] = pd.to_datetime(chain["expiration_date"])
    chain["dte"] = (chain["expiration_date"] - as_of).dt.days
    chain["strike_price"] = chain["strike_price"].astype(float)
    return chain.reset_index(drop=True)

def paired_strikes(e: pd.DataFrame) -> np.ndarray:
    both = e.groupby("strike_price")["contract_type"].nunique()
    return both[both == 2].index.to_numpy(dtype=float)

def contract(e: pd.DataFrame, strike: float, kind: str) -> str:
    return e[(e.strike_price == strike) & (e.contract_type == kind)]["ticker"].iloc[0]

def last_close_on_or_before(opt_ticker: str, day: pd.Timestamp, lookback_days: int = 7) -> float | None:
    bars = option_bars(opt_ticker, day - pd.Timedelta(days=lookback_days), day)
    return float(bars["close"].iloc[-1]) if len(bars) else None

def locate_spot(chain: pd.DataFrame, day: pd.Timestamp, max_iter: int = 8) -> dict | None:
    """Recover the stock price on `day` from the option chain alone, via put-call parity.

    At any strike K with both a call and a put, S = K·e^(−rT) + C − P. The strike that minimises |C − P|
    is at-the-money, and the estimate is best there because both legs are liquid. We start at the
    median strike of the nearest expiry (new weekly series are listed around the current price, so that
    median is already close), then step to the strike nearest our estimate until it stops moving.
    """
    near = chain[chain.dte >= 3]
    if near.empty:
        return None
    near = near[near.dte == near.dte.min()]
    strikes = paired_strikes(near)
    if len(strikes) < 3:
        return None
    T = near.dte.iloc[0] / 365
    k, tried, est = float(np.median(strikes)), set(), None
    k = strikes[np.abs(strikes - k).argmin()]
    for _ in range(max_iter):
        tried.add(k)
        c = last_close_on_or_before(contract(near, k, "call"), day)
        p = last_close_on_or_before(contract(near, k, "put"), day)
        if c is None or p is None:                                   # an illiquid strike: try the next-nearest untried one
            rest = [s for s in strikes if s not in tried]
            if not rest:
                break
            k = rest[int(np.abs(np.array(rest) - k).argmin())]
            continue
        est = k * np.exp(-RISK_FREE * T) + c - p
        k_new = strikes[np.abs(strikes - est).argmin()]
        if k_new == k or k_new in tried:
            break
        k = k_new
    if est is None:
        return None
    return {"spot": float(est), "strike": float(k), "expiry": near.expiration_date.iloc[0], "dte": int(near.dte.iloc[0])}

def pick_expiry(chain: pd.DataFrame, lo: int, hi: int, target: int) -> pd.Timestamp | None:
    cand = chain[(chain.dte >= lo) & (chain.dte <= hi)]
    if cand.empty:
        return None
    dte_of = cand.groupby("expiration_date")["dte"].first()
    ok = [x for x, g in cand.groupby("expiration_date") if len(paired_strikes(g)) >= 3]   # needs a real chain, not a stub
    if not ok:
        return None
    return min(ok, key=lambda x: abs(dte_of[x] - target))

def select_strikes(e: pd.DataFrame, spot: float, otm_pcts: list[float]) -> dict[str, float] | None:
    """ATM strike K (nearest to spot, both legs listed) plus, for each OTM level, the call strike U at or
    above spot·(1+pct) and the put strike L at or below spot·(1−pct)."""
    both = paired_strikes(e)
    both = both[(both >= spot * (1 - STRIKE_WINDOW)) & (both <= spot * (1 + STRIKE_WINDOW))]
    if len(both) == 0:
        return None
    calls = np.sort(e.loc[e.contract_type == "call", "strike_price"].unique())
    puts = np.sort(e.loc[e.contract_type == "put", "strike_price"].unique())
    out = {"K": float(both[np.abs(both - spot).argmin()])}
    for pct in otm_pcts:
        up, dn = calls[calls >= spot * (1 + pct)], puts[puts <= spot * (1 - pct)]
        out[f"U{pct}"] = float(up.min()) if len(up) else float(calls.max())
        out[f"L{pct}"] = float(dn.max()) if len(dn) else float(puts.min())
    return out


@dataclass
class Leg:
    ticker: str
    kind: str                 # "call" or "put"
    strike: float
    bars: pd.DataFrame        # close, volume by session (only sessions with trades)

    def mark(self, day: pd.Timestamp) -> float:
        """Last traded close on or before `day`, if it is at most MAX_STALE_SESSIONS sessions old."""
        b = self.bars.loc[: pd.Timestamp(day)]
        if b.empty or sessions_between(b.index[-1], day) > MAX_STALE_SESSIONS:
            return np.nan
        return float(b["close"].iloc[-1])

    def volume_on(self, day: pd.Timestamp) -> float:
        return float(self.bars["volume"].get(pd.Timestamp(day), 0.0))

@dataclass
class PricedEvent:
    """One event × one expiry bucket: the chain-derived spot, the expiry and every leg a strategy may need."""
    ticker: str
    event_date: pd.Timestamp
    t_pre: pd.Timestamp
    t_0: pd.Timestamp
    bucket: str
    expiry: pd.Timestamp
    expiry_session: pd.Timestamp        # last session on or before expiry
    spot_pre: float                     # chain-implied spot on t_pre
    strikes: dict[str, float]
    legs: dict[str, Leg]                # "C_K", "P_K", "C_U0.05", "P_L0.05", ...

    def marks(self, day) -> dict[str, float]:
        return {name: leg.mark(day) for name, leg in self.legs.items()}

    def synthetic_spot(self, day, m: dict[str, float] | None = None) -> float:
        """Stock price implied by the ATM pair on `day`: K·e^(−rT) + C_K − P_K. Exact at expiry."""
        m = m if m is not None else self.marks(day)
        T = max((self.expiry - pd.Timestamp(day)).days, 0) / 365
        return self.strikes["K"] * np.exp(-RISK_FREE * T) + m["C_K"] - m["P_K"]

def price_event(ticker: str, t_pre: pd.Timestamp, t_0: pd.Timestamp, event_date: pd.Timestamp,
                buckets: dict, otm_pcts: list[float]) -> tuple[list[PricedEvent], list[str]]:
    dte_hi = max(b[1] for b in buckets.values())
    chain = fetch_chain(ticker, t_pre, 2, dte_hi)
    if chain.empty:
        return [], ["no option chain as of the pre-event session"]
    loc = locate_spot(chain, t_pre)
    if loc is None:
        return [], ["could not recover spot from the chain (no liquid near-dated pair)"]
    priced, notes = [], []
    for name, (lo, hi, target) in buckets.items():
        expiry = pick_expiry(chain, lo, hi, target)
        if expiry is None:
            notes.append(f"{name}: no expiry {lo}-{hi} days out")
            continue
        e = chain[chain.expiration_date == expiry]
        strikes = select_strikes(e, loc["spot"], otm_pcts)
        if strikes is None:
            notes.append(f"{name}: no paired strikes near spot")
            continue
        wanted = {"C_K": ("call", strikes["K"]), "P_K": ("put", strikes["K"])}
        for pct in otm_pcts:
            wanted[f"C_U{pct}"] = ("call", strikes[f"U{pct}"])
            wanted[f"P_L{pct}"] = ("put", strikes[f"L{pct}"])
        legs = {}
        for key, (kind, k) in wanted.items():
            tk = contract(e, k, kind)
            legs[key] = Leg(tk, kind, k, option_bars(tk, t_pre - pd.Timedelta(days=10), expiry))
        pe = PricedEvent(ticker, event_date, t_pre, t_0, name, expiry, CAL[CAL.searchsorted(expiry, side="right") - 1],
                         loc["spot"], strikes, legs)
        if np.isnan(pe.legs["C_K"].mark(t_pre)) or np.isnan(pe.legs["P_K"].mark(t_pre)):
            notes.append(f"{name}: ATM pair did not trade on or near the pre-event session")
            continue
        priced.append(pe)
    return priced, notes

def price_events(ev: pd.DataFrame, buckets: dict = None, otm_pcts: list[float] = None, label: str = "events") -> tuple[list[PricedEvent], pd.DataFrame]:
    buckets = buckets or EXPIRY_BUCKETS
    otm_pcts = otm_pcts or OTM_GRID
    priced, dropped, t0 = [], [], time.time()
    for n, row in enumerate(ev.itertuples(index=False), 1):
        got, notes = price_event(row.ticker, row.t_pre, row.t_0, row.event_date if hasattr(row, "event_date") else row.filing_date,
                                 buckets, otm_pcts)
        priced += got
        dropped += [(row.ticker, row.t_0, note) for note in notes]
        if n % 25 == 0:
            print(f"  {label}: {n}/{len(ev)} events, {time.time() - t0:.0f}s")
    print(f"{label}: {len(ev)} events -> {len(priced)} priced (event, bucket) pairs; {len(dropped)} drops; {time.time() - t0:.0f}s")
    return priced, pd.DataFrame(dropped, columns=["ticker", "t_0", "reason"])

def summarize_priced(priced: list[PricedEvent]) -> pd.DataFrame:
    rows = []
    for pe in priced:
        m = pe.marks(pe.t_pre)
        rows.append({"ticker": pe.ticker, "event_date": pe.event_date, "t_pre": pe.t_pre, "t_0": pe.t_0, "bucket": pe.bucket,
                     "expiry": pe.expiry, "dte": (pe.expiry - pe.t_pre).days, "spot_pre": pe.spot_pre, "K": pe.strikes["K"],
                     "moneyness": pe.strikes["K"] / pe.spot_pre - 1, "call": pe.legs["C_K"].ticker, "put": pe.legs["P_K"].ticker,
                     "call_px": m["C_K"], "put_px": m["P_K"], "implied_move": (m["C_K"] + m["P_K"]) / pe.spot_pre,
                     "spot_gap": pe.synthetic_spot(pe.t_pre, m) / pe.spot_pre - 1,     # this bucket's parity spot vs the near-dated one
                     "atm_volume": pe.legs["C_K"].volume_on(pe.t_pre) + pe.legs["P_K"].volume_on(pe.t_pre)})
    return pd.DataFrame(rows)


STRATEGIES = ["stock", "long_call", "covered_call", "protective_put", "collar", "cash_secured_put"]

STRATEGY_LABEL = {"stock": "Stock only (synthetic)", "long_call": "1 · Long call", "covered_call": "2 · Covered call",
                  "protective_put": "3 · Protective put", "collar": "4 · Collar", "cash_secured_put": "5 · Cash-secured put"}

def strategy_pnl(m_e: dict, m_x: dict, S_e: float, S_x: float, otm: float) -> dict[str, float]:
    """P&L per $1 of stock at entry, from entry marks `m_e` to exit marks `m_x`.

    The 100 shares behind the covered call, protective put and collar are replaced by the synthetic
    stock (long ATM call, short ATM put), which is what `S_e` and `S_x` are.
    """
    dS = S_x - S_e
    dC_U = m_x[f"C_U{otm}"] - m_e[f"C_U{otm}"]     # the OTM call we sell
    dP_L = m_x[f"P_L{otm}"] - m_e[f"P_L{otm}"]     # the OTM put we buy (or sell, cash-secured)
    dC_K = m_x["C_K"] - m_e["C_K"]                 # the ATM call we buy
    return {
        "stock":            dS / S_e,
        "long_call":        dC_K / S_e,
        "covered_call":     (dS - dC_U) / S_e,
        "protective_put":   (dS + dP_L) / S_e,
        "collar":           (dS + dP_L - dC_U) / S_e,
        "cash_secured_put": (-dP_L) / S_e,
    }

def evaluate(priced: list[PricedEvent], otm_pcts: list[float] = None) -> pd.DataFrame:
    """Long table: one row per (event, bucket, entry, OTM level, horizon) with every strategy's P&L."""
    otm_pcts = otm_pcts or OTM_GRID
    rows = []
    for pe in priced:
        i0 = CAL.get_loc(pe.t_0)
        exits = {0: pe.t_0}
        exits.update({h: CAL[i0 + h] for h in HORIZONS if i0 + h < len(CAL) and CAL[i0 + h] <= pe.expiry_session})
        exits["exp"] = pe.expiry_session
        for entry, e_day in (("pre", pe.t_pre), ("post", pe.t_0)):
            m_e = pe.marks(e_day)
            S_e = pe.synthetic_spot(e_day, m_e)
            if np.isnan(S_e):
                continue
            implied = (m_e["C_K"] + m_e["P_K"]) / S_e
            dte_sessions = sessions_between(e_day, pe.expiry_session)
            for h, x_day in exits.items():
                if x_day > LAST_SESSION:
                    continue
                m_x = pe.marks(x_day)
                S_x = pe.synthetic_spot(x_day, m_x)
                held = sessions_between(e_day, x_day)
                base = {"ticker": pe.ticker, "event_date": pe.event_date, "t_0": pe.t_0, "bucket": pe.bucket, "expiry": pe.expiry,
                        "entry": entry, "entry_date": e_day, "horizon": h, "exit_date": x_day, "sessions_held": held,
                        "dte_sessions": dte_sessions, "S_entry": S_e, "S_exit": S_x, "realized": S_x / S_e - 1,
                        "implied_move": implied,
                        "implied_scaled": implied * np.sqrt(held / dte_sessions) if dte_sessions else np.nan}
                for otm in otm_pcts:
                    rows.append(dict(base, otm=otm, **strategy_pnl(m_e, m_x, S_e, S_x, otm)))
    res = pd.DataFrame(rows)
    res["ratio"] = res["realized"].abs() / res["implied_scaled"]
    return res


def bootstrap_ci(x, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]).tolist())

def slice_results(res: pd.DataFrame, bucket: str, entry: str, otm: float) -> pd.DataFrame:
    return res[(res.bucket == bucket) & (res.entry == entry) & (res.otm == otm)]

def scoreboard(res: pd.DataFrame, bucket: str = None, entry: str = None, otm: float = None,
               horizons=None, strategies=STRATEGIES) -> pd.DataFrame:
    """Rows: strategy. Columns: horizon. Cells: mean P&L per $1 of spot, with n and a 95% bootstrap CI."""
    r = slice_results(res, bucket or BASELINE_BUCKET, entry or ENTRY, otm or OTM_PCT)
    horizons = horizons or [h for h in HORIZONS + ["exp"] if h in set(r.horizon)]
    rows = []
    for s in strategies:
        for h in horizons:
            x = r.loc[r.horizon == h, s].dropna()
            lo, hi = bootstrap_ci(x)
            rows.append({"strategy": s, "horizon": h, "n": len(x), "mean": x.mean(), "ci_lo": lo, "ci_hi": hi,
                         "median": x.median(), "hit_rate": (x > 0).mean()})
    return pd.DataFrame(rows)

def difference_board(res_a: pd.DataFrame, res_b: pd.DataFrame, bucket: str = None, entry: str = None, otm: float = None,
                     strategies=STRATEGIES, seed: int = 1) -> pd.DataFrame:
    """Mean P&L of group a minus group b, per strategy and horizon, with a bootstrap CI on the difference."""
    a = slice_results(res_a, bucket or BASELINE_BUCKET, entry or ENTRY, otm or OTM_PCT)
    b = slice_results(res_b, bucket or BASELINE_BUCKET, entry or ENTRY, otm or OTM_PCT)
    rng, rows = np.random.default_rng(seed), []
    for s in strategies:
        for h in [h for h in HORIZONS + ["exp"] if h in set(a.horizon) & set(b.horizon)]:
            xa, xb = a.loc[a.horizon == h, s].dropna().to_numpy(), b.loc[b.horizon == h, s].dropna().to_numpy()
            row = {"strategy": s, "horizon": h, "n_a": len(xa), "n_b": len(xb)}
            if len(xa) >= 5 and len(xb) >= 5:
                d = rng.choice(xa, (2000, len(xa))).mean(1) - rng.choice(xb, (2000, len(xb))).mean(1)
                row.update(mean_a=xa.mean(), mean_b=xb.mean(), difference=xa.mean() - xb.mean(),
                           ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5))
            rows.append(row)
    return pd.DataFrame(rows)

def fmt_board(board: pd.DataFrame, value: str = "mean", star_if_ci_excludes_zero: bool = True) -> pd.DataFrame:
    """Wide display table: strategies × horizons, values in % of spot, * where the 95% CI excludes zero."""
    def cell(row):
        v = row[value]
        if pd.isna(v):
            return ""
        star = "*" if star_if_ci_excludes_zero and pd.notna(row.get("ci_lo")) and (row["ci_lo"] > 0 or row["ci_hi"] < 0) else ""
        return f"{v * 100:+.2f}%{star}"
    wide = board.assign(cell=board.apply(cell, axis=1)).pivot(index="strategy", columns="horizon", values="cell")
    wide = wide.reindex([s for s in STRATEGIES if s in wide.index]).rename(index=STRATEGY_LABEL)
    wide.columns = [f"h={c}" if c != "exp" else "expiry" for c in wide.columns]
    return wide.rename_axis(f"P&L per $1 spot", axis=0)

SERIES = {"events": "#2a78d6", "placebo": "#eb6834", "oos": "#1baf7a", "in-sample": "#2a78d6"}

def style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.figure.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)

def plot_scoreboards(boards: dict[str, pd.DataFrame], title: str, strategies=STRATEGIES):
    """Small multiples: one panel per strategy, one line per group, band = 95% bootstrap CI of the first group."""
    horizons = [h for h in HORIZONS + ["exp"] if any(h in set(b.horizon) for b in boards.values())]
    x = np.arange(len(horizons))
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True)
    for ax, s in zip(axes.ravel(), strategies):
        for i, (name, b) in enumerate(boards.items()):
            t = b[b.strategy == s].set_index("horizon").reindex(horizons)
            c = SERIES.get(name, INK2)
            if i == 0:
                ax.fill_between(x, t.ci_lo.to_numpy(float) * 100, t.ci_hi.to_numpy(float) * 100, color=c, alpha=0.10, linewidth=0)
            ax.plot(x, t["mean"].to_numpy(float) * 100, color=c, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, linestyle="-" if i == 0 else "--",
                    label=f"{name} (n={int(t.n.max())})")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(x, [str(h) if h != "exp" else "exp" for h in horizons])
        style(ax, STRATEGY_LABEL[s], "sessions after the filing session" if ax in axes[1] else "",
              "mean P&L, % of spot" if ax in axes[:, 0] else "")
    axes[0, 0].legend(frameon=False, labelcolor=INK2, fontsize=8, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def sample_placebo(ev: pd.DataFrame, n: int, start: str, end: str, gap_days: int = PLACEBO_GAP_DAYS, seed: int = 7) -> pd.DataFrame:
    """Ordinary sessions for the same tickers, drawn in proportion to how often each ticker has an event,
    at least `gap_days` from any of that ticker's events."""
    rng = np.random.default_rng(seed)
    sessions = CAL[(CAL >= pd.Timestamp(start)) & (CAL <= pd.Timestamp(end))]
    by_ticker = ev.groupby("ticker")["filing_date"].apply(list)
    rows = []
    for t in rng.choice(ev["ticker"].to_numpy(), size=n, replace=True):
        for _ in range(25):
            d = sessions[rng.integers(len(sessions))]
            if all(abs((d - a).days) > gap_days for a in by_ticker.get(t, [])):
                rows.append({"ticker": t, "filing_date": d, "event_date": d, "t_0": d, "t_pre": session_before(d)})
                break
    return pd.DataFrame(rows)


def rank_strategies(diff: pd.DataFrame, horizons=(21, 42, "exp")) -> pd.DataFrame:
    """Rank the five strategies by their event-minus-placebo edge, averaged over the given horizons."""
    d = diff[diff.horizon.isin(horizons) & (diff.strategy != "stock")]
    out = (d.groupby("strategy").agg(edge=("difference", "mean"), horizons_ci_excludes_0=("ci_lo", lambda s: int(((s > 0) | (d.loc[s.index, "ci_hi"] < 0)).sum())),
                                     n_events=("n_a", "max"))
             .sort_values("edge", ascending=False))
    out["edge"] = out["edge"].map(lambda v: f"{v * 100:+.2f}%")
    return out.rename(index=STRATEGY_LABEL)


def decay_table(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in [h for h in HORIZONS + ["exp"] if h in set(r.horizon)]:
        x = r.loc[r.horizon == h, "ratio"].dropna()
        lo, hi = bootstrap_ci(x)
        rows.append({"horizon": h, "n": len(x), "mean_ratio": x.mean(), "ci_lo": lo, "ci_hi": hi,
                     "median_ratio": x.median(), "share_above_1": (x > 1).mean()})
    return pd.DataFrame(rows).set_index("horizon")

def fmt_decay(tbl: pd.DataFrame) -> pd.DataFrame:
    out = tbl.copy()
    for c in ["mean_ratio", "ci_lo", "ci_hi", "median_ratio"]:
        out[c] = out[c].map(lambda v: f"{v:.2f}")
    out["share_above_1"] = out["share_above_1"].map(lambda v: f"{v:.0%}")
    return out


def run_study(tag: str, start: str, end: str, universe: list[str] = TOP_100, buckets: dict = EXPIRY_BUCKETS,
              max_events: int | None = None, label: str = None) -> dict:
    """The whole pipeline on a fresh window. Returns the events, priced legs, long results and the scoreboard."""
    ev = build_events(tag, start, end, universe)
    ev["event_date"] = ev["filing_date"]
    if max_events:
        ev = ev.head(max_events).copy()
    pr, drops = price_events(ev, buckets, label=label or f"{tag} {start}..{end}")
    res = evaluate(pr)
    return {"events": ev, "priced": pr, "dropped": drops, "results": res, "board": scoreboard(res)}
