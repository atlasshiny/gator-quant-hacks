"""
Risk manager for the ES burst-reversion strategy (single instrument, seconds-long
holds, marketable orders, netting account).

Built for evaluation/backtest/nautilus_strategy.py (XGBoostReversionStrategy on the
`strategy` branch). Pure Python with no Nautilus import, so it can be unit-tested on
its own (self-test: `python -m python.es_risk_manager` from the repo root). It is not
wired into the strategy yet; the strategy calls it at five points:

    session start           rm.on_session_start(trading_day, equity)   daily reset, drawdown ladder
    every book update       rm.on_book(book)                           data health + short-horizon vol
    in a position           rm.check_exit(book, position)              price stop, time stop, forced flat
    before every entry      rm.pre_trade_check(book, side, ...)        allowed? how many contracts?
    order filled            rm.on_fill(side, qty, fill_px, decision)   slippage beyond the touch
    position closed         rm.on_trade_closed(net_pnl_usd, qty, ts)   P&L, losing streaks, edge monitor

Why these controls (see the risk section of docs/writeup.tex):
  * Price risk per trade is small (1-2 contracts, held seconds). The real ways this
    strategy loses money are: paying the spread on every trade (~1.3 ticks round
    trip), fading bursts that turn out to be informed (adverse selection), trading
    into thin or stressed books, latency, and a model that stops working.
  * So most controls watch COST, BOOK STATE and EDGE, not just P&L.

All $ values assume $1,000,000 capital and 1 ES contract = $50/point, tick = 0.25.
Values marked [calibrate] are starting points to replace with backtest measurements.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

CT = ZoneInfo("America/Chicago")    # CME session clock
ET = ZoneInfo("America/New_York")   # economic release clock


# ============================================================================ #
# Contract and configuration
# ============================================================================ #
@dataclass(frozen=True)
class ContractSpec:
    tick_size: float = 0.25
    multiplier: float = 50.0          # $ per index point
    fee_per_side: float = 2.00        # $ per contract per fill (matches evaluation/metrics.py)

    @property
    def tick_value(self) -> float:    # $12.50
        return self.tick_size * self.multiplier

    def round_trip_cost_ticks(self, spread_ticks: float = 1.0) -> float:
        """Crossing the spread in and out costs one full spread, plus two fees."""
        return spread_ticks + 2 * self.fee_per_side / self.tick_value   # 1 + 0.32 = 1.32 ticks


@dataclass
class RiskConfig:
    capital: float = 1_000_000.0

    # --- Position and exposure limits ---
    max_contracts: int = 2                # 2 x ~$240k notional = ~0.5x leverage
    max_touch_fraction: float = 0.10      # order <= 10% of displayed size at the price we hit
    min_touch_size: int = 20              # [calibrate] skip if fewer contracts at the touch
    allow_pyramiding: bool = False        # one position at a time (matches the strategy)

    # --- Per-trade risk budget (replaces a portfolio volatility target) ---
    risk_per_trade_frac: float = 0.00025  # 0.025% of equity ($250 on $1M) lost if the stop is hit
    min_stop_ticks: float = 4.0           # never tighter than 1 point
    stop_sigma_mult: float = 3.0          # stop = max(min_stop, 3 x expected move over the hold)
    holding_seconds: float = 1.5          # model horizon h (time stop)
    min_edge_to_cost: float = 1.5         # predicted move must be >= 1.5x round-trip cost

    # --- Short-horizon volatility regime ---
    # "Recent" vol (5-minute EWMA) is compared with what is NORMAL FOR THIS TIME OF DAY:
    # the same 30-minute slot of the session on previous days. The open is always busier
    # than overnight, so comparing against an all-day average would cut size every morning.
    vol_short_halflife_s: float = 300.0   # "recent" = ~5 minutes
    vol_bucket_minutes: int = 30          # time-of-day slots, counted from the 17:00 CT open
    vol_baseline_halflife_days: float = 10.0
    vol_baseline_min_days: int = 3        # slot needs 3 sessions of history before it is used
    vol_long_halflife_s: float = 23_400.0 # fallback baseline until the slot has enough history
    vol_warmup_s: int = 1_800
    vol_reduce_ratio: float = 1.5
    vol_block_ratio: float = 2.5

    # --- Loss limits (fractions of equity at the start of the session) ---
    daily_loss_halt_frac: float = 0.02    # lose 2% ($20k on $1M) in a session -> stop until tomorrow
    dd_reduce_frac: float = 0.02          # -2% from peak -> half size
    dd_flat_frac: float = 0.05            # -5% -> flat, then re-audit
    flat_cooldown_sessions: int = 5
    max_consecutive_losses: int = 8
    loss_streak_pause_s: float = 900.0    # 15-minute pause

    # --- Edge and execution monitors ---
    edge_window_trades: int = 200
    edge_reduce_ticks: float = 0.0        # rolling net ticks/contract <= 0 -> half size
    edge_halt_ticks: float = -0.5         # <= -0.5 -> stop for the day
    slippage_window: int = 100
    slippage_reduce_ticks: float = 0.5    # mean fill beyond the touch > 0.5 tick -> half size
    slippage_halt_ticks: float = 1.0      # > 1 tick -> stop for the day (latency problem)

    # --- Throttles (runaway-loop protection) ---
    max_orders_per_minute: int = 20
    max_orders_per_day: int = 1_000

    # --- Data health and book state ---
    max_feed_lag_ms: float = 100.0        # ts_recv - ts_event
    max_quiet_ms: float = 5_000.0         # no book update for 5 s -> treat as stale
    max_spread_ticks: float = 1.0         # ES is 1 tick wide almost always; wider = stress

    # --- Calendar ---
    flatten_before_break_min: int = 3     # CME maintenance break 16:00-17:00 CT
    no_entry_before_break_min: int = 10
    session_blackouts_ct: tuple = ((time(8, 30), 2, 5), (time(15, 0), 2, 2))  # RTH open, settlement
    news_events_et: tuple = ()            # ((datetime_ET, minutes_before, minutes_after), ...)
    roll_blackouts: tuple = ()            # ((first_trading_day, last_trading_day), ...)
    no_trade_days: tuple = ()             # CME trading days skipped entirely (holidays, early closes)


def default_2024h1_calendar() -> dict:
    """
    Scheduled events for the Jan 8 - Jul 8, 2024 sample, checked 2026-10-04 against:
      FOMC  federalreserve.gov/monetarypolicy/fomccalendars.htm (2024 meetings; statements
            print "For release at 2:00 p.m."); press conference follows at 2:30 p.m. ET.
      CPI   bls.gov/bls/news-release/cpi.htm archive (e.g. cpi_01112024 = Dec 2023 CPI), 8:30 a.m. ET.
      Jobs  bls.gov/bls/news-release/empsit.htm archive (empsit_MMDDYYYY), 8:30 a.m. ET.
      Rolls cmegroup.com/trading/equity-index/rolldates.html: roll date = Monday before the
            third Friday; ES trading terminates 9:30 a.m. ET on the third Friday (contract specs).
            Databento ES.c.0 (calendar rule) keeps the EXPIRING contract until it expires and is
            not back-adjusted, so roll week trades a thinning contract and the price jumps after.
      Holidays / early close: NYSE Group 2024 holiday calendar (Nov 10, 2023 release).
    """
    def ev(d, hh, mm, before, after):
        return (datetime(*d, hh, mm, tzinfo=ET), before, after)
    fomc = [(2024, 1, 31), (2024, 3, 20), (2024, 5, 1), (2024, 6, 12)]
    cpi = [(2024, 1, 11), (2024, 2, 13), (2024, 3, 12), (2024, 4, 10), (2024, 5, 15), (2024, 6, 12)]
    jobs = [(2024, 2, 2), (2024, 3, 8), (2024, 4, 5), (2024, 5, 3), (2024, 6, 7), (2024, 7, 5)]
    news = ([ev(d, 14, 0, 5, 45) for d in fomc]        # 2:00 statement + 2:30 press conference
            + [ev(d, 8, 30, 2, 5) for d in cpi + jobs])
    rolls = ((date(2024, 3, 11), date(2024, 3, 15)),  # ESH4 -> ESM4 (roll Mon 3/11, expiry Fri 3/15)
             (date(2024, 6, 17), date(2024, 6, 21)))  # ESM4 -> ESU4 (roll Mon 6/17, expiry Fri 6/21)
    # Exchange holidays (CME runs shortened Globex sessions) and the July 3 early close.
    # Skipped whole: thin, irregular books and halt times we have not modeled.
    no_trade = (date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29), date(2024, 5, 27),
                date(2024, 6, 19), date(2024, 7, 3), date(2024, 7, 4))
    return dict(news_events_et=tuple(news), roll_blackouts=rolls, no_trade_days=no_trade)


def expected_move_ticks(p_short: float, p_long: float, side: int, price: float,
                        barrier_bps: float = 1.5, spec: ContractSpec | None = None) -> float:
    """
    Turn the XGBoost class probabilities into a predicted move for pre_trade_check's cost gate.

    The model (python/model/train_xgb.py) labels a row "long" when the mid moves up by more
    than barrier_bps over the horizon, "short" when it moves down by more, "flat" otherwise.
    We only know the classed moves are AT LEAST the barrier, and treat flat as zero, so

        E[move in our favour] >= barrier_ticks * (P(our direction) - P(against us))

    is a conservative (lower-bound) estimate. At ES 5,000 the 1.5 bps barrier is 3 ticks, so
    passing the 1.98-tick gate needs P(ours) - P(against) >= 0.66.
    """
    spec = spec or ContractSpec()
    barrier_ticks = barrier_bps / 1e4 * price / spec.tick_size
    p_for, p_against = (p_long, p_short) if side > 0 else (p_short, p_long)
    return barrier_ticks * (p_for - p_against)


# ============================================================================ #
# Inputs and outputs
# ============================================================================ #
@dataclass
class Book:
    ts_event_ns: int
    ts_recv_ns: int
    bid_px: float
    ask_px: float
    bid_sz: float          # contracts at the best bid
    ask_sz: float          # contracts at the best ask

    @property
    def mid(self) -> float:
        return (self.bid_px + self.ask_px) / 2


@dataclass
class Position:
    side: int              # +1 long, -1 short
    qty: int
    entry_px: float
    entry_ns: int
    stop_ticks: float


@dataclass
class Decision:
    allowed: bool
    contracts: int = 0
    stop_ticks: float = 0.0
    scale: float = 1.0
    reasons: list = field(default_factory=list)


def _ct(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).astimezone(CT)


def cme_trading_day(t: datetime) -> date:
    """CME trading day: the session that opens at 17:00 CT belongs to the NEXT calendar day."""
    return (t.astimezone(CT) + timedelta(hours=7)).date()


# ============================================================================ #
# Risk manager
# ============================================================================ #
class ESRiskManager:
    def __init__(self, cfg: RiskConfig | None = None, spec: ContractSpec | None = None):
        self.cfg = cfg or RiskConfig()
        self.spec = spec or ContractSpec()
        c = self.cfg
        # account / drawdown state
        self.equity = c.capital
        self.session_start_equity = c.capital
        self.peak = c.capital
        self.cooldown_left = 0
        self.dd_scale = 1.0
        # daily state
        self.daily_pnl = 0.0
        self.day_halt_reason: str | None = None
        self.orders_today = 0
        self.order_times: deque = deque()
        self.consecutive_losses = 0
        self.pause_until_ns = 0
        # monitors
        self.trade_ticks: deque = deque(maxlen=c.edge_window_trades)
        self.slippage: deque = deque(maxlen=c.slippage_window)
        # data / vol state
        self.last_book: Book | None = None
        self.last_lag_ms = 0.0
        self._last_sec: int | None = None
        self._last_sec_mid: float | None = None
        self._var_s = self._var_l = None
        self._vol_samples = 0
        self._x2_sum = 0.0
        self._tod_var: dict[int, float] = {}       # time-of-day slot -> baseline variance
        self._tod_days: dict[int, int] = {}        # slot -> sessions of history
        self._slot_key = None                      # (trading_day, slot) being accumulated
        self._slot_sum = self._slot_dt = 0.0
        self.kill_switch = False

    # ------------------------------------------------------------------ daily
    def on_session_start(self, trading_day: date, equity: float) -> None:
        """Call at 17:00 CT when a new CME trading day begins."""
        c = self.cfg
        self.equity = equity
        self.session_start_equity = equity
        self.daily_pnl, self.day_halt_reason = 0.0, None
        self.orders_today, self.consecutive_losses, self.pause_until_ns = 0, 0, 0
        self.order_times.clear()
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            if self.cooldown_left == 0:
                self.peak, self.dd_scale = equity, 1.0      # restart after re-audit
            return
        self.peak = max(self.peak, equity)
        dd = equity / self.peak - 1
        if dd <= -c.dd_flat_frac:
            self.cooldown_left, self.dd_scale = c.flat_cooldown_sessions, 0.0
        else:
            self.dd_scale = 0.5 if dd <= -c.dd_reduce_frac else 1.0

    # ------------------------------------------------------------------ market data
    def on_book(self, b: Book) -> None:
        self.last_book = b
        self.last_lag_ms = (b.ts_recv_ns - b.ts_event_ns) / 1e6
        sec = b.ts_event_ns // 1_000_000_000
        if self._last_sec is None:
            self._last_sec, self._last_sec_mid = sec, b.mid
            return
        if sec <= self._last_sec:
            return
        dt = sec - self._last_sec                           # seconds since last sample
        dm = (b.mid - self._last_sec_mid) / self.spec.tick_size
        x2 = dm * dm / dt                                   # variance per second, in ticks^2
        a_s = 1 - 0.5 ** (dt / self.cfg.vol_short_halflife_s)
        a_l = 1 - 0.5 ** (dt / self.cfg.vol_long_halflife_s)
        self._vol_samples += dt
        self._x2_sum += x2 * dt
        if self._var_s is None or self._vol_samples <= self.cfg.vol_warmup_s:  # warm-up: plain average
            self._var_s = self._var_l = self._x2_sum / self._vol_samples
        else:
            self._var_s += a_s * (x2 - self._var_s)
            self._var_l += a_l * (x2 - self._var_l)
        self._last_sec, self._last_sec_mid = sec, b.mid
        if dt <= 60:                                       # skip gaps (break, weekend) in slot stats
            self._update_slot(_ct(b.ts_event_ns), x2, dt)

    def _slot(self, t: datetime) -> int:
        minutes_since_open = (t.hour * 60 + t.minute - 17 * 60) % 1440
        return minutes_since_open // self.cfg.vol_bucket_minutes

    def _update_slot(self, t: datetime, x2: float, dt: float) -> None:
        key = (cme_trading_day(t), self._slot(t))
        if key != self._slot_key:
            # close the previous slot: fold that session's variance into the slot baseline
            if self._slot_key is not None and self._slot_dt >= 0.5 * self.cfg.vol_bucket_minutes * 60:
                slot, v = self._slot_key[1], self._slot_sum / self._slot_dt
                a = 1 - 0.5 ** (1 / self.cfg.vol_baseline_halflife_days)
                old = self._tod_var.get(slot)
                self._tod_var[slot] = v if old is None else old + a * (v - old)
                self._tod_days[slot] = self._tod_days.get(slot, 0) + 1
            self._slot_key, self._slot_sum, self._slot_dt = key, 0.0, 0.0
        self._slot_sum += x2 * dt
        self._slot_dt += dt

    def vol_baseline(self) -> tuple[float | None, str]:
        """Normal variance for the current time of day, and where it came from."""
        if self._slot_key is not None:
            slot = self._slot_key[1]
            if self._tod_days.get(slot, 0) >= self.cfg.vol_baseline_min_days:
                return self._tod_var[slot], "time-of-day"
        if self._vol_samples >= self.cfg.vol_warmup_s and self._var_l:
            return self._var_l, "all-day fallback"
        return None, "warming up"

    def vol_ratio(self) -> float:
        base, _ = self.vol_baseline()
        if not base or self._var_s is None:
            return 1.0
        return math.sqrt(self._var_s / base)

    def sigma_h_ticks(self) -> float:
        """Expected size of a move over the holding period, in ticks."""
        if self._var_s is None:
            return self.cfg.min_stop_ticks / self.cfg.stop_sigma_mult
        return math.sqrt(self._var_s * self.cfg.holding_seconds)

    # ------------------------------------------------------------------ calendar
    def _calendar_block(self, now_ns: int) -> str | None:
        c, t = self.cfg, _ct(now_ns)
        d = cme_trading_day(t)
        if d in c.no_trade_days:
            return "holiday / early-close session"
        for first, last in c.roll_blackouts:
            if first <= d <= last:
                return "roll window"
        brk = t.replace(hour=16, minute=0, second=0, microsecond=0)
        if brk - timedelta(minutes=c.no_entry_before_break_min) <= t < brk + timedelta(hours=1):
            return "maintenance break"
        for at, before, after in c.session_blackouts_ct:
            center = t.replace(hour=at.hour, minute=at.minute, second=0, microsecond=0)
            if center - timedelta(minutes=before) <= t <= center + timedelta(minutes=after):
                return f"session blackout ({at.strftime('%H:%M')} CT)"
        for at, before, after in c.news_events_et:
            if at - timedelta(minutes=before) <= t <= at + timedelta(minutes=after):
                return f"news blackout ({at.astimezone(ET).strftime('%Y-%m-%d %H:%M')} ET)"
        return None

    # ------------------------------------------------------------------ halts
    def _halt_reason(self, now_ns: int) -> str | None:
        c = self.cfg
        if self.kill_switch:
            return "manual kill switch"
        if self.cooldown_left > 0:
            return f"drawdown cooldown ({self.cooldown_left} sessions left)"
        if self.day_halt_reason:
            return self.day_halt_reason
        if now_ns < self.pause_until_ns:
            return "losing-streak pause"
        b = self.last_book
        if b is None:
            return "no market data"
        if (now_ns - b.ts_event_ns) / 1e6 > c.max_quiet_ms or self.last_lag_ms > c.max_feed_lag_ms:
            return "stale data"
        return None

    def scale(self) -> tuple[float, list]:
        """Size multiplier in [0, 1]; the most cautious monitor wins."""
        c, notes, s = self.cfg, [], [self.dd_scale]
        if self.dd_scale < 1:
            notes.append(f"drawdown scale {self.dd_scale}")
        r = self.vol_ratio()
        if r >= c.vol_block_ratio:
            s.append(0.0); notes.append(f"vol regime: no entries ({r:.1f}x normal)")
        elif r >= c.vol_reduce_ratio:
            s.append(0.5); notes.append(f"vol regime: half size ({r:.1f}x normal)")
        if len(self.trade_ticks) == self.trade_ticks.maxlen:
            m = sum(self.trade_ticks) / len(self.trade_ticks)
            if m <= c.edge_reduce_ticks:
                s.append(0.5); notes.append(f"edge monitor: half size ({m:+.2f} ticks/contract)")
        if len(self.slippage) >= 20:
            m = sum(self.slippage) / len(self.slippage)
            if m > c.slippage_reduce_ticks:
                s.append(0.5); notes.append(f"slippage monitor: half size ({m:.2f} ticks)")
        return min(s), notes

    # ------------------------------------------------------------------ pre-trade
    def pre_trade_check(self, now_ns: int, side: int, position: Position | None,
                        predicted_move_ticks: float | None = None) -> Decision:
        """
        side                  +1 buy / -1 sell (the fade direction)
        predicted_move_ticks  model's expected reversion in ticks, if available
        """
        c, b = self.cfg, self.last_book
        halt = self._halt_reason(now_ns)
        if halt:
            return Decision(False, reasons=[halt])
        block = self._calendar_block(now_ns)
        if block:
            return Decision(False, reasons=[block])
        if position is not None and position.qty and not c.allow_pyramiding:
            return Decision(False, reasons=["already in a position"])

        spread_ticks = (b.ask_px - b.bid_px) / self.spec.tick_size
        if spread_ticks <= 0:
            return Decision(False, reasons=["crossed or locked book"])
        if spread_ticks > c.max_spread_ticks + 1e-9:
            return Decision(False, reasons=[f"spread too wide ({spread_ticks:.0f} > {c.max_spread_ticks:.0f} ticks)"])
        touch = b.ask_sz if side > 0 else b.bid_sz
        if touch < c.min_touch_size:
            return Decision(False, reasons=[f"thin touch ({touch:.0f} < {c.min_touch_size} contracts)"])

        cost = self.spec.round_trip_cost_ticks(spread_ticks)
        if predicted_move_ticks is not None and predicted_move_ticks < c.min_edge_to_cost * cost:
            return Decision(False, reasons=[f"edge below {c.min_edge_to_cost}x cost "
                                            f"({predicted_move_ticks:.2f} < {c.min_edge_to_cost * cost:.2f} ticks)"])

        while self.order_times and now_ns - self.order_times[0] > 60e9:
            self.order_times.popleft()
        if len(self.order_times) >= c.max_orders_per_minute:
            return Decision(False, reasons=["orders-per-minute limit"])
        if self.orders_today >= c.max_orders_per_day:
            return Decision(False, reasons=["orders-per-day limit"])

        scale, notes = self.scale()
        if scale == 0:
            return Decision(False, scale=0.0, reasons=notes)
        stop = max(c.min_stop_ticks, c.stop_sigma_mult * self.sigma_h_ticks())
        loss_per_contract = (stop + cost) * self.spec.tick_value
        n = math.floor(c.risk_per_trade_frac * self.equity / loss_per_contract * scale)
        n = min(n, c.max_contracts, math.floor(c.max_touch_fraction * touch))
        if n < 1:
            return Decision(False, stop_ticks=stop, scale=scale,
                            reasons=notes + [f"size rounds to 0 (stop {stop:.1f} ticks)"])
        self.order_times.append(now_ns)
        self.orders_today += 1
        return Decision(True, n, stop, scale, notes)

    # ------------------------------------------------------------------ exits
    def check_exit(self, now_ns: int, position: Position) -> tuple[bool, str]:
        b, c = self.last_book, self.cfg
        exit_px = b.bid_px if position.side > 0 else b.ask_px
        pnl_ticks = position.side * (exit_px - position.entry_px) / self.spec.tick_size
        if pnl_ticks <= -position.stop_ticks:
            return True, f"price stop ({pnl_ticks:+.0f} ticks)"
        held = (now_ns - position.entry_ns) / 1e9
        if held >= c.holding_seconds:
            return True, f"time stop ({held:.1f}s)"
        t = _ct(now_ns)
        brk = t.replace(hour=16, minute=0, second=0, microsecond=0)
        if brk - timedelta(minutes=c.flatten_before_break_min) <= t < brk + timedelta(hours=1):
            return True, "flatten before maintenance break"
        halt = self._halt_reason(now_ns)
        if halt and halt != "losing-streak pause":
            return True, f"halt: {halt}"
        block = self._calendar_block(now_ns)
        if block and block.startswith(("news", "roll")):
            return True, f"flatten: {block}"
        return False, ""

    # ------------------------------------------------------------------ post-trade
    def on_fill(self, side: int, fill_px: float, decision_book: Book) -> float:
        """Slippage beyond the touch at decision time (spread itself is not slippage)."""
        touch = decision_book.ask_px if side > 0 else decision_book.bid_px
        slip = side * (fill_px - touch) / self.spec.tick_size
        self.slippage.append(slip)
        if len(self.slippage) >= 20 and sum(self.slippage) / len(self.slippage) > self.cfg.slippage_halt_ticks:
            self.day_halt_reason = "slippage halt (fills far from the touch: latency or depth problem)"
        return slip

    def on_trade_closed(self, net_pnl_usd: float, qty: int, now_ns: int) -> None:
        """net_pnl_usd must already include fees."""
        c = self.cfg
        self.daily_pnl += net_pnl_usd
        self.equity += net_pnl_usd
        self.trade_ticks.append(net_pnl_usd / (qty * self.spec.tick_value))
        self.consecutive_losses = self.consecutive_losses + 1 if net_pnl_usd < 0 else 0
        if self.consecutive_losses >= c.max_consecutive_losses:
            self.pause_until_ns = now_ns + int(c.loss_streak_pause_s * 1e9)
            self.consecutive_losses = 0
        if self.daily_pnl <= -c.daily_loss_halt_frac * self.session_start_equity:
            self.day_halt_reason = f"daily loss halt ({self.daily_pnl:,.0f} USD)"
        if len(self.trade_ticks) == self.trade_ticks.maxlen:
            m = sum(self.trade_ticks) / len(self.trade_ticks)
            if m <= c.edge_halt_ticks:
                self.day_halt_reason = f"edge halt ({m:+.2f} ticks/contract over {len(self.trade_ticks)} trades)"

    def status(self) -> dict:
        s, notes = self.scale()
        return dict(equity=self.equity, daily_pnl=self.daily_pnl, dd_scale=self.dd_scale,
                    vol_ratio=round(self.vol_ratio(), 2), vol_baseline=self.vol_baseline()[1],
                    scale=s, notes=notes,
                    halt=self.day_halt_reason, cooldown=self.cooldown_left,
                    orders_today=self.orders_today)


# ============================================================================ #
# Nautilus wiring (sketch: verify attribute names on your Nautilus version)
# ============================================================================ #
NAUTILUS_SKETCH = """
# in XGBoostReversionStrategy.on_start:
    self.rm = ESRiskManager(RiskConfig(**default_2024h1_calendar()))

# in on_book_depth(depth):
    b = Book(depth.ts_event, depth.ts_init, float(depth.bids[0].price), float(depth.asks[0].price),
             _size(depth.bids[0]), _size(depth.asks[0]))
    self.rm.on_book(b)
    now = self.clock.timestamp_ns()
    if self.pos is not None:
        exit_now, why = self.rm.check_exit(now, self.pos)
        if exit_now:
            self.close_all_positions(self.config.instrument_id)
        return
    p_short, p_flat, p_long = model.predict(...)[0]      # classes 0/1/2 from train_xgb.py
    side = 1 if p_long > p_short else -1
    move = expected_move_ticks(p_short, p_long, side, b.mid)
    d = self.rm.pre_trade_check(now, side, self.pos, move)
    if d.allowed:
        self.decision_book = b
        submit market order for d.contracts; remember d.stop_ticks for the Position

# in on_order_filled(event):   self.rm.on_fill(side, float(event.last_px), self.decision_book)
# in on_position_closed(event): self.rm.on_trade_closed(net_pnl_after_fees, qty, event.ts_event)
# at 17:00 CT each day:         self.rm.on_session_start(trading_day, account_equity)
"""


# ============================================================================ #
# Self-test on synthetic data
# ============================================================================ #
if __name__ == "__main__":
    import random
    from collections import Counter

    random.seed(4)
    rm = ESRiskManager(RiskConfig(**default_2024h1_calendar()))
    spec = rm.spec
    print(f"Round-trip cost at a 1-tick spread: {spec.round_trip_cost_ticks():.2f} ticks "
          f"(${spec.round_trip_cost_ticks() * spec.tick_value:.2f}/contract)")

    def ns(y, mo, d, h, mi, s=0.0, tz=CT):
        return int(datetime(y, mo, d, h, mi, tzinfo=tz).timestamp() * 1e9 + s * 1e9)

    def book(t, mid, spread=1, sz=150):
        half = spread * spec.tick_size / 2
        return Book(t, t + 2_000_000, mid - half, mid + half, sz, sz)

    # --- 1. Targeted checks -------------------------------------------------
    t0 = ns(2024, 1, 9, 9, 0)
    rm.on_session_start(date(2024, 1, 9), 1_000_000)
    mid = 4800.0
    for k in range(3600):                                   # one hour of warm-up data
        mid += random.choice([-0.25, 0, 0, 0.25])
        rm.on_book(book(t0 + k * 1_000_000_000, mid))
    now = t0 + 3600 * 1_000_000_000
    cases = {
        "normal book":            (book(now, mid), now, None),
        "2-tick spread":          (book(now, mid, spread=2), now, None),
        "thin touch":             (book(now, mid, sz=5), now, None),
        "edge below 1.5x cost":   (book(now, mid), now, 1.0),
        "CPI 2024-01-11 08:31 ET": (book(ns(2024, 1, 11, 8, 31, tz=ET), mid), ns(2024, 1, 11, 8, 31, tz=ET), None),
        "roll window 2024-03-12": (book(ns(2024, 3, 12, 10, 0), mid), ns(2024, 3, 12, 10, 0), None),
        "2024-03-08 10:00 CT":    (book(ns(2024, 3, 8, 10, 0), mid), ns(2024, 3, 8, 10, 0), None),
        "Juneteenth 2024-06-19":  (book(ns(2024, 6, 19, 9, 0), mid), ns(2024, 6, 19, 9, 0), None),
        "Sun 2024-03-17 18:00 CT": (book(ns(2024, 3, 17, 18, 0), mid), ns(2024, 3, 17, 18, 0), None),
        "5 min before break":     (book(ns(2024, 1, 9, 15, 55), mid), ns(2024, 1, 9, 15, 55), None),
        "stale feed (8 s quiet)": (book(now - 8_000_000_000, mid), now, None),
    }
    for name, (b, t, edge) in cases.items():
        rm.on_book(b)
        d = rm.pre_trade_check(t, +1, None, edge)
        print(f"  {name:26s} -> {'TRADE ' + str(d.contracts) + ' @ stop ' + format(d.stop_ticks, '.1f') + 't' if d.allowed else 'BLOCK: ' + '; '.join(d.reasons)}")

    # --- 2. A synthetic session: random fades with a small edge -------------
    print("\nSynthetic session (fades every ~20 s; ~1.8 ticks gross edge vs 1.32 cost):")
    rm = ESRiskManager(RiskConfig(**default_2024h1_calendar()))
    rm.on_session_start(date(2024, 1, 16), 1_000_000)
    t, mid, pos = ns(2024, 1, 16, 8, 0), 4800.0, None
    blocks, exits, trades = Counter(), Counter(), 0
    for step in range(int(7 * 3600 / 0.25)):              # 08:00-15:00 CT, 4 updates/second
        t += 250_000_000
        move = random.choice([-1, 0, 0, 0, 1])
        if pos and random.random() < 0.3:                  # reversion edge: extra tick our way
            move += pos.side
        mid += spec.tick_size * move
        spread = 2 if random.random() < 0.02 else 1
        b = book(t, mid, spread)
        rm.on_book(b)
        if pos:
            done, why = rm.check_exit(t, pos)
            if done:
                px = b.bid_px if pos.side > 0 else b.ask_px
                pnl = pos.side * (px - pos.entry_px) * spec.multiplier * pos.qty - 2 * spec.fee_per_side * pos.qty
                rm.on_trade_closed(pnl, pos.qty, t)
                exits[why.split(" (")[0]] += 1
                trades += 1
                pos = None
            continue
        if random.random() < 0.0125:                       # model fires ~ every 20 s
            side = random.choice([-1, 1])
            d = rm.pre_trade_check(t, side, None, predicted_move_ticks=random.uniform(1.5, 4))
            if not d.allowed:
                blocks[d.reasons[0].split(" (")[0]] += 1
                continue
            fill = b.ask_px if side > 0 else b.bid_px
            rm.on_fill(side, fill, b)
            pos = Position(side, d.contracts, fill, t, d.stop_ticks)
    print(f"  trades {trades}, daily P&L ${rm.daily_pnl:,.0f}, status {rm.status()['notes'] or 'normal'}")
    print(f"  exits:  {dict(exits)}")
    print(f"  blocks: {dict(blocks)}")

    # --- 3. Loss limits ------------------------------------------------------
    print("\nLoss limits:")
    rm = ESRiskManager(RiskConfig())
    rm.on_session_start(date(2024, 1, 17), 1_000_000)
    t = ns(2024, 1, 17, 10, 0)
    rm.on_book(book(t, 4800.0))
    for k in range(8):
        rm.on_trade_closed(-40.0, 1, t)
    print(f"  after 8 straight losses: {rm.pre_trade_check(t, 1, None).reasons}")
    t2 = t + 2 * 10**12
    rm.on_book(book(t2, 4800.0))
    rm.on_trade_closed(-19_000.0, 2, t2)
    print(f"  after a -$19.3k day (limit 2% = $20k): {rm.pre_trade_check(t2, 1, None).reasons or 'still trading'}")
    rm.on_trade_closed(-1_000.0, 2, t2)
    print(f"  after a -$20.3k day: {rm.pre_trade_check(t2, 1, None).reasons}")
    for d, eq in [(18, 978_000), (19, 945_000), (22, 945_000)]:
        rm.on_session_start(date(2024, 1, d), eq)
        print(f"  session 2024-01-{d}, equity {eq:,}: dd_scale {rm.dd_scale}, cooldown {rm.cooldown_left}")

    # --- 4. Time-of-day volatility baseline ---------------------------------
    print("\nVolatility check at the open (5 synthetic sessions; open is 2.5x overnight every day):")
    rm = ESRiskManager(RiskConfig())
    sigma = lambda t: 1.0 if 8 <= t.hour < 9 else (0.6 if 9 <= t.hour < 15 else 0.4)  # ticks/sqrt(s)
    mid, start = 5000.0, datetime(2024, 1, 7, 17, 0, tzinfo=CT)          # Sunday 17:00 CT open
    checks = {(2, 8, 45): "day 2, 08:45 CT", (5, 8, 45): "day 5, 08:45 CT (normal open)",
              (5, 11, 15): "day 5, 11:15 CT (real spike: 3x normal since 11:00)"}
    for day in range(1, 6):
        open_t = start + timedelta(days=day - 1)
        for k in range(23 * 3600):                                         # 17:00 -> 16:00 CT
            t = open_t + timedelta(seconds=k)
            vol = sigma(t) * (3.0 if day == 5 and 11 <= t.hour < 12 else 1.0)
            mid += random.gauss(0, vol) * spec.tick_size
            n = int(t.timestamp() * 1e9)
            rm.on_book(Book(n, n + 1_000_000, mid - 0.125, mid + 0.125, 150, 150))
            label = checks.get((day, t.hour, t.minute))
            if label and t.second == 0:
                old = math.sqrt(rm._var_s / rm._var_l)
                new, src = rm.vol_ratio(), rm.vol_baseline()[1]
                act = lambda r: "no entries" if r >= 2.5 else ("half size" if r >= 1.5 else "full size")
                print(f"  {label:48s} old all-day ratio {old:4.1f}x ({act(old)})  |  new {new:4.1f}x "
                      f"vs {src} ({act(new)})")
