"""
Order-book-imbalance strategy driven by OrderBookDepth10 (mbp-10).

    imbalance = (bid size - ask size) / (bid size + ask size)   over the top N levels
    target position = +size if imbalance >  threshold
                      -size if imbalance < -threshold
                       0    otherwise

Market orders only, so fills come from the simulated L2 book and no
queue-position model is involved.
"""

from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import BookType
from nautilus_trader.model import InstrumentId
from nautilus_trader.model import OrderBookDepth10
from nautilus_trader.model import OrderSide
from nautilus_trader.trading import Strategy


class ImbalanceConfig(StrategyConfig):
    def __init__(
        self,
        *,
        instrument_id: InstrumentId,
        threshold: float = 0.8,       # |imbalance| needed to take a position
        levels: int = 3,              # book levels per side to include
        trade_size: int = 1,          # contracts
        min_interval_ms: int = 1000,  # throttle: minimum time between orders
        **_kwargs,                    # lets base fields (strategy_id, ...) pass through
    ) -> None:
        super().__init__()
        self.instrument_id = instrument_id
        self.threshold = threshold
        self.levels = levels
        self.trade_size = trade_size
        self.min_interval_ms = min_interval_ms


def _size(order) -> float:
    """Book-level size as a float (tolerates API differences between v2 RCs)."""
    size = order.size
    as_double = getattr(size, "as_double", None)
    return as_double() if as_double is not None else float(size)


class ImbalanceStrategy(Strategy):
    def __init__(self, config: ImbalanceConfig) -> None:
        super().__init__(config)
        # clock / cache / portfolio / order_factory are unavailable until the strategy is
        # registered (after __init__), so only plain state is initialised here.
        self.instrument = None
        self._last_order_ns = 0
        self._min_gap_ns = config.min_interval_ms * 1_000_000

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found in cache")
            self.stop()
            return
        self.subscribe_book_depth10(self.config.instrument_id, BookType.L2_MBP)

    def on_book_depth(self, depth: OrderBookDepth10) -> None:
        now = self.clock.timestamp_ns()
        if now - self._last_order_ns < self._min_gap_ns:
            return

        n = self.config.levels
        bid = sum(_size(o) for o in depth.bids[:n])
        ask = sum(_size(o) for o in depth.asks[:n])
        total = bid + ask
        if total <= 0:
            return
        imbalance = (bid - ask) / total

        size = Decimal(self.config.trade_size)
        if imbalance > self.config.threshold:
            target = size
        elif imbalance < -self.config.threshold:
            target = -size
        else:
            target = Decimal(0)

        # net_position() returns a Decimal in v2
        current = self.portfolio.net_position(self.config.instrument_id)
        delta = target - current
        if delta == 0:
            return

        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
            quantity=self.instrument.make_qty(abs(delta)),
        )
        self.submit_order(order)
        self._last_order_ns = now

    def on_stop(self) -> None:
        # Finish flat so every window (IS or OOS) is fully realized.
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_book_depth10(self.config.instrument_id)