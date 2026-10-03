"""
nautilus_strategy.py
 
Order-book-imbalance strategy for NautilusTrader, driven by OrderBookDepth10
(mbp-10). Same idea as the earlier pandas example:
 
    imbalance = (bid size - ask size) / (bid size + ask size)   over the top N levels
    target position = +size if imbalance >  threshold
                      -size if imbalance < -threshold
                       0    otherwise
 
Uses market orders only, so fills come from the simulated L2 book and no
queue-position model is involved.
 
NOTE: no `from __future__ import annotations` here on purpose; the config class is a
msgspec Struct and needs real annotations.
"""
 
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import InstrumentId
from nautilus_trader.model.enums import BookType, OrderSide
from nautilus_trader.trading.strategy import Strategy
 
 
class ImbalanceConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    threshold: float = 0.8        # |imbalance| needed to take a position
    levels: int = 3               # book levels per side to include
    trade_size: int = 1           # contracts
    min_interval_ms: int = 1000   # throttle: minimum time between orders
 
 
class ImbalanceStrategy(Strategy):
    def __init__(self, config: ImbalanceConfig) -> None:
        super().__init__(config)
        self.instrument = None
        self._last_order_ns = 0
        self._min_gap_ns = config.min_interval_ms * 1_000_000
 
    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found in cache")
            self.stop()
            return
        # Method/handler names below match the current docs; older Nautilus versions
        # call these subscribe_order_book_depth / on_order_book_depth.
        self.subscribe_book_depth10(self.config.instrument_id, BookType.L2_MBP)
 
    def on_book_depth(self, depth) -> None:
        now = self.clock.timestamp_ns()
        if now - self._last_order_ns < self._min_gap_ns:
            return
 
        n = self.config.levels
        bid = sum(o.size.as_double() for o in depth.bids[:n])
        ask = sum(o.size.as_double() for o in depth.asks[:n])
        total = bid + ask
        if total <= 0:
            return
        imbalance = (bid - ask) / total
 
        size = self.config.trade_size
        if imbalance > self.config.threshold:
            target = size
        elif imbalance < -self.config.threshold:
            target = -size
        else:
            target = 0
 
        current = float(self.portfolio.net_position(self.config.instrument_id))
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
 