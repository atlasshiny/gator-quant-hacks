from decimal import Decimal
import os
import numpy as np
import xgboost as xgb


from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import BookType
from nautilus_trader.model import InstrumentId
from nautilus_trader.model import OrderBookDepth10
from nautilus_trader.model import TradeTick
from nautilus_trader.model import OrderSide
from nautilus_trader.trading import Strategy


class XGBoostReversionConfig(StrategyConfig):
    def __init__(
        self,
        *,
        instrument_id: InstrumentId,
        model_path: str = "model.json",         # Path to your saved XGBoost model
        ml_threshold: float = 0.75,             # Minimum P(revert) required to trade
        levels: int = 5,                        # Book depth levels to evaluate
        trade_size: int = 1,                    # Number of contracts
        holding_period_sec: float = 1.5,        # Time-based exit horizon (h seconds)
        min_interval_ms: int = 500,             # Throttle between orders
        **_kwargs,
    ) -> None:
        super().__init__()
        self.instrument_id = instrument_id
        self.model_path = model_path
        self.ml_threshold = ml_threshold
        self.levels = levels
        self.trade_size = trade_size
        self.holding_period_sec = holding_period_sec
        self.min_gap_ns = min_interval_ms * 1_000_000
        self.holding_period_ns = int(holding_period_sec * 1_000_1000_000) # approx


def _size(order) -> float:
    size = order.size
    as_double = getattr(size, "as_double", None)
    return as_double() if as_double is not None else float(size)


class XGBoostReversionStrategy(Strategy):
    def __init__(self, config: XGBoostReversionConfig) -> None:
        super().__init__(config)
        self.instrument = None
        self.model = None
        self._last_order_ns = 0
        self._entry_timestamp_ns = 0

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.config.instrument_id} not found in cache")
            self.stop()
            return

        # Load the trained XGBoost model
        if xgb is not None and os.path.exists(self.config.model_path):
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.config.model_path)
            self.log.info(f"Successfully loaded XGBoost model from {self.config.model_path}")
        else:
            self.log.warning(f"XGBoost model not found at {self.config.model_path}. Running in dummy/stub mode.")

        # Subscribe to both order book depth (MBP-10) and trade ticks for flow features
        self.subscribe_book_depth10(self.config.instrument_id, BookType.L2_MBP)
        self.subscribe_trade_ticks(self.config.instrument_id)

    def on_book_depth(self, depth: OrderBookDepth10) -> None:
        now = self.clock.timestamp_ns()
        current_position = self.portfolio.net_position(self.config.instrument_id)

        # Manage Time-Based Exit (Close position after h seconds)
        if current_position != 0 and (now - self._entry_timestamp_ns >= self.config.holding_period_ns):
            self.close_all_positions(self.config.instrument_id)
            self.log.info(f"Time-based exit triggered at {now}")
            return

        # Throttle new entries
        if now - self._last_order_ns < self.config.min_gap_ns:
            return

        # Skip if already in a position (don't over-accumulate)
        if current_position != 0:
            return

        # Extract Feature Vector (Depth Imbalance across configured levels)
        n = self.config.levels
        bid_depth = sum(_size(o) for o in depth.bids[:n])
        ask_depth = sum(_size(o) for o in depth.asks[:n])
        total_depth = bid_depth + ask_depth
        if total_depth <= 0:
            return
        
        book_imbalance = (bid_depth - ask_depth) / total_depth

        # Construct feature array matching  GA/XGBoost training columns
        # (Ensure this matches the exact feature order your model expects)
        features = np.array([[book_imbalance, bid_depth, ask_depth, total_depth]])

        # Model Inference
        reversion_prob = 0.5  # Default baseline
        if self.model is not None:
            # Predict probability of class 1 (reversion)
            probs = self.model.predict_proba(features)
            reversion_prob = probs[0][1]

        # Conditional Taker Execution Logic
        # If the model predicts a high probability of reversion, fade the heavy imbalance
        size = Decimal(self.config.trade_size)
        target_side = None

        if book_imbalance > 0.6 and reversion_prob >= self.config.ml_threshold:
            # Heavy buy imbalance predicted to revert -> Fade by selling (short)
            target_side = OrderSide.SELL
        elif book_imbalance < -0.6 and reversion_prob >= self.config.ml_threshold:
            # Heavy sell imbalance predicted to revert -> Fade by buying (long)
            target_side = OrderSide.BUY

        if target_side is not None:
            order = self.order_factory.market(
                instrument_id=self.config.instrument_id,
                order_side=target_side,
                quantity=self.instrument.make_qty(size),
            )
            self.submit_order(order)
            self._last_order_ns = now
            self._entry_timestamp_ns = now
            self.log.info(f"Fading burst: Imbalance={book_imbalance:.2f}, P(revert)={reversion_prob:.2f}")

    def on_trade_tick(self, trade: TradeTick) -> None:
        # Optional: Hook to track high-volume trade bursts in real-time if needed for features
        pass

    def on_stop(self) -> None:
        self.close_all_positions(self.config.instrument_id)
        self.unsubscribe_book_depth10(self.config.instrument_id)
        self.unsubscribe_trade_ticks(self.config.instrument_id)