import time
from pathlib import Path
import numpy as np
import polars as pl

class MicrostructureEngineer:
    """
    Ingests the 73-column raw MBP-10 binary cache and generates 27 stationary 
    microstructure features (OFI, OBI, Spreads, Slopes) for XGBoost training.
    """
    def __init__(
        self,
        raw_bin_path: str = "data/unified_mbp10.bin",
        output_parquet: str = "data/stationary_features.parquet",
    ):
        self.raw_bin_path = Path(raw_bin_path)
        self.output_parquet = Path(output_parquet)
        
        # Exact 73-column schema from RawDatabentoUnifier
        self.core_cols = [
            "ts_event", "ts_recv", "rtype", "publisher_id", "instrument_id", 
            "action", "side", "depth", "price", "size", "flags", "ts_in_delta", "sequence"
        ]
        self.l2_cols = []
        for i in range(10):
            level = f"0{i}"
            self.l2_cols.extend([
                f"bid_px_{level}", f"ask_px_{level}", 
                f"bid_sz_{level}", f"ask_sz_{level}", 
                f"bid_ct_{level}", f"ask_ct_{level}"
            ])
        self.all_columns = self.core_cols + self.l2_cols

    def load_raw_binary(self) -> pl.DataFrame:
        """Loads the raw float32 binary file directly into a Polars DataFrame."""
        print(f"Loading raw binary from {self.raw_bin_path}...")
        # Load memory map (zero-copy)
        matrix = np.memmap(self.raw_bin_path, dtype=np.float32, mode="r")
        matrix = matrix.reshape((-1, len(self.all_columns)))
        
        # Transfer to Polars
        return pl.DataFrame(matrix, schema=self.all_columns, orient="row")

    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Computes 27 stationary microstructure signals using vectorized Polars operations.
        """
        print("Computing stationary microstructure features...")
        
        # 1. Base Prices & Mid-Price
        df = df.with_columns([
            ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2.0).alias("mid_price")
        ])

        # 2. Relative Spreads (Scale-Invariant)
        spread_exprs = [
            ((pl.col(f"ask_px_0{i}") - pl.col(f"bid_px_0{i}")) / pl.col("mid_price")).alias(f"spread_l{i}")
            for i in range(5)  # Top 5 levels
        ]

        # 3. Order Book Imbalance (OBI) - Bounded [-1, 1]
        obi_exprs = [
            ((pl.col(f"bid_sz_0{i}") - pl.col(f"ask_sz_0{i}")) / 
             (pl.col(f"bid_sz_0{i}") + pl.col(f"ask_sz_0{i}") + 1e-8)).alias(f"obi_l{i}")
            for i in range(5)  # Top 5 levels
        ]

        # 4. Multi-Level Cumulative Imbalance
        cum_bid_sz = sum(pl.col(f"bid_sz_0{i}") for i in range(10))
        cum_ask_sz = sum(pl.col(f"ask_sz_0{i}") for i in range(10))
        depth_imbalance = ((cum_bid_sz - cum_ask_sz) / (cum_bid_sz + cum_ask_sz + 1e-8)).alias("depth_imbalance")

        # 5. Order Book Slope (Liquidity Density)
        bid_slope = ((pl.col("bid_px_00") - pl.col("bid_px_09")) / (cum_bid_sz + 1e-8)).alias("bid_slope")
        ask_slope = ((pl.col("ask_px_09") - pl.col("ask_px_00")) / (cum_ask_sz + 1e-8)).alias("ask_slope")

        # Apply Spreads, OBI, Depth, and Slopes
        df = df.with_columns(spread_exprs + obi_exprs + [depth_imbalance, bid_slope, ask_slope])

        # 6. Order Flow Imbalance (OFI) - Top 3 Levels
        # OFI requires comparing current state to previous state (t-1)
        ofi_exprs = []
        for i in range(3):
            level = f"0{i}"
            bid_px = pl.col(f"bid_px_{level}")
            bid_sz = pl.col(f"bid_sz_{level}")
            ask_px = pl.col(f"ask_px_{level}")
            ask_sz = pl.col(f"ask_sz_{level}")
            
            # Bid flow logic
            bid_flow = pl.when(bid_px > bid_px.shift(1)).then(bid_sz) \
                         .when(bid_px == bid_px.shift(1)).then(bid_sz - bid_sz.shift(1)) \
                         .otherwise(0.0)
                         
            # Ask flow logic
            ask_flow = pl.when(ask_px < ask_px.shift(1)).then(ask_sz) \
                         .when(ask_px == ask_px.shift(1)).then(ask_sz - ask_sz.shift(1)) \
                         .otherwise(0.0)
                         
            ofi_exprs.append((bid_flow - ask_flow).alias(f"ofi_l{i}"))

        df = df.with_columns(ofi_exprs)

        # 7. Rolling Micro-Volatility & Z-Scoring OFI
        # Standardize OFI over a rolling 1000-tick window to maintain stationarity
        df = df.with_columns([
            pl.col(f"ofi_l{i}").rolling_mean(window_size=1000).alias(f"ofi_mean_l{i}") for i in range(3)
        ]).with_columns([
            pl.col(f"ofi_l{i}").rolling_std(window_size=1000).alias(f"ofi_std_l{i}") for i in range(3)
        ])

        z_score_exprs = [
            ((pl.col(f"ofi_l{i}") - pl.col(f"ofi_mean_l{i}")) / (pl.col(f"ofi_std_l{i}") + 1e-8)).alias(f"ofi_zscore_l{i}")
            for i in range(3)
        ]
        
        # Action/Cancel Imbalance (Event level)
        cancel_add_ratio = (
            pl.col("action").eq(2).cast(pl.Float32) / 
            (pl.col("action").eq(1).cast(pl.Float32) + 1.0)
        ).alias("cancel_add_ratio")

        df = df.with_columns(z_score_exprs + [cancel_add_ratio])

        trade = pl.col("action") == 5
        buy_volume = pl.when(trade & (pl.col("side") == 2)).then(pl.col("size")).otherwise(0.0)
        sell_volume = pl.when(trade & (pl.col("side") == 1)).then(pl.col("size")).otherwise(0.0)
        trade_volume = (buy_volume + sell_volume).rolling_sum(window_size=1000)
        vpin = ((buy_volume - sell_volume).abs().rolling_sum(window_size=1000) /
                (trade_volume + 1e-8)).alias("vpin")

        bid_dispersion = sum(
            ((pl.col(f"bid_px_0{i}") - pl.col("mid_price")) / pl.col("mid_price")) ** 2
            for i in range(10)
        ).truediv(10).alias("bid_quote_dispersion")
        ask_dispersion = sum(
            ((pl.col(f"ask_px_0{i}") - pl.col("mid_price")) / pl.col("mid_price")) ** 2
            for i in range(10)
        ).truediv(10).alias("ask_quote_dispersion")

        bid_vwap = sum(pl.col(f"bid_px_0{i}") * pl.col(f"bid_sz_0{i}") for i in range(10)) / (cum_bid_sz + 1e-8)
        ask_vwap = sum(pl.col(f"ask_px_0{i}") * pl.col(f"ask_sz_0{i}") for i in range(10)) / (cum_ask_sz + 1e-8)
        buy_slippage = ((ask_vwap - pl.col("ask_px_00")) / pl.col("mid_price")).alias("buy_vwap_slippage")
        sell_slippage = ((pl.col("bid_px_00") - bid_vwap) / pl.col("mid_price")).alias("sell_vwap_slippage")

        cancel_trade_ratio = (
            (pl.when(pl.col("action") == 2).then(pl.col("size")).otherwise(0.0).rolling_sum(window_size=1000))
            / (trade_volume + 1e-8)
        ).alias("cancel_trade_ratio")
        message_intensity = (1.0 / (pl.col("ts_in_delta").abs() + 1.0)).rolling_mean(window_size=1000).alias("message_intensity")
        deep_bid_slope = ((pl.col("bid_px_02") - pl.col("bid_px_09")) / (sum(pl.col(f"bid_sz_0{i}") for i in range(2, 10)) + 1e-8)).alias("deep_bid_slope")
        deep_ask_slope = ((pl.col("ask_px_09") - pl.col("ask_px_02")) / (sum(pl.col(f"ask_sz_0{i}") for i in range(2, 10)) + 1e-8)).alias("deep_ask_slope")
        deep_count_imbalance = (
            (sum(pl.col(f"bid_ct_0{i}") for i in range(2, 10)) - sum(pl.col(f"ask_ct_0{i}") for i in range(2, 10))) /
            (sum(pl.col(f"bid_ct_0{i}") + pl.col(f"ask_ct_0{i}") for i in range(2, 10)) + 1e-8)
        ).alias("deep_count_imbalance")

        df = df.with_columns([
            vpin, bid_dispersion, ask_dispersion, buy_slippage, sell_slippage,
            cancel_trade_ratio, message_intensity, deep_bid_slope,
            deep_ask_slope, deep_count_imbalance,
        ])

        # 8. Filter to Final 27 Features + Metadata
        final_feature_cols = (
            [f"spread_l{i}" for i in range(5)] +
            [f"obi_l{i}" for i in range(5)] +
            ["depth_imbalance", "bid_slope", "ask_slope", "cancel_add_ratio"] +
            [f"ofi_zscore_l{i}" for i in range(3)] +
            [
                "vpin", "bid_quote_dispersion", "ask_quote_dispersion",
                "buy_vwap_slippage", "sell_vwap_slippage", "cancel_trade_ratio",
                "message_intensity", "deep_bid_slope", "deep_ask_slope",
                "deep_count_imbalance",
            ]
        )
        if len(final_feature_cols) != 27:
            raise ValueError(f"Expected 27 engineered features, got {len(final_feature_cols)}.")
        
        # Keep essential core columns needed by `train_xgb.py` for target generation
        retention_cols = ["ts_event", "ts_recv", "mid_price", "ask_px_00", "bid_px_00"] + final_feature_cols
        
        # Drop initial NaN rows caused by rolling windows and shifting
        return (
            df.select(retention_cols)
            .drop_nulls()
            .with_columns(pl.col(final_feature_cols).cast(pl.Float32))
        )

    def run(self):
        start_t = time.perf_counter()
        df_raw = self.load_raw_binary()
        df_features = self.engineer_features(df_raw)
        
        self.output_parquet.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving {df_features.shape[0]:,} rows to {self.output_parquet}...")
        df_features.write_parquet(self.output_parquet)
        
        elapsed = time.perf_counter() - start_t
        print(f"Feature engineering completed in {elapsed:.2f} seconds.")
        print(f"Generated {len(df_features.columns) - 5} engineered features.") # Subtracting the 5 retention metadata columns


if __name__ == "__main__":
    engineer = MicrostructureEngineer()
    engineer.run()