import os
import time
from pathlib import Path
import numpy as np
import polars as pl
import xgboost as xgb
from config import load_config

class MicrostructureXGBTrainer:
    def __init__(
        self,
        features_path: str = "data/stationary_features.parquet",
        output_bin_dir: str = "data/ga_inputs",
        target_horizon_events: int = 100, # e.g., 100 events ahead
        fee_threshold_bps: float = 1.5, # 1.5 bps fee barrier
        xgb_params: dict[str, object] | None = None,
        num_boost_round: int = 800,
        early_stopping_rounds: int = 40,
    ):
        self.features_path = Path(features_path)
        self.output_bin_dir = Path(output_bin_dir)
        self.output_bin_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_horizon = target_horizon_events
        self.fee_threshold = fee_threshold_bps / 10000.0  # Convert bps to decimal
        self.xgb_params = xgb_params or {
            "tree_method": "hist",
            "device": "cuda",
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

    def load_and_label_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        """
        Loads stationary features and computes stationary binary targets.
        Target = 1 if mid-price return over horizon exceeds fee threshold, else 0.
        """
        print(f"Loading feature dataset from {self.features_path}...")
        df = pl.read_parquet(self.features_path)

        # Compute Mid-Price and Forward Return Label
        df = df.with_columns([
            ((pl.col("ask_px_00") + pl.col("bid_px_00")) / 2.0).alias("mid_price")
        ])

        # Shift backward to look ahead H events into the future
        df = df.with_columns([
            ((pl.col("mid_price").shift(-self.target_horizon) - pl.col("mid_price")) 
             / pl.col("mid_price")).alias("future_return")
        ])

        # Create binary directional target above fee friction
        df = df.with_columns([
            (pl.col("future_return") > self.fee_threshold).cast(pl.Int32).alias("target")
        ])

        # Drop the boundary tail where shifted target is null
        df = df.drop_nulls(subset=["future_return"])

        # Separate feature columns from metadata/targets
        ignore_cols = {
            "ts_event", "ts_recv", "mid_price", "future_return", "target"
        }
        feature_cols = [c for c in df.columns if c not in ignore_cols]

        X = df.select(feature_cols).to_numpy().astype(np.float32)
        y = df.select("target").to_numpy().flatten().astype(np.int32)
        forward_returns = (
            df.select("future_return").to_numpy().flatten().astype(np.float32)
        )

        print(f"Dataset shape: {X.shape[0]:,} rows x {X.shape[1]} features")
        print(f"Target Class Balance: {np.mean(y) * 100:.2f}% positive labels")
        return X, y, feature_cols, forward_returns

    def train_purged_cv(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_splits: int = 5, 
        embargo_pct: float = 0.01
    ) -> np.ndarray:
        """
        Trains XGBoost using Purged Walk-Forward splits to prevent time-series data leakage.
        Returns full out-of-fold predicted class probabilities.
        """
        n_samples = len(X)
        oof_preds = np.full(n_samples, np.nan, dtype=np.float32)
        split_size = n_samples // n_splits
        embargo_size = int(n_samples * embargo_pct)

        print(f"\nStarting {n_splits}-Fold Purged Walk-Forward Training...")

        for fold in range(n_splits - 1):
            # Train on historical window, validate on next chronologically adjacent block
            train_end = (fold + 1) * split_size
            val_start = train_end + self.target_horizon + embargo_size  # Purge overlap + Embargo
            val_end = min((fold + 2) * split_size, n_samples)

            if val_start >= n_samples or val_start >= val_end:
                break

            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[val_start:val_end], y[val_start:val_end]

            print(f"--- Fold {fold + 1}/{n_splits - 1} ---")
            print(f"Train Range: [0 : {train_end:,}] | Val Range: [{val_start:,} : {val_end:,}]")

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            model = xgb.train(
                self.xgb_params,
                dtrain,
                num_boost_round=self.num_boost_round,
                evals=[(dval, "val")],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=200,
            )

            # Predict out-of-fold validation probabilities
            dval_predict = xgb.DMatrix(X_val)
            oof_preds[val_start:val_end] = model.predict(dval_predict)

        return oof_preds

    def save_for_triton_ga(
        self,
        X_features: np.ndarray,
        xgb_probs: np.ndarray,
        forward_returns: np.ndarray,
    ) -> None:
        """
        Appends the XGBoost probability column to the microstructure feature matrix
        and writes raw float32 binary arrays (.bin) for Triton loading.
        """
        if X_features.shape[1] < 27:
            raise ValueError("At least 27 base features are required for the Triton GA.")
        valid = np.isfinite(xgb_probs)
        combined_features = np.hstack([
            X_features[valid, :27],
            xgb_probs[valid, None],
        ]).astype(np.float32)
        valid_returns = forward_returns[valid].astype(np.float32)

        feat_path = self.output_bin_dir / "features.bin"
        combined_features.tofile(feat_path)
        valid_returns.tofile(self.output_bin_dir / "returns.bin")
        
        print(f"\nSaved Triton GA Feature Binary -> {feat_path}")
        print(f"Final Tensor Shape: {combined_features.shape[0]:,} rows x {combined_features.shape[1]} columns")
        print(f"Saved GA return series -> {self.output_bin_dir / 'returns.bin'}")

if __name__ == "__main__":
    config = load_config()
    xgb_config = config.xgboost
    trainer = MicrostructureXGBTrainer(
        features_path=str(xgb_config.features_path),
        output_bin_dir=str(xgb_config.output_dir),
        target_horizon_events=xgb_config.target_horizon_events,
        fee_threshold_bps=xgb_config.fee_threshold_bps,
        xgb_params={
            "tree_method": xgb_config.tree_method,
            "device": xgb_config.device,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": xgb_config.max_depth,
            "learning_rate": xgb_config.learning_rate,
            "subsample": xgb_config.subsample,
            "colsample_bytree": xgb_config.colsample_bytree,
        },
        num_boost_round=xgb_config.num_boost_round,
        early_stopping_rounds=xgb_config.early_stopping_rounds,
    )

    start_t = time.perf_counter()
    X, y, feature_names, forward_returns = trainer.load_and_label_data()
    oof_predictions = trainer.train_purged_cv(
        X,
        y,
        n_splits=xgb_config.folds,
        embargo_pct=xgb_config.embargo_pct,
    )
    trainer.save_for_triton_ga(X, oof_predictions, forward_returns)

    final_model = xgb.train(
        trainer.xgb_params,
        xgb.DMatrix(X, label=y),
        num_boost_round=trainer.num_boost_round,
    )
    final_model.save_model(trainer.output_bin_dir / "xgboost_final.json")
    
    print(f"Pipeline complete in {time.perf_counter() - start_t:.2f} seconds.")