"""Generate SHAP explanations for the trained multiclass XGBoost model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

from python.config import load_config
from python.model.train_xgb import MicrostructureXGBTrainer

CLASS_NAMES = ("short", "flat", "long")

def _normalize_shap_values(
    values: list[np.ndarray] | np.ndarray,
    n_samples: int,
    n_features: int,
) -> np.ndarray:
    """Return SHAP values with shape (samples, features, classes)."""
    if isinstance(values, list):
        normalized = np.stack([np.asarray(item) for item in values], axis=-1)
    else:
        normalized = np.asarray(values)
        if normalized.ndim != 3:
            raise ValueError(
                f"Expected multiclass SHAP values with 3 dimensions, got {normalized.shape}."
            )
        if normalized.shape == (n_samples, n_features, len(CLASS_NAMES)):
            pass
        elif normalized.shape == (n_samples, len(CLASS_NAMES), n_features):
            normalized = np.transpose(normalized, (0, 2, 1))
        else:
            raise ValueError(
                "Unexpected multiclass SHAP shape: "
                f"{normalized.shape}; expected samples/features/classes."
            )

    expected_shape = (n_samples, n_features, len(CLASS_NAMES))
    if normalized.shape != expected_shape:
        raise ValueError(
            f"Normalized SHAP shape {normalized.shape} does not match {expected_shape}."
        )
    return normalized.astype(np.float32, copy=False)

def explain_model(
    model_path: Path,
    features_path: Path,
    output_dir: Path,
    target_horizon_events: int,
    fee_threshold_bps: float,
    n_splits: int,
    sample_rows: int,
) -> None:
    """Explain the final model on the final chronological validation window."""
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = MicrostructureXGBTrainer(
        features_path=str(features_path),
        target_horizon_events=target_horizon_events,
        fee_threshold_bps=fee_threshold_bps,
    )
    X, _y, feature_names, _returns = trainer.load_and_label_data()
    if len(X) <= n_splits:
        raise ValueError("The explanation dataset must contain more rows than n_splits.")

    # Match the final chronological WFO validation window without shuffling.
    final_train_indices, validation_indices = list(
        TimeSeriesSplit(n_splits=n_splits).split(X)
    )[-1]
    purge_size = target_horizon_events
    train_end = final_train_indices[-1] + 1
    validation_start = validation_indices[0]
    if train_end - purge_size >= validation_start:
        raise ValueError("The final WFO validation window is not separated by the purge.")

    selected_indices = validation_indices[-sample_rows:]
    X_validation = X[selected_indices]

    model = xgb.Booster()
    model.load_model(model_path)
    explainer = shap.TreeExplainer(model)
    shap_values = _normalize_shap_values(
        explainer.shap_values(X_validation),
        n_samples=len(X_validation),
        n_features=X_validation.shape[1],
    )

    # Positive values push the model toward the named class. The long-class
    # plot is also copied to the stable presentation filename.
    summary_path = output_dir / "shap_summary.png"
    for class_index, class_name in enumerate(CLASS_NAMES):
        plt.figure(figsize=(14, 9))
        shap.summary_plot(
            shap_values[:, :, class_index],
            X_validation,
            feature_names=feature_names,
            max_display=len(feature_names),
            show=False,
            plot_size=None,
        )
        plt.title(f"SHAP Summary: {class_name.title()}-Class XGBoost Decision")
        plt.tight_layout()
        class_path = output_dir / f"shap_summary_{class_name}.png"
        plt.savefig(class_path, dpi=220, bbox_inches="tight")
        if class_name == "long":
            plt.savefig(summary_path, dpi=220, bbox_inches="tight")
        plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pl.DataFrame(
        {
            "feature": feature_names,
            **{
                f"mean_abs_shap_{class_name}": mean_abs[:, class_index]
                for class_index, class_name in enumerate(CLASS_NAMES)
            },
        }
    ).with_columns(
        pl.mean_horizontal(
            [f"mean_abs_shap_{class_name}" for class_name in CLASS_NAMES]
        ).alias("mean_abs_shap_all_classes")
    ).sort("mean_abs_shap_all_classes", descending=True)
    importance.write_csv(output_dir / "shap_feature_importance.csv")

    metadata = {
        "model_path": str(model_path),
        "features_path": str(features_path),
        "validation_window": "final TimeSeriesSplit validation fold",
        "validation_start_row": int(validation_indices[0]),
        "validation_end_row": int(validation_indices[-1] + 1),
        "sample_start_row": int(selected_indices[0]),
        "sample_end_row": int(selected_indices[-1] + 1),
        "sample_rows": int(len(selected_indices)),
        "feature_count": len(feature_names),
        "class_order": list(CLASS_NAMES),
        "summary_plot": str(summary_path),
        "class_summary_plots": {
            class_name: str(output_dir / f"shap_summary_{class_name}.png")
            for class_name in CLASS_NAMES
        },
        "importance_csv": str(output_dir / "shap_feature_importance.csv"),
    }
    (output_dir / "shap_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Saved SHAP summary plot -> {summary_path}")
    print(f"Saved SHAP feature ranking -> {output_dir / 'shap_feature_importance.csv'}")

if __name__ == "__main__":
    config = load_config()
    xgb_config = config.xgboost
    explain_model(
        model_path=xgb_config.model_path,
        features_path=xgb_config.features_path,
        output_dir=xgb_config.explainability_dir,
        target_horizon_events=xgb_config.target_horizon_events,
        fee_threshold_bps=xgb_config.fee_threshold_bps,
        n_splits=xgb_config.folds,
        sample_rows=xgb_config.shap_sample_rows,
    )
