"""Typed application configuration for the data, model, and GA pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr

class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None

class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    dataset: str
    symbol: str
    schema_name: str = Field(alias="schema")
    stype_in: str
    start: str
    end: str
    parquet_dir: Path
    unified_bin: Path

class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stationary_parquet: Path
    expected_base_features: int = Field(ge=1)
    total_kernel_features: int = Field(ge=1)

class XGBoostConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features_path: Path
    output_dir: Path
    model_path: Path
    target_horizon_events: int = Field(gt=0)
    fee_threshold_bps: float = Field(ge=0)
    folds: int = Field(ge=2)
    embargo_pct: float = Field(ge=0, lt=1)
    tree_method: str
    device: str
    num_boost_round: int = Field(gt=0)
    early_stopping_rounds: int = Field(gt=0)
    max_depth: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    subsample: float = Field(gt=0, le=1)
    colsample_bytree: float = Field(gt=0, le=1)
    directional_class_weight: float = Field(default=5.0, ge=1)
    num_class: int = Field(gt=0)
    explainability_dir: Path = Path("output/explainability")
    shap_sample_rows: int = Field(default=50_000, gt=0)

class GeneticAlgorithmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features_path: Path
    returns_path: Path
    output_dir: Path
    population: int = Field(gt=0)
    generations: int = Field(gt=0)
    migration_frequency: int = Field(gt=0)
    chromosome_features: int = Field(ge=28)

class EnvironmentConfig(BaseModel):
    """Values loaded from .env; secrets are never stored in config.yaml."""

    model_config = ConfigDict(extra="forbid")

    databento_api_key: SecretStr | None = Field(
        default_factory=lambda: _optional_secret("DATABENTO_API_KEY")
    )
    databento_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "DATABENTO_BASE_URL", "https://hist.databento.com"
        )
    )
    databento_default_dataset: str = Field(
        default_factory=lambda: os.getenv("DATABENTO_DEFAULT_DATASET", "GLBX.MDP3")
    )
    fred_api_key: SecretStr | None = Field(
        default_factory=lambda: _optional_secret("FRED_API_KEY")
    )
    fred_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "FRED_BASE_URL", "https://api.stlouisfed.org/fred"
        )
    )

class AppConfig(BaseModel):
    """Merged configuration from config.yaml and the repository .env file."""

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentConfig
    data: DataConfig
    features: FeatureConfig
    xgboost: XGBoostConfig
    genetic_algorithm: GeneticAlgorithmConfig
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)

def _optional_secret(name: str) -> SecretStr | None:
    value = os.getenv(name)
    return SecretStr(value) if value else None

def load_config(
    yaml_path: str | Path = "config.yaml",
    env_path: str | Path = ".env",
) -> AppConfig:
    """Load and validate YAML settings together with environment secrets."""
    load_dotenv(dotenv_path=env_path, override=False)

    config_path = Path(yaml_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config: Any = yaml.safe_load(config_file)

    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return AppConfig.model_validate(raw_config)
