# GA-Selected XGBoost on E-mini S&P 500 Order-Book Data

Gator Quant Hacks 2026 · Systematic Trading track
Manny Osorio, Benjamin Friedman, Emily Diaz-Silva

We test whether the shape of the ten-level order book of the CME E-mini S&P 500 future (ES) predicts the next move in the mid-price better than a simple top-of-book rule.

1. **Features:** we compute 25 stationary microstructure features from MBP-10 data: order-flow imbalance, book imbalance, spread and depth slope.
2. **XGBoost:** a 3-class model (sell / hold / buy) is trained on GPU with a purged walk-forward split. Its directional score `P(buy) − P(sell)` becomes one more input.
3. **Genetic algorithm:** a GA runs on GPU with a fused Triton kernel and the island model. Each 64-bit chromosome picks a feature subset, an entry threshold and a warm-up period.
4. **Backtest:** the winning rule is frozen on in-sample days and replayed once, out of sample, in NautilusTrader. Market orders fill against the recorded book, net of $2.00 per contract per fill.

The full write-up is in [`docs/writeup.tex`](docs/writeup.tex).

> **Scope:** compute limits held us to **one month of training data (January 2024)** and **100 GA generations**. Ideally we would use 2+ years of data and run the GA until fitness plateaus. See the Limitations section of the paper.

## Results (out of sample, net of costs)

| Metric | Value |
|---|---|
| Trades | _fill in_ |
| Mean net P&L per trade | _fill in_ |
| Net Sharpe | _fill in_ |
| Max drawdown | _fill in_ |

Reproduce with step 6 below.

---

## Repository layout

| Path | What it does |
|---|---|
| `config.yaml` | Every data path, model setting and GA setting. The pipeline reads it through `python/config.py` |
| `python/data_pipeline/download_databento.py` | Downloads MBP-10 data from Databento to `data/GLBX.MDP3/` (`.dbn.zst`, then `.parquet`) |
| `python/data_pipeline/data_loader.py` | Joins the daily Parquet files into one sorted float32 binary (`data/unified_mbp10_*.bin`) |
| `python/prepare_xgb_features.py` | Raw binary → 25 stationary features (Parquet) |
| `python/model/train_xgb.py` | Labels the data (100-event horizon, 1.5 bps barrier), runs purged walk-forward XGBoost and writes the GA inputs (`features.bin`, `returns.bin`) |
| `python/model/explain_xgb.py` | SHAP summaries and feature importance |
| `python/model/distributed_ga.py` | Island-model GA across GPUs (PyTorch DDP + NCCL) |
| `python/model/evolution_kernel.py`, `mini_backtest.py`, `chromosome.py` | Fused Triton fitness kernel and the 64-bit chromosome layout |
| `evaluation/backtest/nautilus_catalog.py` | Converts DBN files into a NautilusTrader Parquet catalog (one time) |
| `evaluation/backtest/nautilus_strategy.py` | The frozen strategy, plus `compute_stats` / `build_bundle` to freeze GA output |
| `evaluation/backtest/nautilus_backtest.py` | Out-of-sample replay, tearsheet, equity curve, signal log |
| `evaluation/latency_sweep.py` | How sensitive the OOS result is to assumed order latency (0/5/20/50 ms) |
| `evaluation/metrics.py`, `split_sample.py` | Net-of-cost metrics, IS/OOS split on CME trading-day boundaries |
| `python/es_risk_manager.py`, `python/risk_report.py` | Risk limits, P&L decomposition, cost stress, PSR/DSR |
| `scripts/*.sbatch` | Slurm jobs for UF HiPerGator (one per pipeline stage) |

---

## 1. Setup

**Requirements:**
- Python ≥ 3.12 and [`uv`](https://github.com/astral-sh/uv)
- An NVIDIA GPU with CUDA 12.8 for the XGBoost and GA stages. We used B200s on HiPerGator.
- A [Databento](https://databento.com) API key with access to CME Globex (`GLBX.MDP3`)

```bash
git clone https://github.com/atlasshiny/gator-quant-hacks.git
```

```bash
cd gator-quant-hacks && uv sync
```

`uv sync` installs the exact versions pinned in `uv.lock`, which is our dependency file (from `pyproject.toml`). PyTorch comes from the CUDA 12.8 wheel index.

Then set up your API key:

```bash
cp .env.example .env
```

Put your `DATABENTO_API_KEY` in `.env`. `.env` is git-ignored: **never commit API keys.** The FRED key is optional and the main pipeline doesn't use it.

---

## 2. Get the data

The raw data is licensed from Databento, so it is **not in this repo** (`/data` and `/output` are git-ignored). Download it yourself:

| Setting | Value |
|---|---|
| Dataset | `GLBX.MDP3` |
| Schema | `mbp-10` |
| Symbol | `ES.c.0` (continuous front month, maps to ESH4; no roll in our window) |
| Training window | 2024-01-08 → 2024-02-01 |

```bash
uv run python -m python.data_pipeline.download_databento --symbol ES.c.0 --start 2024-01-08T00:00:00Z --end 2024-02-01T00:00:00Z --dataset GLBX.MDP3 --schema mbp-10
```

On HiPerGator, run `sbatch scripts/download_data.sbatch` instead. `scripts/download_data_array.sbatch` pulls the full Jan–Jul 2024 range as six monthly array tasks.

Files land in `data/GLBX.MDP3/`. The next stage reads from `data.parquet_dir` in `config.yaml`, which is currently `data/GLBX.MDP3_january`. Either move the January Parquet files there or point `parquet_dir` at `data/GLBX.MDP3`.

For the Nautilus backtest you also need the instrument definition for ESH4 (`definition` schema). See the docstring at the top of `evaluation/backtest/nautilus_catalog.py`.

---

## 3. Reproduce the results

Every stage reads its paths and settings from `config.yaml`. Run these from the repo root. Each step has a matching Slurm script.

| # | Stage | Command | Slurm |
|---|---|---|---|
| 1 | Join the raw data | `uv run python -m python.data_pipeline.data_loader` | `scripts/data_unifier.sbatch` |
| 2 | Features + XGBoost + SHAP | `uv run python -m python.prepare_xgb_features && uv run python -m python.model.train_xgb && uv run python -m python.model.explain_xgb` | `scripts/train_xgb.sbatch` (1 GPU) |
| 3 | Genetic algorithm (100 generations, population 1,000) | `uv run python -m python.model.distributed_ga` | `scripts/evo_kernel_ddp.sbatch` (2 GPUs) |
| 4 | Freeze the rule | see below | — |
| 5 | Build the Nautilus catalog | `uv run python evaluation/backtest/nautilus_catalog.py` | — |
| 6 | **Out-of-sample backtest (headline results)** | `PYTHONPATH=evaluation:evaluation/backtest uv run python evaluation/backtest/nautilus_backtest.py` | `scripts/nautilus_backtest.sbatch` |
| 7 | Latency robustness | `cd evaluation && uv run python latency_sweep.py` | `sbatch --array=0-3 scripts/nautilus_backtest.sbatch` |

**Step 4: freeze the rule.** The backtest loads two files built only from in-sample data. Run this once after the GA finishes:

```bash
mkdir -p data/ga_inputs
cp output/models/xgboost_january.json data/ga_inputs/xgboost_final.json
PYTHONPATH=evaluation/backtest uv run python - <<'EOF'
import polars as pl
from nautilus_strategy import compute_stats, build_bundle

ignore = {"ts_event", "ts_recv", "mid_price", "ask_px_00", "bid_px_00",
          "future_return", "target"}
feats = [c for c in pl.read_parquet(
    "output/features/stationary_features_january.parquet").columns
    if c not in ignore]                      # the 25 base features, training order
assert len(feats) == 25

compute_stats("output/ga_inputs_january/features.bin", "data/ga_inputs/stats.json")
build_bundle(
    "output/ga_results_january/best_strategy.json",
    "data/ga_inputs/stats.json",
    column_names_28=feats + ["p_sell", "p_hold", "p_buy"],
    xgb_input_names=feats,
    out_path="data/ga_inputs/frozen_bundle.json",
)
EOF
```

**Outputs.** Step 6 writes these files:
- `oos_tearsheet.html` — the Nautilus tearsheet. Its numbers are **gross** of our per-fill fee.
- `equity_curve.png` — gross vs. net equity.
- `signal_log_oos.csv` — one row per entry.
- A printed table of return, volatility, Sharpe, max drawdown and turnover, **net** of fees.

These net numbers are the ones in the paper's Results section.

---

## 4. Methodology guardrails

- **Clean out-of-sample test:** all fitting and tuning happens on in-sample days. The OOS window is replayed **once**, with the rule frozen.
- **No leakage across the split:** the purged walk-forward split leaves an embargo after each training block. IS/OOS splits fall on CME trading-day boundaries (17:00 CT).
- **Multiple-testing count:** we log every chromosome the GA evaluates (`N_eval = population × generations`) so the multiple-testing burden can be reported.
- **Latency is a robustness check, not a tuning knob:** we report every latency in the sweep, not just the best one.

## License and data

The code is for the Gator Quant Hacks 2026 competition. Market data © CME Group, distributed by Databento, and may not be redistributed.
