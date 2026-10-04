import os
import torch
import triton
import triton.language as tl
import numpy as np

# Directly import established pipeline modules
from .chromosome import StrategyChromosome
from .data_memmap import load_memmap_tensor
from .mini_backtest import BacktestConfig, compute_deflated_sharpe_device

C_SCHEMA = StrategyChromosome.CONSTANTS

# TRITON KERNEL: AUTOTUNED EVOLUTION
# The autotuner automatically benchmarks these configs on the first run 
# and compiles the kernel using the fastest one for the detected GPU.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['n_samples', 'n_features']
)
@triton.jit
def evolution_kernel_grid(
    X_ptr,
    bitmasks_ptr,
    returns_ptr,
    fitness_ptr,
    n_samples,
    n_features,
    stride_sample,
    SHIFT_FEAT: tl.constexpr, MASK_FEAT: tl.constexpr,
    SHIFT_LOOK: tl.constexpr, MASK_LOOK: tl.constexpr,
    SHIFT_THRESH: tl.constexpr, MASK_THRESH: tl.constexpr,
    SHIFT_RISK: tl.constexpr, MASK_RISK: tl.constexpr,
    FRICTION_BPS: tl.constexpr,
    ANNUALIZATION: tl.constexpr,
    SKEW_PENALTY_MULT: tl.constexpr,
    KURT_PENALTY_MULT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, # Now dynamically provided by the autotuner
):
    pop_idx = tl.program_id(axis=0)
    bitmask = tl.load(bitmasks_ptr + pop_idx)

    # Decode Chromosome
    feat_flags = (bitmask >> SHIFT_FEAT) & MASK_FEAT
    lookback_gray = (bitmask >> SHIFT_LOOK) & MASK_LOOK
    threshold_raw = (bitmask >> SHIFT_THRESH) & MASK_THRESH
    risk_raw = (bitmask >> SHIFT_RISK) & MASK_RISK

    # Gray Code Decoder for Lookback
    b = lookback_gray ^ (lookback_gray >> 1)
    b = b ^ (b >> 2)
    b = b ^ (b >> 4)
    lookback_decoded = b ^ (b >> 8)

    # Map Strategy Parameters
    z_trigger = threshold_raw.to(tl.float32) / 1000.0
    vol_target = risk_raw.to(tl.float32) / 1000.0
    
    # Accumulators for Higher-Order Moments
    sum_pnl = 0.0
    sum_pnl_2 = 0.0
    sum_pnl_3 = 0.0
    sum_pnl_4 = 0.0
    total_count = 0.0

    # Dynamic Backtest Loop
    for start_sample in range(0, n_samples, BLOCK_SIZE):
        sample_offsets = start_sample + tl.arange(0, BLOCK_SIZE)
        mask = sample_offsets < n_samples
        active_mask = mask & (sample_offsets >= lookback_decoded)

        selected_features = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for feature_idx in range(28):
            feature = tl.load(
                X_ptr + (sample_offsets * stride_sample) + feature_idx,
                mask=mask,
                other=0.0,
            )
            selected_features += tl.where(
                ((feat_flags >> feature_idx) & 1) != 0, feature, 0.0
            )

        ret = tl.load(returns_ptr + sample_offsets, mask=mask, other=0.0)

        has_features = feat_flags != 0
        long_cond = has_features & (selected_features > z_trigger)
        signal = tl.where(
            active_mask & long_cond,
            tl.where(vol_target > 0.0, vol_target, 1.0),
            0.0,
        )
        # Approximate turnover friction within the block. Recompute the
        # previous lane's signal for intra-block transitions; lane zero is
        # explicitly zero so block boundaries have no cross-block dependency.
        previous_offsets = sample_offsets - 1
        previous_mask = mask & (sample_offsets > start_sample)
        previous_active = previous_mask & (previous_offsets >= lookback_decoded)
        previous_features = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for feature_idx in range(28):
            previous_feature = tl.load(
                X_ptr + (previous_offsets * stride_sample) + feature_idx,
                mask=previous_mask,
                other=0.0,
            )
            previous_features += tl.where(
                ((feat_flags >> feature_idx) & 1) != 0,
                previous_feature,
                0.0,
            )
        previous_long_cond = has_features & (previous_features > z_trigger)
        previous_signal = tl.where(
            previous_active & previous_long_cond,
            tl.where(vol_target > 0.0, vol_target, 1.0),
            0.0,
        )
        turnover = tl.abs(signal - previous_signal)
        net_pnl = (signal * ret) - (turnover * FRICTION_BPS)

        sum_pnl += tl.sum(net_pnl, axis=0)
        sum_pnl_2 += tl.sum(net_pnl * net_pnl, axis=0)
        sum_pnl_3 += tl.sum(net_pnl * net_pnl * net_pnl, axis=0)
        sum_pnl_4 += tl.sum(net_pnl * net_pnl * net_pnl * net_pnl, axis=0)
        total_count += tl.sum(tl.where(active_mask, 1.0, 0.0), axis=0)

    final_fitness = compute_deflated_sharpe_device(
        sum_pnl,
        sum_pnl_2,
        sum_pnl_3,
        sum_pnl_4,
        total_count,
        ANNUALIZATION=ANNUALIZATION,
        SKEW_PENALTY_MULT=SKEW_PENALTY_MULT,
        KURT_PENALTY_MULT=KURT_PENALTY_MULT,
    )

    tl.store(fitness_ptr + pop_idx, final_fitness)

# PYTHON DRIVER WRAPPER
def _count_rows_from_binary(file_path: str, n_features: int) -> int:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Binary array not found at: {file_path}")
    if n_features <= 0:
        raise ValueError(f"n_features must be positive. Received: {n_features}")

    bytes_per_row = n_features * np.dtype(np.float32).itemsize
    file_size = os.path.getsize(file_path)

    if file_size % bytes_per_row != 0:
        raise ValueError(
            f"Feature file size {file_size} bytes is not divisible by {bytes_per_row} bytes per row."
        )
    return file_size // bytes_per_row

def evaluate_population(
    data_bin_path: str,
    population_bitmasks: np.ndarray,
    n_features: int = 28,
    returns_bin_path: str | None = None,
    data_gpu: torch.Tensor | None = None,
    returns_gpu: torch.Tensor | None = None,
    n_samples: int | None = None,
):
    population_bitmasks = np.asarray(population_bitmasks, dtype=np.uint64)
    if population_bitmasks.ndim != 1 or population_bitmasks.size == 0:
        raise ValueError("population_bitmasks must be a non-empty one-dimensional array.")
    if n_features < 28:
        raise ValueError("n_features must be at least 28 for the feature flag mask.")
    n_population = population_bitmasks.shape[0]

    if data_gpu is None or returns_gpu is None:
        if returns_bin_path is None:
            raise ValueError("returns_bin_path must be provided.")
        n_samples = _count_rows_from_binary(data_bin_path, n_features)
        n_returns = _count_rows_from_binary(returns_bin_path, 1)
        if n_returns != n_samples:
            raise ValueError("Return series length does not match feature matrix length.")
        data_gpu = load_memmap_tensor(
            file_path=data_bin_path,
            dtype=np.float32,
            shape=(n_samples, n_features),
            device="cuda",
        )
        returns_gpu = load_memmap_tensor(
            file_path=returns_bin_path,
            dtype=np.float32,
            shape=(n_samples,),
            device="cuda",
        ).contiguous()
    elif n_samples is None:
        n_samples = data_gpu.shape[0]

    if data_gpu.ndim != 2 or data_gpu.shape[1] < n_features:
        raise ValueError("Preloaded feature tensor has an incompatible shape.")
    if returns_gpu.ndim != 1 or returns_gpu.shape[0] != n_samples:
        raise ValueError("Preloaded return tensor length does not match features.")

    bitmasks_gpu = torch.from_numpy(population_bitmasks).to("cuda", non_blocking=True)
    fitness_gpu = torch.empty(n_population, device="cuda", dtype=torch.float32)

    grid = (n_population,)

    # BLOCK_SIZE is removed here; the autotuner injects it dynamically.
    evolution_kernel_grid[grid](
        data_gpu, bitmasks_gpu, returns_gpu, fitness_gpu,
        n_samples, n_features, data_gpu.stride(0),
        SHIFT_FEAT=C_SCHEMA["feature_flags"]["shift"],
        MASK_FEAT=C_SCHEMA["feature_flags"]["mask"],
        SHIFT_LOOK=C_SCHEMA["lookback_window"]["shift"],
        MASK_LOOK=C_SCHEMA["lookback_window"]["mask"],
        SHIFT_THRESH=C_SCHEMA["thresholds"]["shift"],
        MASK_THRESH=C_SCHEMA["thresholds"]["mask"],
        SHIFT_RISK=C_SCHEMA["risk_rules"]["shift"],
        MASK_RISK=C_SCHEMA["risk_rules"]["mask"],
        FRICTION_BPS=BacktestConfig.FRICTION_BPS,
        ANNUALIZATION=BacktestConfig.ANNUALIZATION,
        SKEW_PENALTY_MULT=BacktestConfig.SKEW_PENALTY_MULT,
        KURT_PENALTY_MULT=BacktestConfig.KURT_PENALTY_MULT,
    )

    return fitness_gpu

# VERIFICATION AND TESTING
if __name__ == "__main__":
    test_file = "test_features.bin"
    returns_file = "test_returns.bin"
    
    mock_features = np.random.randn(10_000, 28).astype(np.float32)
    mock_features[:, 1] = np.abs(mock_features[:, 1]) + 0.1 
    mock_features.tofile(test_file)
    
    mock_returns = np.random.standard_t(df=3, size=10_000).astype(np.float32) * 0.001
    mock_returns.tofile(returns_file)

    pop_size = 1024
    population = np.zeros(pop_size, dtype=np.uint64)

    for i in range(pop_size):
        feat = np.random.randint(0, 0xFFFFFFF)
        look = np.random.randint(0, 4095)
        thresh = np.random.randint(0, 4095)
        risk = np.random.randint(0, 4095)
        population[i] = StrategyChromosome.encode(feat, look, thresh, risk)

    print(f"Executing Autotuned GA Pipeline for {pop_size} strategies...")
    fitness = evaluate_population(test_file, population, returns_bin_path=returns_file)
    print(f"Top 5 Risk-Adjusted Sharpe Ratios: {torch.topk(fitness, 5).values.cpu().numpy()}")
    
    os.remove(test_file)
    os.remove(returns_file)