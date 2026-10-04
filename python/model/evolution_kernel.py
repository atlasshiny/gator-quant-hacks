import os
import torch
import triton
import triton.language as tl
import numpy as np

# Directly import established pipeline modules
from chromosome import StrategyChromosome
from data_memmap import load_memmap_tensor

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
        strat_ret = signal * ret

        sum_pnl += tl.sum(strat_ret, axis=0)
        sum_pnl_2 += tl.sum(strat_ret * strat_ret, axis=0)
        sum_pnl_3 += tl.sum(strat_ret * strat_ret * strat_ret, axis=0)
        sum_pnl_4 += tl.sum(strat_ret * strat_ret * strat_ret * strat_ret, axis=0)
        total_count += tl.sum(tl.where(active_mask, 1.0, 0.0), axis=0)

    # EXACT STATISTICAL MOMENTS & DEFLATED SHARPE CALCULATION
    count = tl.maximum(total_count, 1.0)
    mean = sum_pnl / count
    variance = (sum_pnl_2 / count) - (mean * mean)
    std_dev = tl.sqrt(tl.maximum(variance, 1e-8))

    e_x2 = sum_pnl_2 / count
    e_x3 = sum_pnl_3 / count
    e_x4 = sum_pnl_4 / count

    mu_3 = e_x3 - (3.0 * mean * e_x2) + (2.0 * mean * mean * mean)
    skew = mu_3 / tl.maximum(std_dev * std_dev * std_dev, 1e-8)

    mu_4 = (
        e_x4
        - (4.0 * mean * e_x3)
        + (6.0 * mean * mean * e_x2)
        - (3.0 * mean * mean * mean * mean)
    )
    kurtosis = mu_4 / tl.maximum(variance * variance, 1e-8)

    annualized_sharpe = (mean / std_dev) * 2432.0

    skew_penalty = tl.where(skew < 0.0, -skew * 0.5, 0.0)
    kurt_penalty = tl.where(kurtosis > 3.0, (kurtosis - 3.0) * 0.1, 0.0)

    computed_fitness = annualized_sharpe - skew_penalty - kurt_penalty
    final_fitness = tl.where(total_count > 1.0, computed_fitness, 0.0)

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
):
    population_bitmasks = np.asarray(population_bitmasks, dtype=np.uint64)
    if population_bitmasks.ndim != 1 or population_bitmasks.size == 0:
        raise ValueError("population_bitmasks must be a non-empty one-dimensional array.")
    if n_features < 28:
        raise ValueError("n_features must be at least 28 for the feature flag mask.")
    n_population = population_bitmasks.shape[0]

    n_samples = _count_rows_from_binary(data_bin_path, n_features)

    X_gpu = load_memmap_tensor(
        file_path=data_bin_path, dtype=np.float32, shape=(n_samples, n_features), device="cuda"
    )

    if returns_bin_path is None:
        raise ValueError("returns_bin_path must be provided.")

    n_returns = _count_rows_from_binary(returns_bin_path, 1)
    if n_returns != n_samples:
        raise ValueError("Return series length does not match feature matrix length.")

    returns_gpu = load_memmap_tensor(
        file_path=returns_bin_path, dtype=np.float32, shape=(n_samples,), device="cuda"
    ).contiguous()

    bitmasks_gpu = torch.from_numpy(population_bitmasks).to("cuda", non_blocking=True)
    fitness_gpu = torch.empty(n_population, device="cuda", dtype=torch.float32)

    grid = (n_population,)

    # BLOCK_SIZE is removed here; the autotuner injects it dynamically.
    evolution_kernel_grid[grid](
        X_gpu, bitmasks_gpu, returns_gpu, fitness_gpu,
        n_samples, n_features, X_gpu.stride(0),
        SHIFT_FEAT=C_SCHEMA["feature_flags"]["shift"],
        MASK_FEAT=C_SCHEMA["feature_flags"]["mask"],
        SHIFT_LOOK=C_SCHEMA["lookback_window"]["shift"],
        MASK_LOOK=C_SCHEMA["lookback_window"]["mask"],
        SHIFT_THRESH=C_SCHEMA["thresholds"]["shift"],
        MASK_THRESH=C_SCHEMA["thresholds"]["mask"],
        SHIFT_RISK=C_SCHEMA["risk_rules"]["shift"],
        MASK_RISK=C_SCHEMA["risk_rules"]["mask"],
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