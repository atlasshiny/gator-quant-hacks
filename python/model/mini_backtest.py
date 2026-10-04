import torch
import triton
import triton.language as tl

# GLOBAL BACKTEST CONFIGURATION
class BacktestConfig:
    FRICTION_BPS = 0.0002 # 2 basis points per position change
    ANNUALIZATION = 2432.0 # Annualization factor for Sharpe
    SKEW_PENALTY_MULT = 0.5 # Weight of penalty for negative skew
    KURT_PENALTY_MULT = 0.1 # Weight of penalty for fat tails > 3.0
    MIN_TRADES = 1.0 # Minimum valid sample size for fitness

# TRITON DEVICE FUNCTIONS (FUSED INTO EVOLUTION KERNEL)
@triton.jit
def compute_deflated_sharpe_device(
    sum_pnl, sum_pnl_2, sum_pnl_3, sum_pnl_4, total_count,
    ANNUALIZATION: tl.constexpr,
    SKEW_PENALTY_MULT: tl.constexpr,
    KURT_PENALTY_MULT: tl.constexpr
):
    """
    Calculates exact statistical moments and penalizes non-normal distributions.
    Can be imported directly into evolution_kernel_2.py.
    """
    count = tl.maximum(total_count, 1.0)
    
    # 1st & 2nd Moments (Mean, Variance)
    mean = sum_pnl / count
    variance = (sum_pnl_2 / count) - (mean * mean)
    std_dev = tl.sqrt(tl.maximum(variance, 1e-8))

    # Raw expected values for 3rd & 4th moments
    e_x2 = sum_pnl_2 / count
    e_x3 = sum_pnl_3 / count
    e_x4 = sum_pnl_4 / count

    # 3rd Moment (Skewness)
    mu_3 = e_x3 - (3.0 * mean * e_x2) + (2.0 * mean * mean * mean)
    skew = mu_3 / tl.maximum(std_dev * std_dev * std_dev, 1e-8)

    # 4th Moment (Kurtosis)
    mu_4 = (
        e_x4
        - (4.0 * mean * e_x3)
        + (6.0 * mean * mean * e_x2)
        - (3.0 * mean * mean * mean * mean)
    )
    kurtosis = mu_4 / tl.maximum(variance * variance, 1e-8)

    # Base Metrics
    annualized_sharpe = (mean / std_dev) * ANNUALIZATION

    # Deflation Penalties (Protects GA from exploiting tail-risk anomalies)
    skew_penalty = tl.where(skew < 0.0, -skew * SKEW_PENALTY_MULT, 0.0)
    kurt_penalty = tl.where(kurtosis > 3.0, (kurtosis - 3.0) * KURT_PENALTY_MULT, 0.0)

    computed_fitness = annualized_sharpe - skew_penalty - kurt_penalty
    
    # Nullify fitness if the strategy didn't trade
    return tl.where(total_count > 1.0, computed_fitness, 0.0)


# VECTORIZED PYTORCH ENGINE (FOR EXACT PATH-DEPENDENCY & VALIDATION)
class VectorizedMiniBacktester:
    """
    Executes identical math to the Triton kernel but explicitly handles
    temporal path-dependency (like turnover friction) via PyTorch vectorization.
    """
    def __init__(self, config=BacktestConfig):
        self.friction_bps = config.FRICTION_BPS
        self.annualization = config.ANNUALIZATION
        self.skew_mult = config.SKEW_PENALTY_MULT
        self.kurt_mult = config.KURT_PENALTY_MULT

    def evaluate_signals(self, signals: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        Evaluates a batch of generated signals.
        :param signals: Tensor of shape (population_size, n_samples)
        :param returns: Tensor of shape (n_samples,)
        :return: Tensor of fitness scores shape (population_size,)
        """
        # Path-Dependent Turnover Calculation
        # Shift signals right by 1 to represent the previous timestep's position
        prev_signals = torch.roll(signals, shifts=1, dims=1)
        prev_signals[:, 0] = 0.0  # Zero out the lookahead artifact on step 0
        
        # Calculate turnover and friction
        turnover = torch.abs(signals - prev_signals)
        transaction_costs = turnover * self.friction_bps

        # Net PnL Calculation
        # Returns broadcast across the population dimension
        gross_pnl = signals * returns
        net_pnl = gross_pnl - transaction_costs

        # Statistical Moments
        mean_pnl = net_pnl.mean(dim=1)
        std_pnl = net_pnl.std(dim=1).clamp(min=1e-8)
        
        # Calculate Base Sharpe
        sharpe = (mean_pnl / std_pnl) * self.annualization

        # Higher-Order Deflation (Skew & Kurtosis)
        # Center the data to calculate exact 3rd and 4th moments
        centered_pnl = net_pnl - mean_pnl.unsqueeze(1)
        variance = std_pnl ** 2
        
        skew = (centered_pnl ** 3).mean(dim=1) / (std_pnl ** 3)
        kurtosis = (centered_pnl ** 4).mean(dim=1) / (variance ** 2)

        # Apply asymmetric penalties
        skew_penalty = torch.where(skew < 0.0, -skew * self.skew_mult, 0.0)
        kurt_penalty = torch.where(kurtosis > 3.0, (kurtosis - 3.0) * self.kurt_mult, 0.0)

        # Final Fitness Score
        fitness = sharpe - skew_penalty - kurt_penalty
        
        # Nullify strategies that did not trade
        trade_counts = (signals != 0).sum(dim=1)
        fitness = torch.where(trade_counts > 1, fitness, 0.0)

        return fitness