import vectorbt as vbt
import pandas as pd
from pathlib import Path

class BacktestEngine:
    """Drives step-by-step market evaluation using LLM backends and runs backtesting using VectorBT."""

    def __init__(
        self,
        data: pd.DataFrame,
        frequency: str = "1D",
        initial_cash: float = 10000.0,
        fees: float = 10,
    ):
        """Initializes the BacktestEngine with dataset parameters and trading costs.

        Args:
            data (pd.DataFrame): DataFrame containing OHLCV market data for a single asset.
                Must include a 'Close' column and a DateTime index.
            frequency (str, optional): The bar frequency of the input data (e.g., '1h', '1D').
                Defaults to "1D".
            initial_cash (float, optional): Starting portfolio cash balance. Defaults to 10000.0.
            fees (float, optional): Commission/slippage fee percentage per trade in basis points (bps). Defaults to 10 bps.
        """

        self.data = data
        self.frequency = frequency
        self.initial_cash = initial_cash
        self.fees = (fees / 10000.0)  # Convert bps to decimal for VectorBT

    def execute_backtest(self, signal_series: pd.Series, cache: bool = True, output_dir: str = "output/backtest") -> vbt.Portfolio:
        """Executes a standard VectorBT portfolio simulation using the generated signal series.

        Args:
            signal_series (pd.Series): Pandas Series containing integer signals (1 for Buy,
                -1 for Sell, 0 for Hold) aligned with the dataset index.
            cache (bool): Whether or not to store backtest results to disk for future analysis.
            output_dir (str): Directory where cache files will be saved. Defaults to "output/backtest".

        Returns:
            vbt.Portfolio: VectorBT Portfolio object containing performance stats,
                trades, orders, and portfolio metrics.
        """
        # Map signal integers to explicit entry and exit booleans
        entries = signal_series == 1
        exits = signal_series == -1

        # Run vectorbt Portfolio simulation
        portfolio = vbt.Portfolio.from_signals(
            close=self.data["Close"],
            entries=entries,
            exits=exits,
            init_cash=self.initial_cash,
            fees=self.fees,
            freq=self.frequency
        )

        # Save portfolio stats and trades to disk for future analysis
        if cache:
            cache_path = Path(output_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

            # Save portfolio stats as CSV
            portfolio.stats().to_csv(cache_path / "portfolio_stats.csv")

            # Save trades and orders as CSV
            portfolio.trades.records_readable.to_csv(cache_path / "trades.csv")
            portfolio.orders.records_readable.to_csv(cache_path / "orders.csv")

        return portfolio