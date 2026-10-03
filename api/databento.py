from typing import Any
import pandas as pd
from databento import DatabentoClient

from api.BaseAPI import BaseAPI
from api.Query import FinancialQuery

class DatabentoAPI(BaseAPI):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        **kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)
        # Pass api_key directly; DatabentoClient loads DATABENTO_API_KEY from environment if None
        self.client = DatabentoClient(api_key=api_key, base_url=base_url)

    def query(self, query: FinancialQuery) -> pd.DataFrame | Any:
        """
        Executes a historical time-series query using Databento's SDK.
        
        Expects `query` to contain:
        - dataset (str): e.g., 'GLBX.MDP3', 'XNAS.ITCH', 'OPRA.PILLAR'
        - symbols (str | list[str]): e.g., 'ES.c.0' or ['AAPL', 'MSFT']
        - schema (str): e.g., 'trades', 'mbp-1', 'ohlcv-1m'
        - start (str | datetime): Start time (ISO 8601 string, nanoseconds integer, or datetime)
        - end (str | datetime, optional): End time
        """
        # Extract query attributes (adapting to dictionary or object attributes)
        dataset = getattr(query, "dataset", None) or getattr(query, "provider_dataset", "GLBX.MDP3")
        symbols = getattr(query, "symbols", None) or getattr(query, "tickers", [])
        schema = getattr(query, "schema", None) or getattr(query, "data_type", "trades")
        start = getattr(query, "start", None) or getattr(query, "start_date", None)
        end = getattr(query, "end", None) or getattr(query, "end_date", None)
        stype_in = getattr(query, "stype_in", "raw_symbol")
        limit = getattr(query, "limit", None)

        if not start:
            raise ValueError("Query must specify a 'start' timestamp.")

        # Ensure symbols is formatted as a list or comma-separated string
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]

        # Call Databento Historical Timeseries API
        # `get_range` returns a DBNStore containing data in binary DBN format
        data_store = self.client.historical.timeseries.get_range(
            dataset=dataset,
            symbols=symbols,
            schema=schema,
            start=start,
            end=end,
            stype_in=stype_in,
            limit=limit,
        )

        # Format output
        # Convert directly to a pandas DataFrame (handles timestamps and schemas automatically)
        return data_store.to_df()