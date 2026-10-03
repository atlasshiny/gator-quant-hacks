from typing import Any
import os
import pandas as pd
from databento import Historical, DBNStore

from BaseAPI import BaseAPI
from Query import FinancialQuery

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
        self.client = Historical(key=api_key)

    def query(self, query: FinancialQuery) -> pd.DataFrame | Any:
        """
        Executes a historical time-series query using Databento's SDK, with local caching in data/[DATASET]/.
        """
        # Extract query attributes (adapting to dictionary or object attributes)
        dataset = getattr(query, "dataset", None) or getattr(query, "provider_dataset", "GLBX.MDP3")
        symbols = getattr(query, "symbols", None) or getattr(query, "tickers", [])
        schema = getattr(query, "schema", None) or getattr(query, "data_type", "mbp-10")
        start = getattr(query, "start", None) or getattr(query, "start_date", None)
        end = getattr(query, "end", None) or getattr(query, "end_date", None)
        stype_in = getattr(query, "stype_in", "raw_symbol")
        limit = getattr(query, "limit", None)

        if not start:
            raise ValueError("Query must specify a 'start' timestamp.")

        # Ensure symbols is formatted as a list or comma-separated string
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]

        # Create directory: data/[DATASET_TYPE]/
        dataset_dir = os.path.join("data", dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        # Generate a unique cache filename
        symbol_str = "_".join(symbols)
        filename = f"{dataset}_{schema}_{symbol_str}_{start}_{end}.dbn.zst".replace(":", "-")
        cache_file = os.path.join(dataset_dir, filename)

        # Check local cache first to prevent duplicate API charges
        if os.path.exists(cache_file):
            print(f"Loading cached data from: {cache_file}")
            return DBNStore.from_file(cache_file).to_df()

        # Call Databento Historical Timeseries API and save directly to data/[DATASET_TYPE]/[FILE]
        print(f"Downloading new data to: {cache_file}")
        self.client.historical.timeseries.get_range_to_file(
            path=cache_file,
            dataset=dataset,
            symbols=symbols,
            schema=schema,
            start=start,
            end=end,
            stype_in=stype_in,
            limit=limit,
        )

        # Load the newly saved file into a DataFrame
        return DBNStore.from_file(cache_file).to_df()

    def query_async(self, query: FinancialQuery) -> pd.DataFrame | Any:
        """
        Executes an asynchronous historical time-series query using Databento's SDK.
        Note: Databento's SDK may not support async natively; this is a placeholder.
        """
        raise NotImplementedError("Async query is not implemented for DatabentoAPI.")

if __name__ == "__main__":
    # Initialize API wrapper
    api = DatabentoAPI()

    # Define Query Parameters
    DATASET = "GLBX.MDP3"
    SYMBOL = "ES.c.0" # Continuous front-month symbol
    SCHEMA = "mbp-10" # Top 10 level order book depth
    STYPE_IN = "continuous" # Instruct Databento to resolve continuous symbology
    START_DATE = "2024-01-08T00:00:00Z" # Monday
    END_DATE = "2024-01-12T23:59:59Z" # Friday

    print("Adaptive GPU Alpha Factory Data Ingestion")
    print(f"Dataset : {DATASET}")
    print(f"Symbol  : {SYMBOL}")
    print(f"Schema  : {SCHEMA}")
    print(f"Window  : {START_DATE} to {END_DATE}")

    query = FinancialQuery(
        dataset=DATASET,
        symbols=[SYMBOL],
        schema=SCHEMA,
        start=START_DATE,
        end=END_DATE,
        stype_in=STYPE_IN,
    )

    try:
        # Execute query (will pull from data/GLBX.MDP3/ if cached, or download new)
        df = api.query(query)

        print("\n[SUCCESS] Data successfully staged!")
        print(f"Total Rows Loaded : {len(df):,}")
        print(f"DataFrame Size    : {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB in RAM")
        print("\nFirst 5 Records:")
        print(df.head())

    except Exception as e:
        print(f"\n[ERROR] Failed to fetch or process data: {e}")    