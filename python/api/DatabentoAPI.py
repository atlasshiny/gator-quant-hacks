from typing import Any
import os
import pandas as pd
import databento as db
from databento import DBNStore

from .BaseAPI import BaseAPI
from .Query import FinancialQuery

class DatabentoAPI(BaseAPI):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        **kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)
        # Pass key directly; Historical picks up DATABENTO_API_KEY from environment if key is None
        self.client = db.Historical(key=api_key)

    def query(self, query: FinancialQuery) -> pd.DataFrame | Any:
        """
        Executes a historical time-series query with daily chunking and automatic
        Parquet caching to disk for ultra-fast GPU/cuDF loading.
        """
        extra = getattr(query, "extra_params", {}) or {}

        dataset = extra.get("dataset") or getattr(query, "dataset", "GLBX.MDP3")
        symbols = getattr(query, "symbols", None) or extra.get("symbols") or []
        
        schema = extra.get("schema") or getattr(query, "data_type", "mbp-10")
        if callable(schema):
            schema = extra.get("schema", "mbp-10")

        start = getattr(query, "start_date", None) or getattr(query, "start", None) or extra.get("start")
        end = getattr(query, "end_date", None) or getattr(query, "end", None) or extra.get("end")
        stype_in = extra.get("stype_in") or getattr(query, "stype_in", "raw_symbol")
        limit = getattr(query, "limit", None) or extra.get("limit")

        if not start or not end:
            raise ValueError("Query must specify both 'start' and 'end' timestamps.")

        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]

        start = pd.to_datetime(start, utc=True)
        end = pd.to_datetime(end, utc=True)

        # Generate daily date intervals
        date_range = pd.date_range(start=start, end=end, freq="1D")
        if len(date_range) < 2:
            date_range = [start, end]

        dataset_dir = os.path.join("data", str(dataset))
        os.makedirs(dataset_dir, exist_ok=True)

        symbol_str = "_".join(symbols)
        dfs = []

        for i in range(len(date_range) - 1):
            chunk_start = date_range[i].isoformat()
            chunk_end = date_range[i + 1].isoformat()

            chunk_date_str = chunk_start.split("T")[0]
            base_filename = f"{dataset}_{schema}_{symbol_str}_{chunk_date_str}"
            
            dbn_cache_file = os.path.join(dataset_dir, f"{base_filename}.dbn.zst")
            parquet_cache_file = os.path.join(dataset_dir, f"{base_filename}.parquet")

            # Load directly from cached .parquet
            if os.path.exists(parquet_cache_file):
                print(f"[CACHE HIT - PARQUET] Loading: {parquet_cache_file}")
                dfs.append(pd.read_parquet(parquet_cache_file))
            
            # Download .dbn.zst -> Convert to .parquet
            else:
                if not os.path.exists(dbn_cache_file):
                    print(f"[DOWNLOADING CHUNK] {chunk_date_str} -> {dbn_cache_file}")
                    self.client.timeseries.get_range(
                        path=dbn_cache_file,
                        dataset=dataset,
                        symbols=symbols,
                        schema=schema,
                        start=chunk_start,
                        end=chunk_end,
                        stype_in=stype_in,
                        limit=limit,
                    )

                print(f"[CONVERTING] Converting {dbn_cache_file} -> {parquet_cache_file}")
                chunk_df = DBNStore.from_file(dbn_cache_file).to_df()
                
                # Save to disk using Snappy compression (optimal for cuDF GPU loads)
                chunk_df.to_parquet(parquet_cache_file, engine="pyarrow", compression="snappy")
                dfs.append(chunk_df)

                # Optional: Remove .dbn.zst to preserve disk space once converted
                # if os.path.exists(dbn_cache_file):
                #     os.remove(dbn_cache_file)

        # Concatenate daily chunks
        full_df = pd.concat(dfs, axis=0)
        return full_df.sort_index() if isinstance(full_df.index, pd.DatetimeIndex) else full_df

    def query_async(self, query: FinancialQuery) -> pd.DataFrame | Any:
        """
        Asynchronous query is not implemented for DatabentoAPI.
        """
        raise NotImplementedError("Async query is not implemented for DatabentoAPI.")

if __name__ == "__main__":
    DATASET = "GLBX.MDP3"
    SYMBOL = "ES.c.0"
    SCHEMA = "mbp-10"
    STYPE_IN = "continuous"
    START_DATE = "2024-01-08T00:00:00Z"
    END_DATE = "2024-01-12T23:59:59Z"

    print("Adaptive GPU Alpha Factory Data Ingestion")
    print(f"Dataset : {DATASET}")
    print(f"Symbol  : {SYMBOL}")
    print(f"Schema  : {SCHEMA}")
    print(f"Window  : {START_DATE} to {END_DATE}")

    query = FinancialQuery(
        symbols=[SYMBOL],
        start_date=START_DATE,
        end_date=END_DATE,
        extra_params={
            "dataset": DATASET,
            "schema": SCHEMA,
            "stype_in": STYPE_IN,
        },
    )

    try:
        api = DatabentoAPI()
        df = api.query(query)

        print("\n[SUCCESS] Data successfully staged!")
        print(f"Total Rows Loaded : {len(df):,}")
        print(f"DataFrame Size    : {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB in RAM")
        print("\nFirst 5 Records:")
        print(df.head())

    except Exception as e:
        print(f"\n[ERROR] Failed to fetch or process data: {e}")