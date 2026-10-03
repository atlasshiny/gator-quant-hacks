import os
from typing import Any
import pandas as pd
import requests

from api.BaseAPI import BaseAPI
from api.Query import FinancialQuery

class FredAPI(BaseAPI):
    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        **kwargs,
    ):
        # Fall back to environment variable if api_key is omitted
        api_key = api_key or os.getenv("FRED_API_KEY")
        base_url = base_url or os.getenv("FRED_BASE_URL", self.BASE_URL)

        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)

        if not self.api_key:
            raise ValueError(
                "FRED API key is required. Pass api_key or set FRED_API_KEY in .env"
            )

        # Requests session for HTTP connections
        self.session = requests.Session()

    def query(self, query: FinancialQuery) -> pd.DataFrame:
        """
        Executes a historical time-series query against the St. Louis Fed (FRED) API.

        Expects `query` to map:
        - symbols / tickers -> FRED Series IDs (e.g. 'GDP', 'CPIAUCSL', 'UNRATE', 'FEDFUNDS')
        - start / start_date -> observation_start (YYYY-MM-DD string)
        - end / end_date -> observation_end (YYYY-MM-DD string)
        - schema / data_type -> units/units aggregation or transformation option
        """
        # Extract symbol/series ID
        symbols = getattr(query, "symbols", None) or getattr(query, "tickers", [])
        if isinstance(symbols, list):
            if not symbols:
                raise ValueError("FRED Query requires at least one symbol/series_id.")
            series_id = symbols[0]  # FRED observations endpoint takes 1 series at a time
        else:
            series_id = str(symbols).strip()

        # Extract dates
        start = getattr(query, "start", None) or getattr(query, "start_date", None)
        end = getattr(query, "end", None) or getattr(query, "end_date", None)

        # Optional FRED transformations / schemas
        # Units: 'lin' (levels), 'chg' (change), 'ch1' (change from year ago), 'pch' (% change), 'pca' (compounded annual % change)
        units = getattr(query, "schema", None) or getattr(query, "units", "lin")
        frequency = getattr(query, "frequency", None)  # 'm' (monthly), 'q' (quarterly), 'a' (annual)
        limit = getattr(query, "limit", 100000)

        # Build parameters
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "limit": limit,
            "units": units if units in ["lin", "chg", "ch1", "pch", "pca", "cch", "cca"] else "lin",
        }

        if start:
            params["observation_start"] = str(start)[:10]
        if end:
            params["observation_end"] = str(end)[:10]
        if frequency:
            params["frequency"] = frequency

        # Pass dynamic extra params (e.g., vintage_dates, realtime_start)
        if hasattr(query, "extra_params") and isinstance(query.extra_params, dict):
            params.update(query.extra_params)

        # Call FRED API observations endpoint
        endpoint = f"{self.base_url.rstrip('/')}/series/observations"
        response = self.session.get(endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        observations = data.get("observations", [])

        if not observations:
            return pd.DataFrame(columns=["date", series_id])

        # Parse observations into a pandas DataFrame
        df = pd.DataFrame(observations)
        df = df[["date", "value"]].copy()

        # Handle FRED missing value placeholders represented by '.'
        df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
        df["date"] = pd.to_datetime(df["date"])

        # Rename 'value' column to the queried series ID
        df.rename(columns={"value": series_id}, inplace=True)
        df.set_index("date", inplace=True)

        return df

if __name__ == "__main__":
    from api.Query import FinancialQuery

    query = FinancialQuery(
        dataset="FRED",
        symbols="GDP",
        schema="lin",
        start="2024-01-01T00:00:00Z"
    )
    api = FredAPI()
    df = api.query(query)
    print(df)