from datetime import date, datetime
from enum import Enum
from typing import Any, Self
from pydantic import BaseModel, Field, ConfigDict, model_validator

class DataInterval(str, Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1m"

class DataType(str, Enum):
    OHLCV = "ohlcv"
    TRADES = "trades"
    QUOTES = "quotes"
    FINANCIALS = "financials"
    NEWS = "news"

class FinancialQuery(BaseModel):
    model_config = ConfigDict(
        use_enum_values=True,
        str_strip_whitespace=True,
        extra="forbid",  # Prevents passing unhandled parameters silently
    )

    # Core Identifiers
    symbols: list[str] = Field(
        ...,
        min_length=1,
        description="List of ticker symbols (e.g., ['AAPL', 'MSFT']). Uppercased automatically.",
    )
    
    # Request Parameters
    data_type: DataType = Field(
        default=DataType.OHLCV,
        description="Type of financial data being requested.",
    )
    interval: DataInterval = Field(
        default=DataInterval.DAY_1,
        description="Bar/candle timeframe interval.",
    )
    
    # Time Ranges
    start_date: date | datetime | None = Field(
        default=None,
        description="Start date/time for historical range.",
    )
    end_date: date | datetime | None = Field(
        default=None,
        description="End date/time for historical range.",
    )
    
    # Pagination / Limits
    limit: int | None = Field(
        default=None,
        ge=1,
        le=50000,
        description="Max records to return per request.",
    )
    
    # Provider-Specific Overflow
    # Pass API-specific flags here (e.g., {"adjusted": True, "feed": "sip"})
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific flags or query parameters.",
    )