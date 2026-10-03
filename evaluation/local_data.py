"""
local_data.py
 
Load Databento data already stored on disk (e.g. UF HiPerGator). 

Set the data directory once via the DATA_DIR environment variable, e.g.
    export DATA_DIR=/blue/<group>/<user>/data/GLBX.MDP3
or pass data_dir=... explicitly.
"""
 
from __future__ import annotations
 
import os
from pathlib import Path
 
import pandas as pd
from databento import DBNStore
 
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "data/GLBX.MDP3")
 
# Top-of-book columns used by the example strategy. Keeping only the columns
TOP_OF_BOOK = ["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
 
 
def list_files(data_dir: str | os.PathLike | None = None, pattern: str = "*.dbn.zst") -> list[Path]:
    d = Path(data_dir or DEFAULT_DATA_DIR)
    if not d.exists():
        raise FileNotFoundError(f"Data directory not found: {d}  (set DATA_DIR or pass data_dir=)")
    files = sorted(d.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {d}")
    return files
 
 
def load_local(
    data_dir: str | os.PathLike | None = None,
    pattern: str = "*.dbn.zst",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    columns: list[str] | None = None,
    symbols: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Read every matching DBN file, trim to [start, end), optionally keep only some
    columns / raw symbols, and return one time-sorted DataFrame.
 
    Files are loaded one at a time and trimmed immediately, so peak memory is
    about one file's worth, not the whole dataset.
 
    start/end : UTC timestamps, end exclusive (same convention as Databento).
    columns   : e.g. TOP_OF_BOOK. None keeps everything.
    symbols   : filter on the 'symbol' column (raw symbols like "ESH4").
    """
    t0 = pd.Timestamp(start, tz="UTC") if start is not None else None
    t1 = pd.Timestamp(end, tz="UTC") if end is not None else None
 
    parts = []
    for f in list_files(data_dir, pattern):
        df = DBNStore.from_file(f).to_df()
        if df.empty:
            continue
        if t0 is not None:
            df = df[df.index >= t0]
        if t1 is not None:
            df = df[df.index < t1]
        if symbols is not None and "symbol" in df.columns:
            df = df[df["symbol"].isin(symbols)]
        if columns is not None:
            df = df[columns]
        if verbose:
            print(f"Loaded {f.name}: {len(df):,} rows")
        if len(df):
            parts.append(df)
 
    if not parts:
        raise ValueError("No rows left after filtering. Check start/end/symbols/pattern.")
 
    out = pd.concat(parts).sort_index()
    if verbose:
        mb = out.memory_usage(deep=True).sum() / 1024**2
        print(f"Total: {len(out):,} rows, {mb:,.0f} MB in RAM")
    return out