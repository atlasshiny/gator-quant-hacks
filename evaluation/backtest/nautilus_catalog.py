"""
nautilus_catalog.py

One-time conversion: Databento DBN files on disk (HiPerGator) -> NautilusTrader
Parquet data catalog.

    mbp-10 files       -> OrderBookDepth10 records
    definition files   -> Instrument objects (tick size, multiplier, price precision)

Environment variables (optional):
    DATA_DIR         folder holding the .dbn.zst files      (default data/GLBX.MDP3)
    CATALOG_DIR      where the Nautilus catalog is written  (default ./catalog)
    PUBLISHERS_JSON  path to Nautilus' publishers.json, only needed if the loader
                     can't find it next to the executable
    INSTRUMENT_ID    Nautilus id of the contract in the files (default ESH4.GLBX)

IMPORTANT: Nautilus needs an instrument definition before it can write or replay
market data. If you don't have a `definition` schema file for the contract, pull it
once from Databento (it is tiny) and drop it in DATA_DIR.

The depth loader labels EVERY record in a file with INSTRUMENT_ID, so each file must
contain a single contract (e.g. January 2024 front month = ESH4). If a file spans a
roll, split it or filter it first.
"""

import os
from pathlib import Path

from nautilus_trader.adapters.databento import DatabentoDataLoader
from nautilus_trader.model import InstrumentId
from nautilus_trader.persistence import ParquetDataCatalog

DATA_DIR = Path(os.environ.get("DATA_DIR", "data/GLBX.MDP3"))
CATALOG_DIR = Path(os.environ.get("CATALOG_DIR", "catalog"))
PUBLISHERS_JSON = os.environ.get("PUBLISHERS_JSON")
INSTRUMENT_ID = os.environ.get("INSTRUMENT_ID", "ESH4.GLBX")


def build_catalog(
    depth_pattern: str = "*mbp-10*.dbn.zst",
    definition_pattern: str = "*definition*.dbn.zst",
) -> ParquetDataCatalog:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(str(CATALOG_DIR))
    loader = (
        DatabentoDataLoader(publishers_filepath=PUBLISHERS_JSON)
        if PUBLISHERS_JSON
        else DatabentoDataLoader()
    )

    # 1) Instruments first: the catalog needs them before it can write market data,
    #    and loading them also seeds the loader's price-precision cache.
    def_files = sorted(DATA_DIR.glob(definition_pattern))
    if not def_files:
        raise FileNotFoundError(
            f"No definition files matching '{definition_pattern}' in {DATA_DIR}. "
            "Pull the Databento 'definition' schema once and place it there."
        )
    for f in def_files:
        instruments = loader.load_instruments(filepath=str(f), use_exchange_as_venue=False)
        catalog.write_instruments(instruments)
        print(f"Wrote {len(instruments)} instruments from {f.name}")

    # 2) mbp-10 -> OrderBookDepth10, one file at a time to keep memory bounded.
    depth_files = sorted(DATA_DIR.glob(depth_pattern))
    if not depth_files:
        raise FileNotFoundError(f"No files matching '{depth_pattern}' in {DATA_DIR}")

    instrument_id = InstrumentId.from_str(INSTRUMENT_ID)
    for f in depth_files:
        depth = loader.load_order_book_depth10(filepath=str(f), instrument_id=instrument_id)
        catalog.write_order_book_depths(depth)
        print(f"Wrote {len(depth):,} OrderBookDepth10 from {f.name}")

    print(f"\nCatalog ready at {CATALOG_DIR.resolve()}")
    print("Instruments:", [str(i.id) for i in catalog.instruments()])
    return catalog


if __name__ == "__main__":
    print(f"PUBLISHERS_JSON={PUBLISHERS_JSON}")
    build_catalog()