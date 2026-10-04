import argparse
import sys
from pathlib import Path

# Ensure repository root is on Python path if running from subdirectories
sys.path.append(str(Path(__file__).resolve().parent))

from api.DatabentoAPI import DatabentoAPI
from api.Query import FinancialQuery

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive GPU Alpha Factory - High-Throughput Data Staging CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core Query Parameters
    parser.add_argument(
        "--symbol",
        type=str,
        default="ES.c.0",
        help="Ticker or continuous symbol (e.g. ES.c.0, NQ.c.0)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2024-01-08T00:00:00Z",
        help="Start ISO timestamp (YYYY-MM-DDTHH:MM:SSZ)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-01-12T23:59:59Z",
        help="End ISO timestamp (YYYY-MM-DDTHH:MM:SSZ)",
    )
    
    # Provider Flags
    parser.add_argument(
        "--dataset",
        type=str,
        default="GLBX.MDP3",
        help="Databento dataset identifier (e.g., GLBX.MDP3)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="mbp-10",
        help="Order book depth schema (e.g., mbp-10, trades, ohlcv-1m)",
    )
    parser.add_argument(
        "--stype_in",
        type=str,
        default="continuous",
        help="Symbology type (e.g., continuous, raw_symbol, parent)",
    )

    args = parser.parse_args()

    print("\n==================================================")
    print("      Adaptive GPU Alpha Factory Data Ingestion   ")
    print("==================================================")
    print(f"Dataset : {args.dataset}")
    print(f"Symbol  : {args.symbol}")
    print(f"Schema  : {args.schema}")
    print(f"Window  : {args.start} to {args.end}")
    print(f"Symbology: {args.stype_in}")
    print("--------------------------------------------------")

    # Construct the query object passing vendor specifics into extra_params
    query = FinancialQuery(
        symbols=[args.symbol],
        start_date=args.start,
        end_date=args.end,
        extra_params={
            "dataset": args.dataset,
            "schema": args.schema,
            "stype_in": args.stype_in,
        },
    )

    try:
        # Initialize API wrapper and execute chunked query/disk-cache load
        api = DatabentoAPI()
        df = api.query(query)

        print("\n[SUCCESS] Data successfully staged to disk/RAM!")
        print(f"Total Records  : {len(df):,}")
        print(f"Memory Usage   : {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        print("\nDataset Preview:")
        print(df.head(3))

    except Exception as e:
        print(f"\n[FATAL ERROR] Data ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()