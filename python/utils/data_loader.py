import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

class RawDatabentoUnifier:
    """
    Polars LazyScan pipeline that unifies daily MBP-10 Parquet files into
    a single, chronologically sorted, float32 binary matrix for GPU loading.
    """
    def __init__(
        self,
        parquet_dir: str = "data/GLBX.MDP3",
        cache_path: str = "data/unified_mbp10.bin",
    ):
        self.parquet_dir = Path(parquet_dir)
        self.cache_path = Path(cache_path)
        
        # Define the 13 Core Event Metadata Columns
        self.core_cols = [
            "ts_event", "ts_recv", "rtype", "publisher_id", "instrument_id", 
            "action", "side", "depth", "price", "size", "flags", "ts_in_delta", "sequence"
        ]
        
        # Define the 60 L2 Order Book Depth Columns
        self.l2_cols = []
        for i in range(10):
            level = f"0{i}"
            self.l2_cols.extend([
                f"bid_px_{level}", f"ask_px_{level}", 
                f"bid_sz_{level}", f"ask_sz_{level}", 
                f"bid_ct_{level}", f"ask_ct_{level}"
            ])
            
        self.all_columns = self.core_cols + self.l2_cols
        self.num_features = len(self.all_columns)  # 73 columns total

    def build_binary_cache(self, force_rebuild: bool = False) -> np.ndarray:
        """
        Lazily scans Parquet files, validates their time order, encodes
        categoricals, and streams to a raw .bin file.
        """
        if not self.parquet_dir.is_dir():
            raise FileNotFoundError(
                f"Parquet directory not found: {self.parquet_dir}"
            )
        parquet_files = sorted(self.parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No Parquet files found in {self.parquet_dir}"
            )

        expected_rows = sum(
            pl.scan_parquet(str(path)).select(pl.len()).collect().item()
            for path in parquet_files
        )
        if self.cache_path.exists() and not force_rebuild:
            item_count, remainder = divmod(
                self.cache_path.stat().st_size, np.dtype(np.float32).itemsize
            )
            cache_rows, cache_remainder = divmod(item_count, self.num_features)
            cache_is_current = (
                remainder == 0
                and cache_remainder == 0
                and cache_rows == expected_rows
                and self.cache_path.stat().st_mtime
                >= max(path.stat().st_mtime for path in parquet_files)
            )
            if cache_is_current:
                print(f"Loading unified binary cache from {self.cache_path}...")
                mmapped = np.memmap(self.cache_path, dtype=np.float32, mode="r")
                return mmapped.reshape((-1, self.num_features))
            print("Existing binary cache is stale or incomplete; rebuilding it.")

        print(f"Scanning and unifying Parquet files in {self.parquet_dir}...")
        start_time = time.perf_counter()
        file_ranges: list[tuple[Path, object, object]] = []
        for path in parquet_files:
            bounds = (
                pl.scan_parquet(str(path))
                .select(
                    pl.col("ts_event").first().alias("first"),
                    pl.col("ts_event").last().alias("last"),
                    pl.col("ts_event").is_sorted().alias("is_sorted"),
                )
                .collect()
                .row(0)
            )
            if not bounds[2]:
                raise ValueError(f"Parquet file is not sorted by ts_event: {path}")
            if file_ranges and bounds[0] < file_ranges[-1][2]:
                raise ValueError(
                    f"Parquet files overlap out of chronological order: "
                    f"{file_ranges[-1][0]} and {path}"
                )
            file_ranges.append((path, bounds[0], bounds[1]))

        # Stream each file in bounded batches so the full dataset does not need
        # to fit in RAM before it can be written.
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as cache_file:
            for path, _, _ in file_ranges:
                file_plan = (
                    pl.scan_parquet(str(path))
                    .with_columns([
                        pl.col("action").replace_strict(
                            {"A": 1, "C": 2, "M": 3, "R": 4, "T": 5},
                            return_dtype=pl.Float32,
                            default=0,
                        ),
                        pl.col("side").replace_strict(
                            {"A": 1, "B": 2, "N": 0},
                            return_dtype=pl.Float32,
                            default=0,
                        ),
                    ])
                    .select([pl.col(c).cast(pl.Float32) for c in self.all_columns])
                )

                def write_batch(batch: pl.DataFrame) -> None:
                    batch.to_numpy().astype(np.float32).tofile(cache_file)

                file_plan.sink_batches(write_batch, chunk_size=16_384)

        matrix = np.memmap(self.cache_path, dtype=np.float32, mode="r")
        matrix = matrix.reshape((-1, self.num_features))

        elapsed = time.perf_counter() - start_time
        print(
            f"Unified {matrix.shape[0]:,} rows x {self.num_features} columns "
            f"in {elapsed:.2f} seconds."
        )
        print(f"Saved binary cache to {self.cache_path}")

        return matrix

    def load_to_vram(self, cpu_matrix: np.ndarray, device: str = "cuda") -> torch.Tensor:
        """Pins host memory and transfers the unified matrix to GPU."""
        # Cache-backed arrays are read-only memory maps; copy before handing the
        # data to PyTorch so the tensor has defined write semantics.
        tensor = torch.from_numpy(np.array(cpu_matrix, dtype=np.float32, copy=True))
        tensor = tensor.contiguous()
        pinned_tensor = tensor.pin_memory()
        return pinned_tensor.to(device=device, non_blocking=True)

if __name__ == "__main__":
    loader = RawDatabentoUnifier()
    
    # First run will take time to stitch and save. Future runs load instantly.
    unified_matrix = loader.build_binary_cache()
    
    # Transfer 73-column schema to HiPerGator GPU
    gpu_tensor = loader.load_to_vram(unified_matrix)
    print(f"GPU Tensor Shape: {gpu_tensor.shape}")