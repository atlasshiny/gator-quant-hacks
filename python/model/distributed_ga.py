import csv
import json
import os
import time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from python.config import load_config

# Import the optimized evaluation pipeline and chromosome schema
from .evolution_kernel import (
    _count_rows_from_binary,
    evaluate_population,
    load_memmap_tensor,
)
from .chromosome import StrategyChromosome

# DISTRIBUTED ENVIRONMENT INITIALIZATION
def setup(rank, world_size):
    """Initializes the PyTorch distributed backend via NCCL."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def crossover_and_mutate(
    population: np.ndarray, 
    fitness: torch.Tensor, 
    elite_ratio: float = 0.1, 
    mutation_rate: float = 0.10
) -> np.ndarray:
    """
    Performs vectorized single-point bitwise crossover and mutation on 64-bit chromosomes.
    """
    n_pop = len(population)
    n_elite = max(1, int(n_pop * elite_ratio))
    n_offspring = n_pop - n_elite

    # Keep the absolute best performers untouched so we never lose good traits
    elite_indices = torch.topk(fitness, k=n_elite, largest=True).indices.cpu().numpy()
    elites = population[elite_indices]

    # Randomly select parents from the elite pool to breed the next generation.
    # (In a more advanced setup, you could use Tournament Selection here)
    p1_indices = np.random.randint(0, n_elite, size=n_offspring)
    p2_indices = np.random.randint(0, n_elite, size=n_offspring)
    
    p1 = elites[p1_indices]
    p2 = elites[p2_indices]

    # Pick a random bit index [1, 63] for each mating pair to split the chromosome
    crossover_points = np.random.randint(1, 64, size=n_offspring).astype(np.uint64)
    
    # Create bitmasks. E.g., if crossover_point = 3, mask = 000...00111
    # Note: Explicit np.uint64(1) prevents Python from defaulting to 32-bit overflow
    masks = (np.uint64(1) << crossover_points) - np.uint64(1)

    # The child inherits the lower bits from Parent 1 and the upper bits from Parent 2
    offspring = (p1 & masks) | (p2 & ~masks)

    # Determine which specific offspring will undergo a random bit-flip
    mutate_mask = np.random.rand(n_offspring) < mutation_rate
    n_mutations = np.sum(mutate_mask)

    if n_mutations > 0:
        # Pick a random bit [0, 63] to flip for each chosen offspring
        bits_to_flip = np.random.randint(0, 64, size=n_mutations).astype(np.uint64)
        
        # Create a mask with a 1 at the specific bit position
        flip_masks = np.uint64(1) << bits_to_flip
        
        # XOR (^) operator flips the bit (0 becomes 1, 1 becomes 0)
        offspring[mutate_mask] ^= flip_masks

    # Combine untouched elites with the newly bred offspring
    return np.concatenate([elites, offspring])

# ISLAND MODEL EXECUTION LOOP (PER GPU)
def run_island_node(rank, world_size, data_bin, returns_bin, total_population_size, generations, migration_freq):
    setup(rank, world_size)
    try:
        if total_population_size <= 0 or total_population_size % world_size != 0:
            raise ValueError("total_population_size must be positive and divisible by world_size.")
        if generations <= 0:
            raise ValueError("generations must be positive.")
        if migration_freq <= 0:
            raise ValueError("migration_freq must be positive.")

        # Divide the total population evenly across available GPUs
        local_pop_size = total_population_size // world_size
        if local_pop_size < world_size:
            raise ValueError("Each GPU must have at least world_size strategies for migration.")

        # Initialize random starting bitmasks for this specific GPU's island
        local_population = np.array([
            StrategyChromosome.encode(
                np.random.randint(0, 0xFFFFFFF),
                np.random.randint(0, 4096),
                np.random.randint(0, 4096),
                np.random.randint(0, 4096)
            ) for _ in range(local_pop_size)
        ], dtype=np.uint64)

        print(f"[GPU {rank}] Island Initialized with {local_pop_size} strategies.")

        output_dir = Path(os.environ["GA_OUTPUT_DIR"])
        history: list[dict[str, float | int]] = []
        global_best: dict[str, float | int] | None = None
        n_features = 28
        n_samples = _count_rows_from_binary(data_bin, n_features)
        data_gpu = load_memmap_tensor(
            data_bin,
            dtype=np.float32,
            shape=(n_samples, n_features),
            device=f"cuda:{rank}",
        )
        returns_gpu = load_memmap_tensor(
            returns_bin,
            dtype=np.float32,
            shape=(n_samples,),
            device=f"cuda:{rank}",
        ).contiguous()
        torch.cuda.synchronize(rank)
        if rank == 0:
            print(
                f"[GA] Loaded {n_samples:,} samples x {n_features} features "
                "onto each GPU.",
                flush=True,
            )

        for gen in range(generations):
            eval_start = time.perf_counter()
            fitness = evaluate_population(
                data_bin_path=data_bin,
                population_bitmasks=local_population,
                n_features=n_features,
                data_gpu=data_gpu,
                returns_gpu=returns_gpu,
                n_samples=n_samples,
            )
            torch.cuda.synchronize(rank)
            eval_seconds = time.perf_counter() - eval_start

            best_idx = torch.argmax(fitness)
            local_best_score = fitness[best_idx].detach()
            local_best_bits = torch.tensor(
                [local_population[best_idx.item() : best_idx.item() + 1].view(np.int64)[0]],
                dtype=torch.int64,
                device=f"cuda:{rank}",
            )
            gathered_scores = [
                torch.empty(1, dtype=torch.float32, device=f"cuda:{rank}")
                for _ in range(world_size)
            ]
            gathered_bits = [
                torch.empty(1, dtype=torch.int64, device=f"cuda:{rank}")
                for _ in range(world_size)
            ]
            dist.all_gather(gathered_scores, local_best_score.reshape(1))
            dist.all_gather(gathered_bits, local_best_bits)

            if rank == 0:
                best_rank = max(
                    range(world_size),
                    key=lambda index: gathered_scores[index].item(),
                )
                best_score = gathered_scores[best_rank].item()
                best_bits = np.array(
                    [gathered_bits[best_rank].item()], dtype=np.int64
                ).view(np.uint64)[0]
                history.append({"generation": gen, "fitness": best_score})
                if global_best is None or best_score > float(global_best["fitness"]):
                    global_best = {
                        "generation": gen,
                        "fitness": best_score,
                        "chromosome": int(best_bits),
                    }
                    print(
                        f"[GA] New all-time best at generation {gen}: "
                        f"{best_score:.3f}",
                        flush=True,
                    )
                print(
                    f"[Generation {gen:03d}] best={best_score:.3f} "
                    f"eval={eval_seconds:.2f}s "
                    f"throughput={local_pop_size * n_samples / eval_seconds:,.0f} "
                    "strategy-samples/s",
                    flush=True,
                )

            local_elite_host = local_population[best_idx.item():best_idx.item() + 1].copy()
            local_population = crossover_and_mutate(local_population, fitness)

            # Migration occurs after breeding so received elites are evaluated
            # by the next generation rather than paired with stale fitness.
            if gen > 0 and gen % migration_freq == 0:
                # NCCL does not support uint64 collectives. Transfer the
                # chromosome bits as int64, then reinterpret them as uint64.
                local_elite = torch.from_numpy(
                    local_elite_host.view(np.int64)
                ).to(device=f"cuda:{rank}")
                gathered_elites = [
                    torch.zeros(1, dtype=torch.int64, device=f"cuda:{rank}")
                    for _ in range(world_size)
                ]
                dist.all_gather(gathered_elites, local_elite)

                if rank == 0:
                    print(f"--- Generation {gen} Migration ---", flush=True)

                for i, elite_tensor in enumerate(gathered_elites):
                    received_bits = np.array(
                        [elite_tensor.item()], dtype=np.int64
                    ).view(np.uint64)[0]
                    local_population[-world_size + i] = received_bits

        if rank == 0 and global_best is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            best_strategy = StrategyChromosome.decode(global_best["chromosome"])
            result = {
                **global_best,
                "world_size": world_size,
                "population": total_population_size,
                "generations": generations,
                "migration_frequency": migration_freq,
                "features_path": data_bin,
                "returns_path": returns_bin,
                "strategy": best_strategy,
            }
            with (output_dir / "best_strategy.json").open("w", encoding="utf-8") as file:
                json.dump(result, file, indent=2)
            with (output_dir / "fitness_history.csv").open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=["generation", "fitness"])
                writer.writeheader()
                writer.writerows(history)
            print(f"Saved best strategy: {output_dir / 'best_strategy.json'}")
            print(f"Saved fitness history: {output_dir / 'fitness_history.csv'}")
    finally:
        cleanup()

# MULTI-PROCESS SPAWNER
if __name__ == "__main__":
    # Cluster Configuration
    WORLD_SIZE = torch.cuda.device_count()
    if WORLD_SIZE < 1:
        raise RuntimeError("No GPUs detected. DDP requires at least 1 GPU.")
    
    print(f"Launching distributed GA across {WORLD_SIZE} GPUs...")
    
    # Run Parameters
    config = load_config()
    ga_config = config.genetic_algorithm
    TOTAL_POPULATION = ga_config.population
    GENERATIONS = ga_config.generations
    MIGRATION_FREQ = ga_config.migration_frequency

    data_bin = str(ga_config.features_path)
    returns_bin = str(ga_config.returns_path)
    os.environ["GA_OUTPUT_DIR"] = str(ga_config.output_dir)
    for path in (data_bin, returns_bin):
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise FileNotFoundError(
                f"Production GA input is missing or empty: {path}. "
                "Run the feature/XGBoost stage first."
            )
    print(f"Using production GA features: {data_bin}")
    print(f"Using production GA returns: {returns_bin}")

    # Spawn a process for each GPU
    mp.spawn(
        run_island_node,
        args=(WORLD_SIZE, data_bin, returns_bin, TOTAL_POPULATION, GENERATIONS, MIGRATION_FREQ),
        nprocs=WORLD_SIZE,
        join=True
    )