import os
import numpy as np
import torch

def load_memmap_tensor(
    file_path: str,
    dtype: np.dtype = np.float32,
    shape: tuple = None,
    device: str = "cuda",
    pin_memory: bool = True
) -> torch.Tensor:
    """
    Memory-maps a binary (.bin/.npy) file and transfers it to a PyTorch tensor.
    
    Args:
        file_path: Path to the binary file on disk.
        dtype: Data type of the stored array (default: np.float32).
        shape: Array dimensions (e.g., (n_timestamps, n_features)). If None, loads 1D.
        device: Destination device ("cuda" or "cpu").
        pin_memory: Whether to lock host RAM for accelerated PCIe transfer.
        
    Returns:
        torch.Tensor on the specified device.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Binary array not found at: {file_path}")

    # Zero-copy memory map on CPU
    mmap_arr = np.memmap(file_path, dtype=dtype, mode="r", shape=shape)

    # Copy the read-only mapping so PyTorch receives writable storage.
    tensor_cpu = torch.from_numpy(np.array(mmap_arr, copy=True))

    # Handle device transfer with optional pinned memory
    if device.startswith("cuda"):
        if pin_memory:
            # Pin host memory for maximum PCIe transfer speeds
            tensor_cpu = tensor_cpu.pin_memory()
        return tensor_cpu.to(device, non_blocking=True)
    
    return tensor_cpu