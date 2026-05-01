## 2024-05-01 - Vectorized Repetition Penalty
**Learning:** Element-wise tensor updates using Python loops and `.tolist()` list conversions (e.g., `set(generated[0].tolist())`) in autoregressive generation loops cause significant performance bottlenecks due to CPU-GPU synchronization and loop overhead.
**Action:** Always use vectorized PyTorch operations like `torch.unique()` and `torch.where()` for token-wise modifications inside generation loops, which avoids syncing and properly supports batch size > 1.
