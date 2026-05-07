
## 2024-05-07 - [PyTorch Tensor Loop Anti-pattern in Generation]
**Learning:** Autoregressive generation loops suffer massive performance penalties when using `for token_id in set(generated[0].tolist()):` because `.tolist()` forces a CPU-GPU synchronization and data transfer on every generated token.
**Action:** Always replace item-wise tensor inspection loops in performance-critical paths (like generation repetition penalties) with vectorized masking (e.g., `torch.zeros(..., dtype=torch.bool).scatter_()`) and `torch.where`.
