## 2024-04-29 - Vectorize repetition penalty in inference loops
**Learning:** In PyTorch, using Python loops and list conversions like `set(tensor.tolist())` for element-wise tensor modifications (such as applying repetition penalty) causes significant performance bottlenecks, particularly in autoregressive generation loops.
**Action:** Always use vectorized operations like `.unique()` and `torch.where()` to process sequence elements in batch, which can yield a ~100x speedup in those specific operations.
