## 2024-05-24 - Vectorized Repetition Penalty
**Learning:** In PyTorch, using Python loops and list conversions (like `set(tensor.tolist())`) for element-wise tensor modifications inside autoregressive generation loops causes significant performance bottlenecks.
**Action:** Always use vectorized tensor operations (e.g., `torch.unique()`, `torch.where()`) instead of Python `for` loops over elements when working with tensors, especially in hot paths like generation loops.
