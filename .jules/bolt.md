## 2024-04-18 - Avoid Python loop & list in autoregressive generation
**Learning:** Python loops over list conversions like `set(tensor.tolist())` in autoregressive generation are major performance bottlenecks.
**Action:** Replace them with vectorized operations, such as identifying unique tokens using `torch.unique` and applying penalties using `torch.where`. This provides significant speedups during token generation.
