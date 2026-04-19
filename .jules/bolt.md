## 2024-06-03 - Vectorizing Autoregressive Tensor Operations
**Learning:** Using Python loops and list conversions (like `set(tensor.tolist())`) for element-wise tensor modifications in an autoregressive generation loop is extremely slow. We found that vectorizing the repetition penalty application is ~168x faster (0.359 ms vs 60.649 ms).
**Action:** Always use vectorized operations like `torch.unique` and `torch.where` for tensor modifications in generation loops instead of falling back to Python primitives.
