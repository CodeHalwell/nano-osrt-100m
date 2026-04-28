## 2024-05-18 - Avoid Python Loops for Element-wise Tensor Operations
**Learning:** Python loops and list conversions (like `set(tensor.tolist())`) are very slow in autoregressive generation loops because they break out of the optimized C++ backend and cause synchronization overhead.
**Action:** Use vectorized operations like boolean masking, `.unique()`, and `torch.where()` for element-wise tensor modifications instead.
