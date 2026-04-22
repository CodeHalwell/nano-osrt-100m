## 2024-05-14 - PyTorch List Conversion Anti-pattern
**Learning:** Using `set(tensor.tolist())` and Python loops for element-wise tensor modifications in an autoregressive loop is a severe performance bottleneck.
**Action:** Always use vectorized operations like `.unique()` and `torch.where()` instead of Python lists and loops to avoid unnecessary CPU-GPU synchronization and Python overhead.
