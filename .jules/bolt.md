
## 2024-05-18 - PyTorch Anti-Pattern: .tolist() in Autoregressive Loops
**Learning:** Using `set(tensor.tolist())` and a python loop inside the `generate` autoregressive loop forces a severe GPU-to-CPU synchronization at every token generation step and incurs huge python overhead. This is a critical performance killer for PyTorch inference loops.
**Action:** Always replace `.tolist()` conversions and python loops with native vectorized PyTorch operations like `torch.unique()`, boolean masking, and `torch.where()` to keep computations purely on device and avoid syncing overhead.
