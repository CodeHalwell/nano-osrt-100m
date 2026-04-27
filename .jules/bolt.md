## 2024-05-24 - PyTorch Autoregressive Loop Bottleneck
**Learning:** Using Python loops and list conversions (like `set(tensor.tolist())`) for element-wise tensor modifications in autoregressive generation loops causes severe CPU-GPU synchronization overhead, slowing down generation significantly (e.g. from 5.6s to 9s in a microbenchmark).
**Action:** Always use pure vectorized PyTorch operations like `torch.unique` and `torch.where` to modify tensors inside performance-critical generation loops.
