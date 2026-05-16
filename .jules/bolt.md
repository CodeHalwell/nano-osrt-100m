
## 2024-05-18 - Optimized RoPE Memory Operations
**Learning:** For element-wise operations on sliced tensors (like applying RoPE), allocating full-sized intermediate tensors (e.g., `x_rot = torch.cat([-x2, x1], dim=-1)`) followed by another full-size addition consumes unnecessary memory bandwidth.
**Action:** Calculate and concatenate the results directly (e.g., `torch.cat([x1*cos1 - x2*sin1, x2*cos2 + x1*sin2], dim=-1)`) to avoid intermediate allocations. This is especially beneficial for bandwidth-bound ops on GPU.
