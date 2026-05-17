## 2024-05-17 - PyTorch RoPE memory allocation bottleneck
**Learning:** In PyTorch generation loops, applying RoPE via `x_rot = torch.cat([-x2, x1], dim=-1)` allocates an intermediate full-size tensor (`x_rot`) and performs element-wise operations on the full size, leading to increased memory bandwidth usage.
**Action:** Calculate and concatenate the results directly (e.g., `torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)`) to reduce intermediate allocations and improve performance.
