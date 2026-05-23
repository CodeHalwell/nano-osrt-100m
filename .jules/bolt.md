## 2025-02-23 - Avoid intermediate full-size tensor allocation in element-wise math
**Learning:** In PyTorch, allocating large intermediate tensors (like `torch.cat([-x2, x1], dim=-1)` for RoPE) inside high-frequency functions causes memory bandwidth bottlenecks.
**Action:** When performing element-wise math on sliced tensors, calculate and concatenate the results directly (`torch.cat([math_part1, math_part2], dim=-1)`) instead of allocating intermediate full-size tensors. This reduces memory bandwidth usage and lowers latency, especially on GPU hardware.
