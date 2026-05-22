## 2024-05-15 - [Initial]
**Learning:** Initial memory check.
**Action:** Proceed.
## 2024-05-15 - [RoPE Memory Bandwidth Optimization]
**Learning:** PyTorch transformer implementations like RoPE can bottleneck on intermediate memory allocations rather than FLOPs. A naive `torch.cat([-x2, x1], dim=-1)` followed by element-wise operations allocates temporary full-size tensors that waste memory bandwidth.
**Action:** When implementing element-wise math on sliced tensors, calculate and concatenate the results directly (`torch.cat([math_part1, math_part2], dim=-1)`) to avoid intermediate allocations and reduce latency.
