## 2024-03-XX - PyTorch RoPE Operation Memory Bandwidth Optimization
**Learning:** To optimize PyTorch operations involving element-wise math on sliced tensors (like RoPE), calculate and concatenate the results directly (e.g., `torch.cat([math_part1, math_part2], dim=-1)`) instead of allocating intermediate full-size tensors, as this significantly reduces memory bandwidth usage. However, the exact performance is hardware and device dependent (e.g. GPU might see improvement while CPU sees regression due to caching differences).
**Action:** When attempting memory bandwidth optimizations like in RoPE, only apply them if there's a proven bottleneck or on appropriate hardware where memory bandwidth limits execution speed. Do not apply memory optimizations without hardware validation.

## 2024-03-XX - Vectorized batch dimension checks in stopping conditions
**Learning:** In PyTorch generation loops, avoid using `.item()` on tensors to check stopping conditions, as it causes a RuntimeError for batch sizes > 1. Use vectorized checks with None-safety, such as `eos_token_id is not None and (next_token == eos_token_id).all()`.
**Action:** Avoid `.item()` for multi-dimensional operations unless you explicitly know the output is a single element. Instead, use `.any()` or `.all()`.
