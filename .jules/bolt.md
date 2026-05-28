## 2025-02-23 - Avoid intermediate full-size tensor allocation in element-wise math
**Learning:** In PyTorch, allocating large intermediate tensors (like `torch.cat([-x2, x1], dim=-1)` for RoPE) inside high-frequency functions causes memory bandwidth bottlenecks.
**Action:** When performing element-wise math on sliced tensors, calculate and concatenate the results directly (`torch.cat([math_part1, math_part2], dim=-1)`) instead of allocating intermediate full-size tensors. This reduces memory bandwidth usage and lowers latency, especially on GPU hardware.

## 2024-05-07 - Vectorise repetition penalty in autoregressive generation loops
**Learning:** Using `for token_id in set(generated[0].tolist())` in generation loops forces a CPU-GPU synchronisation on every step, causes O(N) Python overhead, and silently breaks for batch sizes > 1 due to hardcoded `[0]` indexing.
**Action:** Replace with vectorised boolean masking: `mask.scatter_(1, generated.clamp(...), True)` + `torch.where(mask, penalised, original)`. Eliminates sync, runs on-device, and correctly handles any batch size.
## 2025-05-27 - Remove one_hot in favor of bincount
**Learning:** Using `F.one_hot(indices).sum()` to count token assignments to experts creates a large 3D intermediate tensor (e.g., `(B*S, K, E)`), which causes unnecessary peak memory spikes and slows down MoE routing via memory bandwidth constraints.
**Action:** Replace `F.one_hot(indices, num_classes=E).sum(...)` with `torch.bincount(indices.view(-1), minlength=E)`. This computes the assignment counts directly using integers, saving memory and offering a speedup. Always check for downstream references to the removed one-hot variable before deleting it completely (e.g., it might be used to construct sequential routing signals).

## 2026-05-28 - Optimize batch sequence counts in PyTorch with scatter_add_
**Learning:** Using `F.one_hot(...).sum()` to count sequence-wise token assignments creates large intermediate tensors (e.g., `(B, S, K, E)`), which slows down MoE routing via memory bandwidth constraints.
**Action:** When a 2D batch count `(B, E)` is needed, replace `F.one_hot(...).sum()` with an initialized zero tensor `torch.zeros(B, E)` and `scatter_add_`.
