## 2024-06-11 - Torch bincount over F.one_hot for MoE Router Counts
**Learning:** In PyTorch, when calculating expert fractions or aggregating occurrences in MoE routing, `F.one_hot(indices, num_classes=E).sum()` creates massive 3D intermediate tensors of shape `(Batch * SeqLen, TopK, NumExperts)`.
**Action:** Always replace `F.one_hot(...).sum()` with `torch.bincount(indices.view(-1), minlength=E)` for 1D counts, or use `scatter_add_` for 2D batch grouping, to save memory bandwidth and speed up forward passes.

## 2025-02-23 - Avoid intermediate full-size tensor allocation in element-wise math
**Learning:** In PyTorch, allocating large intermediate tensors (like `torch.cat([-x2, x1], dim=-1)` for RoPE) inside high-frequency functions causes memory bandwidth bottlenecks.
**Action:** When performing element-wise math on sliced tensors, calculate and concatenate the results directly (`torch.cat([math_part1, math_part2], dim=-1)`) instead of allocating intermediate full-size tensors. This reduces memory bandwidth usage and lowers latency, especially on GPU hardware.

## 2024-05-07 - Vectorise repetition penalty in autoregressive generation loops
**Learning:** Using `for token_id in set(generated[0].tolist())` in generation loops forces a CPU-GPU synchronisation on every step, causes O(N) Python overhead, and silently breaks for batch sizes > 1 due to hardcoded `[0]` indexing.
**Action:** Replace with vectorised boolean masking: `mask.scatter_(1, generated.clamp(...), True)` + `torch.where(mask, penalised, original)`. Eliminates sync, runs on-device, and correctly handles any batch size.

## 2025-05-27 - Remove one_hot in favor of bincount
**Learning:** Using `F.one_hot(indices).sum()` to count token assignments to experts creates a large 3D intermediate tensor (e.g., `(B*S, K, E)`), which causes unnecessary peak memory spikes and slows down MoE routing via memory bandwidth constraints.
**Action:** Replace `F.one_hot(indices, num_classes=E).sum(...)` with `torch.bincount(indices.view(-1), minlength=E)`. This computes the assignment counts directly using integers, saving memory and offering a speedup. Always check for downstream references to the removed one-hot variable before deleting it completely (e.g., it might be used to construct sequential routing signals).

## 2025-05-29 - Avoid one_hot for 2D aggregation
**Learning:** Using `F.one_hot(...).sum()` to count occurrences where a 2D grouping (like per batch item) is needed creates a large 4D intermediate tensor.
**Action:** Initialize a zero tensor (`torch.zeros(B, E)`) and use `scatter_add_` with a tensor of ones to accumulate counts. This provides a significant speedup and prevents out-of-memory bottlenecks.

## $(date +%Y-%m-%d) - RoPE Intermediate Tensor Allocation Bottleneck
**Learning:** In the Rotary Positional Embedding (RoPE) implementation, constructing a rotated tensor via `x_rot = torch.cat([-x2, x1], dim=-1)` followed by `x * cos + x_rot * sin` incurs a massive memory allocation and bandwidth penalty, essentially allocating a full-sized hidden state tensor just for intermediate operations.
**Action:** Always compute RoPE directly on slice references. Restructuring the math to `torch.cat([x1 * cos1 - x2 * sin1, x2 * cos2 + x1 * sin2], dim=-1)` entirely avoids the intermediate full-size tensor allocation, significantly reducing memory bandwidth bottlenecks and resulting in faster execution on both CPU and GPU environments without changing mathematical output.
