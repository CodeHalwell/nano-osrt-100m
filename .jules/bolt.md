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

## 2025-07-11 - Pre-allocate tensor and use torch.isin to avoid implicit CPU-GPU synchronization
**Learning:** In PyTorch, using a Python for-loop (`for sid in stop_token_ids: finished = finished | (nt == sid)`) inside an autoregressive generation loop causes implicit device-to-host synchronization and high O(N) Python overhead, leading to significant latency and memory bandwidth bottlenecks.
**Action:** When repeatedly checking values against a list of options in a hot PyTorch loop (like `stop_token_ids` in text generation), allocate a `stop_tensor` on the appropriate device outside the loop and use `torch.isin(nt, stop_tensor)` inside the loop. This keeps the logic fully vectorized and on-device. Ensure you filter out `None` values before converting to a tensor to avoid `TypeError`s.
