## 2025-02-23 - Avoid intermediate full-size tensor allocation in element-wise math
**Learning:** In PyTorch, allocating large intermediate tensors (like `torch.cat([-x2, x1], dim=-1)` for RoPE) inside high-frequency functions causes memory bandwidth bottlenecks.
**Action:** When performing element-wise math on sliced tensors, calculate and concatenate the results directly (`torch.cat([math_part1, math_part2], dim=-1)`) instead of allocating intermediate full-size tensors. This reduces memory bandwidth usage and lowers latency, especially on GPU hardware.

## 2024-05-07 - Vectorise repetition penalty in autoregressive generation loops
**Learning:** Using `for token_id in set(generated[0].tolist())` in generation loops forces a CPU-GPU synchronisation on every step, causes O(N) Python overhead, and silently breaks for batch sizes > 1 due to hardcoded `[0]` indexing.
**Action:** Replace with vectorised boolean masking: `mask.scatter_(1, generated.clamp(...), True)` + `torch.where(mask, penalised, original)`. Eliminates sync, runs on-device, and correctly handles any batch size.

## 2026-05-26 - Use torch.bincount instead of F.one_hot().sum()
**Learning:** Computing expert fractions in MoE routing via `F.one_hot(indices).sum()` creates an unnecessary 3D intermediate tensor ([N, K, E]), which stresses memory bandwidth and slows down routing.
**Action:** Replace `F.one_hot(indices, num_classes=E).sum()` with `torch.bincount(indices.view(-1), minlength=E)` to directly accumulate counts. This avoids intermediate allocations and is up to ~8x faster on both CPU and GPU.
