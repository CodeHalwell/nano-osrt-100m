## 2024-06-11 - Torch bincount over F.one_hot for MoE Router Counts
**Learning:** In PyTorch, when calculating expert fractions or aggregating occurrences in MoE routing, `F.one_hot(indices, num_classes=E).sum()` creates massive 3D intermediate tensors of shape `(Batch * SeqLen, TopK, NumExperts)`.
**Action:** Always replace `F.one_hot(...).sum()` with `torch.bincount(indices.view(-1), minlength=E)` for 1D counts, or use `scatter_add_` for 2D batch grouping, to save memory bandwidth and speed up forward passes.
