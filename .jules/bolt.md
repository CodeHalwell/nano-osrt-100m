## 2025-04-14 - MoE Dispatch Memory Allocation
**Learning:** In PyTorch, using `.expand()` and then `.reshape()` to duplicate tokens for top-k MoE dispatch forces a contiguous memory allocation because the reshape cannot be represented as a view of the 0-stride expanded tensor.
**Action:** Instead of expanding the tensor, compute the target indices and use direct indexing (e.g., `flat_x[torch.div(sorted_order, top_k, rounding_mode='floor')]`). This avoids allocating the large intermediate tensor and significantly improves dispatch performance.
