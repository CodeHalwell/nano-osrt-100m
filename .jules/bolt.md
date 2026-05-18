## 2025-02-20 - PyTorch RoPE Operation Memory Bandwidth
**Learning:** PyTorch operations that slice tensors (like `x[..., :d]`) and allocate intermediate full-sized tensors (like `torch.cat([-x2, x1], dim=-1)`) incur high memory bandwidth costs.
**Action:** When applying Rotational Positional Embeddings (RoPE), compute the rotation mathematically directly into the output tensor by calculating the parts and using a single `torch.cat(...)` rather than allocating and operating on a full-size intermediate rotated tensor. This can reduce execution time by avoiding memory allocation overhead.
