## 2025-03-02 - Avoiding Full-Size Intermediate Tensor Allocations in RoPE
**Learning:** Computing Rotary Position Embedding by concatenating the rotated halves before element-wise multiplication creates a full-size intermediate tensor `x_rot`, consuming unnecessary memory bandwidth.
**Action:** Compute the two halves separately (`x1 * cos - x2 * sin` and `x2 * cos + x1 * sin`) and concatenate them at the very end to save memory bandwidth and improve latency in CPU environments (about ~8% faster locally).
