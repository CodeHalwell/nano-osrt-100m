## 2024-04-26 - [Vectorized Repetition Penalty]
**Learning:** PyTorch iteration over tensor.tolist() is very slow, especially inside autoregressive generation loops. Using boolean masking and `torch.where` can be significantly faster.
**Action:** Avoid python loops over tensor elements. Use vectorized operations like `torch.unique` and `torch.where` instead.
