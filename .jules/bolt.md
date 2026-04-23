## 2024-04-23 - Python Loops in PyTorch Autoregressive Generation
**Learning:** Element-wise PyTorch tensor modifications within a python loop (e.g. `for token_id in set(generated[0].tolist()):`) are incredibly slow, taking orders of magnitude more time compared to equivalent vectorized operations, particularly during auto-regressive generation.
**Action:** When adjusting generation loop probabilities or handling repetition penalties, use vectorized tensor operations such as `.unique()` combined with index masking and `torch.where()`.
