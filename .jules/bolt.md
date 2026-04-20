## 2023-10-24 - Avoid `tolist()` in PyTorch Autoregressive Loops
**Learning:** Found a performance bottleneck where `set(tensor.tolist())` was used inside a Python `for` loop to apply repetition penalties during autoregressive generation. In PyTorch, using Python loops and list conversions for element-wise modifications creates significant bottlenecks, particularly inside text generation loops where performance is critical.
**Action:** Always use vectorized operations like boolean masking, `.unique()`, and `torch.where()` instead of converting tensors to Python lists and looping over them.
