## 2024-07-08 - PyTorch Generation Loop Bottleneck
**Learning:** Checking for stop tokens using list comprehensions like `[t.item() in stop_set for t in tensor]` causes severe performance bottlenecks due to implicit device-to-host synchronization on every iteration for each token.
**Action:** Use `torch.isin(tensor, stop_tensor)` instead to keep truncation logic fully on the GPU/device and prevent CPU blocking.
