## 2024-05-24 - [Avoid Device-to-Host syncs in generation loop]
**Learning:** In PyTorch, calling `.tolist()` inside an autoregressive generation loop forces a slow Device-to-Host (GPU to CPU) synchronization on every single step. This completely breaks down the performance.
**Action:** When applying operations like repetition penalty, rely exclusively on vectorized PyTorch functions (like `.unique()`, boolean masking, and `torch.where()`) to keep all computation on the accelerator and avoid CPU loops.
