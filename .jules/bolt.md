## 2024-06-25 - Avoid Python Loops for Repetition Penalty in PyTorch Autoregressive Generation
**Learning:** Using Python loops and list conversions (like `set(generated[0].tolist())`) for applying repetition penalty to logits in the hot loop of text generation is a massive performance bottleneck and assumes batch_size=1.
**Action:** Always vectorize repetition penalty logic using boolean masking (e.g. `mask = torch.zeros(...); mask.scatter_(...); next_logits[mask] = torch.where(...)`) to improve generation speed and correctly support batched generation.
