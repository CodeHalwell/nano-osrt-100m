
## 2024-05-06 - Batched Vectorization in PyTorch Generation Loops
**Learning:** Found a severe unvectorized bottleneck in `NanoOSRTForCausalLM.generate` using an O(N) host-to-device Python loop (`for token_id in set(generated[0].tolist()):`) to apply repetition penalty, which causes heavy performance degradation on GPUs and failed to correctly support batch sizes > 1.
**Action:** Replace unvectorized loops during element-wise tensor updates with `mask.scatter_` and `torch.where` to provide O(1) performance and full batch size support. Ensure comments are added describing the optimization before submitting PR.
