import torch
import time

def apply_rope_original(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    x_rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + x_rot * sin

def apply_rope_optimized(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos1, cos2 = cos[..., :d], cos[..., d:]
    sin1, sin2 = sin[..., :d], sin[..., d:]

    return torch.cat([
        x1 * cos1 - x2 * sin1,
        x2 * cos2 + x1 * sin2
    ], dim=-1)

# Warmup
bsz, seq_len, heads, head_dim = 16, 1024, 12, 64
x = torch.randn(bsz, seq_len, heads, head_dim, device="cpu")
cos = torch.randn(1, seq_len, 1, head_dim, device="cpu")
sin = torch.randn(1, seq_len, 1, head_dim, device="cpu")

# Verify correctness
out_orig = apply_rope_original(x, cos, sin)
out_opt = apply_rope_optimized(x, cos, sin)
print("Max diff:", (out_orig - out_opt).abs().max().item())

def benchmark(fn, name, iters=1000):
    start = time.perf_counter()
    for _ in range(iters):
        _ = fn(x, cos, sin)
    end = time.perf_counter()
    print(f"{name}: {end - start:.4f}s")

benchmark(apply_rope_original, "Original")
benchmark(apply_rope_optimized, "Optimized")
