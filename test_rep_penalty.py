import torch
import time

def slow_rep_penalty(next_logits, generated, repetition_penalty):
    for token_id in set(generated[0].tolist()):
        if token_id < next_logits.shape[-1]:
            if next_logits[0, token_id] > 0:
                next_logits[0, token_id] /= repetition_penalty
            else:
                next_logits[0, token_id] *= repetition_penalty
    return next_logits

def fast_rep_penalty(next_logits, generated, repetition_penalty):
    unique_tokens = generated[0].unique()
    valid_tokens = unique_tokens[unique_tokens < next_logits.shape[-1]]
    if len(valid_tokens) > 0:
        score = next_logits[0, valid_tokens]
        next_logits[0, valid_tokens] = torch.where(
            score > 0,
            score / repetition_penalty,
            score * repetition_penalty
        )
    return next_logits

vocab_size = 32000
seq_len = 1000

print("Generating dummy data...")
torch.manual_seed(42)
generated = torch.randint(0, vocab_size, (1, seq_len))

# Pre-warm
slow_next_logits = torch.randn(1, vocab_size)
fast_next_logits = slow_next_logits.clone()
slow_rep_penalty(slow_next_logits.clone(), generated, 1.2)
fast_rep_penalty(fast_next_logits.clone(), generated, 1.2)

# Benchmark slow
start = time.time()
for _ in range(100):
    slow_next_logits = torch.randn(1, vocab_size)
    slow_rep_penalty(slow_next_logits, generated, 1.2)
slow_time = time.time() - start

# Benchmark fast
start = time.time()
for _ in range(100):
    fast_next_logits = torch.randn(1, vocab_size)
    fast_rep_penalty(fast_next_logits, generated, 1.2)
fast_time = time.time() - start

print(f"Slow time: {slow_time*1000:.2f} ms")
print(f"Fast time: {fast_time*1000:.2f} ms")
print(f"Speedup: {slow_time / fast_time:.2f}x")

# Check correctness
slow_out = slow_rep_penalty(slow_next_logits.clone(), generated, 1.2)
fast_out = fast_rep_penalty(slow_next_logits.clone(), generated, 1.2)
print("Max diff:", (slow_out - fast_out).abs().max().item())
