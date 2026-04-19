import torch
import time

# Mock data
V = 50277
S = 2000
next_logits_orig = torch.randn(1, V)
generated = torch.randint(0, V, (1, S))
repetition_penalty = 1.2

next_logits_orig = next_logits_orig * 10 # scale it up

# original method
def original(next_logits, generated):
    for token_id in set(generated[0].tolist()):
        if token_id < next_logits.shape[-1]:
            if next_logits[0, token_id] > 0:
                next_logits[0, token_id] /= repetition_penalty
            else:
                next_logits[0, token_id] *= repetition_penalty

# vectorized method
def vectorized(next_logits, generated):
    gen_tokens = torch.unique(generated[0])
    gen_tokens = gen_tokens[gen_tokens < next_logits.shape[-1]]

    token_logits = next_logits[0, gen_tokens]
    penalized_logits = torch.where(
        token_logits > 0,
        token_logits / repetition_penalty,
        token_logits * repetition_penalty
    )
    next_logits[0, gen_tokens] = penalized_logits

logits_a = next_logits_orig.clone()
logits_b = next_logits_orig.clone()

# correctness
original(logits_a, generated)
vectorized(logits_b, generated)
print("Correctness:", torch.allclose(logits_a, logits_b))

# perf
t0 = time.time()
for _ in range(100):
    original(next_logits_orig.clone(), generated)
t1 = time.time()
print(f"Original: {(t1-t0)*1000/100:.3f} ms/iter")

t0 = time.time()
for _ in range(100):
    vectorized(next_logits_orig.clone(), generated)
t1 = time.time()
print(f"Vectorized: {(t1-t0)*1000/100:.3f} ms/iter")
