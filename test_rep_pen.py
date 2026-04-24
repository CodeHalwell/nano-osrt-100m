import torch
import time

def slow_rep_pen(next_logits, generated, repetition_penalty):
    for token_id in set(generated[0].tolist()):
        if token_id < next_logits.shape[-1]:
            if next_logits[0, token_id] > 0:
                next_logits[0, token_id] /= repetition_penalty
            else:
                next_logits[0, token_id] *= repetition_penalty
    return next_logits

def fast_rep_pen(next_logits, generated, repetition_penalty):
    gen_tokens = generated[0].unique()
    valid_tokens = gen_tokens[gen_tokens < next_logits.shape[-1]]
    scores = next_logits[0, valid_tokens]
    next_logits[0, valid_tokens] = torch.where(
        scores > 0,
        scores / repetition_penalty,
        scores * repetition_penalty
    )
    return next_logits

V = 50277
S = 512
next_logits_slow = torch.randn(1, V)
next_logits_fast = next_logits_slow.clone()
generated = torch.randint(0, V + 100, (1, S))
repetition_penalty = 1.2

t0 = time.perf_counter()
for _ in range(100):
    slow_rep_pen(next_logits_slow, generated, repetition_penalty)
t1 = time.perf_counter()

t2 = time.perf_counter()
for _ in range(100):
    fast_rep_pen(next_logits_fast, generated, repetition_penalty)
t3 = time.perf_counter()

print("Slow:", t1 - t0)
print("Fast:", t3 - t2)
print("Max diff:", torch.max(torch.abs(next_logits_slow - next_logits_fast)).item())
