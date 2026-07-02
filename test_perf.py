import time

import torch


def test_isin():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    comp_region = torch.randint(0, 100, (1000,), device=device)
    stop_set = {10, 20, 30}
    stop_tensor = torch.tensor(list(stop_set), device=device)

    # warmup
    for _ in range(10):
        _ = torch.tensor([t.item() in stop_set for t in comp_region], device=device)
        _ = torch.isin(comp_region, stop_tensor)

    start = time.time()
    for _ in range(100):
        _ = torch.tensor([t.item() in stop_set for t in comp_region], device=device)
    t1 = time.time() - start

    start = time.time()
    for _ in range(100):
        _ = torch.isin(comp_region, stop_tensor)
    t2 = time.time() - start

    print(f"List comp with .item(): {t1:.4f}s")
    print(f"torch.isin: {t2:.4f}s")
    print(f"Speedup: {t1 / t2:.2f}x")


if __name__ == "__main__":
    test_isin()
