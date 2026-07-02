import re
import time


def test_redos():
    # Construct an adversarial payload
    text = "\n " * 20000 + "A"

    start = time.time()
    try:
        re.findall(r"(?:^|\n)[ \t]*(?:\d+[\.\):]|step\s+\d+)", text, re.IGNORECASE)
    except Exception as e:
        print("New failed:", e)
    end = time.time()
    print(f"New took {end - start:.4f} seconds")


test_redos()
