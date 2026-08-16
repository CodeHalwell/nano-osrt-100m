def test_extract_numeric_answer_strict_redos():
    from nano_osrt.rewards import extract_numeric_answer_strict
    import time

    # Construct an adversarial payload
    text = "<|answer|> answer is 1" + " . " * 10000 + "A<|/answer|>"

    start = time.time()
    try:
        extract_numeric_answer_strict(text)
    except Exception as e:
        print("Failed:", e)
    end = time.time()
    print(f"Took {end - start:.4f} seconds")
    assert end - start < 0.5
