## 2024-06-21 - Python Sandbox Bypass Risk in MBPP Eval
**Vulnerability:** The MBPP verifiable reward evaluation used `subprocess.Popen` to evaluate Python code but did not explicitly block Python from loading user site-packages or set a restricted `PYTHONPATH`.
**Learning:** Even when starting a sandboxed environment without env vars like `HOME`, Python defaults to checking some site-package directories unless disabled. This could allow malicious model outputs to load pre-planted or arbitrary dependencies outside the restricted execution context.
**Prevention:** Always add `"PYTHONNOUSERSITE": "1"` and `"PYTHONPATH": ""` to the restricted environment dictionary when executing Python code in a sandboxed subprocess via `subprocess.Popen`.
