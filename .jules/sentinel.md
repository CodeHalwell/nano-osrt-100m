## 2025-06-14 - Regex ReDoS in count_reasoning_steps
**Vulnerability:** The regex `(?:^|\n)\s*(?:\d+[\.\):]|step\s+\d+)` uses `\s*` next to `\n` which creates a ReDoS vulnerability due to catastrophic backtracking on adversarial input.
**Learning:** `\s*` includes newlines, which overlaps with the `(?:^|\n)` boundary condition, causing O(N^2) complexity.
**Prevention:** Use `[ \t]*` or `[^\S\n]*` instead of `\s*` when matching horizontal whitespace next to newlines.

## 2024-06-23 - Prevent Sandbox Evasion via PYTHONPATH / PYTHONNOUSERSITE
**Vulnerability:** Python sandboxed subprocess execution via `subprocess.Popen` in `mbpp_test_reward` (in `archive/v5/src/nano_osrt/rewards.py`) allows untrusted code to run. While the environment is restricted (cleared env dict), it relies on Python's default behavior for module loading. By default, Python adds the current working directory to `sys.path` and attempts to load packages from the user site-packages directory. Even though `cwd` is set to a tempdir, if an attacker can write a file to `~/.local/lib/pythonX.Y/site-packages` or if an existing user package is present, they could execute arbitrary code when an innocent module is imported by standard library imports.
**Learning:** When creating a minimal, restricted environment for Python `subprocess.Popen` to execute untrusted code securely, relying on a stripped `os.environ` is insufficient if Python-specific module loading environment variables aren't explicitly mitigated.
**Prevention:** Always explicitly set `PYTHONNOUSERSITE=1` and `PYTHONPATH=""` in the `sandbox_env` when executing untrusted Python code to prevent sandbox evasion via local module shadowing and pre-installed user site packages.

## 2024-05-22 - Gradio UI Input Length Limits
**Vulnerability:** Missing input length validation in Gradio UI.
**Learning:** Gradio inputs do not have inherent length limits, allowing excessively large payloads that can lead to Denial of Service (DoS) or memory exhaustion during model inference.
**Prevention:** Always explicitly validate input string lengths and raise `gr.Error` for excessively large payloads before processing.

## 2024-05-25 - Gradio UI State Payload Limits
**Vulnerability:** Missing length validation for historical state payloads in Gradio UI.
**Learning:** Gradio stateful components (like `gr.Chatbot` or `gr.State`) pass history arrays back from the client, allowing malicious actors to bypass immediate input limits by injecting large payloads into the history, leading to memory exhaustion.
**Prevention:** Always explicitly validate the length of both immediate input strings and historical state payloads (including individual message contents) before processing.

## 2024-05-29 - ReDoS in Whitespace Regex
**Vulnerability:** Regular Expression Denial of Service (ReDoS) due to overlapping whitespace matches.
**Learning:** Using `\s*` adjacent to `(?:^|\n)` causes overlapping backtracking paths (O(N^2) complexity) because `\s` includes `\n`. This makes parsing vulnerable to crafted inputs with alternating spaces and newlines.
**Prevention:** Use `[ \t]*` or `[^\S\n]*` instead of `\s*` when you only want to match horizontal whitespace next to a newline boundary.
## 2025-06-14 - Sandbox Evasion via Resource Exhaustion
**Vulnerability:** Untrusted Python code evaluation via `subprocess.Popen` lacks CPU and memory limits, leaving it vulnerable to denial-of-service (DoS) via resource exhaustion (e.g. fork bombs, infinite loops, and massive memory allocation).
**Learning:** Even with an empty environment and timeouts, a child process can exhaust system resources (RAM/CPU) before the parent process timeout is reached, affecting the stability of the entire system.
**Prevention:** Use `resource.setrlimit` in a `preexec_fn` callback to enforce strict system resource limits (like `RLIMIT_AS`, `RLIMIT_CPU`, and `RLIMIT_FSIZE`) on untrusted subprocesses.
