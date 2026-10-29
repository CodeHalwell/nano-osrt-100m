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

## 2024-05-24 - Resource Limits for Untrusted Subprocesses
**Vulnerability:** The MBPP evaluation engine runs untrusted model-generated Python code in a subprocess (`subprocess.Popen(..., start_new_session=True)`) without enforcing any resource limits on the child process beyond a simple execution timeout. A malicious model could cause an out-of-memory error (OOM), fork bomb, or fill the disk by abusing OS resources within the timeout period.
**Learning:** `subprocess.Popen` allows injecting a `preexec_fn` callback to execute in the child process just before `exec()`, which can be used to set resource limits using `resource.setrlimit`. However, `preexec_fn` and `start_new_session=True` are explicitly incompatible. To start a new session while using `preexec_fn`, we must omit `start_new_session=True` from the `Popen` arguments and instead call `os.setsid()` directly inside the `preexec_fn` callback. `resource.setrlimit` must also be used carefully: the `resource` module should be conditionally imported *outside* the callback (to prevent deadlocks after `fork()`), and limits should be generous to allow normal code execution while blocking catastrophic resource exhaustion.
**Prevention:** Whenever running untrusted code in a subprocess, enforce limits on CPU time (`RLIMIT_CPU`), memory (`RLIMIT_AS`), and file size (`RLIMIT_FSIZE`) using a `preexec_fn` callback that includes `os.setsid()` if a new process group is needed.
