## 2025-06-14 - Regex ReDoS in count_reasoning_steps
**Vulnerability:** The regex `(?:^|\n)\s*(?:\d+[\.\):]|step\s+\d+)` uses `\s*` next to `\n` which creates a ReDoS vulnerability due to catastrophic backtracking on adversarial input.
**Learning:** `\s*` includes newlines, which overlaps with the `(?:^|\n)` boundary condition, causing O(N^2) complexity.
**Prevention:** Use `[ \t]*` or `[^\S\n]*` instead of `\s*` when matching horizontal whitespace next to newlines.
