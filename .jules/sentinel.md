## 2025-02-27 - ReDoS in Concluding Answer Extraction
**Vulnerability:** Catastrophic exponential backtracking (ReDoS) triggered by overlapping adjacent quantifiers in regex `\s*[.\s]*$` at the end of the `extract_numeric_answer_strict` matching pattern.
**Learning:** Overlapping sets, like `\s*` next to `[.\s]*`, cause the regex engine to explore every possible permutation of whitespace distribution when the terminal anchor `$` fails to match.
**Prevention:** Always combine adjacent optional sets if one is a subset of the other. Replaced `\s*[.\s]*$` with `[.\s]*$` to eliminate the exponential complexity while maintaining functional equivalence.

## 2025-02-27 - ReDoS when Handling Line Endings in Regex
**Vulnerability:** Overlapping backtracking triggered by `\s*` next to `(?:^|\n)` in the regex pattern for `count_reasoning_steps`.
**Learning:** `\s` includes `\n`. Thus, `\s*` can match the newline character and overlap with `(?:^|\n)`. This overlap creates a catastrophic backtracking vector when parsing long inputs.
**Prevention:** Avoid `\s*` adjacent to newline anchors. Use explicit non-newline spaces like `[ \t]*`. To support Windows CRLF without matching the newline, include the carriage return specifically: `[ \t\r]*`.
