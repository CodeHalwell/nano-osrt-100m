## 2025-02-20 - PyTorch Tensors File Modification Artifacts
**Learning:** When using `cat -A` to pipe content into a new Python file via a bash heredoc during scratchpad testing, the `$` end-of-line marker characters get physically written into the file. This creates syntax errors (`SyntaxError: invalid syntax`) because Python attempts to parse the `$` symbols.
**Action:** Always use standard `cat` for writing files via heredocs (e.g. `cat << 'EOF' > file.py`) to prevent corrupting the source file with terminal control/display characters. Use `cat -A` strictly for viewing file content in the console.

## 2025-02-20 - RoPE Memory Bandwidth Optimization
**Learning:** PyTorch's `torch.cat([-x2, x1], dim=-1)` allocates an entirely new tensor (`x_rot`) that matches the dimensions of `x`, only to multiply it by `sin` on the very next line and add it to the final result. At sequence lengths like 4096, this intermediate tensor creates a significant memory bandwidth bottleneck, specifically slowing down CPU execution by ~2x compared to direct slice math.
**Action:** When working with Rotary Positional Embeddings or similar slice-and-combine operations, calculate the final components individually using the slices (`x1 * cos1 - x2 * sin1` and `x2 * cos2 + x1 * sin2`) and concatenate them directly at the end. This entirely avoids allocating the intermediate `x_rot` tensor.
