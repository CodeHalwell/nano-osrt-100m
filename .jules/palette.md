## 2025-02-28 - Gradio Component Polish
**Learning:** Adding tooltips using `info="..."` to technical slider components (like temperature, top-p, top-k) significantly improves usability by helping non-expert users understand what the parameters do. Also, `autofocus=True` is a simple win for chat interfaces to allow immediate input.
**Action:** When working with Gradio settings, always consider adding `info` tooltips to clarify confusing parameters, and ensure the primary input component has `autofocus=True` where appropriate.
