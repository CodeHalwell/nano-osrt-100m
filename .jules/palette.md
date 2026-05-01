## 2025-03-08 - Gradio UI Parameter Usage
**Learning:** Gradio UI framework (version 6.9.0) provides native, zero-dependency ways to improve UX and accessibility through `autofocus=True` on inputs (reduces user interaction friction) and `info` parameters on components like `gr.Slider` (acts as an accessible tooltip/description).
**Action:** Default to utilizing `autofocus=True` for primary text inputs and populate `info` parameters on configuration settings instead of building custom tooltip components or writing external helper text blocks.
