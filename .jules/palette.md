## 2025-02-18 - Subtle UI Enhancements with Gradio
**Learning:** Gradio provides powerful native attributes to enhance accessibility and user experience, which are often overlooked:
- `autofocus=True` for `gr.Textbox` enables users to start typing immediately, removing friction on app load.
- `info` parameter for components like `gr.Slider` gives a native tooltip-like explanation for complex settings (like LLM temperature and top-p), dramatically improving usability for non-technical users without requiring extra HTML/CSS work.
**Action:** Always check if a Gradio component supports `info` before adding custom helper text markdown. Apply `autofocus=True` to the primary chat input in all new conversational interfaces.
