## 2024-05-10 - Gradio Slider Info Tooltips
**Learning:** Gradio Sliders natively support an `info` parameter to act as a tooltip/helper text, which is an accessible way to explain complex AI generation parameters like top-p and top-k without cluttering the UI with separate Markdown blocks. The Textbox component also cleanly accepts `autofocus=True` to immediately engage users on page load.
**Action:** Use `info` directly on Gradio inputs for parameter descriptions to improve inline UX, and apply `autofocus=True` to primary chat inputs.
