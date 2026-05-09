
## 2024-05-09 - Gradio UI Component Configuration
**Learning:** Gradio components like `gr.Textbox` support an `autofocus=True` attribute which immediately focuses the input field, reducing user clicks. Components like `gr.Slider` support an `info` parameter which natively handles tooltip-style helper text, making the UI more accessible and self-documenting without needing additional markdown or complex layout elements.
**Action:** Always prefer using these native configuration arguments over building custom tooltips or workarounds to keep UI code clean and accessible.
