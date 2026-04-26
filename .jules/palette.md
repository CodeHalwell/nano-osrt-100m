## $(date +%Y-%m-%d) - Adding Tooltips to Gradio Sliders
**Learning:** In Gradio applications (like demo.py), users often don't understand complex model parameters like top-p or repetition penalty. The `info` parameter on components like `gr.Slider` provides an excellent, built-in way to add inline tooltips/helper text without cluttering the UI with extra markdown blocks.
**Action:** When working on Gradio interfaces with technical settings, always utilize the `info` parameter to explain the functionality of the setting to the end user.
