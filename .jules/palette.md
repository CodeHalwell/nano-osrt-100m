## 2025-02-12 - Adding Helpful Context to Technical Gradio Settings
**Learning:** Users often struggle with advanced model generation parameters (like top-p, top-k, repetition penalty) because the names are technical and lack built-in context in default Gradio sliders. Additionally, forcing users to click into a primary input field breaks their flow.
**Action:** Always utilize Gradio's `info` parameter on technical UI components (like Sliders) to provide plain-english tooltips, and use `autofocus=True` on primary text inputs to eliminate unnecessary clicks and improve immediate interactivity.
