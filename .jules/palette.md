## 2026-05-18 - Tooltips and Autofocus for Gradio UI
**Learning:** Gradio sliders representing complex model generation parameters (like top-p, top-k, repetition penalty) benefit significantly from inline tooltips (`info` parameter) to improve usability. Adding `autofocus=True` to the primary input textbox removes friction on initial load.
**Action:** When implementing generation UIs with Gradio, utilize the `info` parameter on technical sliders and `autofocus` on primary inputs by default.
