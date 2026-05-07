## 2025-05-07 - Autofocus in Chat UIs
**Learning:** Found that chat UIs without autofocus require an extra click before the user can start typing. For Gradio, `gr.Textbox` supports `autofocus=True`.
**Action:** Always add `autofocus=True` to the primary input in chat interfaces to save users a click and improve the interaction flow.
