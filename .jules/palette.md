## 2024-05-18 - Gradio Autofocus for Chat UX
**Learning:** Conversational UI users expect to be able to start typing immediately upon page load. Without autofocus on the main input field, users are forced into an unnecessary click, interrupting the interaction flow.
**Action:** When building conversational interfaces with Gradio, always add `autofocus=True` to the primary input `gr.Textbox` to reduce interaction friction and improve accessibility for keyboard users.
