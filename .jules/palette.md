## 2024-04-19 - Gradio Chat UX Standardization
**Learning:** Gradio 6.x explicitly supports `type="messages"` for `gr.Chatbot` to adhere to standard role-based chat history data structures. Additionally, using `autofocus=True` for `gr.Textbox` significantly improves immediate interaction upon load.
**Action:** Always verify `gr.Chatbot` is using `type="messages"` for better interoperability and set `autofocus=True` for main input textboxes to optimize user entry flow in Gradio demos.
