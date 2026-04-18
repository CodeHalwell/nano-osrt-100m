## 2024-11-23 - Gradio Autofocus and Chatbot type
**Learning:** For Gradio `gr.Chatbot`, setting `type="messages"` structures history as standard role-based chat, while setting `autofocus=True` on `gr.Textbox` significantly improves input UX for chat interfaces by immediately directing user attention and input without requiring an extra click.
**Action:** Always utilize `type="messages"` for clear message structures and consider `autofocus=True` for primary chat input fields.
