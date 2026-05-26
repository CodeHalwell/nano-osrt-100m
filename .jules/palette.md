## 2025-03-01 - Add text input shortcuts and multiline
**Learning:** Adding keyboard shortcut hints in form field `info` parameters clarifies expected behavior for ambiguous submission patterns. In this app, the `gr.Textbox` handles both single-line submissions and multi-line paragraphs, so exposing `max_lines` along with clear instructions directly addresses user friction.
**Action:** When adding text inputs that support multiline behavior (e.g. Chatbot inputs), include `max_lines` and a clear shortcut hint in the `info` parameter to guide interaction.
