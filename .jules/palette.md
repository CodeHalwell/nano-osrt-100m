## 2025-03-01 - Add text input shortcuts and multiline
**Learning:** Adding keyboard shortcut hints in form field `info` parameters clarifies expected behavior for ambiguous submission patterns. In this app, the `gr.Textbox` handles both single-line submissions and multi-line paragraphs, so exposing `max_lines` along with clear instructions directly addresses user friction.
**Action:** When adding text inputs that support multiline behavior (e.g. Chatbot inputs), include `max_lines` and a clear shortcut hint in the `info` parameter to guide interaction.
## 2024-05-18 - Tooltip added to icon-only button
**Learning:** Adding a tooltip to an icon-only button improves accessibility and user experience.
**Action:** When adding icon-only buttons, always ensure an aria-label or tooltip is present to describe the action.
## 2024-05-18 - Missing interactive component feedback
**Learning:** Adding explicit visual feedback (like clear empty states and interactive element cues) makes the interface much more approachable for users unfamiliar with AI models.
**Action:** Consistently review input fields and button states to ensure clear feedback mechanisms are provided.
