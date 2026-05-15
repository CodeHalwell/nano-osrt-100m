## 2024-05-24 - [Preventing Gradio Input Denial of Service]
**Vulnerability:** Gradio Textbox inputs lack inherent length limits. Without validation, attackers can submit excessively large payloads.
**Learning:** This can lead to Denial of Service (DoS) or memory exhaustion during model inference processing.
**Prevention:** Always explicitly validate input string lengths and raise `gr.Error` for excessively large payloads before processing.
