## 2024-05-22 - Gradio UI Input Length Limits
**Vulnerability:** Missing input length validation in Gradio UI.
**Learning:** Gradio inputs do not have inherent length limits, allowing excessively large payloads that can lead to Denial of Service (DoS) or memory exhaustion during model inference.
**Prevention:** Always explicitly validate input string lengths and raise `gr.Error` for excessively large payloads before processing.
