## 2024-05-22 - Gradio UI Input Length Limits
**Vulnerability:** Missing input length validation in Gradio UI.
**Learning:** Gradio inputs do not have inherent length limits, allowing excessively large payloads that can lead to Denial of Service (DoS) or memory exhaustion during model inference.
**Prevention:** Always explicitly validate input string lengths and raise `gr.Error` for excessively large payloads before processing.

## 2024-05-25 - Gradio UI State Payload Limits
**Vulnerability:** Missing length validation for historical state payloads in Gradio UI.
**Learning:** Gradio stateful components (like `gr.Chatbot` or `gr.State`) pass history arrays back from the client, allowing malicious actors to bypass immediate input limits by injecting large payloads into the history, leading to memory exhaustion.
**Prevention:** Always explicitly validate the length of both immediate input strings and historical state payloads (including individual message contents) before processing.
