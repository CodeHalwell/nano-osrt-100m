## 2024-05-17 - Missing Input Length Limits in Gradio
**Vulnerability:** Gradio inputs (e.g., in demo.py) do not have inherent length limits.
**Learning:** This can lead to Denial of Service (DoS) or memory exhaustion during model inference if a user sends an excessively large payload.
**Prevention:** Always explicitly validate input string lengths and raise `gr.Error` for excessively large payloads before processing.