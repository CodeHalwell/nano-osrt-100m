## 2025-02-28 - Missing Gradio Input Length Limits
**Vulnerability:** Missing input length validation on Gradio Chatbot `respond` endpoint (DoS risk).
**Learning:** Gradio UI inputs do not inherently limit payload size, allowing excessively large strings to exhaust memory or cause DoS during downstream tokenization and model inference.
**Prevention:** Always explicitly validate input string lengths and raise `gr.Error` early for large payloads before processing them in the Gradio backend.
