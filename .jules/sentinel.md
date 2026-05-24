## 2024-05-22 - Gradio UI Input Length Limits
**Vulnerability:** Missing input length validation in Gradio UI.
**Learning:** Gradio inputs do not have inherent length limits, allowing excessively large payloads that can lead to Denial of Service (DoS) or memory exhaustion during model inference.
**Prevention:** Always explicitly validate input string lengths and raise `gr.Error` for excessively large payloads before processing.

## 2025-02-23 - Gradio UI Chat History Limits
**Vulnerability:** Missing input validation on Gradio chat history state.
**Learning:** Even if the immediate user message is validated for length, stateful components like `gr.State` or `gr.Chatbot` (which pass history back to the server) can carry massive payloads injected directly into API requests. This can lead to DoS or memory exhaustion during prompt building or tokenization.
**Prevention:** Always explicitly validate the lengths and sizes of *all* state arrays passed from the client before processing them.
