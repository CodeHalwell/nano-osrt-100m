## 2024-05-18 - Missing input length validation in Gradio UI
**Vulnerability:** User inputs in the Gradio demo lacked length limits, allowing excessively large inputs to exhaust memory or cause a Denial of Service (DoS) during tokenization/inference.
**Learning:** Gradio components do not enforce inherent length limits.
**Prevention:** Always validate input string length early and raise a `gr.Error` if the payload exceeds reasonable thresholds before processing.
