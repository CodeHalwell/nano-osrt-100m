## 2025-02-28 - [Add Input Validation]
**Vulnerability:** Missing input limit constraint in public-facing UI.
**Learning:** By default, Gradio input elements accept any amount of text, which when passed to the model could cause Denial of Service (DoS) due to high processing time or memory exhaustion.
**Prevention:** Apply input length validation on UI forms before processing user inputs.
