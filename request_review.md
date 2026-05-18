## Plan to fix Gradio input length limitation (Denial of Service DoS risk)

Looking at the codebase and the Sentinel instructions, I identified the following:
* **Missing input length limits (DoS risk)** is a Medium priority security issue, or we can consider it an enhancement.
* Based on memory: "Gradio inputs (e.g., in `demo.py`) do not have inherent length limits. To prevent Denial of Service (DoS) or memory exhaustion during model inference, always explicitly validate input string lengths and raise `gr.Error` for excessively large payloads before processing."
* The Gradio demo in `archive/v3/demo.py` takes user messages without validating length in `respond(message, ...)`. A huge input message will be encoded by the tokenizer and can lead to OOM.

**Proposed solution:**
1. Add input length validation in `archive/v3/demo.py` inside `def respond(...)`.
2. Specifically:
```python
        def respond(message, chat_history, display_history,
                    temperature, top_p, top_k, max_tokens, repetition_penalty):
            if len(message) > 4000:
                raise gr.Error("Message too long! Please keep it under 4000 characters.")
            # Add user message to both histories
            chat_history = chat_history + [{"role": "user", "content": message}]
```
3. Verify changes.
4. Record critical codebase-specific security learnings in `.jules/sentinel.md`.

Let me submit this plan for review.
