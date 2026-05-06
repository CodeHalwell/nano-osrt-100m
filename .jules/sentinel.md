## 2025-02-28 - Add input length limit to Gradio Chatbot to prevent DoS
**Vulnerability:** The Gradio demo (`demo.py`) accepted user inputs of arbitrary length. Extremely large string payloads could be submitted.
**Learning:** Gradio text components and Chatbot interfaces do not enforce length limits by default. Because inputs are fed into the tokenizer and then run through the autoregressive generation loop, an excessively large input could trigger massive memory allocations, causing an Out Of Memory (OOM) error or tying up GPU/CPU resources, leading to Denial of Service (DoS).
**Prevention:** Always explicitly validate lengths of `message` strings in Gradio functions (`len(message) > N`) and raise `gr.Error` to abort before expensive tokenization or inference starts.
