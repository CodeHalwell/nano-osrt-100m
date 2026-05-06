#!/usr/bin/env python3
"""Gradio demo for NanoOSRT v3.

Usage:
    uv run python demo.py
    uv run python demo.py --share  # public URL
"""

import argparse

import gradio as gr
import torch
from transformers import AutoTokenizer

from src.nano_osrt.hf_model import NanoOSRTForCausalLM

# ── Model loading ───────────────────────────────────────────────────────

MODEL_PATH = "./nano-osrt-model"
DEVICE = "cpu"
MAX_CONTEXT_TOKENS = 3072  # sliding window: keep last N tokens of conversation


def load_model():
    global DEVICE
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"

    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    model = NanoOSRTForCausalLM.from_pretrained(MODEL_PATH, device=DEVICE)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


MODEL, TOKENIZER = load_model()

# ── Generation ──────────────────────────────────────────────────────────


def build_prompt(message: str, history: list[dict]) -> str:
    """Build prompt string from chat history with sliding window.

    Keeps the most recent turns that fit within MAX_CONTEXT_TOKENS,
    dropping oldest messages first.
    """
    # Build all turns
    turns = []
    for entry in history:
        role = entry.get("role", "user")
        content = entry.get("content", "")
        if role == "user":
            turns.append(f"user: {content}")
        elif role == "assistant" and content:
            turns.append(f"assistant: {content}")

    turns.append(f"user: {message}")
    turns.append("assistant:")

    # Sliding window: drop oldest turns until it fits
    while len(turns) > 2:  # always keep at least current user + assistant:
        candidate = "\n".join(turns)
        token_count = len(TOKENIZER.encode(candidate, add_special_tokens=False))
        if token_count <= MAX_CONTEXT_TOKENS:
            break
        # Drop the oldest turn
        turns.pop(0)

    return "\n".join(turns)


def generate_stream(
    message: str,
    history: list[dict],
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    repetition_penalty: float,
):
    """Generate a response with streaming output."""
    prompt_text = build_prompt(message, history)
    input_ids = TOKENIZER.encode(prompt_text, return_tensors="pt").to(DEVICE)
    generated = input_ids.clone()
    response_tokens = []

    with torch.no_grad():
        for i in range(max_tokens):
            context = generated[:, -MODEL.config.seq_len :]
            out = MODEL.forward(context)
            next_logits = out["logits"][:, -1, : MODEL.config.real_vocab_size].float()

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    if token_id < next_logits.shape[-1]:
                        if next_logits[0, token_id] > 0:
                            next_logits[0, token_id] /= repetition_penalty
                        else:
                            next_logits[0, token_id] *= repetition_penalty

            if temperature > 0:
                next_logits = next_logits / temperature

                if top_k > 0:
                    topk_vals, _ = torch.topk(
                        next_logits, min(top_k, next_logits.size(-1))
                    )
                    next_logits[next_logits < topk_vals[:, -1:]] = float("-inf")

                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cumprobs - sorted_probs >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                next_logits.scatter_(1, sorted_indices, sorted_logits)

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == TOKENIZER.eos_token_id:
                break

            response_tokens.append(next_token.item())

            if i % 2 == 0:
                yield TOKENIZER.decode(response_tokens, skip_special_tokens=True)

    if response_tokens:
        yield TOKENIZER.decode(response_tokens, skip_special_tokens=True)


# ── Gradio UI ───────────────────────────────────────────────────────────


def create_demo():
    with gr.Blocks(title="NanoOSRT v3") as demo:
        gr.Markdown(
            """
            # NanoOSRT v3 — Recursive Transformer Demo
            **115.7M parameters** (104.5M base + 11.2M HRA) | ~302M effective via recursive weight sharing (2 blocks x 6 loops = 12 layers)

            Trained: TinyStories -> FineWeb-Edu -> SmolTalk -> Math SFT -> Code SFT
            """
        )

        # Store history as plain list of dicts (not Gradio objects)
        chat_state = gr.State(value=[])

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="Chat")
                msg = gr.Textbox(
                    placeholder="Ask me anything... (try code or math questions)",
                    label="Message",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Column(scale=1):
                gr.Markdown("### Generation Settings")
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.2,
                    step=0.05,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top-p",
                )
                top_k = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=5,
                    label="Top-k",
                )
                max_tokens = gr.Slider(
                    minimum=32,
                    maximum=1024,
                    value=512,
                    step=32,
                    label="Max tokens",
                )
                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.2,
                    step=0.05,
                    label="Repetition penalty",
                )

                gr.Markdown(
                    "### Model Info\n"
                    f"- **Device:** {DEVICE}\n"
                    f"- **Parameters:** {sum(p.numel() for p in MODEL.parameters()):,}\n"
                    "- **Architecture:** 2 blocks x 6 loops\n"
                    "- **Context:** 4096 tokens\n"
                    f"- **Sliding window:** {MAX_CONTEXT_TOKENS} tokens\n"
                    "- **Format:** `<think>...</think>`\n"
                )

        gr.Examples(
            examples=[
                ["Write a Python function to check if a number is prime"],
                ["What is the derivative of x^3 + 2x^2 - 5x + 3?"],
                ["Explain what recursion is in programming"],
                ["Write a binary search algorithm in Python"],
                ["What is 15 * 23?"],
                ["Write a Python class for a linked list"],
            ],
            inputs=msg,
            label="Try these examples",
        )

        def respond(
            message,
            chat_history,
            display_history,
            temperature,
            top_p,
            top_k,
            max_tokens,
            repetition_penalty,
        ):
            if len(message) > 4000:
                raise gr.Error(
                    "Input message is too long. Please limit to 4000 characters."
                )

            # Add user message to both histories
            chat_history = chat_history + [{"role": "user", "content": message}]
            display_history = display_history + [
                gr.ChatMessage(role="user", content=message),
                gr.ChatMessage(role="assistant", content=""),
            ]
            yield chat_history, display_history, ""

            # Stream generation
            for partial in generate_stream(
                message,
                chat_history[
                    :-1
                ],  # exclude the user msg we just added (it's in the prompt builder)
                temperature,
                top_p,
                top_k,
                max_tokens,
                repetition_penalty,
            ):
                display_history[-1] = gr.ChatMessage(role="assistant", content=partial)
                yield chat_history, display_history, ""

            # Save final assistant response to chat_history
            final = (
                display_history[-1].content
                if hasattr(display_history[-1], "content")
                else ""
            )
            chat_history = chat_history + [{"role": "assistant", "content": final}]
            yield chat_history, display_history, ""

        msg.submit(
            respond,
            [
                msg,
                chat_state,
                chatbot,
                temperature,
                top_p,
                top_k,
                max_tokens,
                repetition_penalty,
            ],
            [chat_state, chatbot, msg],
        )
        submit_btn.click(
            respond,
            [
                msg,
                chat_state,
                chatbot,
                temperature,
                top_p,
                top_k,
                max_tokens,
                repetition_penalty,
            ],
            [chat_state, chatbot, msg],
        )
        clear_btn.click(lambda: ([], [], ""), outputs=[chat_state, chatbot, msg])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public URL")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = create_demo()
    demo.launch(share=args.share, server_port=args.port)
