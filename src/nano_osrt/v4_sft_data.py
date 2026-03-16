"""SFT data pipeline for NanoOSRT v4 with native token tags.

Chat format:
    <|user|>{prompt}<|assistant|><|think|>{reasoning}<|/think|><|answer|>{answer}<|/answer|><|end_of_text|>

Loss masking:
    - Everything from <|user|> to <|assistant|> (inclusive): IGNORE_INDEX
    - <|think|> through <|/answer|>: trained on (real labels)

All structural tags are single tokens in the v4 tokenizer.
"""

import random

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerFast

IGNORE_INDEX = -100


# ── Format functions ────────────────────────────────────────────────────
# Each returns (question, reasoning, answer) triple.
# The SFTStream handles wrapping with tags.


def format_gsm8k(example: dict) -> tuple[str, str, str]:
    """GSM8K: question + step-by-step reasoning + final numeric answer."""
    question = example["question"]
    answer_raw = example["answer"]

    if "####" in answer_raw:
        reasoning, final = answer_raw.rsplit("####", 1)
        return question, reasoning.strip(), final.strip()
    return question, answer_raw, answer_raw


def format_numina_math(example: dict) -> tuple[str, str, str]:
    """NuminaMath-CoT: problem + solution with reasoning."""
    import re
    question = example.get("problem", "")
    solution = example.get("solution", "")

    if not question or not solution:
        return "", "", ""

    lines = solution.strip().split("\n")
    if len(lines) > 1:
        reasoning = "\n".join(lines[:-1]).strip()
        final = lines[-1].strip()
        match = re.search(r"\\boxed\{(.+?)\}", final)
        if match:
            final = match.group(1)
    else:
        reasoning = solution.strip()
        final = solution.strip()

    return question, reasoning, final


def format_orca_math(example: dict) -> tuple[str, str, str]:
    """Orca-Math: question + worked solution."""
    question = example["question"]
    answer = example["answer"]

    lines = answer.strip().split("\n")
    if len(lines) > 1:
        reasoning = "\n".join(lines[:-1]).strip()
        final = lines[-1].strip()
    else:
        reasoning = answer.strip()
        final = answer.strip()

    return question, reasoning, final


def format_math_instruct(example: dict) -> tuple[str, str, str]:
    """MathInstruct: instruction + output."""
    question = example.get("instruction", "")
    output = example.get("output", "")

    if not question or not output:
        return "", "", ""

    lines = output.strip().split("\n")
    if len(lines) > 1:
        reasoning = "\n".join(lines[:-1]).strip()
        final = lines[-1].strip()
    else:
        reasoning = output.strip()
        final = output.strip()

    return question, reasoning, final


def format_evol_code(example: dict) -> tuple[str, str, str]:
    """Evol-Instruct-Code: instruction + code output."""
    question = example.get("instruction", "")
    output = example.get("output", "")

    if not question or not output:
        return "", "", ""

    parts = output.split("```")
    if len(parts) >= 3:
        reasoning = parts[0].strip() or "Let me write the code for this."
        code = "```" + "```".join(parts[1:])
        return question, reasoning, code

    paragraphs = output.strip().split("\n\n")
    if len(paragraphs) > 1:
        return question, paragraphs[0].strip(), "\n\n".join(paragraphs[1:]).strip()

    return question, "Let me solve this step by step.", output


def format_alpaca_code(example: dict) -> tuple[str, str, str]:
    """Alpaca-style code: instruction + optional input + output."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")

    if not instruction or not output:
        return "", "", ""

    question = f"{instruction}\n{inp}".strip() if inp else instruction

    parts = output.split("```")
    if len(parts) >= 3:
        reasoning = parts[0].strip() or "I'll write the code to solve this."
        code = "```" + "```".join(parts[1:])
        return question, reasoning, code

    lines = output.strip().split("\n")
    code_indicators = ("def ", "class ", "import ", "from ", "#", "if __name__")
    if lines and any(lines[0].strip().startswith(ind) for ind in code_indicators):
        return question, "Here's the solution.", output

    if len(lines) > 3:
        return question, lines[0].strip(), "\n".join(lines[1:]).strip()

    return question, "Here's my approach.", output


def format_alpaca(example: dict) -> tuple[str, str, str]:
    """Alpaca: instruction + optional input + output."""
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example["output"]

    question = f"{instruction}\n{inp}".strip() if inp else instruction

    sentences = output.strip().split(". ")
    if len(sentences) > 2:
        reasoning = ". ".join(sentences[:-1]).strip() + "."
        final = sentences[-1].strip()
        if not final.endswith("."):
            final += "."
        return question, reasoning, final

    return question, "Let me work through this.", output


def format_openhermes(example: dict) -> tuple[str, str, str]:
    """OpenHermes: conversations list."""
    conversations = example.get("conversations", [])
    if len(conversations) < 2:
        return "", "", ""

    question = ""
    answer = ""
    for msg in conversations:
        role = msg.get("from", "")
        value = msg.get("value", "")
        if role == "human" and not question:
            question = value
        elif role == "gpt" and question and not answer:
            answer = value

    if not question or not answer:
        return "", "", ""

    paragraphs = answer.strip().split("\n\n")
    if len(paragraphs) > 1:
        reasoning = "\n\n".join(paragraphs[:-1]).strip()
        final = paragraphs[-1].strip()
        return question, reasoning, final

    return question, "Let me think about this.", answer


def format_ifeval(example: dict) -> tuple[str, str, str]:
    """IFEval-like: prompt + response with constraints."""
    question = example.get("prompt", "")
    response = example.get("response", "")

    if not question or not response:
        return "", "", ""

    paragraphs = response.strip().split("\n\n")
    if len(paragraphs) > 1:
        reasoning = "\n\n".join(paragraphs[:-1]).strip()
        final = paragraphs[-1].strip()
        return question, reasoning, final

    return question, "Let me follow the instructions carefully.", response


def format_longform(example: dict) -> tuple[str, str, str]:
    """LongForm: input + long output."""
    question = example["input"]
    output = example["output"]

    paragraphs = output.strip().split("\n\n")
    if len(paragraphs) > 1:
        reasoning = "\n\n".join(paragraphs[:-1]).strip()
        final = paragraphs[-1].strip()
        return question, reasoning, final

    return question, output.strip(), output.strip()


FORMAT_FN = {
    "gsm8k": format_gsm8k,
    "numina_math": format_numina_math,
    "orca_math": format_orca_math,
    "math_instruct": format_math_instruct,
    "evol_code": format_evol_code,
    "alpaca_code": format_alpaca_code,
    "alpaca": format_alpaca,
    "openhermes": format_openhermes,
    "ifeval": format_ifeval,
    "longform": format_longform,
}


# ── SFT Stream ──────────────────────────────────────────────────────────


class V4SFTStream(IterableDataset):
    """Streaming SFT dataset with native token tags and loss masking.

    Formats each example as:
        <|user|>{question}<|assistant|><|think|>{reasoning}<|/think|><|answer|>{answer}<|/answer|><|end_of_text|>

    Loss masking:
        - <|user|> through <|assistant|> (inclusive): IGNORE_INDEX
        - Everything after <|assistant|>: real labels (think + answer)
    """

    def __init__(
        self,
        dataset_configs: list[dict],
        seq_len: int,
        tokenizer: PreTrainedTokenizerFast,
        seed: int,
        user_tag: str = "<|user|>",
        assistant_tag: str = "<|assistant|>",
        think_open: str = "<|think|>",
        think_close: str = "<|/think|>",
        answer_open: str = "<|answer|>",
        answer_close: str = "<|/answer|>",
    ) -> None:
        self.dataset_configs = dataset_configs
        self.seq_len = seq_len
        self.tok = tokenizer
        self.seed = seed
        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.think_open = think_open
        self.think_close = think_close
        self.answer_open = answer_open
        self.answer_close = answer_close

    def __iter__(self):  # noqa: ANN204
        from datasets import load_dataset

        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed if worker_info is None else self.seed + worker_info.id
        rng = random.Random(seed)

        # Load all dataset streams
        streams = []
        weights = []
        for ds_cfg in self.dataset_configs:
            print(f"[SFTWorker] Connecting to {ds_cfg['hf_id']}...")
            load_kwargs = {"split": ds_cfg.get("split", "train"), "streaming": True}
            if ds_cfg.get("hf_config"):
                load_kwargs["name"] = ds_cfg["hf_config"]
            ds = load_dataset(ds_cfg["hf_id"], **load_kwargs)
            ds = ds.shuffle(buffer_size=2_000, seed=seed)
            streams.append(iter(ds))
            weights.append(ds_cfg["weight"])
            print(f"[SFTWorker] Stream ready for {ds_cfg['hf_id']}")

        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        format_fns = [FORMAT_FN[cfg["format"]] for cfg in self.dataset_configs]

        while True:
            idx = rng.choices(range(len(streams)), weights=weights, k=1)[0]

            try:
                example = next(streams[idx])
            except StopIteration:
                ds_cfg = self.dataset_configs[idx]
                load_kwargs = {"split": ds_cfg.get("split", "train"), "streaming": True}
                if ds_cfg.get("hf_config"):
                    load_kwargs["name"] = ds_cfg["hf_config"]
                ds = load_dataset(ds_cfg["hf_id"], **load_kwargs)
                ds = ds.shuffle(buffer_size=2_000, seed=seed + rng.randint(0, 10000))
                streams[idx] = iter(ds)
                try:
                    example = next(streams[idx])
                except StopIteration:
                    continue

            try:
                question, reasoning, answer = format_fns[idx](example)
            except (KeyError, TypeError):
                continue

            if not question or not answer:
                continue

            # Build token sequence with native tags
            # Prompt (masked): <|user|>{question}<|assistant|>
            # Response (trained): <|think|>{reasoning}<|/think|><|answer|>{answer}<|/answer|><|end_of_text|>
            prompt_text = f"{self.user_tag}{question}{self.assistant_tag}"
            response_text = (
                f"{self.think_open}{reasoning}{self.think_close}"
                f"{self.answer_open}{answer}{self.answer_close}"
                f"{self.tok.eos_token}"
            )

            prompt_ids = self.tok.encode(prompt_text, add_special_tokens=False)
            response_ids = self.tok.encode(response_text, add_special_tokens=False)

            full_ids = prompt_ids + response_ids

            # Truncate to seq_len
            if len(full_ids) > self.seq_len:
                full_ids = full_ids[: self.seq_len]
                # Make sure we don't cut mid-response — need at least some response tokens
                if len(prompt_ids) >= self.seq_len - 10:
                    continue  # prompt too long, skip

            # Build labels with masking
            labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
            labels = labels[: len(full_ids)]

            # Pad to seq_len
            pad_len = self.seq_len - len(full_ids)
            if pad_len > 0:
                full_ids = full_ids + [self.tok.pad_token_id] * pad_len
                labels = labels + [IGNORE_INDEX] * pad_len

            yield (
                torch.tensor(full_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            )


def make_v4_sft_loader(
    dataset_configs: list[dict],
    seq_len: int,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int,
    seed: int,
    user_tag: str = "<|user|>",
    assistant_tag: str = "<|assistant|>",
    think_open: str = "<|think|>",
    think_close: str = "<|/think|>",
    answer_open: str = "<|answer|>",
    answer_close: str = "<|/answer|>",
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build a streaming DataLoader for v4 SFT.

    Args:
        dataset_configs: List of dataset config dicts.
        seq_len: Sequence length.
        tokenizer: The v4 tokenizer.
        batch_size: Micro-batch size.
        seed: Random seed.
        user_tag: User turn tag.
        assistant_tag: Assistant turn tag.
        think_open: Think open tag.
        think_close: Think close tag.
        answer_open: Answer open tag.
        answer_close: Answer close tag.

    Returns:
        DataLoader yielding (input_ids, labels) batches.
    """
    ds = V4SFTStream(
        dataset_configs, seq_len, tokenizer, seed,
        user_tag=user_tag,
        assistant_tag=assistant_tag,
        think_open=think_open,
        think_close=think_close,
        answer_open=answer_open,
        answer_close=answer_close,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
