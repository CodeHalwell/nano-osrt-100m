"""Streaming SFT data pipeline with loss masking for chain-of-thought training."""

import random

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

IGNORE_INDEX = -100  # PyTorch cross_entropy ignores this label


def format_gsm8k(example: dict, think_open: str, think_close: str) -> tuple[str, str]:
    """Format a GSM8K example into user/assistant with CoT."""
    question = example["question"]
    answer_raw = example["answer"]

    if "####" in answer_raw:
        reasoning, final = answer_raw.rsplit("####", 1)
        reasoning = reasoning.strip()
        final = final.strip()
        assistant = f"{think_open}{reasoning}{think_close}\n{final}"
    else:
        assistant = f"{think_open}{answer_raw}{think_close}\n{answer_raw}"

    return question, assistant


def format_orca_math(
    example: dict, think_open: str, think_close: str
) -> tuple[str, str]:
    """Format an Orca-Math example into user/assistant with CoT."""
    question = example["question"]
    answer = example["answer"]

    lines = answer.strip().split("\n")
    if len(lines) > 1:
        reasoning = "\n".join(lines[:-1]).strip()
        final = lines[-1].strip()
    else:
        reasoning = answer.strip()
        final = answer.strip()

    assistant = f"{think_open}{reasoning}{think_close}\n{final}"
    return question, assistant


def format_numina_math(
    example: dict, think_open: str, think_close: str
) -> tuple[str, str]:
    """Format a NuminaMath-CoT example into user/assistant with CoT.

    NuminaMath-CoT has 'problem' and 'solution' columns.
    """
    question = example.get("problem", "")
    solution = example.get("solution", "")

    if not question or not solution:
        return "", ""

    # Split solution into reasoning and final answer
    lines = solution.strip().split("\n")
    if len(lines) > 1:
        reasoning = "\n".join(lines[:-1]).strip()
        final = lines[-1].strip()
        # Clean up common patterns like "The answer is ..."
        if final.startswith("\\boxed"):
            # Extract from \boxed{...}
            import re
            match = re.search(r"\\boxed\{(.+?)\}", final)
            if match:
                final = match.group(1)
    else:
        reasoning = solution.strip()
        final = solution.strip()

    assistant = f"{think_open}{reasoning}{think_close}\n{final}"
    return question, assistant


def format_ifeval(
    example: dict, think_open: str, think_close: str
) -> tuple[str, str]:
    """Format an ifeval-like-data example into user/assistant with CoT.

    Has 'prompt' and 'response' columns. The key value is teaching
    the model to follow precise constraints in the prompt.
    """
    question = example.get("prompt", "")
    response = example.get("response", "")

    if not question or not response:
        return "", ""

    # Wrap response in thinking format
    paragraphs = response.strip().split("\n\n")
    if len(paragraphs) > 1:
        reasoning = "\n\n".join(paragraphs[:-1]).strip()
        final = paragraphs[-1].strip()
        assistant = f"{think_open}{reasoning}{think_close}\n{final}"
    else:
        assistant = f"{think_open}Let me follow the instructions carefully.{think_close}\n{response}"

    return question, assistant


def format_math_instruct(
    example: dict, think_open: str, think_close: str
) -> tuple[str, str]:
    """Format a TIGER-Lab/MathInstruct example into user/assistant with CoT."""
    question = example.get("instruction", "")
    output = example.get("output", "")

    if not question or not output:
        return "", ""

    # Split reasoning from final answer
    lines = output.strip().split("\n")
    if len(lines) > 1:
        reasoning = "\n".join(lines[:-1]).strip()
        final = lines[-1].strip()
    else:
        reasoning = output.strip()
        final = output.strip()

    assistant = f"{think_open}{reasoning}{think_close}\n{final}"
    return question, assistant


def format_longform(
    example: dict, think_open: str, think_close: str
) -> tuple[str, str]:
    """Format a LongForm example into user/assistant with CoT.

    LongForm has 'input' and 'output' columns with long-form responses.
    We wrap the output as thinking with the final paragraph as the answer.
    """
    question = example["input"]
    output = example["output"]

    paragraphs = output.strip().split("\n\n")
    if len(paragraphs) > 1:
        reasoning = "\n\n".join(paragraphs[:-1]).strip()
        final = paragraphs[-1].strip()
    else:
        reasoning = output.strip()
        final = output.strip()

    assistant = f"{think_open}{reasoning}{think_close}\n{final}"
    return question, assistant


def format_alpaca(
    example: dict, think_open: str, think_close: str
) -> tuple[str, str]:
    """Format an Alpaca example. Wraps the output with brief thinking."""
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example["output"]

    question = f"{instruction}\n{inp}".strip() if inp else instruction

    # For short outputs, thinking is the planning step
    sentences = output.strip().split(". ")
    if len(sentences) > 2:
        reasoning = ". ".join(sentences[:-1]).strip() + "."
        final = sentences[-1].strip()
        if not final.endswith("."):
            final += "."
        assistant = f"{think_open}{reasoning}{think_close}\n{final}"
    else:
        # Short answer — still wrap in think to maintain format consistency
        assistant = f"{think_open}Let me work through this.{think_close}\n{output}"

    return question, assistant


def format_openhermes(
    example: dict, think_open: str, think_close: str
) -> tuple[str, str]:
    """Format an OpenHermes example from conversations list."""
    conversations = example.get("conversations", [])
    if len(conversations) < 2:
        return "", ""

    # Find first human/gpt pair
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
        return "", ""

    paragraphs = answer.strip().split("\n\n")
    if len(paragraphs) > 1:
        reasoning = "\n\n".join(paragraphs[:-1]).strip()
        final = paragraphs[-1].strip()
        assistant = f"{think_open}{reasoning}{think_close}\n{final}"
    else:
        sentences = answer.strip().split(". ")
        if len(sentences) > 2:
            reasoning = ". ".join(sentences[:-1]).strip() + "."
            final = sentences[-1].strip()
            assistant = f"{think_open}{reasoning}{think_close}\n{final}"
        else:
            assistant = f"{think_open}Let me think about this.{think_close}\n{answer}"

    return question, assistant


def format_slimorca(
    example: dict, think_open: str, think_close: str
) -> tuple[str, str]:
    """Format a SlimOrca example from conversations list."""
    conversations = example.get("conversations", [])
    if len(conversations) < 2:
        return "", ""

    # SlimOrca uses system/human/gpt roles
    question = ""
    answer = ""
    system = ""
    for msg in conversations:
        role = msg.get("from", "")
        value = msg.get("value", "")
        if role == "system":
            system = value
        elif role == "human" and not question:
            question = f"{system}\n{value}".strip() if system else value
        elif role == "gpt" and question and not answer:
            answer = value

    if not question or not answer:
        return "", ""

    paragraphs = answer.strip().split("\n\n")
    if len(paragraphs) > 1:
        reasoning = "\n\n".join(paragraphs[:-1]).strip()
        final = paragraphs[-1].strip()
        assistant = f"{think_open}{reasoning}{think_close}\n{final}"
    else:
        assistant = f"{think_open}Let me think step by step.{think_close}\n{answer}"

    return question, assistant


def format_evol_code(
    example: dict, think_open: str, think_close: str
) -> tuple[str, str]:
    """Format an Evol-Instruct-Code example.

    Has 'instruction' and 'output' columns. The output typically contains
    code with explanation — we split into reasoning (approach) and code.
    """
    question = example.get("instruction", "")
    output = example.get("output", "")

    if not question or not output:
        return "", ""

    # Look for code blocks to separate reasoning from code
    parts = output.split("```")
    if len(parts) >= 3:
        # Has code block: before is reasoning, code block is answer
        reasoning = parts[0].strip()
        code_and_rest = "```" + "```".join(parts[1:])
        if reasoning:
            assistant = f"{think_open}{reasoning}{think_close}\n{code_and_rest}"
        else:
            assistant = f"{think_open}Let me write the code for this.{think_close}\n{code_and_rest}"
    else:
        # No code block — treat first paragraph as reasoning
        paragraphs = output.strip().split("\n\n")
        if len(paragraphs) > 1:
            reasoning = paragraphs[0].strip()
            code = "\n\n".join(paragraphs[1:]).strip()
            assistant = f"{think_open}{reasoning}{think_close}\n{code}"
        else:
            assistant = f"{think_open}Let me solve this step by step.{think_close}\n{output}"

    return question, assistant


def format_alpaca_code(
    example: dict, think_open: str, think_close: str
) -> tuple[str, str]:
    """Format an Alpaca-style code instruction example.

    Has 'instruction', optional 'input', and 'output' columns.
    """
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")

    if not instruction or not output:
        return "", ""

    question = f"{instruction}\n{inp}".strip() if inp else instruction

    # Look for code blocks
    parts = output.split("```")
    if len(parts) >= 3:
        reasoning = parts[0].strip()
        code_and_rest = "```" + "```".join(parts[1:])
        if reasoning:
            assistant = f"{think_open}{reasoning}{think_close}\n{code_and_rest}"
        else:
            assistant = f"{think_open}I'll write the code to solve this.{think_close}\n{code_and_rest}"
    else:
        # Many code outputs don't use ``` blocks — check for common code patterns
        lines = output.strip().split("\n")
        # If output looks like code (starts with def, class, import, #, etc.)
        code_indicators = ("def ", "class ", "import ", "from ", "#", "if __name__")
        if any(lines[0].strip().startswith(ind) for ind in code_indicators):
            assistant = f"{think_open}Let me write the solution.{think_close}\n{output}"
        elif len(lines) > 3:
            # First line as reasoning, rest as code
            reasoning = lines[0].strip()
            code = "\n".join(lines[1:]).strip()
            assistant = f"{think_open}{reasoning}{think_close}\n{code}"
        else:
            assistant = f"{think_open}Here's my approach.{think_close}\n{output}"

    return question, assistant


FORMAT_FN = {
    "gsm8k": format_gsm8k,
    "orca_math": format_orca_math,
    "numina_math": format_numina_math,
    "math_instruct": format_math_instruct,
    "longform": format_longform,
    "alpaca": format_alpaca,
    "openhermes": format_openhermes,
    "slimorca": format_slimorca,
    "ifeval": format_ifeval,
    "evol_code": format_evol_code,
    "alpaca_code": format_alpaca_code,
}


class SFTStream(IterableDataset):
    """Streaming SFT dataset with loss masking.

    Formats each example as::

        user: {prompt}
        assistant: <think>{reasoning}</think>
        {answer}<|endoftext|>

    Yields ``(input_ids, labels)`` where ``labels[i] = IGNORE_INDEX``
    for all positions corresponding to user prompt tokens.
    """

    def __init__(
        self,
        dataset_configs: list[dict],
        seq_len: int,
        tok_name: str,
        seed: int,
        think_open: str = "<think>",
        think_close: str = "</think>",
        user_prefix: str = "user: ",
        assistant_prefix: str = "assistant: ",
    ) -> None:
        self.dataset_configs = dataset_configs
        self.seq_len = seq_len
        self.tok_name = tok_name
        self.seed = seed
        self.think_open = think_open
        self.think_close = think_close
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix

    def __iter__(self):  # noqa: ANN204
        from datasets import load_dataset

        tok = AutoTokenizer.from_pretrained(self.tok_name)
        tok.pad_token = tok.eos_token

        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed if worker_info is None else self.seed + worker_info.id
        rng = random.Random(seed)

        # Load all datasets as streaming iterators
        streams = []
        weights = []
        for ds_cfg in self.dataset_configs:
            print(f"[SFTWorker] Connecting to {ds_cfg['hf_id']}...")
            load_kwargs = {"split": ds_cfg["split"], "streaming": True}
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
                reload_kwargs = {"split": ds_cfg["split"], "streaming": True}
                if ds_cfg.get("hf_config"):
                    reload_kwargs["name"] = ds_cfg["hf_config"]
                ds = load_dataset(ds_cfg["hf_id"], **reload_kwargs)
                ds = ds.shuffle(
                    buffer_size=2_000, seed=seed + rng.randint(0, 10000)
                )
                streams[idx] = iter(ds)
                example = next(streams[idx])

            try:
                question, assistant_text = format_fns[idx](
                    example, self.think_open, self.think_close
                )
            except (KeyError, TypeError):
                continue

            if not question or not assistant_text:
                continue

            user_part = f"{self.user_prefix}{question}\n{self.assistant_prefix}"
            full_text = f"{user_part}{assistant_text}{tok.eos_token}"

            user_tokens = tok.encode(user_part, add_special_tokens=False)
            full_tokens = tok.encode(full_text, add_special_tokens=False)

            if len(full_tokens) > self.seq_len + 1:
                full_tokens = full_tokens[: self.seq_len + 1]

            if len(full_tokens) < 10:
                continue

            input_ids = full_tokens[:-1]
            labels = full_tokens[1:]

            # Mask user prompt positions
            mask_len = len(user_tokens) - 1
            for i in range(min(mask_len, len(labels))):
                labels[i] = IGNORE_INDEX

            # Pad to seq_len
            pad_len = self.seq_len - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [tok.eos_token_id] * pad_len
                labels = labels + [IGNORE_INDEX] * pad_len

            yield (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            )


def make_sft_loader(
    dataset_configs: list[dict],
    seq_len: int,
    tokenizer_name: str,
    batch_size: int,
    seed: int = 42,
    think_open: str = "<think>",
    think_close: str = "</think>",
    user_prefix: str = "user: ",
    assistant_prefix: str = "assistant: ",
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build a streaming DataLoader for SFT training."""
    ds = SFTStream(
        dataset_configs=dataset_configs,
        seq_len=seq_len,
        tok_name=tokenizer_name,
        seed=seed,
        think_open=think_open,
        think_close=think_close,
        user_prefix=user_prefix,
        assistant_prefix=assistant_prefix,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
