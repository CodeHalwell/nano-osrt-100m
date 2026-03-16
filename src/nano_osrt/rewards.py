"""Verifiable reward functions for GRPO training.

Uses rule-based rewards (no learned reward model):
- Correctness: extract numeric answer and compare to ground truth
- Format: check for proper <think>...</think> usage
- Reasoning quality: reward multi-step thinking when correct
- Truncation penalty: penalize hitting near seq_len (degenerate loops)
- Empty thinking penalty: penalize gaming format with no actual reasoning
"""

import re


def extract_numeric_answer(text: str) -> str | None:
    """Extract the final numeric answer from model output.

    Looks for the answer after </think> tag, or falls back to
    the last number in the text.
    """
    # Try to get answer after </think>
    think_split = text.split("</think>")
    if len(think_split) > 1:
        after_think = think_split[-1].strip()
        # Find numbers (including negatives, decimals, commas)
        numbers = re.findall(r"-?[\d,]+\.?\d*", after_think)
        if numbers:
            return numbers[0].replace(",", "")

    # Fallback: last number in entire text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def extract_gsm8k_answer(answer_text: str) -> str | None:
    """Extract the ground truth answer from GSM8K #### format."""
    if "####" in answer_text:
        final = answer_text.rsplit("####", 1)[1].strip()
        return final.replace(",", "").strip()
    return None


def extract_thinking(text: str, think_open: str = "<think>", think_close: str = "</think>") -> str:
    """Extract the content between think tags."""
    if think_open in text and think_close in text:
        start = text.index(think_open) + len(think_open)
        end = text.index(think_close)
        return text[start:end].strip()
    return ""


def check_format(text: str, think_open: str = "<think>", think_close: str = "</think>") -> bool:
    """Check if the completion uses proper thinking format."""
    return think_open in text and think_close in text


def numeric_match(predicted: str | None, ground_truth: str | None) -> bool:
    """Check if two numeric answers match (handles float comparison)."""
    if predicted is None or ground_truth is None:
        return False

    try:
        pred_val = float(predicted)
        gt_val = float(ground_truth)
        if pred_val == gt_val:
            return True
        if abs(gt_val) > 1e-8:
            return abs(pred_val - gt_val) / abs(gt_val) < 1e-4
        return abs(pred_val - gt_val) < 1e-8
    except (ValueError, OverflowError):
        return predicted.strip() == ground_truth.strip()


def count_reasoning_steps(thinking: str) -> int:
    """Count reasoning steps in the thinking block.

    Looks for numbered steps, newlines with content, or sentence breaks
    that indicate multi-step reasoning.
    """
    if not thinking:
        return 0

    # Count numbered steps (1. 2. 3. or Step 1, Step 2)
    numbered = re.findall(r"(?:^|\n)\s*(?:\d+[\.\):]|step\s+\d+)", thinking, re.IGNORECASE)
    if len(numbered) >= 2:
        return len(numbered)

    # Fallback: count non-empty lines as steps
    lines = [l.strip() for l in thinking.split("\n") if l.strip()]
    return len(lines)


def compute_reward(
    completion: str,
    ground_truth_answer: str,
    correctness_weight: float = 1.0,
    format_weight: float = 0.2,
    length_penalty: float = 0.0,
    think_open: str = "<think>",
    think_close: str = "</think>",
    max_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_bonus: float = 0.3,
    truncation_penalty: float = -0.5,
    empty_think_penalty: float = -0.1,
) -> tuple[float, dict]:
    """Compute the total reward for a single completion.

    Reward components:
        1. Correctness (+1.0): correct final answer
        2. Format (+0.2): proper <think>...</think> tags
        3. Reasoning bonus (+0.3): multi-step thinking when correct
        4. Truncation penalty (-0.5): output hit 90% of max tokens
        5. Empty thinking penalty (-0.1): <think></think> with no content
        6. Length penalty (0.0 default): disabled, reasoning needs room

    Returns:
        (total_reward, reward_breakdown_dict)
    """
    reward = 0.0
    breakdown = {}

    # 1. Correctness reward (core signal)
    predicted = extract_numeric_answer(completion)
    gt = extract_gsm8k_answer(ground_truth_answer)
    if gt is None:
        gt = ground_truth_answer.replace(",", "").strip()

    correct = numeric_match(predicted, gt)
    correctness_r = correctness_weight if correct else 0.0
    reward += correctness_r
    breakdown["correct"] = correct
    breakdown["correctness_reward"] = correctness_r

    # 2. Format reward
    has_format = check_format(completion, think_open, think_close)
    format_r = format_weight if has_format else 0.0
    reward += format_r
    breakdown["has_format"] = has_format
    breakdown["format_reward"] = format_r

    # 3. Reasoning quality bonus (only when correct — reward thinking that works)
    thinking = extract_thinking(completion, think_open, think_close)
    n_steps = count_reasoning_steps(thinking)
    breakdown["reasoning_steps"] = n_steps
    breakdown["thinking_length"] = len(thinking.split()) if thinking else 0

    if correct and n_steps >= 2:
        # Graduated bonus: more steps = more reward, capped at 3+ steps
        step_bonus = reasoning_bonus * min(n_steps / 3.0, 1.0)
        reward += step_bonus
        breakdown["reasoning_bonus"] = step_bonus
    else:
        breakdown["reasoning_bonus"] = 0.0

    # 4. Truncation penalty (hit 90% of max tokens — likely degenerate loop)
    if max_tokens > 0 and completion_tokens >= int(max_tokens * 0.9):
        reward += truncation_penalty
        breakdown["truncated"] = True
        breakdown["truncation_penalty"] = truncation_penalty
    else:
        breakdown["truncated"] = False
        breakdown["truncation_penalty"] = 0.0

    # 5. Empty thinking penalty (gaming format with no content)
    if has_format and len(thinking.split()) < 3:
        reward += empty_think_penalty
        breakdown["empty_thinking"] = True
        breakdown["empty_think_penalty"] = empty_think_penalty
    else:
        breakdown["empty_thinking"] = False
        breakdown["empty_think_penalty"] = 0.0

    # 6. Legacy length penalty (disabled by default)
    n_words = len(completion.split())
    length_r = length_penalty * n_words
    reward += length_r
    breakdown["n_words"] = n_words
    breakdown["length_reward"] = length_r

    breakdown["total_reward"] = reward
    return reward, breakdown


def compute_group_advantages(rewards: list[float]) -> list[float]:
    """Compute advantages within a group using GRPO normalization.

    GRPO normalizes rewards within each group to zero mean, unit variance.
    This removes the need for a value function baseline.
    """
    n = len(rewards)
    if n == 0:
        return []
    if n == 1:
        return [0.0]

    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5

    if std < 1e-8:
        return [0.0] * n

    return [(r - mean) / std for r in rewards]
