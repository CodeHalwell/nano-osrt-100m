"""Verifiable reward functions for GRPO training.

Uses rule-based rewards (no learned reward model):
- Correctness: extract numeric answer and compare to ground truth
- Format: check for proper <think>...</think> usage
- Length: mild penalty to encourage concise reasoning
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
        # Clean up commas and whitespace
        return final.replace(",", "").strip()
    return None


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
        # Exact match for integers, close match for floats
        if pred_val == gt_val:
            return True
        # Relative tolerance for floats
        if abs(gt_val) > 1e-8:
            return abs(pred_val - gt_val) / abs(gt_val) < 1e-4
        return abs(pred_val - gt_val) < 1e-8
    except (ValueError, OverflowError):
        return predicted.strip() == ground_truth.strip()


def compute_reward(
    completion: str,
    ground_truth_answer: str,
    correctness_weight: float = 1.0,
    format_weight: float = 0.2,
    length_penalty: float = -0.001,
    think_open: str = "<think>",
    think_close: str = "</think>",
) -> tuple[float, dict]:
    """Compute the total reward for a single completion.

    Returns:
        (total_reward, reward_breakdown_dict)
    """
    reward = 0.0
    breakdown = {}

    # 1. Correctness reward
    predicted = extract_numeric_answer(completion)
    gt = extract_gsm8k_answer(ground_truth_answer)
    if gt is None:
        # If no #### format, try raw number
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

    # 3. Length penalty (number of tokens approximated by whitespace split)
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
