"""Verifiable reward functions for GRPO training.

Uses rule-based rewards (no learned reward model):
- Correctness: extract numeric answer and compare to ground truth
- Format: check for proper <think>...</think> usage
- Reasoning quality: reward multi-step thinking when correct
- Truncation penalty: penalize hitting near seq_len (degenerate loops)
- Empty thinking penalty: penalize gaming format with no actual reasoning
"""

import re


def extract_numeric_answer(
    text: str,
    think_close: str = "</think>",
    answer_open: str = "<|answer|>",
    answer_close: str = "<|/answer|>",
) -> str | None:
    """Extract the final numeric answer from model output.

    Priority order:
        1. LAST number inside answer_open / answer_close tags
           (many correct responses say e.g. "After 3 steps, the answer
           is 12" — returning the first number would score 3, wrong.
           The last number inside the answer tag is the model's final
           committed answer.)
        2. Last number after think_close (v3 format)
        3. Last number in the entire text (fallback)
    """
    # Strategy 1: look inside explicit answer tags. Use rindex on the
    # opening tag so a model that emits multiple <|answer|>...<|/answer|>
    # pairs (e.g. self-correcting mid-completion) gets credit for its
    # FINAL committed answer, not the first attempt. With rindex, end is
    # always the first close after the last open — correct for nested or
    # repeated tag pairs.
    if answer_open in text and answer_close in text:
        start = text.rindex(answer_open) + len(answer_open)
        end = (
            text.index(answer_close, start)
            if answer_close in text[start:]
            else len(text)
        )
        inside = text[start:end]
        numbers = re.findall(r"-?[\d,]+\.?\d*", inside)
        if numbers:
            return numbers[-1].replace(",", "")

    # Strategy 2: last number after the think_close tag
    if think_close and think_close in text:
        after_think = text.split(think_close, 1)[1].strip()
        numbers = re.findall(r"-?[\d,]+\.?\d*", after_think)
        if numbers:
            return numbers[-1].replace(",", "")

    # Strategy 3: fallback to last number in entire text
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


def extract_thinking(
    text: str,
    think_open: str = "<think>",
    think_close: str = "</think>",
) -> str:
    """Extract the content between think tags."""
    if think_open in text and think_close in text:
        start = text.index(think_open) + len(think_open)
        end = text.index(think_close)
        return text[start:end].strip()
    return ""


def check_format(
    text: str,
    think_open: str = "<think>",
    think_close: str = "</think>",
) -> bool:
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
    # Using [ \t]* instead of \s* to prevent ReDoS from overlapping newline matches
    numbered = re.findall(
        r"(?:^|\n)[ \t]*(?:\d+[\.\):]|step\s+\d+)",
        thinking,
        re.IGNORECASE,
    )
    if len(numbered) >= 2:
        return len(numbered)

    # Fallback: count non-empty lines as steps
    lines = [line.strip() for line in thinking.split("\n") if line.strip()]
    return len(lines)


def correctness_partial_credit(
    predicted: str | None,
    ground_truth: str | None,
) -> tuple[float, str]:
    """Unsloth-style tiered correctness reward.

    Instead of binary correct/wrong (which produces zero advantage
    when all rollouts in a group are wrong — see GRPO run 2/3 stuck
    pattern), this gives partial credit for "close" answers and
    negative reward for "way off" or unparseable.

    Returns (score, tier_label). Score ranges from +5.0 (exact) to
    -2.5 (no extractable number).

    Why partial credit matters
    ──────────────────────────
    At ~5 % gsm8k baseline accuracy with group_size=8, most groups
    have ZERO correct rollouts under binary scoring → all rollouts
    get reward 0 (or just format reward 0.2) → group-relative
    advantage normalisation gives all zeros → no gradient → policy
    frozen. Partial credit ensures variance in rollout rewards even
    when none are exactly right: a rollout that produces "53"
    against ground truth "18" gets a different reward from one that
    produces "144" or "blah". Variance → advantages → gradient.

    Tier schedule (matches the Unsloth GRPO tutorial pattern):
        exact match     +5.0     ← what we want
        within   5 %    +3.5
        within  10 %    +2.5
        within  20 %    +1.5
        0.5x – 2.0x     -0.5     ← still in same order of magnitude
        wrong number    -2.0     ← off by an order of magnitude+
        no number       -2.5     ← couldn't even parse one
    """
    if predicted is None or predicted == "":
        return -2.5, "no_extraction"
    if ground_truth is None or ground_truth == "":
        return 0.0, "no_ground_truth"

    # String-strip exact match before numeric coercion (handles "53"
    # vs " 53" vs "53.0" cases the comma-strip in numeric_match
    # already handles).
    if str(predicted).strip() == str(ground_truth).strip():
        return 5.0, "exact"

    try:
        gt = float(str(ground_truth).strip().replace(",", ""))
        pr = float(str(predicted).strip().replace(",", ""))
    except (ValueError, TypeError):
        return -2.0, "non_numeric"

    if gt == 0:
        # Special-case division-by-zero: only exact match is valid
        return (5.0, "exact") if pr == 0 else (-2.0, "zero_gt_wrong")

    if pr == gt:
        return 5.0, "exact_numeric"

    ratio = pr / gt
    if 0.95 <= ratio <= 1.05:
        return 3.5, "within_5_pct"
    if 0.90 <= ratio <= 1.10:
        return 2.5, "within_10_pct"
    if 0.80 <= ratio <= 1.20:
        return 1.5, "within_20_pct"
    if 0.50 <= ratio <= 2.00:
        return -0.5, "wrong_same_order"
    return -2.0, "wrong_far_off"


def length_ramp_penalty(completion_tokens: int, max_tokens: int) -> float:
    """Smooth length penalty that ramps as the completion approaches the
    output budget.

    Designed to discourage the model from padding its <|think|> block to
    fill the entire generation budget — observed in math reasoning
    models that learn "longer reasoning = more reward" without a
    counter-pressure. The ramp targets the *use of the budget*, not
    the absolute token count, so it scales with whatever max_gen_len
    is configured.

    Schedule (linear between breakpoints):
        ≤ 80 % of max_tokens : 0.0   (think freely, no penalty)
          80 %               : -0.5
          90 %               : -0.75
          100 %              : -1.0  (caps; model hit the cap)

    Returns 0.0 if max_tokens is 0 or completion_tokens is 0 (used as
    a no-op when GRPO doesn't pass token info — e.g. unit tests).
    """
    if max_tokens <= 0 or completion_tokens <= 0:
        return 0.0
    pct = completion_tokens / max_tokens
    if pct < 0.80:
        return 0.0
    if pct >= 1.00:
        return -1.0
    # Linear ramp from -0.5 at 80 % to -1.0 at 100 %:
    #   slope = (−1.0 − (−0.5)) / (1.00 − 0.80) = −2.5
    # Verifies: 0.80→-0.5, 0.90→-0.75, 1.00→-1.0
    return -0.5 + (pct - 0.80) * (-2.5)


def compute_reward(
    completion: str,
    ground_truth_answer: str,
    correctness_weight: float = 1.0,
    format_weight: float = 0.2,
    length_penalty: float = 0.0,
    think_open: str = "<think>",
    think_close: str = "</think>",
    answer_open: str = "<|answer|>",
    answer_close: str = "<|/answer|>",
    max_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_bonus: float = 0.3,
    truncation_penalty: float = -0.5,
    empty_think_penalty: float = -0.1,
) -> tuple[float, dict]:
    """Compute the total reward for a single completion.

    Reward components:
        1. Correctness (+1.0): correct final answer
        2. Format (+0.2): proper think_open/think_close tags present
        3. Reasoning bonus (+0.3): multi-step thinking when correct
        4. Truncation penalty (-0.5): output hit 90% of max tokens
        5. Empty thinking penalty (-0.1): thinking block has no content
        6. Length penalty (0.0 default): disabled, reasoning needs room

    IMPORTANT: pass the actual tag strings used during training (e.g. v4 uses
    single-token native tags '<|think|>' and '<|/think|>'). The defaults match
    v3's plain-string tags and will silently fail for v4 if not overridden.

    Returns:
        (total_reward, reward_breakdown_dict)
    """
    reward = 0.0
    breakdown = {}

    # 1. Correctness reward (core signal) — Unsloth-style partial credit.
    # Replaces the original binary correctness (+1 if exact, 0 otherwise)
    # which caused GRPO collapse on this model: at ~5 % gsm8k baseline
    # acc with group_size=8, most groups had zero correct rollouts →
    # uniform rewards → zero advantage → frozen updates. See
    # correctness_partial_credit() for the tier schedule (+5.0 exact
    # down to -2.5 no-extract). correctness_weight now scales the
    # ENTIRE tier output (default 1.0 keeps the raw Unsloth schedule).
    predicted = extract_numeric_answer(
        completion,
        think_close=think_close,
        answer_open=answer_open,
        answer_close=answer_close,
    )
    gt = extract_gsm8k_answer(ground_truth_answer)
    if gt is None:
        gt = ground_truth_answer.replace(",", "").strip()

    tier_score, tier_label = correctness_partial_credit(predicted, gt)
    correctness_r = tier_score * correctness_weight
    correct = tier_label in ("exact", "exact_numeric")
    reward += correctness_r
    breakdown["correct"] = correct
    breakdown["correctness_tier"] = tier_label
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

    # 4. Length-ramp penalty (smooth, scales toward output token budget)
    # Replaces the old binary truncation_penalty (which only fired at
    # 90 %). Now: 0 below 80 %, then linear ramp from -0.5 at 80 %
    # → -0.75 at 90 % → -1.0 at 100 %. Discourages padding the think
    # block to fill the budget without harshly penalising completions
    # that legitimately need 60-75 % of the room.
    # `truncation_penalty` config field is now unused (kept for
    # backwards compat with saved configs); the ramp is hardcoded
    # in length_ramp_penalty() above.
    length_pen = length_ramp_penalty(completion_tokens, max_tokens)
    reward += length_pen
    breakdown["length_ramp_penalty"] = length_pen
    breakdown["length_ramp_pct"] = (
        completion_tokens / max_tokens if max_tokens > 0 else 0.0
    )
    # Keep `truncated` flag for compatibility with downstream logging
    # — true iff we hit the cap (≥ 100 %).
    breakdown["truncated"] = max_tokens > 0 and completion_tokens >= max_tokens
    breakdown["truncation_penalty"] = 0.0  # deprecated; see length_ramp_penalty

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
    std = var**0.5

    if std < 1e-8:
        return [0.0] * n

    return [(r - mean) / std for r in rewards]
