"""Evaluation metrics for multi-hop QA."""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison.

    Applies standard HotpotQA normalization:
    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Remove extra whitespace

    Args:
        s: Answer string to normalize

    Returns:
        Normalized string
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in string.punctuation)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score.

    Args:
        prediction: Predicted answer
        ground_truth: Gold answer

    Returns:
        1.0 if normalized strings match, 0.0 otherwise
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score.

    Args:
        prediction: Predicted answer
        ground_truth: Gold answer

    Returns:
        F1 score between 0.0 and 1.0
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    # Handle empty cases
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    # Count common tokens
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def supporting_fact_metrics(
    pred_facts: list[tuple[str, int]],
    gold_facts: list[tuple[str, int]],
) -> tuple[float, float]:
    """Compute EM and F1 for supporting facts.

    Args:
        pred_facts: Predicted supporting facts as (title, sent_idx) tuples
        gold_facts: Gold supporting facts

    Returns:
        Tuple of (exact_match, f1_score)
    """
    pred_set = set(pred_facts)
    gold_set = set(gold_facts)

    # Exact match
    em = float(pred_set == gold_set)

    # F1
    if len(pred_set) == 0 and len(gold_set) == 0:
        f1 = 1.0
    elif len(pred_set) == 0 or len(gold_set) == 0:
        f1 = 0.0
    else:
        intersection = pred_set & gold_set
        precision = len(intersection) / len(pred_set)
        recall = len(intersection) / len(gold_set)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

    return em, f1


def joint_metrics(
    answer_em: float,
    answer_f1: float,
    sp_em: float | None,
    sp_f1: float | None,
) -> tuple[float | None, float | None]:
    """Compute joint answer + supporting fact metrics.

    Args:
        answer_em: Answer exact match
        answer_f1: Answer F1
        sp_em: Supporting fact EM
        sp_f1: Supporting fact F1

    Returns:
        Tuple of (joint_em, joint_f1)
    """
    if sp_em is None or sp_f1 is None:
        return None, None

    joint_em = answer_em * sp_em
    joint_f1 = answer_f1 * sp_f1

    return joint_em, joint_f1
