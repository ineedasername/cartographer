"""
Prediction comparison utilities: categorize changes between baseline and intervention.
"""

from collections import Counter
from typing import List, Optional, Sequence, Union


def categorize_changes(
    baseline_preds: Sequence,
    intervention_preds: Sequence,
    gold_labels: Sequence,
    label_names: Optional[dict] = None,
) -> list:
    """Classify each prediction change between baseline and intervention runs.

    For each example, assigns one of 8 categories based on the pattern
    (gold, baseline_pred, intervention_pred):

      rescued        — was wrong, now correct
      stuck_wrong    — was wrong, still wrong (same prediction)
      flipped_wrong  — was wrong, changed to a different wrong answer
      degraded       — was correct, now wrong
      held_correct   — was correct, still correct
      changed_correct— was correct, changed but still correct (multi-label edge case)
      other_changed  — changed but doesn't fit above patterns
      unchanged      — no change in prediction

    Args:
        baseline_preds: baseline predictions (any comparable type)
        intervention_preds: intervention predictions (same type)
        gold_labels: ground truth labels (same type)
        label_names: optional dict mapping values to display names

    Returns:
        List of dicts with keys: index, gold, baseline, intervention, category, changed
    """
    if not (len(baseline_preds) == len(intervention_preds) == len(gold_labels)):
        raise ValueError("All inputs must have the same length")

    def _name(val):
        if label_names and val in label_names:
            return label_names[val]
        return val

    results = []
    for i, (base, intv, gold) in enumerate(
        zip(baseline_preds, intervention_preds, gold_labels)
    ):
        changed = base != intv
        base_correct = base == gold
        intv_correct = intv == gold

        if not changed:
            category = "held_correct" if base_correct else "stuck_wrong"
        elif not base_correct and intv_correct:
            category = "rescued"
        elif not base_correct and not intv_correct:
            category = "flipped_wrong"
        elif base_correct and not intv_correct:
            category = "degraded"
        elif base_correct and intv_correct:
            category = "changed_correct"
        else:
            category = "other_changed"

        results.append({
            "index": i,
            "gold": _name(gold),
            "baseline": _name(base),
            "intervention": _name(intv),
            "category": category,
            "changed": changed,
        })

    return results


def summarize_categories(categorized: list) -> dict:
    """Return counts and rates for each category.

    Args:
        categorized: output of categorize_changes()

    Returns:
        Dict with 'counts' (Counter), 'total', and 'rates' (fraction per category).
    """
    counts = Counter(r["category"] for r in categorized)
    total = len(categorized)
    rates = {k: v / total if total > 0 else 0.0 for k, v in counts.items()}
    return {"counts": dict(counts), "total": total, "rates": rates}
