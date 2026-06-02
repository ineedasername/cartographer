"""
Sweep analysis utilities: transition detection and Pareto frontier computation.

Source: negation/analyze_sweep.py (transition detection logic)
        negation/analyze_sweep_failures.py (Pareto frontier bucketing)
"""

from typing import List, Tuple, Union


def _to_list_of_dicts(df_or_rows) -> list:
    """Accept either a list of dicts or a pandas DataFrame, return list of dicts."""
    if hasattr(df_or_rows, "to_dict"):
        return df_or_rows.to_dict("records")
    return list(df_or_rows)


def find_transitions(
    df_or_rows,
    accuracy_col: str,
    target_col: str,
    threshold: float = 0.005,
) -> list:
    """Detect where a sweep curve changes character.

    Scans sorted rows for points where:
      - target_col first exceeds baseline by > threshold
      - accuracy_col first drops below baseline by > threshold
      - accuracy_col peaks (region within threshold of max)
      - target_col crosses accuracy_col

    Returns a list of dicts, each with keys:
      type  — one of 'target_improves', 'accuracy_degrades', 'peak_region', 'crossover'
      index — row index where the transition occurs
      row   — the row dict at that point
      detail — human-readable description

    Source pattern: negation/analyze_sweep.py lines 84-126
    """
    rows = _to_list_of_dicts(df_or_rows)
    if not rows:
        return []

    transitions = []
    baseline = rows[0]
    base_acc = baseline[accuracy_col]
    base_tgt = baseline[target_col]

    # First improvement in target
    for i, r in enumerate(rows):
        if r[target_col] > base_tgt + threshold:
            transitions.append({
                "type": "target_improves",
                "index": i,
                "row": r,
                "detail": f"{target_col} exceeds baseline ({base_tgt:.4f}) at index {i}",
            })
            break

    # First degradation in accuracy
    for i, r in enumerate(rows):
        if r[accuracy_col] < base_acc - threshold:
            transitions.append({
                "type": "accuracy_degrades",
                "index": i,
                "row": r,
                "detail": f"{accuracy_col} drops below baseline ({base_acc:.4f}) at index {i}",
            })
            break

    # Peak region
    best_acc = max(r[accuracy_col] for r in rows)
    peak_indices = [i for i, r in enumerate(rows) if r[accuracy_col] >= best_acc - threshold]
    if peak_indices:
        transitions.append({
            "type": "peak_region",
            "index": peak_indices[0],
            "row": rows[peak_indices[0]],
            "detail": f"Peak region spans indices [{peak_indices[0]}, {peak_indices[-1]}], "
                      f"best {accuracy_col}={best_acc:.4f}",
        })

    # Crossover: where target_col overtakes accuracy_col
    for i in range(1, len(rows)):
        prev_diff = rows[i - 1][target_col] - rows[i - 1][accuracy_col]
        curr_diff = rows[i][target_col] - rows[i][accuracy_col]
        if prev_diff < 0 and curr_diff >= 0:
            transitions.append({
                "type": "crossover",
                "index": i,
                "row": rows[i],
                "detail": f"{target_col} crosses above {accuracy_col} at index {i}",
            })
            break

    return transitions


def pareto_frontier(
    df_or_rows,
    x_col: str,
    y_col: str,
    maximize_both: bool = True,
) -> list:
    """Compute Pareto-optimal points from a set of rows.

    Returns list of row dicts that are not dominated by any other row.
    A point dominates another if it is >= on both objectives (when maximize_both=True)
    and strictly > on at least one.

    Source pattern: negation/analyze_sweep_failures.py lines 167-186
    """
    rows = _to_list_of_dicts(df_or_rows)
    if not rows:
        return []

    sign = 1.0 if maximize_both else -1.0

    # Sort by x descending (if maximizing)
    sorted_rows = sorted(rows, key=lambda r: sign * r[x_col], reverse=True)

    frontier = []
    best_y = float("-inf")

    for r in sorted_rows:
        y_val = sign * r[y_col]
        if y_val >= best_y:
            frontier.append(r)
            best_y = y_val

    # Return sorted by x ascending
    frontier.sort(key=lambda r: r[x_col])
    return frontier
