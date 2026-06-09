"""
Model-agnostic sweep harness for intervention experiments.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .hooks import InterventionHook


@dataclass
class SweepResult:
    """One row from a sweep — strength value plus arbitrary metrics."""
    strength: float
    metrics: Dict[str, Any]


@dataclass
class SweepReport:
    """Full sweep output."""
    results: List[SweepResult] = field(default_factory=list)
    baseline_metrics: Optional[Dict[str, Any]] = None
    elapsed_seconds: float = 0.0


def run_sweep(
    model,
    strengths: Sequence[float],
    hook_factory: Callable[[float], InterventionHook],
    evaluate_fn: Callable[[Any], Dict[str, Any]],
    baseline_fn: Optional[Callable[[], Dict[str, Any]]] = None,
    csv_path: Optional[str] = None,
    csv_fields: Optional[List[str]] = None,
    progress_every: int = 10,
    progress_fn: Optional[Callable[[int, int, float, Dict], None]] = None,
) -> SweepReport:
    """Sweep over intervention strengths, evaluate, collect metrics.

    Args:
        model: The model to intervene on (passed to hook.install).
        strengths: Sequence of strength values to sweep.
        hook_factory: Callable(strength) -> InterventionHook. Called once per
            strength. The hook is installed, evaluation runs, then it's removed.
        evaluate_fn: Callable(model) -> dict of metrics. Runs inference and
            returns a metrics dict.
        baseline_fn: Optional callable that returns baseline metrics (run once
            before sweep, no hooks installed).
        csv_path: If provided, write results row-by-row to this CSV.
        csv_fields: Column names for CSV. If None, inferred from first result.
        progress_every: Print/call progress every N strengths.
        progress_fn: Optional callback(step, total, strength, metrics) for
            custom progress reporting. If None, prints to stdout.

    Returns:
        SweepReport with all results and optional baseline.
    """
    report = SweepReport()
    t0 = time.time()

    # Baseline
    if baseline_fn is not None:
        report.baseline_metrics = baseline_fn()

    # CSV setup
    writer = None
    csv_file = None
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')

    try:
        for i, strength in enumerate(strengths):
            hook = hook_factory(strength)
            hook.install(model)

            try:
                metrics = evaluate_fn(model)
            finally:
                hook.remove()

            metrics['strength'] = strength
            report.results.append(SweepResult(strength=strength, metrics=metrics))

            # CSV
            if csv_file is not None:
                if writer is None:
                    fields = csv_fields or sorted(metrics.keys())
                    writer = csv.DictWriter(csv_file, fieldnames=fields,
                                            extrasaction='ignore')
                    writer.writeheader()
                writer.writerow(metrics)
                csv_file.flush()

            # Progress
            if (i + 1) % progress_every == 0:
                if progress_fn:
                    progress_fn(i + 1, len(strengths), strength, metrics)
                else:
                    acc = metrics.get('overall_acc', metrics.get('accuracy', '?'))
                    print(f"  {i+1}/{len(strengths)}  s={strength:+.2f}  acc={acc}")

    finally:
        if csv_file is not None:
            csv_file.close()

    report.elapsed_seconds = time.time() - t0
    return report


def make_strength_range(start: float, end: float, step: float) -> List[float]:
    """Generate a list of rounded strength values (avoids float drift)."""
    strengths = []
    s = start
    while s <= end + 1e-6:
        strengths.append(round(s, 4))
        s += step
    return strengths
