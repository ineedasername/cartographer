"""Rank-displacement scan modules — Spearman rank-displacement analysis.

Measures how the token ranking produced by each attention head/layer shifts
relative to a null-prompt baseline, by projecting per-head outputs through the
final norm and unembedding and comparing the resulting rank orderings.

Submodules:
  scan          — Database init, baseline capture, prompt scanning
  compare       — Full-softmax comparison between coordinates
  summary       — Dashboard and summary reporting
  queries       — Cross-model DB query utilities
  multi_model   — Sequential multi-model scanning and comparison
"""

from cartographer.rankd.scan import (
    init_db, capture_baseline, run_scan,
    compute_spearman_fast, get_model_internals,
)
from cartographer.rankd.compare import (
    get_full_probs, load_baseline_ranks, get_scan_input_ids,
)
from cartographer.rankd.summary import (
    get_scan_summary, get_phase_summary, get_layer_summary, print_dashboard,
    heat,
)
