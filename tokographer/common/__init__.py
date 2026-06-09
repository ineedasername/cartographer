"""Common utilities: tokens, statistics, CLI helpers."""

from tokographer.common.cli import (
    C_H, C_B, C_G, C_Y, C_C, C_E, C_BLD, C_DIM,
    safe_print, setup_stdout, heat, prob_color, fmt_tok, fmt_pct,
)
from tokographer.common.stats import (
    compute_spearman_fast, cohens_d, jaccard_topk, rank_tokens,
)
from tokographer.common.tokens import (
    categorize_token, broader_category, char_script, char_width, display_width,
)
