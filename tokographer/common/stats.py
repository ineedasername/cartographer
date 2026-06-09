"""Shared statistical functions used across the tokographer ecosystem."""

import numpy as np


def rank_tokens(probs):
    """Convert probability array to rank array (0 = highest probability).

    This pattern appears ~15 times across the codebase as:
        np.argsort(np.argsort(-probs)).astype(np.int32)

    Args:
        probs: 1D array of probabilities (float)

    Returns:
        1D array of ranks (int32), where 0 = highest probability
    """
    return np.argsort(np.argsort(-probs)).astype(np.int32)


def compute_spearman_fast(base_ranks, prompt_probs):
    """Compute Spearman rho between baseline ranks and prompt probabilities.

    Pure numpy implementation optimized for vocab-scale (262k) rank comparison.
    Uses the classical formula: rho = 1 - 6*sum(d²) / (n*(n²-1))

    Args:
        base_ranks: [vocab_size] int32 — rank ordering from baseline (0 = highest)
        prompt_probs: [vocab_size] float32 — probability distribution from prompt

    Returns:
        float — Spearman rho in [-1, 1]
    """
    prompt_ranks = np.argsort(np.argsort(-prompt_probs)).astype(np.int64)
    base = base_ranks.astype(np.int64)
    d_sq_sum = np.sum((prompt_ranks - base) ** 2, dtype=np.float64)
    n = float(len(base_ranks))
    return 1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1.0))


def cohens_d(group_a, group_b, bessel=False):
    """Compute Cohen's d effect size between two groups.

    Args:
        group_a: array-like — values for group A
        group_b: array-like — values for group B
        bessel: if True, use Bessel-corrected pooled variance
                (better for small samples)

    Returns:
        float — Cohen's d (positive means group_a > group_b)
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)

    mean_a = np.mean(a)
    mean_b = np.mean(b)

    if bessel:
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return 0.0
        s1 = np.var(a, ddof=1)
        s2 = np.var(b, ddof=1)
        pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    else:
        std_a = np.std(a) + 1e-8
        std_b = np.std(b) + 1e-8
        pooled = np.sqrt((std_a**2 + std_b**2) / 2)

    return (mean_a - mean_b) / pooled


def jaccard_topk(sets_a, sets_b):
    """Compute average Jaccard overlap between paired token sets.

    For each pair of sets, computes |intersection|/|union|,
    then averages across all pairs where both sets are non-empty.

    Args:
        sets_a: list of sets — token ID sets for register A per example
        sets_b: list of sets — token ID sets for register B per example

    Returns:
        float — average Jaccard similarity in [0, 1]
    """
    n = len(sets_a)
    if n == 0:
        return 0.0

    total = 0.0
    count = 0
    for i in range(n):
        if sets_a[i] and sets_b[i]:
            inter = len(sets_a[i] & sets_b[i])
            union = len(sets_a[i] | sets_b[i])
            if union > 0:
                total += inter / union
                count += 1

    return total / count if count > 0 else 0.0
