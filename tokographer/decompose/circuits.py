"""Circuit mapping via pairwise register-to-register correlation.

Three correlation methods:
  - Pearson on numeric columns (probabilities, norms)
  - Jaccard on categorical columns (token ID set overlap)
  - Spearman rank correlation on token ID columns
"""

from collections import defaultdict

import numpy as np


def compute_circuit_correlations(X, col_names, min_corr=0.3):
    """Compute pairwise correlations using appropriate methods per column type.

    Token ID columns ('_tid_'): Jaccard overlap + Spearman rank correlation
    Probability/norm columns: Pearson correlation

    Args:
        X: [n_examples, n_columns] data matrix
        col_names: list of column name strings (e.g. 'L00_H2_tid_r0', 'L00_H2_prob_r0')
        min_corr: minimum absolute correlation to retain

    Returns:
        list of (abs_corr, corr, col_a, col_b, method) sorted by strength
    """
    n_cols = X.shape[1]
    n_examples = X.shape[0]

    # Classify columns by type
    is_tid = ['_tid_' in name for name in col_names]
    is_prob = ['_prob_' in name or name.endswith('_Id') or name.endswith('_Lt')
               or name.endswith('_Mt') or name.endswith('_Hp') for name in col_names]
    is_norm = ['_norm' in name for name in col_names]

    numeric_idx = [i for i in range(n_cols) if is_prob[i] or is_norm[i]]
    categ_idx = [i for i in range(n_cols) if is_tid[i]]

    results = []

    # --- Pearson on numeric columns ---
    if numeric_idx:
        Xnum = X[:, numeric_idx]
        num_names = [col_names[i] for i in numeric_idx]
        means = Xnum.mean(axis=0)
        stds = Xnum.std(axis=0)
        stds[stds < 1e-8] = 1.0
        Xn = (Xnum - means) / stds

        CHUNK = 500
        for i_start in range(0, len(numeric_idx), CHUNK):
            i_end = min(i_start + CHUNK, len(numeric_idx))
            for j_start in range(i_start, len(numeric_idx), CHUNK):
                j_end = min(j_start + CHUNK, len(numeric_idx))
                corr_block = (Xn[:, i_start:i_end].T @ Xn[:, j_start:j_end]) / n_examples

                for ci in range(corr_block.shape[0]):
                    for cj in range(corr_block.shape[1]):
                        ai = i_start + ci
                        aj = j_start + cj
                        if aj <= ai:
                            continue
                        r = corr_block[ci, cj]
                        if abs(r) < min_corr:
                            continue
                        na, nb = num_names[ai], num_names[aj]
                        la, lb = na[:3], nb[:3]
                        ra = na[4:].split('_')[0]
                        rb = nb[4:].split('_')[0]
                        if la == lb and ra == rb:
                            continue
                        results.append((abs(r), r, na, nb, 'pearson'))

    # --- Token co-occurrence (Jaccard) on categorical columns ---
    reg_token_sets = {}
    if categ_idx:
        cat_names = [col_names[i] for i in categ_idx]
        Xcat = X[:, categ_idx].astype(np.int64)

        for ci, cname in enumerate(cat_names):
            parts = cname.rsplit('_r', 1)
            base = parts[0]
            if base not in reg_token_sets:
                reg_token_sets[base] = [set() for _ in range(n_examples)]
            for ex in range(n_examples):
                tid = int(Xcat[ex, ci])
                if tid != 0:
                    reg_token_sets[base][ex].add(tid)

        reg_bases = sorted(reg_token_sets.keys())

        for i, base_a in enumerate(reg_bases):
            sets_a = reg_token_sets[base_a]
            for j in range(i + 1, len(reg_bases)):
                base_b = reg_bases[j]
                sets_b = reg_token_sets[base_b]

                la, lb = base_a[:3], base_b[:3]
                ra = base_a[4:].split('_')[0]
                rb = base_b[4:].split('_')[0]
                if la == lb and ra == rb:
                    continue

                overlaps = 0
                for ex in range(n_examples):
                    if sets_a[ex] and sets_b[ex]:
                        inter = len(sets_a[ex] & sets_b[ex])
                        union = len(sets_a[ex] | sets_b[ex])
                        if union > 0:
                            overlaps += inter / union

                jaccard = overlaps / n_examples
                if jaccard >= min_corr:
                    results.append((jaccard, jaccard, base_a, base_b, 'jaccard'))

    # --- Spearman rank correlation on token ID columns ---
    if categ_idx and reg_token_sets:
        from scipy.stats import spearmanr

        reg_bases = sorted(reg_token_sets.keys())

        for i, base_a in enumerate(reg_bases):
            for j in range(i + 1, len(reg_bases)):
                base_b = reg_bases[j]

                la, lb = base_a[:3], base_b[:3]
                ra = base_a[4:].split('_')[0]
                rb = base_b[4:].split('_')[0]
                if la == lb and ra == rb:
                    continue

                vals_a = np.array([list(reg_token_sets[base_a][ex])[0]
                                   if reg_token_sets[base_a][ex] else 0
                                   for ex in range(n_examples)])
                vals_b = np.array([list(reg_token_sets[base_b][ex])[0]
                                   if reg_token_sets[base_b][ex] else 0
                                   for ex in range(n_examples)])

                if np.std(vals_a) < 1e-8 or np.std(vals_b) < 1e-8:
                    continue

                rho, pval = spearmanr(vals_a, vals_b)
                if not np.isnan(rho) and abs(rho) >= min_corr:
                    results.append((abs(rho), rho, base_a, base_b, 'spearman'))

    results.sort(key=lambda x: -x[0])
    return results
