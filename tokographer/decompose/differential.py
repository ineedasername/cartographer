"""Differential analysis -- compare register states between outcome groups."""

from collections import defaultdict

import numpy as np


def group_by(examples, key_fn):
    """Group example indices by a key function.

    Args:
        examples: list of dicts, each must have 'index' key
        key_fn: callable(example) -> hashable key or None (None = skip)

    Returns:
        dict of {key: [index, ...]}
    """
    groups = defaultdict(list)
    for ex in examples:
        key = key_fn(ex)
        if key is not None:
            groups[key].append(ex['index'])
    return groups


def diff_analysis(matrices, reg_names, group_a_indices, group_b_indices,
                  label_a="A", label_b="B", top_n=15):
    """Compare mean register values between two groups across all layers.

    Computes Cohen's d effect size for each (layer, register) pair.

    Args:
        matrices: [n_examples, n_layers, n_registers] array
        reg_names: list of register name strings
        group_a_indices: list of example indices for group A
        group_b_indices: list of example indices for group B
        label_a: name for group A (for documentation only)
        label_b: name for group B (for documentation only)
        top_n: number of top results to return

    Returns:
        list of (abs_d, d, layer, reg_name, mean_a, mean_b) sorted by |d|,
        truncated to top_n.
    """
    n_layers = matrices.shape[1]
    n_regs = len(reg_names)

    results = []
    for l in range(n_layers):
        for r in range(n_regs):
            vals_a = matrices[group_a_indices, l, r]
            vals_b = matrices[group_b_indices, l, r]

            mean_a = np.mean(vals_a)
            mean_b = np.mean(vals_b)
            std_a = np.std(vals_a) + 1e-8
            std_b = np.std(vals_b) + 1e-8

            # Cohen's d
            pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
            d = (mean_a - mean_b) / pooled_std

            if abs(d) > 0.1:
                results.append((abs(d), d, l, reg_names[r], mean_a, mean_b))

    results.sort(key=lambda x: -x[0])
    return results[:top_n]
