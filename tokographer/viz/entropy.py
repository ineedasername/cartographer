"""Entropy landscape across layers x generation steps."""

import numpy as np


def compute_entropy_grid(scan, col_idx):
    """Compute entropy (bits) at every step x layer cell.

    Args:
        scan: ndarray (steps, layers, cols, vocab_size)
        col_idx: int column index

    Returns:
        ndarray (steps, layers) float32 entropy in bits
    """
    num_steps, num_layers, num_cols, vocab = scan.shape
    entropy = np.zeros((num_steps, num_layers), dtype=np.float32)

    for step_i in range(num_steps):
        for layer_i in range(num_layers):
            probs = scan[step_i, layer_i, col_idx, :].astype(np.float32)
            probs = probs + 1e-10
            ent = -(probs * np.log2(probs)).sum()
            entropy[step_i, layer_i] = ent

    return entropy


def plot_entropy(entropy, meta, col_name, ax=None, title=None, vmin=None, vmax=None):
    """Render entropy heatmap.

    Args:
        entropy: ndarray (steps, layers) from compute_entropy_grid()
        meta: dict scan metadata
        col_name: str column name
        ax: matplotlib Axes (created if None)
        title: str optional override title
        vmin, vmax: float optional color scale bounds

    Returns:
        (fig, ax)
    """
    import matplotlib.pyplot as plt

    num_steps, num_layers = entropy.shape
    steps_info = meta['steps']

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(max(14, num_steps * 0.5), 8))
    else:
        fig = ax.figure

    if vmin is None:
        vmin = entropy.min()
    if vmax is None:
        vmax = entropy.max()

    im = ax.imshow(
        entropy.T,
        aspect='auto',
        cmap='inferno',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
        origin='lower',
    )

    step_labels = []
    for s in steps_info[:num_steps]:
        tok = s['token'].replace('\n', '\\n').replace(' ', '_')
        if len(tok) > 8:
            tok = tok[:7] + '.'
        step_labels.append(tok)

    ax.set_xticks(range(num_steps))
    ax.set_xticklabels(step_labels, rotation=60, ha='right', fontsize=7, fontfamily='monospace')
    ax.set_xlabel('Generation Step', fontsize=9)

    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f'L{i:02d}' for i in range(num_layers)], fontsize=7, fontfamily='monospace')
    ax.set_ylabel('Layer', fontsize=9)

    if title is None:
        title = f'Entropy (bits) by layer x step  [{col_name}]'
    ax.set_title(title, fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Entropy (bits)', fontsize=9)

    max_pos = np.unravel_index(entropy.argmax(), entropy.shape)
    min_pos = np.unravel_index(entropy.argmin(), entropy.shape)
    ax.plot(max_pos[0], max_pos[1], 'w^', markersize=8, label=f'Max: {entropy.max():.1f}b')
    ax.plot(min_pos[0], min_pos[1], 'wv', markersize=8, label=f'Min: {entropy.min():.1f}b')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.7)

    return fig, ax


def plot_entropy_profiles(entropy, meta, col_name, ax=None):
    """Plot per-layer and per-step entropy profiles.

    Args:
        entropy: ndarray (steps, layers) from compute_entropy_grid()
        meta: dict scan metadata
        col_name: str column name
        ax: matplotlib Axes for layer profile (step profile on ax2 if ax is None)

    Returns:
        fig
    """
    import matplotlib.pyplot as plt

    num_steps, num_layers = entropy.shape

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig = ax.figure
        ax1 = ax
        ax2 = None

    layer_avg = entropy.mean(axis=0)
    layer_std = entropy.std(axis=0)
    ax1.barh(range(num_layers), layer_avg, xerr=layer_std, height=0.7,
             color=plt.cm.inferno(layer_avg / layer_avg.max()), alpha=0.8, capsize=2)
    ax1.set_yticks(range(num_layers))
    ax1.set_yticklabels([f'L{i:02d}' for i in range(num_layers)], fontsize=7)
    ax1.set_xlabel('Entropy (bits)', fontsize=9)
    ax1.set_title(f'Average Entropy by Layer [{col_name}]', fontsize=10)
    ax1.invert_yaxis()

    if ax2 is not None:
        step_avg = entropy.mean(axis=1)
        step_labels = []
        for s in meta['steps'][:num_steps]:
            tok = s['token'].replace('\n', '\\n').replace(' ', '_')
            if len(tok) > 6:
                tok = tok[:5] + '.'
            step_labels.append(tok)

        colors = plt.cm.inferno(step_avg / step_avg.max())
        ax2.bar(range(num_steps), step_avg, color=colors, alpha=0.8)
        ax2.set_xticks(range(num_steps))
        ax2.set_xticklabels(step_labels, rotation=60, ha='right', fontsize=7, fontfamily='monospace')
        ax2.set_ylabel('Entropy (bits)', fontsize=9)
        ax2.set_title(f'Average Entropy by Step [{col_name}]', fontsize=10)

    return fig
