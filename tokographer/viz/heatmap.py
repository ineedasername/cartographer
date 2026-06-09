"""Token rank heatmap across layers x generation steps."""

import numpy as np


def compute_rank_grid(scan, token_id, col_idx):
    """Compute rank of a specific token at every step x layer cell.

    Args:
        scan: ndarray (steps, layers, cols, vocab_size)
        token_id: int vocabulary index
        col_idx: int column index

    Returns:
        ndarray (steps, layers) int32 ranks (1-indexed)
    """
    num_steps, num_layers, num_cols, vocab = scan.shape
    ranks = np.zeros((num_steps, num_layers), dtype=np.int32)

    for step_i in range(num_steps):
        for layer_i in range(num_layers):
            cell = scan[step_i, layer_i, col_idx, :].astype(np.float32)
            sorted_idx = np.argsort(cell)[::-1]
            ranks[step_i, layer_i] = int(np.where(sorted_idx == token_id)[0][0]) + 1

    return ranks


def make_heatmap(rank_grid, meta, token_name, col_name, vmax=None, ax=None):
    """Render a single token's rank heatmap on the given axes.

    Args:
        rank_grid: ndarray (steps, layers) from compute_rank_grid()
        meta: dict scan metadata
        token_name: str display name
        col_name: str column name
        vmax: int optional max rank for color scale
        ax: matplotlib Axes (created if None)

    Returns:
        (fig, ax)
    """
    import matplotlib.pyplot as plt

    num_steps, num_layers = rank_grid.shape
    steps_info = meta['steps']

    log_ranks = np.log10(rank_grid.T.astype(np.float64))

    if vmax is None:
        vmax_log = np.log10(meta.get('vocab_size', 262144))
    else:
        vmax_log = np.log10(vmax)

    cmap = plt.cm.magma_r

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(max(14, num_steps * 0.5), 8))
    else:
        fig = ax.figure

    im = ax.imshow(
        log_ranks,
        aspect='auto',
        cmap=cmap,
        vmin=0,
        vmax=vmax_log,
        interpolation='nearest',
        origin='lower',
    )

    step_labels = []
    for s in steps_info[:num_steps]:
        tok = s['token'].replace('\n', '\\n').replace(' ', '_')
        if len(tok) > 8:
            tok = tok[:7] + '.'
        step_labels.append(f"{tok}")

    ax.set_xticks(range(num_steps))
    ax.set_xticklabels(step_labels, rotation=60, ha='right', fontsize=7, fontfamily='monospace')
    ax.set_xlabel('Generation Step (output token)', fontsize=9)

    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f'L{i:02d}' for i in range(num_layers)], fontsize=7, fontfamily='monospace')
    ax.set_ylabel('Layer', fontsize=9)

    ax.set_title(f'"{token_name}" rank by layer x step  [{col_name}]', fontsize=11, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    tick_values = [0, 1, 2, 3, 4, 5]
    tick_labels = ['1', '10', '100', '1K', '10K', '100K']
    valid_ticks = [(v, l) for v, l in zip(tick_values, tick_labels) if v <= vmax_log]
    cbar.set_ticks([v for v, l in valid_ticks])
    cbar.set_ticklabels([l for v, l in valid_ticks])
    cbar.set_label('Rank (log scale)', fontsize=9)

    for step_i in range(num_steps):
        for layer_i in range(num_layers):
            rank = rank_grid[step_i, layer_i]
            if rank <= 50:
                color = 'white' if log_ranks[layer_i, step_i] > vmax_log * 0.5 else 'black'
                ax.text(step_i, layer_i, str(rank), ha='center', va='center',
                        fontsize=5, color=color, fontweight='bold')

    return fig, ax
