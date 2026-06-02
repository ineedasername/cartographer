"""Token identity river flowing through layers across generation.

Lifted from: viz_river.py
  - token_to_color() (L44-56)
  - build_river_image() (L74-86)
  - build_multi_col_image() (L89-110)
  - plot_river() (L113-148)
  - make_animated_gif() (L151-201)
"""

import hashlib

import numpy as np


def token_to_color(token_id, prob, vocab_size=262144):
    """Map a token ID to an HSV color, with brightness modulated by probability.

    Args:
        token_id: int vocabulary index
        prob: float probability of this token
        vocab_size: int (unused, kept for API compat)

    Returns:
        (r, g, b) float tuple
    """
    import matplotlib.colors as mcolors

    hue_bytes = hashlib.md5(str(token_id).encode()).digest()
    hue = int.from_bytes(hue_bytes[:2], 'big') / 65535.0
    saturation = min(1.0, prob * 10)
    value = 0.2 + 0.8 * min(1.0, prob * 5)

    return mcolors.hsv_to_rgb([hue, saturation, value])


def build_river_image(scan, col_idx, vocab_size):
    """Build the full river: (num_layers, num_steps, 3) RGB image.

    Args:
        scan: ndarray (steps, layers, cols, vocab_size)
        col_idx: int column index
        vocab_size: int vocabulary size

    Returns:
        ndarray (layers, steps, 3) float RGB
    """
    num_steps, num_layers = scan.shape[0], scan.shape[1]
    image = np.zeros((num_layers, num_steps, 3))

    for step_i in range(num_steps):
        for layer_i in range(num_layers):
            cell = scan[step_i, layer_i, col_idx, :].astype(np.float32)
            argmax_id = int(np.argmax(cell))
            prob = float(cell[argmax_id])
            image[layer_i, step_i] = token_to_color(argmax_id, prob, vocab_size)

    return image


def build_multi_col_image(scan, col_indices, col_labels, vocab_size):
    """Build river for multiple columns stacked vertically with gaps.

    Args:
        scan: ndarray (steps, layers, cols, vocab_size)
        col_indices: list of int column indices
        col_labels: list of str column names
        vocab_size: int vocabulary size

    Returns:
        (image, y_labels) where image is (total_height, steps, 3) and
        y_labels is list of (y_pos, label) tuples
    """
    num_steps, num_layers = scan.shape[0], scan.shape[1]
    n_cols = len(col_indices)
    gap = 1

    total_height = n_cols * num_layers + (n_cols - 1) * gap
    image = np.ones((total_height, num_steps, 3)) * 0.15

    y_labels = []
    for ci, col_idx in enumerate(col_indices):
        y_offset = ci * (num_layers + gap)
        for step_i in range(num_steps):
            for layer_i in range(num_layers):
                cell = scan[step_i, layer_i, col_idx, :].astype(np.float32)
                argmax_id = int(np.argmax(cell))
                prob = float(cell[argmax_id])
                image[y_offset + layer_i, step_i] = token_to_color(argmax_id, prob, vocab_size)

        y_labels.append((y_offset + num_layers // 2, col_labels[ci]))

    return image, y_labels


def plot_river(image, meta, col_name, y_labels=None, num_layers=None, ax=None):
    """Render the river image.

    Args:
        image: ndarray (height, steps, 3) RGB from build_river_image/build_multi_col_image
        meta: dict scan metadata
        col_name: str column name for title
        y_labels: optional list of (y_pos, label) for multi-column
        num_layers: optional int for single-column layer labels
        ax: matplotlib Axes (created if None)

    Returns:
        (fig, ax)
    """
    import matplotlib.pyplot as plt

    steps_info = meta['steps']
    num_steps = image.shape[1]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(max(14, num_steps * 0.45), 8))
    else:
        fig = ax.figure

    ax.imshow(image, aspect='auto', interpolation='nearest', origin='lower')

    step_labels = []
    for s in steps_info[:num_steps]:
        tok = s['token'].replace('\n', '\\n').replace(' ', '_')
        if len(tok) > 8:
            tok = tok[:7] + '.'
        step_labels.append(tok)

    ax.set_xticks(range(num_steps))
    ax.set_xticklabels(step_labels, rotation=60, ha='right', fontsize=7, fontfamily='monospace')
    ax.set_xlabel('Generation Step', fontsize=9)

    if y_labels:
        ax.set_yticks([y for y, l in y_labels])
        ax.set_yticklabels([l for y, l in y_labels], fontsize=8, fontweight='bold')
    elif num_layers:
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels([f'L{i:02d}' for i in range(num_layers)], fontsize=7, fontfamily='monospace')

    ax.set_ylabel('Layer', fontsize=9)
    ax.set_title(f'Token Identity River  [{col_name}]', fontsize=11, fontweight='bold')

    return fig, ax


def make_animated_gif(scan, meta, col_idx, col_name, vocab_size, output_path, dpi=100):
    """Create an animated GIF building up the river step by step.

    Args:
        scan: ndarray (steps, layers, cols, vocab_size)
        meta: dict scan metadata
        col_idx: int column index
        col_name: str column name for title
        vocab_size: int vocabulary size
        output_path: str path to write .gif
        dpi: int resolution
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    num_steps, num_layers = scan.shape[0], scan.shape[1]
    full_image = build_river_image(scan, col_idx, vocab_size)

    fig, ax = plt.subplots(1, 1, figsize=(max(12, num_steps * 0.4), 7))

    display = np.zeros_like(full_image)
    display[:, :, :] = 0.05
    im = ax.imshow(display, aspect='auto', interpolation='nearest', origin='lower')

    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f'L{i:02d}' for i in range(num_layers)], fontsize=6, fontfamily='monospace')
    ax.set_ylabel('Layer', fontsize=9)
    ax.set_title(f'Token Identity River  [{col_name}]', fontsize=11, fontweight='bold')

    title_text = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                          ha='center', fontsize=9, fontstyle='italic', color='gray')

    def update(frame):
        step_i = frame
        display[:, step_i, :] = full_image[:, step_i, :]
        im.set_data(display)

        steps_info = meta['steps']
        labels = []
        for s in steps_info[:step_i + 1]:
            tok = s['token'].replace('\n', '\\n').replace(' ', '_')
            if len(tok) > 6:
                tok = tok[:5] + '.'
            labels.append(tok)
        labels += [''] * (num_steps - len(labels))
        ax.set_xticks(range(num_steps))
        ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=6, fontfamily='monospace')

        output_so_far = ''.join(s['token'] for s in steps_info[:step_i + 1])
        if len(output_so_far) > 60:
            output_so_far = output_so_far[:57] + '...'
        title_text.set_text(f'"{output_so_far}"')

        return [im, title_text]

    anim = FuncAnimation(fig, update, frames=num_steps, interval=300, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=3), dpi=dpi)
    plt.close()
