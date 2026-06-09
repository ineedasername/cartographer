"""Map any token sequence's probability terrain through an MRI scan."""

import numpy as np


def tokenize_sequence(tokenizer, text):
    """Tokenize a sequence, return list of (token_id, decoded_str) pairs.

    Args:
        tokenizer: HF tokenizer
        text: str input text

    Returns:
        list of (int, str) tuples
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = []
    for tid in ids:
        decoded = tokenizer.decode(tid)
        tokens.append((tid, decoded))
    return tokens


def compute_token_proximity(scan, token_id, col_indices, num_layers):
    """Compute proximity of a single token across all steps and cells.

    Args:
        scan: ndarray (steps, layers, cols, vocab_size)
        token_id: int vocabulary index
        col_indices: list of int column indices
        num_layers: int number of layers

    Returns:
        ndarray (steps, cells) float64 where higher = closer to surface
    """
    num_steps = scan.shape[0]
    max_log = np.log10(scan.shape[3] + 1)
    num_cells = num_layers * len(col_indices)

    surface = np.zeros((num_steps, num_cells), dtype=np.float64)

    for step_i in range(num_steps):
        cell_idx = 0
        for c_idx in col_indices:
            for layer_i in range(num_layers):
                cell = scan[step_i, layer_i, c_idx, :].astype(np.float32)
                sorted_idx = np.argsort(cell)[::-1]
                rank = int(np.where(sorted_idx == token_id)[0][0]) + 1
                surface[step_i, cell_idx] = max_log - np.log10(rank + 1)
                cell_idx += 1

    return surface


def compute_sequence_terrain(scan, token_ids, col_indices, num_layers):
    """Compute joint proximity terrain for a multi-token sequence.

    For an N-token sequence, at each step S:
      token[0] proximity at step S, token[1] at S+1, etc.
    Joint = geometric mean of all N surfaces shifted by position.

    Args:
        scan: ndarray (steps, layers, cols, vocab_size)
        token_ids: list of int token IDs
        col_indices: list of int column indices
        num_layers: int number of layers

    Returns:
        ndarray (valid_steps, cells) or None if sequence too long
    """
    n_tokens = len(token_ids)
    num_steps = scan.shape[0]
    num_cells = num_layers * len(col_indices)
    valid_steps = num_steps - n_tokens + 1

    if valid_steps <= 0:
        return None

    surfaces = []
    for tid in token_ids:
        surfaces.append(compute_token_proximity(scan, tid, col_indices, num_layers))

    joint = np.ones((valid_steps, num_cells), dtype=np.float64)

    for seq_pos, surface in enumerate(surfaces):
        shifted = surface[seq_pos:seq_pos + valid_steps, :]
        joint *= np.maximum(shifted, 1e-10)

    joint = np.power(joint, 1.0 / n_tokens)

    return joint


def render_single_terrain(joint, meta, cell_labels, sequence_str, token_strs,
                          save_path=None):
    """Render a single sequence's terrain.

    Args:
        joint: ndarray (steps, cells) from compute_sequence_terrain()
        meta: dict scan metadata
        cell_labels: list of str cell labels
        sequence_str: str original input text
        token_strs: list of str decoded tokens
        save_path: str optional output path

    Returns:
        plotly Figure (if plotly available)
    """
    import plotly.graph_objects as go

    num_steps, num_cells = joint.shape
    steps_info = meta['steps']

    step_labels = []
    for i in range(num_steps):
        if i < len(steps_info):
            tok = steps_info[i]['token'].replace('\n', '\\n').replace(' ', '_')[:6]
            step_labels.append(f"S{i:02d} {tok}")
        else:
            step_labels.append(f"S{i:02d}")

    x = np.arange(num_cells)
    y = np.arange(num_steps)
    X, Y = np.meshgrid(x, y)

    colorscale = [
        [0.0, '#1a0533'], [0.15, '#2d1b69'], [0.30, '#3b528b'],
        [0.45, '#21918c'], [0.55, '#5ec962'], [0.70, '#fde725'],
        [0.85, '#f98e09'], [1.0, '#f0f921'],
    ]

    fig = go.Figure(data=[
        go.Surface(
            z=joint, x=X, y=Y,
            colorscale=colorscale,
            colorbar=dict(title='Joint<br>Proximity'),
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
            lighting=dict(ambient=0.4, diffuse=0.5, specular=0.3),
        )
    ])

    seq_display = ' -> '.join(f'"{t}"' for t in token_strs)

    fig.update_layout(
        title=dict(text=f'Multiverse Timeline: {seq_display}', font=dict(size=14)),
        scene=dict(
            xaxis=dict(
                title='Network Position',
                tickvals=list(range(0, num_cells, max(1, num_cells // 10))),
                ticktext=[cell_labels[i] if i < len(cell_labels) else ''
                          for i in range(0, num_cells, max(1, num_cells // 10))],
            ),
            yaxis=dict(title='Generation Step', autorange='reversed'),
            zaxis=dict(title='Joint Proximity'),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
            aspectratio=dict(x=2, y=1.5, z=0.7),
        ),
        width=1200, height=800,
    )

    if save_path:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, scale=2)

    return fig


def render_comparison(terrains, meta, cell_labels, save_path=None):
    """Render multiple sequence terrains overlaid for comparison.

    Args:
        terrains: list of (label, ndarray) tuples
        meta: dict scan metadata
        cell_labels: list of str cell labels
        save_path: str optional output path

    Returns:
        plotly Figure
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    colors = [
        ([[0, '#2d1b69'], [0.5, '#e34a33'], [1, '#fee8c8']], 'Reds'),
        ([[0, '#1b3a4b'], [0.5, '#3b8ea5'], [1, '#d4f1f9']], 'Blues'),
        ([[0, '#1a3c1a'], [0.5, '#5ec962'], [1, '#e5f5e0']], 'Greens'),
        ([[0, '#4a1a4a'], [0.5, '#9c6ade'], [1, '#f0e6ff']], 'Purples'),
    ]

    for i, (label, terrain) in enumerate(terrains):
        num_steps, num_cells = terrain.shape
        x = np.arange(num_cells)
        y = np.arange(num_steps)
        X, Y = np.meshgrid(x, y)

        cs, name = colors[i % len(colors)]
        fig.add_trace(go.Surface(
            z=terrain, x=X, y=Y,
            opacity=0.5 if len(terrains) > 1 else 0.9,
            colorscale=cs,
            showscale=(i == 0),
            name=label,
        ))

    all_labels = ' vs '.join(label for label, _ in terrains)

    fig.update_layout(
        title=dict(text=f'Multiverse Comparison: {all_labels}', font=dict(size=14)),
        scene=dict(
            xaxis=dict(title='Network Position'),
            yaxis=dict(title='Generation Step', autorange='reversed'),
            zaxis=dict(title='Joint Proximity'),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
            aspectratio=dict(x=2, y=1.5, z=0.7),
        ),
        width=1200, height=800,
    )

    if save_path:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, scale=2)

    return fig
