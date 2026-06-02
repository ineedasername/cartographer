"""3D topological map of token proximity across the network.

Lifted from: viz_terrain.py
  - compute_proximity_surface() (L45-65)
  - render_terrain_plotly() (L103-203)
  - render_pair_terrain() (L206-282)
"""

import numpy as np


def compute_proximity_surface(scan, token_id, col_indices, col_labels):
    """Compute proximity (inverted log-rank) at every step x cell.

    Args:
        scan: ndarray (steps, layers, cols, vocab_size)
        token_id: int vocabulary index
        col_indices: list of int column indices
        col_labels: list of str column names (unused, kept for API compat)

    Returns:
        ndarray (steps, cells) float64 where high = close to surface
    """
    num_steps, num_layers = scan.shape[0], scan.shape[1]
    num_cells = num_layers * len(col_indices)
    max_log = np.log10(scan.shape[3] + 1)

    surface = np.zeros((num_steps, num_cells), dtype=np.float64)

    for step_i in range(num_steps):
        cell_idx = 0
        for c_pos, c_idx in enumerate(col_indices):
            for layer_i in range(num_layers):
                cell = scan[step_i, layer_i, c_idx, :].astype(np.float32)
                sorted_idx = np.argsort(cell)[::-1]
                rank = int(np.where(sorted_idx == token_id)[0][0]) + 1
                surface[step_i, cell_idx] = max_log - np.log10(rank + 1)
                cell_idx += 1

    return surface


def build_cell_labels(col_labels_used, num_layers):
    """Build readable labels for cell positions (Layer/Col pairs)."""
    labels = []
    for col_name in col_labels_used:
        for layer_i in range(num_layers):
            labels.append(f"L{layer_i:02d}/{col_name}")
    return labels


def render_terrain_plotly(surface, meta, cell_labels, token_name, title_suffix="",
                          save_path=None, colorscale=None):
    """Render 3D terrain using plotly.

    Args:
        surface: ndarray (steps, cells) from compute_proximity_surface()
        meta: dict scan metadata
        cell_labels: list of str cell labels
        token_name: str display name
        title_suffix: str appended to title
        save_path: str optional output path (.html or image)
        colorscale: list optional plotly colorscale

    Returns:
        plotly Figure
    """
    import plotly.graph_objects as go

    num_steps, num_cells = surface.shape
    steps_info = meta['steps']

    step_labels = []
    for s in steps_info[:num_steps]:
        tok = s['token'].replace('\n', '\\n').replace(' ', '_')
        if len(tok) > 8: tok = tok[:7] + '.'
        step_labels.append(f"S{s['step']:02d} {tok}")

    x = np.arange(num_cells)
    y = np.arange(num_steps)
    X, Y = np.meshgrid(x, y)

    if colorscale is None:
        colorscale = [
            [0.0, '#1a0533'], [0.15, '#2d1b69'], [0.30, '#3b528b'],
            [0.45, '#21918c'], [0.55, '#5ec962'], [0.70, '#fde725'],
            [0.85, '#f98e09'], [1.0, '#f0f921'],
        ]

    fig = go.Figure(data=[
        go.Surface(
            z=surface, x=X, y=Y,
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text='Proximity<br>(log scale)', font=dict(size=11)),
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=['buried', 'r100K', 'r10K', 'r1K', 'r100', 'r1'],
            ),
            hovertemplate=(
                'Cell: %{customdata}<br>'
                'Step: %{y}<br>'
                'Proximity: %{z:.2f}<br>'
                '<extra></extra>'
            ),
            customdata=np.array([[cell_labels[xi] for xi in range(num_cells)]
                                  for _ in range(num_steps)]),
            lighting=dict(ambient=0.4, diffuse=0.5, specular=0.3, roughness=0.7),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor='white',
                       project_z=True, start=0, end=surface.max(), size=0.5)
            ),
        )
    ])

    fig.update_layout(
        title=dict(
            text=f'Token Proximity Terrain: "{token_name}" {title_suffix}',
            font=dict(size=14),
        ),
        scene=dict(
            xaxis=dict(
                title='Network Position (Layer/Head)',
                tickvals=list(range(0, num_cells, 26)),
                ticktext=[cell_labels[i].split('/')[1] if i < len(cell_labels) else ''
                          for i in range(0, num_cells, 26)],
            ),
            yaxis=dict(
                title='Generation Step',
                tickvals=list(range(num_steps)),
                ticktext=step_labels,
                autorange='reversed',
            ),
            zaxis=dict(title='Proximity (higher = closer to surface)'),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0), up=dict(x=0, y=0, z=1)),
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


def render_pair_terrain(surface_a, surface_b, meta, cell_labels,
                        token_a_name, token_b_name, save_path=None):
    """Render overlaid terrains for a token pair (A at step N, B at step N+1).

    Args:
        surface_a, surface_b: ndarray (steps, cells) from compute_proximity_surface()
        meta: dict scan metadata
        cell_labels: list of str cell labels
        token_a_name, token_b_name: str display names
        save_path: str optional output path

    Returns:
        plotly Figure
    """
    import plotly.graph_objects as go

    num_steps = min(surface_a.shape[0], surface_b.shape[0]) - 1
    num_cells = surface_a.shape[1]

    sa = surface_a[:num_steps, :]
    sb = surface_b[1:num_steps + 1, :]
    joint = np.sqrt(sa * sb)

    x = np.arange(num_cells)
    y = np.arange(num_steps)
    X, Y = np.meshgrid(x, y)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=sa, x=X, y=Y, opacity=0.4,
        colorscale='Blues', showscale=False,
        name=f'"{token_a_name}" at step N',
    ))

    fig.add_trace(go.Surface(
        z=sb, x=X, y=Y, opacity=0.4,
        colorscale='Reds', showscale=False,
        name=f'"{token_b_name}" at step N+1',
    ))

    fig.add_trace(go.Surface(
        z=joint, x=X, y=Y, opacity=0.8,
        colorscale='YlOrRd',
        colorbar=dict(title='Joint<br>Proximity'),
        name='Joint (both close)',
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
    ))

    fig.update_layout(
        title=dict(
            text=f'Sequence Terrain: "{token_a_name}" -> "{token_b_name}"',
            font=dict(size=14),
        ),
        scene=dict(
            xaxis=dict(title='Network Position'),
            yaxis=dict(title='Generation Step', autorange='reversed'),
            zaxis=dict(title='Proximity'),
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
