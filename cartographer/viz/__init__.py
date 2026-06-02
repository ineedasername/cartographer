"""Visualization modules for cartographer scan data.

Submodules:
  heatmap    — Token rank heatmaps (layers x steps)
  entropy    — Entropy landscape visualization
  river      — Token identity river (argmax color flow)
  terrain    — 3D topological proximity maps (plotly)
  multiverse — Multi-token sequence terrain comparison
  motion     — Rank displacement over time (placeholder)

These submodules require the optional [viz] extra (matplotlib, plotly):
    pip install cartographer-interp[viz]
If those are not installed, importing `cartographer.viz` raises a clear
ImportError telling you what to install, rather than failing obscurely.
"""

try:
    from cartographer.viz.heatmap import compute_rank_grid, make_heatmap
    from cartographer.viz.entropy import compute_entropy_grid, plot_entropy, plot_entropy_profiles
    from cartographer.viz.river import (
        token_to_color, build_river_image, build_multi_col_image, plot_river, make_animated_gif,
    )
    from cartographer.viz.terrain import (
        compute_proximity_surface, build_cell_labels, render_terrain_plotly, render_pair_terrain,
    )
    from cartographer.viz.multiverse import (
        tokenize_sequence, compute_token_proximity, compute_sequence_terrain,
        render_single_terrain, render_comparison,
    )
except ImportError as _e:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "cartographer.viz requires the optional visualization dependencies "
        "(matplotlib, plotly). Install them with:  pip install cartographer-interp[viz]"
    ) from _e
