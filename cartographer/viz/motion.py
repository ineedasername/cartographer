"""Rank displacement over time visualization.

Visualizes how token ranks shift across generation steps — the "motion"
of the vocabulary through the network. Renders rank displacement vectors
as animated or static flow fields, showing which tokens are rising and
falling at each layer/step.

NOT YET IMPLEMENTED — placeholder for the pattern James explored manually
in Untitled.png (rank delta heatmaps, displacement trajectories).

Planned API:
  - compute_displacement_field(scan, col_idx) -> (steps-1, layers, vocab) deltas
  - plot_displacement_heatmap(field, meta, token_ids) -> fig
  - plot_rank_trajectories(scan, meta, token_ids, col_idx) -> fig
"""
