"""Database query utilities for rank-displacement scan cross-model comparison.

Provides clean access to rank-displacement scan DB data for analysis and visualization.
Schema: scans(scan_id, prompt, n_input_tokens, n_gen_tokens, model, created_at),
        positions(scan_id, position, phase, token_text, token_id, layer, head,
                  spearman, net_motion, top1_id, top1_prob),
        baselines(layer, head, ranks BLOB)
"""

import sqlite3
from collections import defaultdict

import numpy as np


def _connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def list_scans(db_path):
    """List all scans with metadata.

    Returns:
        list of dicts: {scan_id, prompt, model, n_input_tokens, n_gen_tokens,
                        created_at, n_positions}
    """
    conn = _connect(db_path)
    scans = conn.execute("""
        SELECT s.*, COUNT(DISTINCT p.position) as n_positions
        FROM scans s
        LEFT JOIN positions p ON s.scan_id = p.scan_id
        GROUP BY s.scan_id
        ORDER BY s.scan_id
    """).fetchall()
    conn.close()
    return [dict(s) for s in scans]


def get_positions(db_path, scan_id, phase=None):
    """Get all position rows for a scan.

    Args:
        db_path: path to rank-displacement scan DB
        scan_id: which scan
        phase: optional filter — 'input' or 'gen'

    Returns:
        list of dicts with all position columns
    """
    conn = _connect(db_path)
    if phase:
        rows = conn.execute(
            "SELECT * FROM positions WHERE scan_id = ? AND phase = ? ORDER BY position, layer, head",
            (scan_id, phase)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM positions WHERE scan_id = ? ORDER BY position, layer, head",
            (scan_id,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_spearman_grid(db_path, scan_id, head=None):
    """Build a (positions, layers) Spearman matrix for heatmap display.

    If head is specified, returns that head only. Otherwise averages across heads.

    Returns:
        (grid, position_labels) where:
        - grid: ndarray (n_positions, n_layers) of Spearman rho values
        - position_labels: list of (position_idx, phase, token_text) tuples
    """
    conn = _connect(db_path)

    if head is not None:
        rows = conn.execute(
            "SELECT position, layer, phase, token_text, spearman "
            "FROM positions WHERE scan_id = ? AND head = ? "
            "ORDER BY position, layer",
            (scan_id, head)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT position, layer, phase, token_text, AVG(spearman) as spearman "
            "FROM positions WHERE scan_id = ? "
            "GROUP BY position, layer "
            "ORDER BY position, layer",
            (scan_id,)
        ).fetchall()

    conn.close()

    if not rows:
        return np.array([]), []

    # Build grid
    positions = sorted(set(r['position'] for r in rows))
    layers = sorted(set(r['layer'] for r in rows))
    pos_map = {p: i for i, p in enumerate(positions)}
    lay_map = {l: i for i, l in enumerate(layers)}

    grid = np.full((len(positions), len(layers)), np.nan)
    position_labels = []
    seen_pos = set()

    for r in rows:
        pi = pos_map[r['position']]
        li = lay_map[r['layer']]
        grid[pi, li] = r['spearman']
        if r['position'] not in seen_pos:
            position_labels.append((r['position'], r['phase'], r['token_text']))
            seen_pos.add(r['position'])

    return grid, position_labels


def get_motion_grid(db_path, scan_id, head=None):
    """Build a (positions, layers) net_motion matrix.

    Same structure as get_spearman_grid but for net_motion values.
    """
    conn = _connect(db_path)

    if head is not None:
        rows = conn.execute(
            "SELECT position, layer, phase, token_text, net_motion "
            "FROM positions WHERE scan_id = ? AND head = ? "
            "ORDER BY position, layer",
            (scan_id, head)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT position, layer, phase, token_text, AVG(net_motion) as net_motion "
            "FROM positions WHERE scan_id = ? "
            "GROUP BY position, layer "
            "ORDER BY position, layer",
            (scan_id,)
        ).fetchall()

    conn.close()

    if not rows:
        return np.array([]), []

    positions = sorted(set(r['position'] for r in rows))
    layers = sorted(set(r['layer'] for r in rows))
    pos_map = {p: i for i, p in enumerate(positions)}
    lay_map = {l: i for i, l in enumerate(layers)}

    grid = np.full((len(positions), len(layers)), np.nan)
    position_labels = []
    seen_pos = set()

    for r in rows:
        grid[pos_map[r['position']], lay_map[r['layer']]] = r['net_motion']
        if r['position'] not in seen_pos:
            position_labels.append((r['position'], r['phase'], r['token_text']))
            seen_pos.add(r['position'])

    return grid, position_labels


def get_diagonal_propagation(db_path, scan_id, head=None):
    """Extract diagonal Spearman deltas: H[n]L[m] -> H[n+1]L[m+1].

    For each position, computes Spearman(layer+1) - Spearman(layer) along the
    diagonal of the layer×head grid — the change in rank displacement from one
    layer to the next.

    Returns:
        dict with:
        - deltas: ndarray (n_positions, n_layers-1) — Spearman delta per position
        - position_labels: list of (position_idx, phase, token_text)
        - layers: list of layer indices
    """
    grid, labels = get_spearman_grid(db_path, scan_id, head=head)
    if grid.size == 0:
        return {"deltas": np.array([]), "position_labels": [], "layers": []}

    n_pos, n_layers = grid.shape
    deltas = np.diff(grid, axis=1)  # (n_pos, n_layers-1)

    return {
        "deltas": deltas,
        "position_labels": labels,
        "layers": list(range(n_layers - 1)),
    }
