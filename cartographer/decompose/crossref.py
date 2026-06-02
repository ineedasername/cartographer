"""Metadata-enriched rank cross-reference analysis.

Joins baseline rank orderings from a rank-displacement scan DB with token
metadata from tokens.db or lexicon.db, then provides grouping, aggregation,
filtering, and pivot table computation across arbitrary metadata dimensions.

Lifted from:
  - baseline_crossref.py L124-133 (load_baseline_dimensions)
  - baseline_crossref.py L136-159 (parse_range_spec)
  - baseline_crossref.py L181-190 (load_metadata_map)
  - baseline_crossref.py L193-215 (fetch_baseline_ranks)
  - baseline_crossref.py L218-235 (iter_hits)
  - baseline_crossref.py L238-242 (normalize_value)
  - baseline_crossref.py L253-280 (collect_hits_with_metadata)
  - baseline_crossref.py L283-318 (summarize_groups)
  - baseline_crossref.py L321-329 (sort_summary_rows)
  - baseline_crossref.py L358-379 (pivot_metric_aggregates)
  - baseline_crossref.py L382-432 (build_pivot — extracted from render_pivot_table)
  - baseline_crossref.py L729-744 (apply_filters)

Metrics:
  hits, hit_pct, unique_tokens, unique_pct, avg_rank, best_rank, worst_rank
"""

import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


# ─── Database loading ─────────────────────────────────────────────────────────

def load_baseline_dimensions(conn):
    """Extract available layer range, head range, and vocab_size from baselines table.

    Args:
        conn: sqlite3.Connection to a rank-displacement scan DB (row_factory=sqlite3.Row)

    Returns:
        (layers: sorted list[int], heads: sorted list[int], vocab_size: int)
    """
    rows = conn.execute(
        "SELECT layer, head, ranks FROM baselines ORDER BY layer, head"
    ).fetchall()
    if not rows:
        raise ValueError("No baseline rows found in scan DB.")
    layers = sorted({row["layer"] for row in rows})
    heads = sorted({row["head"] for row in rows})
    vocab_size = len(np.frombuffer(rows[0]["ranks"], dtype=np.int32))
    return layers, heads, vocab_size


def fetch_baseline_ranks(conn, layers, heads):
    """Load rank arrays for selected (layer, head) cells from baselines table.

    Args:
        conn: sqlite3.Connection to a rank-displacement scan DB
        layers: list of layer indices
        heads: list of head indices

    Returns:
        dict[(layer, head)] -> ndarray int32 of shape (vocab_size,)
    """
    layer_marks = ",".join("?" for _ in layers)
    head_marks = ",".join("?" for _ in heads)
    params = list(layers) + list(heads)
    rows = conn.execute(
        f"SELECT layer, head, ranks FROM baselines "
        f"WHERE layer IN ({layer_marks}) AND head IN ({head_marks}) "
        f"ORDER BY layer, head",
        params,
    ).fetchall()
    out = {}
    for row in rows:
        out[(row["layer"], row["head"])] = np.frombuffer(row["ranks"], dtype=np.int32).copy()
    return out


def load_metadata_map(source, path, requested_fields):
    """Load all rows from tokens or lexicon DB into memory as a lookup dict.

    Args:
        source: "tokens" or "lexicon"
        path: Path to the SQLite database
        requested_fields: list of column names to validate

    Returns:
        (dict[token_id -> sqlite3.Row], list of available columns)
    """
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    table = "tokens" if source == "tokens" else "words"

    cursor = conn.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cursor.fetchall()]

    missing = [f for f in requested_fields if f not in cols]
    if missing:
        conn.close()
        raise ValueError(f"Unknown {source} field(s): {', '.join(missing)}")

    rows = conn.execute(f"SELECT * FROM {table}").fetchall()
    conn.close()
    return {row["token_id"]: row for row in rows}, cols


# ─── Range parsing ────────────────────────────────────────────────────────────

def parse_range_spec(spec, allowed, label="value"):
    """Parse range specs like 'all', '0,3,8:12' into validated lists.

    Args:
        spec: string like "all", "0", "0,3", "0:5", "0,3,8:12"
        allowed: list of valid values
        label: name for error messages

    Returns:
        sorted list of selected values
    """
    if spec == "all":
        return list(allowed)

    chosen = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = part.split(":")
            if len(bits) != 2:
                raise ValueError(f"Invalid {label} range '{part}'")
            start, end = int(bits[0]), int(bits[1])
            step = 1 if end >= start else -1
            chosen.update(range(start, end + step, step))
        else:
            chosen.add(int(part))

    bad = sorted(v for v in chosen if v not in allowed)
    if bad:
        raise ValueError(f"{label.title()} values out of range: {bad}")
    return sorted(chosen)


# ─── Hit enumeration ──────────────────────────────────────────────────────────

def normalize_value(value):
    """Normalize a metadata value for display: None -> '<null>', empty -> '<empty>'."""
    if value is None:
        return "<null>"
    text = str(value).strip()
    return text if text else "<empty>"


def iter_hits(baseline_map, layers, heads, rank_min, rank_max):
    """Enumerate tokens within a rank band across selected (layer, head) cells.

    Yields dicts: {layer, head, token_id, rank} sorted by rank within each cell.

    Args:
        baseline_map: dict[(layer, head)] -> ndarray of ranks
        layers: list of layer indices to scan
        heads: list of head indices to scan
        rank_min: inclusive minimum rank (1-indexed)
        rank_max: inclusive maximum rank
    """
    for layer in layers:
        for head in heads:
            ranks = baseline_map[(layer, head)]
            mask = (ranks >= rank_min) & (ranks <= rank_max)
            token_ids = np.nonzero(mask)[0]
            if len(token_ids) == 0:
                continue
            local_ranks = ranks[token_ids]
            order = np.argsort(local_ranks, kind="stable")
            for token_id, rank in zip(token_ids[order], local_ranks[order], strict=True):
                yield {"layer": layer, "head": head, "token_id": int(token_id), "rank": int(rank)}


def collect_hits_with_metadata(conn, source, meta_path, layers, heads,
                                rank_min, rank_max, needed_fields, dedupe="none"):
    """Join baseline ranks with token metadata, optionally deduplicating.

    Args:
        conn: sqlite3.Connection to a rank-displacement scan DB
        source: "tokens" or "lexicon"
        meta_path: Path to metadata SQLite DB
        layers, heads: selection
        rank_min, rank_max: rank window
        needed_fields: metadata columns to include
        dedupe: "none" (every occurrence) or "token" (one per token_id)

    Returns:
        (list of hit dicts with metadata, list of available columns)
    """
    metadata_map, available_cols = load_metadata_map(source, meta_path, needed_fields)
    baseline_map = fetch_baseline_ranks(conn, layers, heads)

    rows = []
    seen_tokens = set()
    for hit in iter_hits(baseline_map, layers, heads, rank_min, rank_max):
        meta = metadata_map.get(hit["token_id"])
        if meta is None:
            continue
        if dedupe == "token" and hit["token_id"] in seen_tokens:
            continue
        seen_tokens.add(hit["token_id"])
        row = dict(hit)
        for field in needed_fields:
            row[field] = normalize_value(meta[field])
        rows.append(row)
    return rows, available_cols


# ─── Aggregation ──────────────────────────────────────────────────────────────

def summarize_groups(hits, group_fields):
    """Aggregate hit statistics by grouping across metadata dimensions.

    Args:
        hits: list of hit dicts (from collect_hits_with_metadata)
        group_fields: list of metadata column names to group by

    Returns:
        (rows, total_hits, total_unique) where rows is a list of dicts with:
        key, label, hits, hit_pct, unique_tokens, unique_pct, avg_rank, best_rank, worst_rank
    """
    counts = Counter()
    unique_tokens_by_group = defaultdict(set)
    ranks_by_group = defaultdict(list)
    total_unique_tokens = set()

    for hit in hits:
        key = tuple(hit[field] for field in group_fields) if group_fields else ("<all>",)
        counts[key] += 1
        unique_tokens_by_group[key].add(hit["token_id"])
        ranks_by_group[key].append(hit["rank"])
        total_unique_tokens.add(hit["token_id"])

    total_hits = len(hits)
    total_unique = len(total_unique_tokens)
    rows = []
    for key, hit_count in counts.items():
        unique_count = len(unique_tokens_by_group[key])
        avg_rank = sum(ranks_by_group[key]) / len(ranks_by_group[key])
        rows.append({
            "key": key,
            "label": " | ".join(str(x) for x in key),
            "hits": hit_count,
            "hit_pct": hit_count / total_hits if total_hits else 0.0,
            "unique_tokens": unique_count,
            "unique_pct": unique_count / total_unique if total_unique else 0.0,
            "avg_rank": avg_rank,
            "best_rank": min(ranks_by_group[key]),
            "worst_rank": max(ranks_by_group[key]),
        })
    return rows, total_hits, total_unique


def sort_summary_rows(rows, sort_key="hits"):
    """Sort summary rows in-place by the specified criterion.

    Args:
        rows: list from summarize_groups
        sort_key: "hits", "unique_tokens", "avg_rank", or "label"
    """
    if sort_key == "hits":
        rows.sort(key=lambda r: (-r["hits"], r["avg_rank"], r["label"]))
    elif sort_key == "unique_tokens":
        rows.sort(key=lambda r: (-r["unique_tokens"], r["avg_rank"], r["label"]))
    elif sort_key == "avg_rank":
        rows.sort(key=lambda r: (r["avg_rank"], -r["hits"], r["label"]))
    else:
        rows.sort(key=lambda r: r["label"])


# ─── Pivot ────────────────────────────────────────────────────────────────────

def pivot_metric_aggregates(subset):
    """Compute aggregate metrics for a subset of hits (one pivot cell).

    Returns dict with: hits, hit_pct (0, normalized later), unique_tokens,
    unique_pct (0, normalized later), avg_rank, best_rank, worst_rank
    """
    if not subset:
        return {
            "hits": 0, "hit_pct": 0.0,
            "unique_tokens": 0, "unique_pct": 0.0,
            "avg_rank": float("nan"),
            "best_rank": None, "worst_rank": None,
        }
    unique_tokens = {row["token_id"] for row in subset}
    ranks = [int(row["rank"]) for row in subset]
    return {
        "hits": len(subset),
        "hit_pct": 0.0,
        "unique_tokens": len(unique_tokens),
        "unique_pct": 0.0,
        "avg_rank": sum(ranks) / len(ranks),
        "best_rank": min(ranks),
        "worst_rank": max(ranks),
    }


def build_pivot(hits, row_fields, col_fields, metrics, limit=50):
    """Build a 2D pivot table from enriched hits.

    Computation only — no rendering. Returns structured data for display
    by any frontend (TUI, terminal, HTML).

    Args:
        hits: list of enriched hit dicts
        row_fields: metadata dimensions for rows
        col_fields: metadata dimensions for columns
        metrics: list of metric names to compute
        limit: max row groups

    Returns:
        dict with:
        - total_hits, total_unique
        - row_keys: sorted list of row key tuples
        - col_keys: sorted list of col key tuples
        - row_aggregates: dict[row_key] -> metric dict
        - cells: dict[(row_key, col_key)] -> metric dict
    """
    total_hits = len(hits)
    total_unique = len({row["token_id"] for row in hits})

    row_groups = defaultdict(list)
    col_keys = set()

    for hit in hits:
        row_key = tuple(hit[f] for f in row_fields) if row_fields else ("<all>",)
        col_key = tuple(hit[f] for f in col_fields) if col_fields else ("<all>",)
        row_groups[row_key].append(hit)
        col_keys.add(col_key)

    sorted_row_keys = sorted(
        row_groups.keys(),
        key=lambda k: (-len(row_groups[k]), k),
    )[:limit]
    sorted_col_keys = sorted(col_keys)

    row_aggregates = {}
    cells = {}

    for row_key in sorted_row_keys:
        row_subset = row_groups[row_key]
        agg = pivot_metric_aggregates(row_subset)
        agg["hit_pct"] = agg["hits"] / total_hits if total_hits else 0.0
        agg["unique_pct"] = agg["unique_tokens"] / total_unique if total_unique else 0.0
        row_aggregates[row_key] = agg

        for col_key in sorted_col_keys:
            cell_subset = [
                row for row in row_subset
                if (tuple(row[f] for f in col_fields) if col_fields else ("<all>",)) == col_key
            ]
            if not cell_subset:
                continue
            cell = pivot_metric_aggregates(cell_subset)
            cell["hit_pct"] = cell["hits"] / total_hits if total_hits else 0.0
            cell["unique_pct"] = cell["unique_tokens"] / total_unique if total_unique else 0.0
            cells[(row_key, col_key)] = cell

    return {
        "total_hits": total_hits,
        "total_unique": total_unique,
        "row_keys": sorted_row_keys,
        "col_keys": sorted_col_keys,
        "row_aggregates": row_aggregates,
        "cells": cells,
    }


# ─── Filtering ────────────────────────────────────────────────────────────────

def apply_filters(hits, filters):
    """Filter hit list by exact-match metadata constraints (AND logic).

    Args:
        hits: list of hit dicts
        filters: dict[field_name] -> set of allowed values

    Returns:
        filtered list
    """
    if not filters:
        return hits
    filtered = []
    for hit in hits:
        ok = True
        for field, values in filters.items():
            if hit.get(field) not in values:
                ok = False
                break
        if ok:
            filtered.append(hit)
    return filtered
