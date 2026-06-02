"""Token motion analysis — characterize WHICH tokens move and WHAT KIND they are.

Given two consecutive rank orderings from an MRI scan, identifies the tokens
that moved most (up or down), then breaks them down by lexical properties
(script, language, category) using the token metadata database.

The rank-displacement scan tells you HOW MUCH changed (Spearman ρ). This module
tells you WHO changed, and what kind of thing they are.

All functions are pure computation: numpy arrays in, structured dicts out.
No CLI, no display, no model loading.
"""

import sqlite3
from collections import Counter, defaultdict

import numpy as np


# ─── Rank computation ─────────────────────────────────────────────────────────

def compute_rank_arrays(scan, col_idx):
    """Convert softmax distributions to rank arrays for one column across all steps/layers.

    Args:
        scan: ndarray (steps, layers, cols, vocab_size) float16 — MRI scan
        col_idx: which column (0-3 for H0-H3, 4 for Layer)

    Returns:
        ndarray (steps, layers, vocab_size) int32 — rank at each position (0 = highest prob)
    """
    num_steps, num_layers, num_cols, vocab = scan.shape
    ranks = np.zeros((num_steps, num_layers, vocab), dtype=np.int32)

    for s in range(num_steps):
        for l in range(num_layers):
            probs = scan[s, l, col_idx, :].astype(np.float32)
            ranks[s, l, :] = np.argsort(np.argsort(-probs)).astype(np.int32)

    return ranks


def compute_rank_deltas(rank_arrays):
    """Compute per-token rank change between consecutive steps.

    Args:
        rank_arrays: ndarray (steps, layers, vocab_size) int32

    Returns:
        ndarray (steps-1, layers, vocab_size) int32
        Delta < 0 = token rose toward surface. Delta > 0 = token sank.
    """
    return np.diff(rank_arrays, axis=0)


# ─── Metadata loading ─────────────────────────────────────────────────────────

def load_token_metadata(db_path, vocab_size,
                         fields=('script', 'language', 'category', 'is_ascii')):
    """Load token metadata into numpy arrays for O(1) lookup.

    Strings are encoded as integer codes with lookup dicts.

    Args:
        db_path: path to tokens.db
        vocab_size: number of tokens (to size arrays)
        fields: tuple of field names to load

    Returns:
        dict with:
        - For string fields: {field: ndarray int32, field + '_labels': dict[int->str]}
        - For int fields: {field: ndarray int32}
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Determine which fields are string vs int
    col_info = {c[1]: c[2] for c in conn.execute("PRAGMA table_info(tokens)").fetchall()}

    result = {}
    for field in fields:
        if field not in col_info:
            continue

        rows = conn.execute(f"SELECT token_id, {field} FROM tokens").fetchall()

        if col_info[field] == 'INTEGER':
            arr = np.zeros(vocab_size, dtype=np.int32)
            for r in rows:
                tid = r['token_id']
                if 0 <= tid < vocab_size:
                    arr[tid] = int(r[field]) if r[field] is not None else 0
            result[field] = arr
        else:
            # String field — encode as int codes
            values = set()
            for r in rows:
                val = r[field] if r[field] else '<null>'
                values.add(val)
            label_to_code = {v: i for i, v in enumerate(sorted(values))}
            code_to_label = {i: v for v, i in label_to_code.items()}

            arr = np.zeros(vocab_size, dtype=np.int32)
            for r in rows:
                tid = r['token_id']
                val = r[field] if r[field] else '<null>'
                if 0 <= tid < vocab_size:
                    arr[tid] = label_to_code[val]

            result[field] = arr
            result[f'{field}_labels'] = code_to_label

    conn.close()
    return result


# ─── Per-cell analysis ────────────────────────────────────────────────────────

def get_top_movers(rank_deltas, step, layer, top_k=500):
    """Identify tokens with largest rank changes at one cell.

    Args:
        rank_deltas: ndarray (steps-1, layers, vocab) int32
        step: delta step index (0 = between original step 0 and step 1)
        layer: layer index
        top_k: how many top movers to return

    Returns:
        (top_risers, top_sinkers) where each is ndarray of (token_id, delta) pairs
        Risers have negative delta (moved toward surface).
        Sinkers have positive delta (moved away from surface).
    """
    deltas = rank_deltas[step, layer, :]

    # Top risers: most negative deltas
    riser_idx = np.argsort(deltas)[:top_k]
    risers = np.column_stack([riser_idx, deltas[riser_idx]])

    # Top sinkers: most positive deltas
    sinker_idx = np.argsort(deltas)[::-1][:top_k]
    sinkers = np.column_stack([sinker_idx, deltas[sinker_idx]])

    return risers, sinkers


def compute_displacement_stats(rank_deltas, step, layer):
    """Aggregate displacement metrics for one cell.

    Returns:
        dict with total_displacement, mean_displacement, std_displacement,
        gini_displacement, n_moved (tokens with any delta != 0)
    """
    deltas = rank_deltas[step, layer, :]
    abs_deltas = np.abs(deltas).astype(np.float64)

    total = float(abs_deltas.sum())
    mean = float(abs_deltas.mean())
    std = float(abs_deltas.std())
    n_moved = int(np.count_nonzero(deltas))

    # Gini coefficient
    sorted_d = np.sort(abs_deltas)
    n = len(sorted_d)
    if total > 0:
        index = np.arange(1, n + 1)
        gini = float((2 * np.sum(index * sorted_d) / (n * total)) - (n + 1) / n)
    else:
        gini = 0.0

    return {
        'total_displacement': total,
        'mean_displacement': mean,
        'std_displacement': std,
        'gini_displacement': gini,
        'n_moved': n_moved,
    }


def characterize_movers(token_ids, deltas, metadata):
    """Break down a set of movers by lexical properties.

    Args:
        token_ids: ndarray of token IDs
        deltas: ndarray of rank deltas (parallel to token_ids)
        metadata: dict from load_token_metadata

    Returns:
        dict with per-dimension breakdowns:
        {dimension: {label: {count, total_delta, mean_delta, fraction}}}
    """
    result = {}

    for field in metadata:
        if field.endswith('_labels'):
            continue

        codes = metadata[field]
        labels = metadata.get(f'{field}_labels', None)

        counts = Counter()
        delta_sums = defaultdict(float)

        for tid, d in zip(token_ids, deltas):
            if 0 <= tid < len(codes):
                code = int(codes[tid])
                label = labels[code] if labels else str(code)
                counts[label] += 1
                delta_sums[label] += float(d)

        total = len(token_ids)
        breakdown = {}
        for label in sorted(counts, key=lambda x: -counts[x]):
            c = counts[label]
            breakdown[label] = {
                'count': c,
                'fraction': c / total if total > 0 else 0.0,
                'total_delta': delta_sums[label],
                'mean_delta': delta_sums[label] / c if c > 0 else 0.0,
            }

        result[field] = breakdown

    return result


# ─── Head profiles ────────────────────────────────────────────────────────────

def build_head_profile(rank_deltas, col_idx, metadata, top_k=500):
    """Aggregate traffic character across all steps for one head.

    Args:
        rank_deltas: ndarray (steps-1, layers, vocab) — for one column
        col_idx: not used (rank_deltas should already be for one column)
        metadata: dict from load_token_metadata
        top_k: top movers per cell to characterize

    Returns:
        dict with:
        - mean_magnitude: average total_displacement
        - mean_gini: average gini (uniformity of motion)
        - script_distribution: {script: mean_fraction across steps}
        - language_distribution: {lang: mean_fraction}
        - ascii_ratio: fraction of top movers that are ASCII
        - per_layer: list of per-layer aggregate profiles
    """
    num_steps, num_layers, vocab = rank_deltas.shape

    magnitudes = []
    ginis = []
    script_accum = defaultdict(list)
    lang_accum = defaultdict(list)
    ascii_counts = []
    total_counts = []

    per_layer = []

    for layer in range(num_layers):
        layer_magnitudes = []
        layer_ginis = []
        layer_script = defaultdict(int)
        layer_ascii = 0
        layer_total = 0

        for step in range(num_steps):
            stats = compute_displacement_stats(rank_deltas, step, layer)
            magnitudes.append(stats['total_displacement'])
            ginis.append(stats['gini_displacement'])
            layer_magnitudes.append(stats['total_displacement'])
            layer_ginis.append(stats['gini_displacement'])

            risers, sinkers = get_top_movers(rank_deltas, step, layer, top_k)
            all_movers = np.concatenate([risers[:, 0], sinkers[:, 0]]).astype(np.int32)
            all_deltas = np.concatenate([risers[:, 1], sinkers[:, 1]])

            char = characterize_movers(all_movers, all_deltas, metadata)

            if 'script' in char:
                for label, info in char['script'].items():
                    script_accum[label].append(info['fraction'])
                    layer_script[label] += info['count']

            if 'language' in char:
                for label, info in char['language'].items():
                    lang_accum[label].append(info['fraction'])

            if 'is_ascii' in char:
                for label, info in char['is_ascii'].items():
                    if str(label) == '1':
                        ascii_counts.append(info['count'])
                        layer_ascii += info['count']
                total_counts.append(len(all_movers))
                layer_total += len(all_movers)

        per_layer.append({
            'layer': layer,
            'mean_magnitude': np.mean(layer_magnitudes) if layer_magnitudes else 0,
            'mean_gini': np.mean(layer_ginis) if layer_ginis else 0,
            'ascii_ratio': layer_ascii / layer_total if layer_total > 0 else 0,
            'top_scripts': dict(sorted(layer_script.items(), key=lambda x: -x[1])[:5]),
        })

    # Aggregate
    script_dist = {s: np.mean(fracs) for s, fracs in script_accum.items()}
    lang_dist = {l: np.mean(fracs) for l, fracs in lang_accum.items()}
    total_ascii = sum(ascii_counts)
    total_all = sum(total_counts)

    return {
        'mean_magnitude': np.mean(magnitudes) if magnitudes else 0,
        'mean_gini': np.mean(ginis) if ginis else 0,
        'script_distribution': dict(sorted(script_dist.items(), key=lambda x: -x[1])),
        'language_distribution': dict(sorted(lang_dist.items(), key=lambda x: -x[1])),
        'ascii_ratio': total_ascii / total_all if total_all > 0 else 0,
        'per_layer': per_layer,
    }


def build_all_head_profiles(scan, metadata, col_labels=None, top_k=500):
    """Build traffic profiles for all columns (H0-H3 + Layer).

    Args:
        scan: ndarray (steps, layers, cols, vocab) — full MRI scan
        metadata: dict from load_token_metadata
        col_labels: list of column names (default: H0-H3 + Layer)
        top_k: top movers per cell

    Returns:
        dict[col_label -> head_profile]
    """
    num_steps, num_layers, num_cols, vocab = scan.shape
    if col_labels is None:
        col_labels = [f'H{i}' for i in range(num_cols - 1)] + ['Layer']

    profiles = {}
    for col_idx, label in enumerate(col_labels):
        rank_arrays = compute_rank_arrays(scan, col_idx)
        deltas = compute_rank_deltas(rank_arrays)
        profiles[label] = build_head_profile(deltas, col_idx, metadata, top_k)

    return profiles


# ─── Cross-head comparison ────────────────────────────────────────────────────

def head_specialization(head_profiles, dimension='script'):
    """Pairwise Jensen-Shannon divergence between heads' distribution profiles.

    High JSD = heads specialize in different token types.

    Args:
        head_profiles: dict[label -> profile] from build_all_head_profiles
        dimension: 'script' or 'language'

    Returns:
        (labels, divergence_matrix) where matrix[i][j] = JSD between head i and j
    """
    dist_key = f'{dimension}_distribution'
    labels = list(head_profiles.keys())
    n = len(labels)

    # Collect all categories across all heads
    all_cats = set()
    for profile in head_profiles.values():
        all_cats.update(profile.get(dist_key, {}).keys())
    all_cats = sorted(all_cats)

    # Build probability vectors
    vectors = []
    for label in labels:
        dist = head_profiles[label].get(dist_key, {})
        vec = np.array([dist.get(c, 0.0) for c in all_cats], dtype=np.float64)
        total = vec.sum()
        if total > 0:
            vec /= total
        else:
            vec = np.ones_like(vec) / len(vec)
        # Add small epsilon to avoid log(0)
        vec = vec + 1e-10
        vec /= vec.sum()
        vectors.append(vec)

    # JSD matrix
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m = 0.5 * (vectors[i] + vectors[j])
            kl_im = np.sum(vectors[i] * np.log2(vectors[i] / m))
            kl_jm = np.sum(vectors[j] * np.log2(vectors[j] / m))
            jsd = 0.5 * (kl_im + kl_jm)
            matrix[i][j] = jsd
            matrix[j][i] = jsd

    return labels, matrix


# ─── Timeline ─────────────────────────────────────────────────────────────────

def motion_timeline(scan, col_idx, metadata, top_k=500):
    """Per-step motion magnitude + lexical composition for all layers.

    Args:
        scan: full MRI scan array
        col_idx: column to analyze
        metadata: dict from load_token_metadata
        top_k: top movers per cell

    Returns:
        list of per-step dicts, each with:
        - step: step index
        - per_layer: list of {layer, displacement_stats, script_breakdown}
    """
    rank_arrays = compute_rank_arrays(scan, col_idx)
    deltas = compute_rank_deltas(rank_arrays)
    num_steps, num_layers, vocab = deltas.shape

    timeline = []
    for step in range(num_steps):
        step_data = {'step': step, 'per_layer': []}
        for layer in range(num_layers):
            stats = compute_displacement_stats(deltas, step, layer)
            risers, sinkers = get_top_movers(deltas, step, layer, top_k)
            all_movers = np.concatenate([risers[:, 0], sinkers[:, 0]]).astype(np.int32)
            all_deltas_arr = np.concatenate([risers[:, 1], sinkers[:, 1]])
            char = characterize_movers(all_movers, all_deltas_arr, metadata)
            step_data['per_layer'].append({
                'layer': layer,
                **stats,
                'breakdowns': char,
            })
        timeline.append(step_data)

    return timeline


# ─── Per-token layer profiles (vibration analysis) ────────────────────────────

def compute_token_layer_profile(scan, col_idx, position, token_ids=None, top_k=1000):
    """For each token, compute its rank at every layer for one position,
    then derive layer-to-layer motion characteristics.

    Args:
        scan: ndarray (steps, layers, cols, vocab) — MRI scan
        col_idx: column to analyze
        position: which step/position to examine (int index into steps axis)
        token_ids: specific token IDs to profile (default: auto-select top_k most active)
        top_k: if token_ids is None, select the top_k tokens with most layer-to-layer motion

    Returns:
        list of dicts sorted by total_layer_motion descending:
        [{token_id, ranks[26], layer_deltas[25],
          mean_amplitude, sign_change_rate, rank_range, net_drift,
          peak_layer, peak_rank, trough_layer, trough_rank}]
    """
    num_layers = scan.shape[1]
    vocab = scan.shape[3]

    # Get ranks at every layer for this position
    probs_at_pos = scan[position, :, col_idx, :].astype(np.float32)  # (layers, vocab)
    ranks_at_pos = np.zeros((num_layers, vocab), dtype=np.int32)
    for l in range(num_layers):
        ranks_at_pos[l, :] = np.argsort(np.argsort(-probs_at_pos[l, :])).astype(np.int32)

    # Layer-to-layer deltas for ALL tokens
    layer_deltas = np.diff(ranks_at_pos, axis=0)  # (layers-1, vocab)

    # Select tokens to profile
    if token_ids is None:
        # Select by total layer motion (sum of |delta| across layers)
        total_motion = np.sum(np.abs(layer_deltas), axis=0)  # (vocab,)
        token_ids = np.argsort(total_motion)[::-1][:top_k]

    profiles = []
    for tid in token_ids:
        tid = int(tid)
        ranks = ranks_at_pos[:, tid]          # (num_layers,)
        deltas = layer_deltas[:, tid]          # (num_layers-1,)
        abs_deltas = np.abs(deltas)

        # Sign changes: count where consecutive deltas have opposite signs
        signs = np.sign(deltas)
        nonzero_signs = signs[signs != 0]
        if len(nonzero_signs) > 1:
            sign_changes = int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1]))
            sign_change_rate = sign_changes / (len(nonzero_signs) - 1)
        else:
            sign_changes = 0
            sign_change_rate = 0.0

        peak_layer = int(np.argmin(ranks))
        trough_layer = int(np.argmax(ranks))

        profiles.append({
            'token_id': tid,
            'ranks': ranks.tolist(),
            'layer_deltas': deltas.tolist(),
            'mean_amplitude': float(abs_deltas.mean()),
            'max_amplitude': float(abs_deltas.max()),
            'sign_changes': sign_changes,
            'sign_change_rate': sign_change_rate,
            'rank_range': int(ranks.max() - ranks.min()),
            'net_drift': int(ranks[-1] - ranks[0]),
            'peak_layer': peak_layer,
            'peak_rank': int(ranks[peak_layer]),
            'trough_layer': trough_layer,
            'trough_rank': int(ranks[trough_layer]),
            'total_layer_motion': float(abs_deltas.sum()),
        })

    profiles.sort(key=lambda p: -p['total_layer_motion'])
    return profiles


def profile_all_heads_at_position(scan, position, metadata, top_k=1000):
    """Profile all heads at one position.

    Args:
        scan: full MRI scan
        position: step index
        metadata: from load_token_metadata
        top_k: tokens to profile per head

    Returns:
        dict[col_idx -> list of token profiles with metadata attached]
    """
    num_cols = scan.shape[2]
    result = {}

    for col_idx in range(num_cols):
        profiles = compute_token_layer_profile(scan, col_idx, position, top_k=top_k)

        # Attach metadata
        for p in profiles:
            tid = p['token_id']
            for field in metadata:
                if field.endswith('_labels'):
                    continue
                codes = metadata[field]
                labels = metadata.get(f'{field}_labels')
                if 0 <= tid < len(codes):
                    code = int(codes[tid])
                    p[field] = labels[code] if labels else code

        result[col_idx] = profiles

    return result


def classify_motion_type(profile):
    """Classify a token's layer-to-layer motion pattern.

    Returns one of:
        'vibrator'    — high sign-change rate, small amplitude relative to range
        'sweeper'     — low sign-change rate, large net drift
        'peaker'      — rises to a specific layer then falls (or vice versa)
        'flat'        — minimal motion across layers
        'chaotic'     — high amplitude, high sign changes, large range

    This is a heuristic classification for exploration, not a formal taxonomy.
    """
    scr = profile['sign_change_rate']
    amp = profile['mean_amplitude']
    rng = profile['rank_range']
    drift = abs(profile['net_drift'])

    if rng < 1000:
        return 'flat'
    if scr > 0.6 and amp < rng * 0.15:
        return 'vibrator'
    if scr < 0.3 and drift > rng * 0.5:
        return 'sweeper'
    if scr < 0.4 and drift < rng * 0.2:
        return 'peaker'
    return 'chaotic'


# ─── Full-vocabulary layer profile ────────────────────────────────────────────

def full_vocab_layer_profile(scan, col_idx, position, metadata):
    """Compute layer-to-layer motion statistics for the ENTIRE vocabulary
    at one (position, column), broken down by metadata dimensions.

    No sampling, no top-K — all 262k tokens contribute.

    Args:
        scan: ndarray (steps, layers, cols, vocab) — MRI scan
        col_idx: column to analyze
        position: step/position index
        metadata: dict from load_token_metadata

    Returns:
        dict with:
        - per_layer: list of 25 dicts (one per layer transition L->L+1), each with:
            - layer_from, layer_to
            - mean_abs_delta, median_abs_delta, std_abs_delta
            - pct_moved_gt_1000, pct_moved_gt_10000, pct_moved_gt_100000
            - mean_positive_delta, mean_negative_delta (direction split)
            - n_rising, n_sinking, n_stationary (count by direction)
            - by_script: {script: {count, mean_abs_delta, mean_signed_delta, fraction_of_script}}
            - by_category: {cat: {count, mean_abs_delta, mean_signed_delta, fraction_of_cat}}
            - by_ascii: {0: {...}, 1: {...}}
        - overall: aggregate across all layer transitions
    """
    num_layers = scan.shape[1]
    vocab = scan.shape[3]

    # Get ranks at every layer for this position
    probs_at_pos = scan[position, :, col_idx, :].astype(np.float32)
    ranks_at_pos = np.zeros((num_layers, vocab), dtype=np.int32)
    for l in range(num_layers):
        ranks_at_pos[l, :] = np.argsort(np.argsort(-probs_at_pos[l, :])).astype(np.int32)

    # Layer-to-layer deltas
    layer_deltas = np.diff(ranks_at_pos, axis=0)  # (num_layers-1, vocab)

    # Get metadata arrays
    script_codes = metadata.get('script', np.zeros(vocab, dtype=np.int32))
    script_labels = metadata.get('script_labels', {})
    cat_codes = metadata.get('category', np.zeros(vocab, dtype=np.int32))
    cat_labels = metadata.get('category_labels', {})
    ascii_arr = metadata.get('is_ascii', np.zeros(vocab, dtype=np.int32))

    per_layer = []
    for l in range(num_layers - 1):
        deltas = layer_deltas[l, :]  # (vocab,)
        abs_d = np.abs(deltas).astype(np.float64)

        rising = deltas < 0   # negative delta = rose toward surface
        sinking = deltas > 0
        stationary = deltas == 0

        layer_result = {
            'layer_from': l,
            'layer_to': l + 1,
            'mean_abs_delta': float(abs_d.mean()),
            'median_abs_delta': float(np.median(abs_d)),
            'std_abs_delta': float(abs_d.std()),
            'pct_moved_gt_1000': float(np.mean(abs_d > 1000) * 100),
            'pct_moved_gt_10000': float(np.mean(abs_d > 10000) * 100),
            'pct_moved_gt_100000': float(np.mean(abs_d > 100000) * 100),
            'mean_positive_delta': float(deltas[sinking].mean()) if sinking.any() else 0,
            'mean_negative_delta': float(deltas[rising].mean()) if rising.any() else 0,
            'n_rising': int(rising.sum()),
            'n_sinking': int(sinking.sum()),
            'n_stationary': int(stationary.sum()),
        }

        # Script breakdown — all tokens, not sampled
        by_script = {}
        for code, label in script_labels.items():
            mask = script_codes == code
            n = int(mask.sum())
            if n == 0:
                continue
            script_deltas = deltas[mask].astype(np.float64)
            script_abs = abs_d[mask]
            by_script[label] = {
                'count': n,
                'mean_abs_delta': float(script_abs.mean()),
                'mean_signed_delta': float(script_deltas.mean()),
                'std_abs_delta': float(script_abs.std()),
                'pct_rising': float(np.mean(script_deltas < 0) * 100),
                'pct_sinking': float(np.mean(script_deltas > 0) * 100),
            }
        layer_result['by_script'] = by_script

        # Category breakdown
        by_category = {}
        for code, label in cat_labels.items():
            mask = cat_codes == code
            n = int(mask.sum())
            if n == 0:
                continue
            cat_deltas = deltas[mask].astype(np.float64)
            cat_abs = abs_d[mask]
            by_category[label] = {
                'count': n,
                'mean_abs_delta': float(cat_abs.mean()),
                'mean_signed_delta': float(cat_deltas.mean()),
                'pct_rising': float(np.mean(cat_deltas < 0) * 100),
                'pct_sinking': float(np.mean(cat_deltas > 0) * 100),
            }
        layer_result['by_category'] = by_category

        # ASCII breakdown
        by_ascii = {}
        for val in [0, 1]:
            mask = ascii_arr == val
            n = int(mask.sum())
            if n == 0:
                continue
            a_deltas = deltas[mask].astype(np.float64)
            a_abs = abs_d[mask]
            by_ascii[val] = {
                'count': n,
                'mean_abs_delta': float(a_abs.mean()),
                'mean_signed_delta': float(a_deltas.mean()),
                'pct_rising': float(np.mean(a_deltas < 0) * 100),
            }
        layer_result['by_ascii'] = by_ascii

        per_layer.append(layer_result)

    # Overall aggregate
    all_abs = np.abs(layer_deltas).astype(np.float64)
    overall = {
        'mean_abs_delta': float(all_abs.mean()),
        'median_abs_delta': float(np.median(all_abs)),
        'total_positions': num_layers - 1,
    }

    return {'per_layer': per_layer, 'overall': overall}


def full_vocab_oscillation_profile(scan, col_idx, position):
    """Characterize the oscillation pattern of EVERY token across layers
    at one (position, column). No sampling.

    For each token, computes:
    - sign_change_rate: fraction of consecutive layer deltas with opposite sign (1.0 = perfect flip-flop)
    - mean_amplitude: average |delta| across layer transitions
    - period_estimate: dominant oscillation period via zero-crossing count
    - rank at each layer (for tokens of interest)

    Args:
        scan: MRI scan array
        col_idx: column to analyze
        position: step index

    Returns:
        dict with:
        - sign_change_rates: ndarray (vocab,) — 0 to 1 per token
        - mean_amplitudes: ndarray (vocab,) — average |layer delta| per token
        - amplitude_bands: {band_label: {count, mean_scr, median_scr, pct_flipflop}}
            where bands are <100, 100-500, 500-2000, 2000-10000, 10000-50000, 50000+
        - flipflop_tokens: list of token_ids with scr > 0.8 and mean_amp in each band
    """
    num_layers = scan.shape[1]
    vocab = scan.shape[3]

    # Ranks at every layer
    probs = scan[position, :, col_idx, :].astype(np.float32)
    ranks = np.zeros((num_layers, vocab), dtype=np.int32)
    for l in range(num_layers):
        ranks[l, :] = np.argsort(np.argsort(-probs[l, :])).astype(np.int32)

    # Layer deltas: (num_layers-1, vocab)
    deltas = np.diff(ranks, axis=0).astype(np.float64)
    abs_deltas = np.abs(deltas)

    # Per-token mean amplitude
    mean_amps = abs_deltas.mean(axis=0)  # (vocab,)

    # Per-token sign change rate
    signs = np.sign(deltas)  # (num_layers-1, vocab), values in {-1, 0, 1}
    # For sign changes: compare consecutive signs, count disagreements
    # Only count where both are nonzero
    n_transitions = deltas.shape[0] - 1  # number of consecutive pairs
    sign_changes = np.zeros(vocab, dtype=np.float64)
    sign_counts = np.zeros(vocab, dtype=np.float64)

    for i in range(n_transitions):
        s1 = signs[i, :]
        s2 = signs[i + 1, :]
        both_nonzero = (s1 != 0) & (s2 != 0)
        changed = both_nonzero & (s1 != s2)
        sign_changes += changed.astype(np.float64)
        sign_counts += both_nonzero.astype(np.float64)

    scr = np.where(sign_counts > 0, sign_changes / sign_counts, 0.0)

    # Amplitude bands
    bands = [
        ('<100', 0, 100),
        ('100-500', 100, 500),
        ('500-2K', 500, 2000),
        ('2K-10K', 2000, 10000),
        ('10K-50K', 10000, 50000),
        ('50K+', 50000, float('inf')),
    ]

    band_stats = {}
    flipflop_by_band = {}
    for label, lo, hi in bands:
        mask = (mean_amps >= lo) & (mean_amps < hi)
        n = int(mask.sum())
        if n == 0:
            band_stats[label] = {'count': 0, 'mean_scr': 0, 'median_scr': 0,
                                  'pct_flipflop': 0, 'pct_sweeper': 0}
            flipflop_by_band[label] = []
            continue

        band_scr = scr[mask]
        flipflop_mask = band_scr > 0.8
        sweeper_mask = band_scr < 0.2

        band_stats[label] = {
            'count': n,
            'mean_scr': float(band_scr.mean()),
            'median_scr': float(np.median(band_scr)),
            'pct_flipflop': float(flipflop_mask.sum() / n * 100),
            'pct_sweeper': float(sweeper_mask.sum() / n * 100),
        }

        # Collect flip-flop token IDs in this band
        full_mask = mask & (scr > 0.8)
        flipflop_by_band[label] = np.where(full_mask)[0].tolist()

    return {
        'sign_change_rates': scr,
        'mean_amplitudes': mean_amps,
        'amplitude_bands': band_stats,
        'flipflop_by_band': flipflop_by_band,
        'ranks': ranks,  # (layers, vocab) for detailed inspection
    }


def compute_token_motion_table(scan, col_idx, position):
    """Compute motion metrics for EVERY token at one (position, column).

    Returns a structured numpy array with one row per token (262k rows),
    columns for all motion metrics. This is the raw data — attach metadata
    separately for grouping/filtering.

    Args:
        scan: ndarray (steps, layers, cols, vocab)
        col_idx: column index
        position: step index

    Returns:
        numpy structured array with fields:
            token_id        int32
            mean_amplitude  float32   average |layer delta|
            max_amplitude   int32     largest single layer jump
            sign_change_rate float32  0-1, fraction of consecutive deltas with opposite sign
            rank_range      int32     max_rank - min_rank across layers
            net_drift       int32     rank[last_layer] - rank[first_layer]
            peak_layer      int8      layer with lowest rank (closest to surface)
            peak_rank       int32     rank at peak layer
            trough_layer    int8      layer with highest rank (most buried)
            trough_rank     int32     rank at trough layer
            total_motion    int64     sum of |delta| across all layer transitions
            mean_rank       float32   average rank across all layers
    """
    num_layers = scan.shape[1]
    vocab = scan.shape[3]

    # Ranks at every layer
    probs = scan[position, :, col_idx, :].astype(np.float32)
    ranks = np.zeros((num_layers, vocab), dtype=np.int32)
    for l in range(num_layers):
        ranks[l, :] = np.argsort(np.argsort(-probs[l, :])).astype(np.int32)

    # Layer deltas
    deltas = np.diff(ranks, axis=0).astype(np.int64)  # (num_layers-1, vocab)
    abs_deltas = np.abs(deltas)

    # Compute all metrics vectorized across vocab
    mean_amp = abs_deltas.mean(axis=0).astype(np.float32)
    max_amp = abs_deltas.max(axis=0).astype(np.int32)
    total_motion = abs_deltas.sum(axis=0)
    rank_range = (ranks.max(axis=0) - ranks.min(axis=0)).astype(np.int32)
    net_drift = (ranks[-1, :] - ranks[0, :]).astype(np.int32)
    peak_layer = ranks.argmin(axis=0).astype(np.int8)
    peak_rank = ranks.min(axis=0).astype(np.int32)
    trough_layer = ranks.argmax(axis=0).astype(np.int8)
    trough_rank = ranks.max(axis=0).astype(np.int32)
    mean_rank = ranks.mean(axis=0).astype(np.float32)

    # Sign change rate — vectorized
    signs = np.sign(deltas)  # (num_layers-1, vocab)
    n_trans = signs.shape[0] - 1
    sign_changes = np.zeros(vocab, dtype=np.float32)
    sign_pairs = np.zeros(vocab, dtype=np.float32)
    for i in range(n_trans):
        s1 = signs[i, :]
        s2 = signs[i + 1, :]
        both_nz = (s1 != 0) & (s2 != 0)
        changed = both_nz & (s1 != s2)
        sign_changes += changed.astype(np.float32)
        sign_pairs += both_nz.astype(np.float32)
    scr = np.where(sign_pairs > 0, sign_changes / sign_pairs, 0.0).astype(np.float32)

    # Build structured array
    dtype = np.dtype([
        ('token_id', np.int32),
        ('mean_amplitude', np.float32),
        ('max_amplitude', np.int32),
        ('sign_change_rate', np.float32),
        ('rank_range', np.int32),
        ('net_drift', np.int32),
        ('peak_layer', np.int8),
        ('peak_rank', np.int32),
        ('trough_layer', np.int8),
        ('trough_rank', np.int32),
        ('total_motion', np.int64),
        ('mean_rank', np.float32),
    ])

    table = np.zeros(vocab, dtype=dtype)
    table['token_id'] = np.arange(vocab, dtype=np.int32)
    table['mean_amplitude'] = mean_amp
    table['max_amplitude'] = max_amp
    table['sign_change_rate'] = scr
    table['rank_range'] = rank_range
    table['net_drift'] = net_drift
    table['peak_layer'] = peak_layer
    table['peak_rank'] = peak_rank
    table['trough_layer'] = trough_layer
    table['trough_rank'] = trough_rank
    table['total_motion'] = total_motion
    table['mean_rank'] = mean_rank

    return table


def attach_metadata_to_table(motion_table, metadata):
    """Attach lexical metadata columns to a motion table.

    Args:
        motion_table: structured array from compute_token_motion_table
        metadata: dict from load_token_metadata

    Returns:
        (motion_table, meta_arrays) where meta_arrays is dict[field -> values_array]
        Both indexed by position in the table (same order as motion_table).
    """
    vocab = len(motion_table)
    meta_arrays = {}
    for field in metadata:
        if field.endswith('_labels'):
            continue
        codes = metadata[field]
        labels = metadata.get(f'{field}_labels')
        if labels:
            # Decode to strings
            decoded = np.array([labels.get(int(codes[i]), '?') for i in range(vocab)], dtype=object)
            meta_arrays[field] = decoded
        else:
            meta_arrays[field] = codes[:vocab].copy()
    return motion_table, meta_arrays


def group_motion_by(motion_table, meta_arrays, group_field, sort_by='mean_amplitude'):
    """Group tokens by a metadata field and compute aggregate motion stats.

    Args:
        motion_table: structured array from compute_token_motion_table
        meta_arrays: dict from attach_metadata_to_table
        group_field: metadata field to group by (e.g., 'script', 'language', 'category')
        sort_by: metric to sort groups by

    Returns:
        list of dicts sorted by sort_by descending:
        [{group, count, mean_amplitude, median_amplitude, mean_scr, median_scr,
          mean_rank_range, mean_net_drift, mean_peak_rank, pct_flipflop, pct_sweeper}]
    """
    if group_field not in meta_arrays:
        return []

    labels = meta_arrays[group_field]
    unique_labels = sorted(set(labels))

    groups = []
    for label in unique_labels:
        mask = labels == label
        subset = motion_table[mask]
        n = len(subset)
        if n == 0:
            continue

        # Build group stats from whatever fields the table has
        fields = subset.dtype.names
        entry = {'group': str(label), 'count': n}

        for fname in fields:
            if fname == 'token_id':
                continue
            vals = subset[fname].astype(np.float64)
            entry[f'mean_{fname}'] = float(vals.mean())

        # Motion-specific stats if available
        if 'mean_amplitude' in fields:
            amps = subset['mean_amplitude']
            entry['median_amplitude'] = float(np.median(amps))
            entry['std_amplitude'] = float(amps.std())
        if 'sign_change_rate' in fields:
            scrs = subset['sign_change_rate']
            entry['median_scr'] = float(np.median(scrs))
            entry['pct_flipflop'] = float(np.sum(scrs > 0.8) / n * 100)
            entry['pct_sweeper'] = float(np.sum(scrs < 0.2) / n * 100)

        groups.append(entry)

    groups.sort(key=lambda g: -g.get(sort_by, 0))
    return groups


def compute_all_coordinates_motion(scan, position):
    """Compute token motion table at ALL 104 (layer×head) coordinates for one position.

    Instead of looking at layer-to-layer motion (rank change from L0 to L25),
    this computes the motion metrics for each token at each individual
    (layer, head) coordinate using the softmax at that single cell.

    But wait — motion requires comparing two states. At a single cell there's
    no "motion." What we compute here is each token's RANK at each of the 104
    coordinates, then derive cross-coordinate metrics:

    For each token:
    - rank at each of 104 coordinates: (26 layers × 4 heads)
    - How variable is this token across heads at the same layer?
    - How variable across layers at the same head?
    - What's its best/worst coordinate?

    Args:
        scan: ndarray (steps, layers, cols, vocab) — MRI scan
        position: step index

    Returns:
        dict with:
        - ranks: ndarray (num_layers, num_heads, vocab) int32
        - per_token: structured array (vocab,) with:
            token_id, mean_rank, std_rank_across_coords,
            best_coord_layer, best_coord_head, best_rank,
            worst_coord_layer, worst_coord_head, worst_rank,
            head_variance (mean std across heads at same layer),
            layer_variance (mean std across layers at same head)
    """
    num_layers = scan.shape[1]
    # cols 0-3 are H0-H3, col 4 is Layer (combined)
    num_heads = min(4, scan.shape[2] - 1)  # individual heads only
    vocab = scan.shape[3]

    # Compute ranks at all 104 coordinates
    ranks = np.zeros((num_layers, num_heads, vocab), dtype=np.int32)
    for l in range(num_layers):
        for h in range(num_heads):
            probs = scan[position, l, h, :].astype(np.float32)
            ranks[l, h, :] = np.argsort(np.argsort(-probs)).astype(np.int32)

    # Per-token metrics across all coordinates
    flat_ranks = ranks.reshape(-1, vocab)  # (104, vocab)

    mean_rank = flat_ranks.mean(axis=0).astype(np.float32)
    std_rank = flat_ranks.std(axis=0).astype(np.float32)

    # Best and worst coordinates
    flat_best = flat_ranks.argmin(axis=0)  # index into flattened (104,)
    flat_worst = flat_ranks.argmax(axis=0)
    best_layer = (flat_best // num_heads).astype(np.int8)
    best_head = (flat_best % num_heads).astype(np.int8)
    best_rank = flat_ranks.min(axis=0).astype(np.int32)
    worst_layer = (flat_worst // num_heads).astype(np.int8)
    worst_head = (flat_worst % num_heads).astype(np.int8)
    worst_rank = flat_ranks.max(axis=0).astype(np.int32)

    # Head variance: for each layer, std across 4 heads; then average across layers
    head_stds = np.zeros((num_layers, vocab), dtype=np.float32)
    for l in range(num_layers):
        head_stds[l, :] = ranks[l, :, :].std(axis=0)
    head_variance = head_stds.mean(axis=0)

    # Layer variance: for each head, std across 26 layers; then average across heads
    layer_stds = np.zeros((num_heads, vocab), dtype=np.float32)
    for h in range(num_heads):
        layer_stds[h, :] = ranks[:, h, :].std(axis=0)
    layer_variance = layer_stds.mean(axis=0)

    dtype = np.dtype([
        ('token_id', np.int32),
        ('mean_rank', np.float32),
        ('std_rank', np.float32),
        ('best_layer', np.int8),
        ('best_head', np.int8),
        ('best_rank', np.int32),
        ('worst_layer', np.int8),
        ('worst_head', np.int8),
        ('worst_rank', np.int32),
        ('rank_range', np.int32),
        ('head_variance', np.float32),
        ('layer_variance', np.float32),
    ])

    table = np.zeros(vocab, dtype=dtype)
    table['token_id'] = np.arange(vocab, dtype=np.int32)
    table['mean_rank'] = mean_rank
    table['std_rank'] = std_rank
    table['best_layer'] = best_layer
    table['best_head'] = best_head
    table['best_rank'] = best_rank
    table['worst_layer'] = worst_layer
    table['worst_head'] = worst_head
    table['worst_rank'] = worst_rank
    table['rank_range'] = (worst_rank - best_rank).astype(np.int32)
    table['head_variance'] = head_variance
    table['layer_variance'] = layer_variance

    return {'ranks': ranks, 'per_token': table}


def full_vocab_head_comparison(scan, position, metadata):
    """Run full_vocab_layer_profile for all columns and compare.

    Args:
        scan: MRI scan
        position: step index
        metadata: from load_token_metadata

    Returns:
        dict[col_idx -> full_vocab_layer_profile result]
    """
    num_cols = scan.shape[2]
    return {col: full_vocab_layer_profile(scan, col, position, metadata)
            for col in range(num_cols)}
