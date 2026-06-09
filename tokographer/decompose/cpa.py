"""Correlation Power Analysis on transformer register traces.

Adapts side-channel analysis to find which input token properties cause
which register changes at which layers.
"""

import numpy as np


def extract_input_properties(meta):
    """Extract 21 numerical properties from each generated token for CPA.

    Properties: token_id, output_prob, token_len, is_ascii, is_alpha, is_digit,
    is_punct, is_special, is_whitespace, has_upper, position, ctx_len,
    token_bit_0..7, token_low_byte.

    Returns dict of property_name -> np.array[n_steps].
    """
    steps = meta['steps']
    n = len(steps)
    props = {}

    # Token ID (raw)
    props['token_id'] = np.array([s['token_id'] for s in steps], dtype=np.float64)

    # Output probability (model confidence)
    props['output_prob'] = np.array([s['prob'] for s in steps], dtype=np.float64)

    # Token length in characters
    props['token_len'] = np.array([len(s['token']) for s in steps], dtype=np.float64)

    # Is ASCII
    props['is_ascii'] = np.array([
        1.0 if all(ord(c) < 128 for c in s['token']) else 0.0
        for s in steps
    ], dtype=np.float64)

    # Is alphabetic
    props['is_alpha'] = np.array([
        1.0 if s['token'].strip().isalpha() else 0.0
        for s in steps
    ], dtype=np.float64)

    # Is digit
    props['is_digit'] = np.array([
        1.0 if s['token'].strip().isdigit() else 0.0
        for s in steps
    ], dtype=np.float64)

    # Is punctuation
    props['is_punct'] = np.array([
        1.0 if s['token'].strip() and all(not c.isalnum() for c in s['token'].strip()) else 0.0
        for s in steps
    ], dtype=np.float64)

    # Is special token
    props['is_special'] = np.array([
        1.0 if s['token'].startswith('<') and s['token'].endswith('>') else 0.0
        for s in steps
    ], dtype=np.float64)

    # Is whitespace/newline
    props['is_whitespace'] = np.array([
        1.0 if s['token'].strip() == '' else 0.0
        for s in steps
    ], dtype=np.float64)

    # Has uppercase
    props['has_upper'] = np.array([
        1.0 if any(c.isupper() for c in s['token']) else 0.0
        for s in steps
    ], dtype=np.float64)

    # Step position (normalized 0-1)
    props['position'] = np.linspace(0, 1, n)

    # Context length
    props['ctx_len'] = np.array([s['ctx_len'] for s in steps], dtype=np.float64)

    # Token ID bit decomposition (low 8 bits) -- classic CPA approach
    for bit in range(8):
        props[f'token_bit_{bit}'] = np.array([
            float((s['token_id'] >> bit) & 1)
            for s in steps
        ], dtype=np.float64)

    # Token ID byte (low byte, normalized)
    props['token_low_byte'] = np.array([
        float(s['token_id'] & 0xFF) / 255.0
        for s in steps
    ], dtype=np.float64)

    return props


def compute_cpa(traces_dict, register_names, extract_trace_fn, target_layers=None):
    """Compute Pearson correlation between register values and input token properties.

    For each (prompt, layer, register, property): Pearson correlation.
    Then aggregate across prompts.

    Args:
        traces_dict: dict of {name: {'meta': meta_dict, 'path': filepath}}
        register_names: list of register name strings
        extract_trace_fn: callable(path, mode, layer) -> (trace, _, _, labels)
        target_layers: optional list of layer indices to restrict to

    Returns:
        corr_matrix: [n_layers, n_registers, n_properties] mean abs correlation
        property_names: sorted list of property names
        layer_list: sorted list of layer indices
    """
    all_data = []

    for pname, data in traces_dict.items():
        meta = data['meta']
        props = extract_input_properties(meta)
        target = meta['target_layers']

        if target_layers:
            target = [l for l in target if l in target_layers]

        traces_by_layer = {}
        for l in target:
            trace, _, _, labels = extract_trace_fn(data['path'], mode='prob', layer=l)
            traces_by_layer[l] = trace

        all_data.append((traces_by_layer, props, pname))

    # Get dimensions
    layer_list = sorted(set(
        l for tbyl, _, _ in all_data for l in tbyl.keys()
    ))
    n_layers = len(layer_list)
    n_regs = len(register_names)
    prop_names = sorted(all_data[0][1].keys())
    n_props = len(prop_names)

    # Accumulate correlations
    corr_sum = np.zeros((n_layers, n_regs, n_props))
    corr_count = np.zeros((n_layers, n_regs, n_props))

    for traces_by_layer, props, pname in all_data:
        for l_idx, layer in enumerate(layer_list):
            if layer not in traces_by_layer:
                continue
            trace = traces_by_layer[layer]  # [n_steps, n_regs]
            n_steps = trace.shape[0]

            for p_idx, pname_prop in enumerate(prop_names):
                prop_vals = props[pname_prop]
                plen = min(len(prop_vals), n_steps)
                if plen < 3:
                    continue

                for r in range(n_regs):
                    reg_vals = trace[:plen, r]
                    p_vals = prop_vals[:plen]

                    if np.std(reg_vals) < 1e-8 or np.std(p_vals) < 1e-8:
                        continue

                    corr = np.corrcoef(reg_vals, p_vals)[0, 1]
                    if not np.isnan(corr):
                        corr_sum[l_idx, r, p_idx] += abs(corr)
                        corr_count[l_idx, r, p_idx] += 1

    # Average
    mask = corr_count > 0
    corr_matrix = np.zeros_like(corr_sum)
    corr_matrix[mask] = corr_sum[mask] / corr_count[mask]

    return corr_matrix, prop_names, layer_list
