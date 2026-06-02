"""Compare ranked token stacks between coordinates in the rank-displacement DB.

Helpers for reading full-softmax distributions and baseline rank arrays back
out of a scan database for post-hoc comparison between (layer, head) cells.
"""

import sqlite3

import numpy as np
import torch
import torch.nn.functional as F


def get_full_probs(model, tokenizer, inner, final_norm, lm_head, device,
                   num_heads, head_dim, hidden_dim,
                   input_ids, layer, head, position=-1):
    """Get full softmax for one head at one position.

    Args:
        model: HF model
        tokenizer: HF tokenizer (unused, kept for API compat)
        inner: model inner (e.g. model.model)
        final_norm: final layer norm
        lm_head: language model head
        device: torch device
        num_heads, head_dim, hidden_dim: int model dimensions
        input_ids: tensor [1, seq_len]
        layer: int layer index
        head: int head index
        position: int sequence position (-1 for last)

    Returns:
        ndarray [vocab_size] float32 probabilities
    """
    captured = {}
    def hook_fn(module, args, output):
        captured['ctx'] = args[0].detach()

    handle = inner.layers[layer].self_attn.o_proj.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    ctx = captured['ctx']
    o_weight = inner.layers[layer].self_attn.o_proj.weight
    weight_view = o_weight.view(hidden_dim, num_heads, head_dim)

    ctx_pos = ctx[0, position, :]
    multi = ctx_pos.view(num_heads, head_dim)
    head_out = multi[head]
    head_w = weight_view[:, head, :]
    projected = torch.matmul(head_out, head_w.t()) * num_heads

    with torch.no_grad():
        normed = final_norm(projected.unsqueeze(0))
        logits = lm_head(normed).squeeze()
        probs = F.softmax(logits.float(), dim=-1)

    return probs.cpu().numpy()


def load_baseline_ranks(conn, layer, head):
    """Load baseline rank array from DB.

    Args:
        conn: sqlite3.Connection
        layer: int
        head: int

    Returns:
        ndarray [vocab_size] int32 or None
    """
    row = conn.execute("SELECT ranks FROM baselines WHERE layer=? AND head=?",
                       (layer, head)).fetchone()
    if row:
        return np.frombuffer(row['ranks'], dtype=np.int32).copy()
    return None


def get_scan_input_ids(conn, tokenizer, device, scan_id):
    """Reconstruct input_ids for a scan.

    Args:
        conn: sqlite3.Connection
        tokenizer: HF tokenizer
        device: torch device
        scan_id: int

    Returns:
        tensor [1, seq_len]
    """
    scan = conn.execute("SELECT * FROM scans WHERE scan_id=?", (scan_id,)).fetchone()
    prompt = scan['prompt']
    if prompt == '' or prompt is None:
        return torch.tensor([[tokenizer.bos_token_id]], device=device)
    else:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
        return torch.cat([bos, ids.to(device)], dim=1)
