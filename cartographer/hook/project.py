"""All projection functions for converting hidden states to token space.

Consolidated from:
  - projection_map.py L122-194 (project_to_token, _rich, _softmax, _full)
  - telescope.py L48-110 (project_batch_topk, _hidden_topk, _heads_batch, _head_topk, _kv_topk)
  - rank_scan.py L74-103 (project_all_heads_all_positions)
"""

import numpy as np
import torch
import torch.nn.functional as F


# ─── Single-vector projections ────────────────────────────────────────────────
# Source: projection_map.py L122-194

def project_to_token(projected_hidden, final_norm, lm_head, tokenizer):
    """Project hidden state to argmax token.

    Args:
        projected_hidden: tensor — hidden state (already decomposed/combined)
        final_norm: nn.Module — the model's final layer norm
        lm_head: nn.Module — the output projection (unembedding)
        tokenizer: tokenizer for decoding

    Returns:
        (token_str, probability)
    """
    normed = final_norm(projected_hidden)
    logits = lm_head(normed)
    probs = F.softmax(logits, dim=-1)
    p, tid = torch.max(probs, dim=-1)
    tok = tokenizer.decode(tid.item())
    return tok, p.item()


def project_to_token_rich(projected_hidden, final_norm, lm_head, tokenizer, top_k=5):
    """Rich projection: top-k tokens, entropy, distribution shape.

    Returns dict with:
        'top_k': list of (token_str, prob) for top k
        'entropy': Shannon entropy of the distribution (bits)
        'argmax': (token_str, prob) — same as project_to_token
        'mass_top5': total probability mass in top 5
        'mass_top50': total probability mass in top 50
    """
    normed = final_norm(projected_hidden)
    logits = lm_head(normed)
    probs = F.softmax(logits, dim=-1)

    # Entropy in bits
    log_probs = torch.log2(probs + 1e-10)
    entropy = -(probs * log_probs).sum().item()

    # Top-k
    topk_probs, topk_ids = torch.topk(probs.squeeze(), top_k)
    top_k_list = []
    for i in range(top_k):
        tok = tokenizer.decode(topk_ids[i].item())
        top_k_list.append((tok, topk_probs[i].item()))

    # Mass in top-50
    top50_probs, _ = torch.topk(probs.squeeze(), min(50, probs.shape[-1]))
    mass_top50 = top50_probs.sum().item()

    return {
        'top_k': top_k_list,
        'entropy': entropy,
        'argmax': top_k_list[0],
        'mass_top5': sum(p for _, p in top_k_list),
        'mass_top50': mass_top50,
    }


def project_to_softmax(projected_hidden, final_norm, lm_head):
    """Project hidden state to full softmax distribution.

    Returns the raw probability tensor (vocab_size,) on CPU as float16.
    This is the complete recording — every possible analysis is a post-hoc query.
    """
    normed = final_norm(projected_hidden)
    logits = lm_head(normed)
    probs = F.softmax(logits, dim=-1)
    return probs.squeeze().detach().cpu().half()


def project_full(projected_hidden, final_norm, lm_head):
    """Project hidden state and return BOTH the projected vector and softmax.

    Returns:
        (projected_vector, softmax_probs) where:
        - projected_vector: (hidden_dim,) float32 on CPU — the normed hidden state,
          directly comparable to lm_head weight vectors (token embeddings).
          This is what you search against the Lance token index.
        - softmax_probs: (vocab_size,) float16 on CPU — the full distribution.
    """
    normed = final_norm(projected_hidden)
    logits = lm_head(normed)
    probs = F.softmax(logits, dim=-1)
    return (
        normed.squeeze().detach().cpu().float(),
        probs.squeeze().detach().cpu().half(),
    )


# ─── Batch projections (telescope) ───────────────────────────────────────────
# Source: telescope.py L48-110

def project_batch_topk(vecs, final_norm, lm_head, k=20):
    """Project a batch of hidden vectors to top-k token IDs and probs.

    Args:
        vecs: tensor [N, hidden_dim] — batch of hidden state vectors
        final_norm: nn.Module — final layer norm
        lm_head: nn.Module — output projection
        k: number of top tokens to return

    Returns:
        (ids, probs) where:
        - ids: ndarray [N, k] int32 — token IDs
        - probs: ndarray [N, k] float32 — probabilities
    """
    with torch.no_grad():
        if vecs.dim() == 1:
            vecs = vecs.unsqueeze(0)
        normed = final_norm(vecs)
        logits = lm_head(normed)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, k, dim=-1)
    return topk_ids.cpu().numpy().astype(np.int32), topk_probs.cpu().float().numpy()


def project_hidden_topk(vec, final_norm, lm_head, k=20):
    """Project single hidden vector to top-k (convenience wrapper)."""
    ids, probs = project_batch_topk(
        vec.unsqueeze(0) if vec.dim() == 1 else vec,
        final_norm, lm_head, k
    )
    return ids[0], probs[0]


def project_heads_batch(head_vecs, head_indices, o_proj, final_norm, lm_head,
                         num_heads=4, head_dim=256, k=20):
    """Project multiple per-head vectors in one batch via o_proj.

    Args:
        head_vecs: list of (head_vec, head_idx) tuples
        head_indices: unused (kept for API compat)
        o_proj: nn.Module — the output projection that maps concat-heads to hidden
        final_norm, lm_head: as usual
        num_heads, head_dim: model dimensions
        k: top-k

    Returns:
        list of (ids[k], probs[k]) tuples
    """
    if not head_vecs:
        return []

    device = head_vecs[0][0].device
    dtype = head_vecs[0][0].dtype
    n = len(head_vecs)
    full = torch.zeros(n, num_heads * head_dim, device=device, dtype=dtype)
    for i, (vec, h_idx) in enumerate(head_vecs):
        full[i, h_idx * head_dim:(h_idx + 1) * head_dim] = vec

    with torch.no_grad():
        projected = o_proj(full)  # [N, hidden_dim]
        ids, probs = project_batch_topk(projected, final_norm, lm_head, k)

    return [(ids[i], probs[i]) for i in range(n)]


def project_head_topk(head_vec, head_idx, o_proj, final_norm, lm_head,
                       num_heads=4, head_dim=256, k=20):
    """Project single per-head vector via o_proj (convenience wrapper)."""
    results = project_heads_batch(
        [(head_vec, head_idx)], None, o_proj, final_norm, lm_head,
        num_heads, head_dim, k
    )
    return results[0]


def project_kv_topk(kv_vec, o_proj, final_norm, lm_head,
                     num_heads=4, num_kv_heads=1, head_dim=256, k=20):
    """Project shared K or V vector via head slot 0."""
    return project_head_topk(kv_vec, 0, o_proj, final_norm, lm_head,
                              num_heads, head_dim, k)


# ─── All-positions projection ─────────────────────────────────────────────────
# Source: rank_scan.py L74-103

def project_all_heads_all_positions(captured_ctx, layer_idx, inner, final_norm, lm_head,
                                     num_heads, head_dim, hidden_dim):
    """Project all heads at all positions for one layer.

    Decomposes the captured o_proj input into per-head contributions using
    o_proj weight matrix factorization, then projects each through norm+lm_head.

    Args:
        captured_ctx: tensor [1, seq_len, attn_out_dim] — captured o_proj input
        layer_idx: which layer (for accessing o_proj weights)
        inner: the inner model (e.g., model.model)
        final_norm: final layer norm
        lm_head: output projection
        num_heads, head_dim, hidden_dim: model dimensions

    Returns:
        ndarray [seq_len, num_heads, vocab_size] float32
    """
    ctx = captured_ctx[0]  # [seq_len, attn_out_dim]
    seq_len = ctx.shape[0]
    o_weight = inner.layers[layer_idx].self_attn.o_proj.weight
    weight_view = o_weight.view(hidden_dim, num_heads, head_dim)

    multi = ctx.view(seq_len, num_heads, head_dim)

    results = []
    for h in range(num_heads):
        head_out = multi[:, h, :]       # [seq_len, head_dim]
        head_w = weight_view[:, h, :]   # [hidden_dim, head_dim]
        projected = torch.matmul(head_out, head_w.t()) * num_heads  # [seq_len, hidden_dim]

        with torch.no_grad():
            normed = final_norm(projected)
            logits = lm_head(normed)
            probs = F.softmax(logits.float(), dim=-1)

        results.append(probs.cpu().numpy())

    stacked = np.stack(results, axis=0)
    return np.transpose(stacked, (1, 0, 2))  # [seq_len, num_heads, vocab_size]


# ─── Delta-V projection (layer contribution to residual stream) ───────────────

def project_delta_v(hidden_states, layer_idx, final_norm, lm_head, tokenizer=None, top_k=3):
    """Project the delta-v of a layer: what the complete layer (attention + MLP + skips)
    actually contributed to the residual stream, projected through final_norm → lm_head.

    Delta-v = hidden_states[layer+1] - hidden_states[layer]
    This is the true force vector of the layer on the residual stream.

    Args:
        hidden_states: list/tuple from model output with output_hidden_states=True
            hidden_states[0] = embedding, hidden_states[N+1] = residual after layer N
        layer_idx: which layer's contribution to compute
        final_norm: model's final RMSNorm
        lm_head: model's output projection
        tokenizer: optional, for decoding
        top_k: number of top tokens to return

    Returns:
        list of (token_str_or_id, probability) tuples
    """
    # Delta: residual after this layer minus residual before
    before = hidden_states[layer_idx][:, -1, :]      # [batch, hidden] at last position
    after = hidden_states[layer_idx + 1][:, -1, :]   # [batch, hidden] at last position
    delta = after - before

    with torch.no_grad():
        normed = final_norm(delta)
        logits = lm_head(normed)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, top_k)

    results = []
    for k in range(top_k):
        tid = top_ids[0, k].item()
        p = top_probs[0, k].item()
        tok = tokenizer.decode(tid) if tokenizer else tid
        results.append((tok, p))
    return results


def project_residual_at_layer(hidden_states, layer_idx, final_norm, lm_head, tokenizer=None, top_k=3):
    """Project the full residual stream at a given layer through final_norm → lm_head.

    This shows what the model would output if generation stopped at this layer —
    the cumulative result of all layers up to this point.

    Args:
        hidden_states: from output_hidden_states=True
        layer_idx: layer (0 = after embedding, N = after layer N-1)
        final_norm, lm_head: model components
        tokenizer: optional
        top_k: number of top tokens

    Returns:
        list of (token_str_or_id, probability) tuples
    """
    residual = hidden_states[layer_idx + 1][:, -1, :]  # after this layer

    with torch.no_grad():
        normed = final_norm(residual)
        logits = lm_head(normed)
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, top_k)

    results = []
    for k in range(top_k):
        tid = top_ids[0, k].item()
        p = top_probs[0, k].item()
        tok = tokenizer.decode(tid) if tokenizer else tid
        results.append((tok, p))
    return results


# ─── Constrained vocabulary projection ───────────────────────────────────────

def project_to_constrained_vocab(hidden, final_norm, lm_head, token_ids):
    """Project hidden state to a SUBSET of the vocabulary.

    Instead of projecting to all 262k tokens, project to a chosen subset
    (e.g., top 5000 English words). Softmax is computed only over the subset,
    so probabilities redistribute among the included tokens.

    This gives a "translation" of what the head is saying if it could only
    speak in the constrained vocabulary. Useful for interpreting heads that
    project to multilingual computational tokens — the constrained projection
    shows the nearest English (or other language) equivalent.

    Args:
        hidden: tensor — hidden state (already decomposed/combined)
        final_norm: nn.Module — final layer norm
        lm_head: nn.Module — full output projection
        token_ids: list/tensor of token IDs to include in the constrained vocab

    Returns:
        (token_ids_array, probabilities) where:
        - token_ids_array: ndarray of the constrained token IDs
        - probabilities: ndarray of softmax probabilities over the constrained set
    """
    with torch.no_grad():
        normed = final_norm(hidden)
        if isinstance(token_ids, (list, np.ndarray)):
            token_ids_t = torch.tensor(token_ids, dtype=torch.long, device=lm_head.weight.device)
        else:
            token_ids_t = token_ids

        constrained_weights = lm_head.weight[token_ids_t]  # (N, hidden_dim)
        logits = torch.matmul(normed, constrained_weights.T)  # (..., N)
        probs = F.softmax(logits, dim=-1)

    ids_out = token_ids_t.cpu().numpy() if isinstance(token_ids_t, torch.Tensor) else np.array(token_ids)
    return ids_out, probs.squeeze().cpu().numpy()


def project_to_constrained_topk(hidden, final_norm, lm_head, token_ids, k=10, tokenizer=None):
    """Project to constrained vocab and return top-k with optional decoding.

    Args:
        hidden: hidden state tensor
        final_norm: final layer norm
        lm_head: output projection
        token_ids: constrained vocab token IDs
        k: number of top tokens to return
        tokenizer: optional, for decoding token IDs to strings

    Returns:
        list of (token_id, probability, token_str_or_None) tuples
    """
    ids, probs = project_to_constrained_vocab(hidden, final_norm, lm_head, token_ids)
    topk_idx = np.argsort(probs)[::-1][:k]
    results = []
    for i in topk_idx:
        tid = int(ids[i])
        p = float(probs[i])
        s = tokenizer.decode(tid) if tokenizer else None
        results.append((tid, p, s))
    return results
