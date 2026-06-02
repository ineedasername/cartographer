"""Rank-displacement scan — multi-position Spearman capture across all layers/heads.

Projects each attention head's output (at every sequence position) through the
final norm and unembedding, ranks the resulting vocabulary distribution, and
compares it to a null-prompt baseline via Spearman rho. The result is a record
of how far each head/layer displaces the token ranking for a given prompt.

Includes model helpers and the Spearman computation for self-containment.
"""

import sqlite3
import time

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_MODEL = "google/gemma-3-1b-it"
TOP_K = 100


# ─── Model helpers ───────────────────────────────────────────────────────────

def get_model_internals(model, num_layers, num_heads):
    """Resolve inner model path, norm, lm_head, head_dim."""
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        inner = model.model.language_model
    elif hasattr(model, 'model') and hasattr(model.model, 'norm'):
        inner = model.model
    else:
        inner = model.model

    cfg = model.config
    if hasattr(cfg, 'text_config'):
        cfg = cfg.text_config
    head_dim = cfg.head_dim if hasattr(cfg, 'head_dim') else cfg.hidden_size // cfg.num_attention_heads

    return inner, inner.norm, model.lm_head, head_dim


def project_all_heads_all_positions(captured_ctx, layer_idx, inner, final_norm, lm_head,
                                     num_heads, head_dim, hidden_dim):
    """Project all heads at all positions for one layer.

    captured_ctx: [1, seq_len, attn_out_dim]
    Returns: probs_array [seq_len, num_heads, vocab_size] as float32 numpy
    """
    ctx = captured_ctx[0]
    seq_len = ctx.shape[0]
    o_weight = inner.layers[layer_idx].self_attn.o_proj.weight
    weight_view = o_weight.view(hidden_dim, num_heads, head_dim)

    multi = ctx.view(seq_len, num_heads, head_dim)

    results = []
    for h in range(num_heads):
        head_out = multi[:, h, :]
        head_w = weight_view[:, h, :]
        projected = torch.matmul(head_out, head_w.t()) * num_heads

        with torch.no_grad():
            normed = final_norm(projected)
            logits = lm_head(normed)
            probs = F.softmax(logits.float(), dim=-1)

        results.append(probs.cpu().numpy())

    stacked = np.stack(results, axis=0)
    return np.transpose(stacked, (1, 0, 2))


def compute_spearman_fast(base_ranks, prompt_probs):
    """Compute Spearman rho between base ranks and prompt probs.

    Args:
        base_ranks: [vocab_size] int32 (0 = highest prob in baseline)
        prompt_probs: [vocab_size] float32

    Returns:
        float (Spearman rho)
    """
    prompt_ranks = np.argsort(np.argsort(-prompt_probs)).astype(np.int64)
    base = base_ranks.astype(np.int64)
    d_sq_sum = np.sum((prompt_ranks - base) ** 2, dtype=np.float64)
    n = float(len(base_ranks))
    return 1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1.0))


# ─── Database ────────────────────────────────────────────────────────────────

def init_db(path):
    """Initialize the rank-displacement scan SQLite database.

    Args:
        path: str path to .db file

    Returns:
        sqlite3.Connection with row_factory set
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT
    );

    CREATE TABLE IF NOT EXISTS baselines (
        layer INTEGER,
        head INTEGER,
        ranks BLOB,
        PRIMARY KEY (layer, head)
    );

    CREATE TABLE IF NOT EXISTS scans (
        scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt TEXT NOT NULL,
        n_input_tokens INTEGER,
        n_gen_tokens INTEGER,
        model TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS positions (
        scan_id INTEGER,
        position INTEGER,
        phase TEXT,
        token_text TEXT,
        token_id INTEGER,
        layer INTEGER,
        head INTEGER,
        spearman REAL,
        net_motion INTEGER,
        top1_id INTEGER,
        top1_prob REAL,
        PRIMARY KEY (scan_id, position, phase, layer, head)
    );
    """)
    conn.commit()
    return conn


# ─── Core scan flows ─────────────────────────────────────────────────────────

def capture_baseline(model, tokenizer, device, num_layers, num_heads, db_path,
                     model_name=DEFAULT_MODEL):
    """Capture null-prompt baseline for all registers.

    Args:
        model: HF model
        tokenizer: HF tokenizer
        device: torch device
        num_layers, num_heads: int model dimensions
        db_path: str path to SQLite DB
        model_name: str model identifier for metadata

    Returns:
        float elapsed seconds
    """
    inner, final_norm, lm_head, head_dim = get_model_internals(model, num_layers, num_heads)
    hidden_dim = inner.norm.weight.shape[0]
    vocab_size = lm_head.weight.shape[0]

    conn = init_db(db_path)
    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)

    t0 = time.time()
    for layer in range(num_layers):
        captured = {}
        def hook_fn(module, hook_args, output):
            captured['ctx'] = hook_args[0].detach()

        handle = inner.layers[layer].self_attn.o_proj.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            handle.remove()

        ctx = captured['ctx']
        o_weight = inner.layers[layer].self_attn.o_proj.weight
        weight_view = o_weight.view(hidden_dim, num_heads, head_dim)
        ctx_pos = ctx[0, -1, :]
        multi = ctx_pos.view(num_heads, head_dim)

        for h in range(num_heads):
            head_out = multi[h]
            head_w = weight_view[:, h, :]
            projected = torch.matmul(head_out, head_w.t()) * num_heads

            with torch.no_grad():
                normed = final_norm(projected.unsqueeze(0))
                logits = lm_head(normed).squeeze()
                probs = F.softmax(logits.float(), dim=-1).cpu().numpy()

            ranks = np.argsort(np.argsort(-probs)).astype(np.int32)
            conn.execute("INSERT OR REPLACE INTO baselines (layer, head, ranks) VALUES (?,?,?)",
                         (layer, h, ranks.tobytes()))

    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                 ('model', model_name))
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                 ('vocab_size', str(vocab_size)))
    conn.commit()
    conn.close()
    return time.time() - t0


def run_scan(model, tokenizer, device, num_layers, num_heads,
             prompt, db_path, gen_tokens=30, model_name=DEFAULT_MODEL):
    """Scan a prompt: capture Spearman at every input position + generation.

    Args:
        model: HF model
        tokenizer: HF tokenizer
        device: torch device
        num_layers, num_heads: int model dimensions
        prompt: str input prompt
        db_path: str path to SQLite DB
        gen_tokens: int number of tokens to generate
        model_name: str model identifier

    Returns:
        (scan_id, elapsed_seconds)
    """
    inner, final_norm, lm_head, head_dim = get_model_internals(model, num_layers, num_heads)
    hidden_dim = inner.norm.weight.shape[0]
    vocab_size = lm_head.weight.shape[0]

    conn = init_db(db_path)

    # Check baseline exists
    base_count = conn.execute("SELECT COUNT(*) as c FROM baselines").fetchone()['c']
    if base_count == 0:
        raise RuntimeError("No baseline found. Run capture_baseline() first.")

    # Load all baseline ranks
    base_ranks = {}
    for row in conn.execute("SELECT layer, head, ranks FROM baselines"):
        base_ranks[(row['layer'], row['head'])] = np.frombuffer(row['ranks'], dtype=np.int32).copy()

    # Tokenize
    if prompt == '' or prompt is None:
        input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
        tokens = ['<bos>']
    else:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
        input_ids = torch.cat([bos, ids.to(device)], dim=1)
        tokens = ['<bos>'] + [tokenizer.decode(input_ids[0, i].item()) for i in range(1, input_ids.shape[1])]

    n_input = input_ids.shape[1]

    cur = conn.execute("INSERT INTO scans (prompt, n_input_tokens, n_gen_tokens, model) VALUES (?,?,?,?)",
                       (prompt, n_input, gen_tokens, model_name))
    scan_id = cur.lastrowid

    t0 = time.time()

    # Phase A: Input positions
    for layer in range(num_layers):
        captured = {}
        def hook_fn(module, hook_args, output):
            captured['ctx'] = hook_args[0].detach()

        handle = inner.layers[layer].self_attn.o_proj.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                model(input_ids)
        finally:
            handle.remove()

        all_pos_probs = project_all_heads_all_positions(
            captured['ctx'], layer, inner, final_norm, lm_head,
            num_heads, head_dim, hidden_dim)

        prev_ranks = {}

        for pos in range(n_input):
            for h in range(num_heads):
                probs = all_pos_probs[pos, h, :]
                rho = compute_spearman_fast(base_ranks[(layer, h)], probs)
                current_ranks = np.argsort(np.argsort(-probs)).astype(np.int32)

                if pos == 0:
                    ref_ranks = base_ranks[(layer, h)]
                else:
                    ref_ranks = prev_ranks.get(h, base_ranks[(layer, h)])

                moved_up = int(np.sum(current_ranks < ref_ranks))
                moved_down = int(np.sum(current_ranks > ref_ranks))
                net_motion = moved_up - moved_down
                prev_ranks[h] = current_ranks

                top1_idx = np.argmax(probs)
                top1_prob = float(probs[top1_idx])

                conn.execute("""INSERT OR REPLACE INTO positions
                    (scan_id, position, phase, token_text, token_id, layer, head,
                     spearman, net_motion, top1_id, top1_prob)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (scan_id, pos, 'input', tokens[pos] if pos < len(tokens) else '',
                     input_ids[0, pos].item(), layer, h, rho, net_motion, int(top1_idx), top1_prob))

    conn.commit()

    # Phase B: Generation
    if gen_tokens > 0:
        current_ids = input_ids.clone()

        layer_captured = {}
        handles = []
        for layer in range(num_layers):
            def make_hook(l):
                def hook_fn(module, hook_args, output):
                    layer_captured[l] = hook_args[0].detach()
                return hook_fn
            handle = inner.layers[layer].self_attn.o_proj.register_forward_hook(make_hook(layer))
            handles.append(handle)

        gen_prev_ranks = {}

        try:
            for step in range(gen_tokens):
                with torch.no_grad():
                    outputs = model(current_ids)

                logits_out = outputs.logits[0, -1, :]
                next_probs = F.softmax(logits_out.float(), dim=-1)
                next_id = torch.argmax(next_probs).item()
                next_token = tokenizer.decode(next_id)

                gen_pos = n_input + step
                for layer in range(num_layers):
                    if layer not in layer_captured:
                        continue
                    ctx = layer_captured[layer]
                    o_weight = inner.layers[layer].self_attn.o_proj.weight
                    weight_view = o_weight.view(hidden_dim, num_heads, head_dim)
                    ctx_pos = ctx[0, -1, :]
                    multi = ctx_pos.view(num_heads, head_dim)

                    for h in range(num_heads):
                        head_out = multi[h]
                        head_w = weight_view[:, h, :]
                        projected = torch.matmul(head_out, head_w.t()) * num_heads

                        with torch.no_grad():
                            normed = final_norm(projected.unsqueeze(0))
                            head_logits = lm_head(normed).squeeze()
                            probs = F.softmax(head_logits.float(), dim=-1).cpu().numpy()

                        rho = compute_spearman_fast(base_ranks[(layer, h)], probs)
                        current_ranks = np.argsort(np.argsort(-probs)).astype(np.int32)

                        ref_key = (layer, h)
                        if ref_key in gen_prev_ranks:
                            ref_ranks = gen_prev_ranks[ref_key]
                        else:
                            ref_ranks = base_ranks[ref_key]

                        moved_up = int(np.sum(current_ranks < ref_ranks))
                        moved_down = int(np.sum(current_ranks > ref_ranks))
                        net_motion = moved_up - moved_down
                        gen_prev_ranks[ref_key] = current_ranks

                        top1_idx = np.argmax(probs)
                        top1_prob = float(probs[top1_idx])

                        conn.execute("""INSERT OR REPLACE INTO positions
                            (scan_id, position, phase, token_text, token_id, layer, head,
                             spearman, net_motion, top1_id, top1_prob)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                            (scan_id, gen_pos, 'gen', next_token, next_id,
                             layer, h, rho, net_motion, int(top1_idx), top1_prob))

                current_ids = torch.cat(
                    [current_ids, torch.tensor([[next_id]], device=device)], dim=1)

                if next_id == tokenizer.eos_token_id:
                    break

        finally:
            for h in handles:
                h.remove()

    conn.commit()
    conn.close()
    return scan_id, time.time() - t0
