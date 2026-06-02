"""Multi-model rank-displacement scanning and comparison.

Orchestrates: load model → capture baseline → run scan → unload → next model.
Sequential execution (one model at a time) due to GPU memory constraints.
"""

import gc
import logging
import time

import torch

log = logging.getLogger("cartographer.rankd.multi_model")


def scan_model(model_id, prompt, db_path, gen_tokens=30, label=None, dtype=None):
    """Load a model, capture baseline, run scan, unload.

    One-shot convenience function for scanning a single model. Handles the
    full lifecycle including memory cleanup.

    Args:
        model_id: HuggingFace model ID (e.g., 'google/gemma-3-1b-it')
        prompt: str prompt to scan
        db_path: path to SQLite DB (shared across models)
        gen_tokens: number of tokens to generate
        label: optional label for the scan (defaults to model_id)
        dtype: optional torch dtype override

    Returns:
        dict with: model_id, label, scan_id, baseline_time, scan_time, total_time
    """
    from cartographer.inspect.model import load_model_simple
    from cartographer.rankd.scan import init_db, capture_baseline, run_scan

    label = label or model_id
    init_db(db_path)

    log.info("Loading %s...", model_id)
    t0 = time.time()

    # Load
    if dtype is not None:
        # Custom dtype — use ModelEngine for control
        from cartographer.inspect.model import ModelEngine
        engine = ModelEngine()
        engine.load_model(model_id, dtype=dtype)
        model, tokenizer, device = engine.model, engine.tokenizer, engine.device
        num_layers = engine.profile.num_layers
        num_heads = engine.profile.num_heads
    else:
        model, tokenizer, device, num_layers, num_heads = load_model_simple(model_id)

    t_load = time.time() - t0
    log.info("Loaded in %.1fs: %d layers, %d heads", t_load, num_layers, num_heads)

    # Baseline
    log.info("Capturing baseline...")
    t_baseline = capture_baseline(model, tokenizer, device, num_layers, num_heads,
                                   db_path, model_name=label)

    # Scan
    log.info("Running scan: '%s' (%d gen tokens)...", prompt[:50], gen_tokens)
    scan_id, t_scan = run_scan(model, tokenizer, device, num_layers, num_heads,
                                prompt, db_path, gen_tokens=gen_tokens, model_name=label)

    # Unload
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("Unloaded %s", model_id)

    total = time.time() - t0
    return {
        "model_id": model_id,
        "label": label,
        "scan_id": scan_id,
        "baseline_time": t_baseline,
        "scan_time": t_scan,
        "total_time": total,
    }


def scan_models(model_ids, prompt, db_path, gen_tokens=30, labels=None, dtype=None):
    """Run rank-displacement scans across multiple models sequentially.

    Args:
        model_ids: list of HuggingFace model IDs
        prompt: str prompt (same for all models)
        db_path: shared DB path
        gen_tokens: tokens to generate per scan
        labels: optional list of labels (parallel to model_ids)
        dtype: optional torch dtype override (applied to all)

    Returns:
        list of result dicts from scan_model()
    """
    if labels is None:
        labels = [None] * len(model_ids)

    results = []
    for i, (mid, label) in enumerate(zip(model_ids, labels)):
        log.info("=== Model %d/%d: %s ===", i + 1, len(model_ids), mid)
        result = scan_model(mid, prompt, db_path, gen_tokens, label, dtype)
        results.append(result)
        log.info("  scan_id=%d, total=%.1fs", result['scan_id'], result['total_time'])

    return results


def scan_with_perturbation(model_id, prompt, db_path, token_id_or_str,
                            custom_vector, gen_tokens=30, label=None,
                            tokenizer_for_token=None):
    """Load a model, inject an embedding perturbation, scan, unload.

    Args:
        model_id: base model to load
        prompt: prompt to scan
        db_path: shared DB path
        token_id_or_str: which token to perturb (int or str like '<unused42>')
        custom_vector: tensor [embedding_dim] — the replacement embedding
        gen_tokens: tokens to generate
        label: optional scan label (defaults to model_id + '-perturbed')
        tokenizer_for_token: only needed if token_id_or_str is a string

    Returns:
        result dict from scan (same structure as scan_model)
    """
    from cartographer.inspect.model import load_model_simple
    from cartographer.rankd.scan import init_db, capture_baseline, run_scan
    from cartographer.intervene.embedding import EmbeddingPerturbation

    label = label or f"{model_id}-perturbed"
    init_db(db_path)

    log.info("Loading %s for perturbation...", model_id)
    t0 = time.time()
    model, tokenizer, device, num_layers, num_heads = load_model_simple(model_id)

    tok = tokenizer_for_token or tokenizer

    with EmbeddingPerturbation(model, token_id_or_str, custom_vector, tok):
        log.info("Embedding perturbation active")

        # Baseline (with perturbation active)
        t_baseline = capture_baseline(model, tokenizer, device, num_layers, num_heads,
                                       db_path, model_name=label)

        # Scan (with perturbation active)
        scan_id, t_scan = run_scan(model, tokenizer, device, num_layers, num_heads,
                                    prompt, db_path, gen_tokens=gen_tokens, model_name=label)

    # Embedding restored automatically by context manager

    # Unload
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total = time.time() - t0
    return {
        "model_id": model_id,
        "label": label,
        "scan_id": scan_id,
        "baseline_time": t_baseline,
        "scan_time": t_scan,
        "total_time": total,
    }
