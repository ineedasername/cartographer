"""Rank-displacement scan — tokographer's distinctive lens.

Most logit-lens tooling reads off the *top* tokens or a single target token's rank.
This instead projects every attention head at every position through the unembedding
and measures how far the **entire vocabulary's rank ordering** shifts relative to a
null-prompt baseline (Spearman rho), per (layer, head, position). The dashboard shows
where in the stack a prompt most reorders what the model is "about" to say.

    pip install tokographer
    python examples/02_rank_displacement.py
"""
import os
from tokographer.inspect import load_model_simple
from tokographer.rankd import init_db, capture_baseline, run_scan, print_dashboard

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DB = "rank_scan.db"

model, tok, device, n_layers, n_heads = load_model_simple(MODEL)
print(f"loaded {MODEL}: {n_layers} layers x {n_heads} heads\n")

if os.path.exists(DB):
    os.remove(DB)
init_db(DB)
capture_baseline(model, tok, device, n_layers, n_heads, DB)          # null-prompt baseline
run_scan(model, tok, device, n_layers, n_heads, "The capital of France is", DB)
print_dashboard(DB)
print(f"\n(full per-cell rank-displacement data persisted in {DB})")
