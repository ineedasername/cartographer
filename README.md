# Cartographer

A library for inspecting transformer internals via logit-lens projection and rank-displacement analysis — architecture-agnostic, so model structure is discovered at load time rather than hardcoded.

Cartographer projects hidden states and per-head attention outputs through a model's final norm and unembedding (logit-lens style), then measures how the resulting token rankings shift across layers and heads. The core measurement is plain: it tracks how the rank ordering of the vocabulary changes from one layer/head to the next when projected to token space.

## Install

```bash
pip install cartographer-interp
```

Core dependencies are `numpy`, `torch`, and `transformers`. Optional extras pull in the plotting and storage backends only if you need them:

```bash
pip install "cartographer-interp[viz]"        # matplotlib + plotly for cartographer.viz
pip install "cartographer-interp[decompose]"  # scipy for cartographer.decompose.circuits
pip install "cartographer-interp[lance]"      # lancedb for the token-index helper
pip install "cartographer-interp[all]"        # everything
```

To install from source:

```bash
git clone https://github.com/ineedasername/cartographer
cd cartographer
pip install -e .
```

## Quick start

Load any supported causal LM, run a forward pass, and read out what each layer's residual stream "wants to say" by projecting it through the model's final norm and unembedding (a logit lens):

```python
import torch
from cartographer.inspect import ModelEngine
from cartographer.hook import project_to_token

# Architecture is probed at load time — no per-model wiring needed.
engine = ModelEngine()
profile = engine.load_model("HuggingFaceTB/SmolLM2-135M")
model, tokenizer = engine.model, engine.tokenizer
print(profile.architecture, profile.num_layers, "layers")

ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to(engine.device)
with torch.no_grad():
    out = model(ids, output_hidden_states=True)

norm, lm_head = profile.get_norm(), profile.get_lm_head()

# Project the residual stream at the last position, layer by layer.
for layer in range(0, profile.num_layers + 1, 5):
    h = out.hidden_states[layer][:, -1, :]
    token, prob = project_to_token(h, norm, lm_head, tokenizer)
    print(f"layer {layer:2d}: {token!r} (p={prob:.3f})")
```

For a per-head, per-position rank-displacement scan across all layers (writes to a SQLite DB and prints a summary dashboard):

```python
from cartographer.rankd import capture_baseline, run_scan, print_dashboard
from cartographer.inspect import load_model_simple

model, tok, device, n_layers, n_heads = load_model_simple("google/gemma-3-1b-it")
db = "rank_scan.db"

capture_baseline(model, tok, device, n_layers, n_heads, db)
run_scan(model, tok, device, n_layers, n_heads, "The capital of France is", db)
print_dashboard(db)
```

## What's in it

- **`cartographer.common`** — shared utilities: Spearman rank comparison, Cohen's d, Jaccard, token categorization, Unicode-script detection, terminal formatting.
- **`cartographer.inspect`** — model loading and architecture probing; discovers layers, norm, and unembedding for a model family without hardcoding paths.
- **`cartographer.hook`** — hook lifecycle management and projection functions that map hidden states and per-head outputs to token space.
- **`cartographer.io`** — file and database helpers for scan files (`.npz`), SQLite scan/benchmark DBs, and an optional LanceDB token index.
- **`cartographer.evaluate`** — a lightweight benchmark database plus sweep-curve transition detection and Pareto-frontier utilities.
- **`cartographer.viz`** — rank heatmaps, entropy landscapes, token-identity "river" plots, and 3D proximity terrains (matplotlib / plotly).
- **`cartographer.intervene`** — reusable hooks for scaling, zeroing, subtracting, or nudging attention heads / MLPs / logits, with a sweep harness.
- **`cartographer.rankd`** — the rank-displacement scan: per-head, per-position Spearman capture across all layers, with cross-model comparison and summary reporting.
- **`cartographer.decompose`** — analysis on captured traces: pairwise register correlation (`circuits`), correlation power analysis (`cpa`), group differential analysis (`differential`), metadata cross-reference (`crossref`), and per-token rank-displacement traffic analysis (`traffic`).

A small set of per-script token-ID vocabularies ships in `cartographer/data/constrained_vocabs/` for constrained-vocabulary projection.

## Status

v0.1 — extracted and consolidated from research tooling. The APIs work and the core paths are exercised, but they may shift between minor versions as the surface is cleaned up. Treat function signatures as provisional.

A hosted demo is planned: [link: HF Space].

## License

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 James J. Davison.

Built with Claude (Anthropic) as coding collaborator. **Responsibility for the code and its design choices rests with the human author, James J. Davison.**
