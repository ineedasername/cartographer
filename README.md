# tokographer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**A dependency-light toolkit for reading a transformer's internals with the logit lens** — project any layer or attention head to vocabulary space, on a stock Hugging Face causal LM, with the architecture discovered at load time so there's no per-model wiring.

tokographer provides an accessible range of analytical interpretability tools  — logit-lens projection, per-head decomposition, activation interventions, rank-displacement analysis — behind one small, architecture-agnostic API. It runs directly on stock `transformers` models (including multimodal wrappers), so nothing has to be converted into a special format first. The architecture probe is validated on the Gemma, Qwen, and Llama families; other families may need a pattern added to the probe — a few lines, not a rewrite.

*Functionality includes:*

- **logit-lens projection** of hidden states and isolated per-head attention outputs (through the model's own final norm + unembedding)
- *projection rank tracking below argmax:** across token generation how the vocabulary's full rank ordering shifts from layer to layer and head to head, against a null-prompt baseline
- a trainable **tuned lens** with a *swappable training objective* — KL (the canonical tuned lens) vs. other targets that capture alternative portions of the distribution
- **head-to-head pairwise correlations** and per-token output movement
- **activation interventions** — scale, zero, subtract, or nudge heads / MLPs / logits, with a sweep harness
- **visualizations** — rank heatmaps, entropy landscapes, token-identity "river" plots, 3-D proximity terrains
- an optional **LanceDB** token index for capture and post-inference analysis
- **multilingual constrained-vocabulary projection** — *"if the model had to surface this token in a target language, where in that language's vocabulary does the latent space place it?"*

## Install

From source:

```bash
git clone https://github.com/ineedasername/tokographer
cd tokographer
pip install -e .
```

Core dependencies: `numpy`, `torch`, `transformers`. Optional extras: `pip install -e ".[viz]"` (matplotlib + plotly), `".[decompose]"` (scipy), `".[lance]"` (lancedb), or `".[all]"`.

*(A PyPI release — `pip install tokographer` — is on the way.)*

## Quick start

Read what each layer's residual stream "wants to say" by projecting it through the model's final norm and unembedding — a **logit lens**. The architecture is probed at load time, so the same code runs on any supported causal LM:

```python
import torch
from tokographer.inspect import ModelEngine
from tokographer.hook import project_to_token

engine = ModelEngine()
profile = engine.load_model("HuggingFaceTB/SmolLM2-135M-Instruct")
model, tokenizer = engine.model, engine.tokenizer

ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to(engine.device)
with torch.no_grad():
    out = model(ids, output_hidden_states=True)
norm, lm_head = profile.get_norm(), profile.get_lm_head()

for layer in range(0, profile.num_layers + 1, profile.num_layers // 6):
    tok, p = project_to_token(out.hidden_states[layer][:, -1, :], norm, lm_head, tokenizer)
    print(f"layer {layer:2d}: {tok!r:10} (p={p:.3f})")
```

```
layer  0: ' is'    (p=1.000)
layer 15: ' '      (p=0.882)
layer 25: ' the'   (p=0.462)
```

Each line is the top token the logit lens reads from that layer's residual. (Raw logit-lens readouts are noisy on a small 135M model and sharpen with scale — the very noise the *tuned lens* was built to reduce; pass a larger model for crisper trajectories.)

Runnable scripts are in [`examples/`](examples/): the logit lens above, a per-layer / per-head rank-displacement scan, a head-muting intervention (zero one head's contribution and watch the prediction move), and a **tuned lens** trained with a swappable objective.

## What's in it

- **`tokographer.inspect`** — model loading and architecture probing: locates the layer stack, final norm, and unembedding for a model family without hardcoded paths (works on stock Hugging Face models, including multimodal wrappers).
- **`tokographer.hook`** — forward-hook lifecycle and **logit-lens** projection: map a hidden state, or a single attention head's isolated output, to a vocabulary distribution.
- **`tokographer.intervene`** — reusable hooks to scale, zero, subtract, or nudge attention heads / MLPs / logits, plus a sweep harness; architecture-agnostic via a layer accessor.
- **`tokographer.rankd`** — a per-(layer, head, position) scan over a forward pass. For each layer and head it reads the **vocabulary ranking** the logit lens implies and computes the **Spearman rank-correlation** — the strength and direction of the monotonic relationship between two orderings — against a null-prompt baseline. Also includes a trainable **tuned lens** (`tokographer.rankd.tuned_lens`, after Belrose et al. 2023) whose *training objective* is swappable: KL (the canonical tuned lens, head-dominated) versus rank/geometry objectives (pairwise-rank, Pearson, soft-Spearman) that target the long tail KL ignores — a small instrument for asking which layer preserves which structure of the distribution.
- **`tokographer.decompose`** — analysis over captured traces: pairwise head-to-head correlation (`circuits`), correlation power analysis (`cpa`), group differential analysis (`differential`), metadata cross-reference (`crossref`), and per-token movement (`traffic`).
- **`tokographer.viz`** — heatmaps, entropy landscapes, token-identity "river" plots, and 3-D proximity terrains (matplotlib / plotly).
- **`tokographer.io`** — `.npz` scan files, SQLite stores, and an optional LanceDB token index.
- **`tokographer.evaluate`** — a lightweight benchmark database plus sweep-curve transition detection and Pareto-frontier utilities.
- **`tokographer.common`** — shared statistics (Spearman, Cohen's d, Jaccard), token categorization, Unicode-script detection, terminal formatting.

A small set of token-ID vocabularies ships in `tokographer/data/constrained_vocabs/` (English, Chinese, Arabic, Cyrillic, Devanagari) for constrained-vocabulary projection — restrict the unembedding to one language and ask where a layer's representation would surface as a token in that script.

## Status

v0 — extracted and consolidated from research tooling. It is research code: the core paths run and are exercised by the examples, but the API surface is provisional and may shift between versions. The architecture probe is exercised on the Gemma, Qwen, and Llama families; other families may need a pattern added. Built and used in the open as a working instrument, not a polished framework.

## Citation

If you use tokographer in your work, you can cite it as:

```bibtex
@software{davison2026tokographer,
  author = {Davison, James J.},
  title  = {tokographer: reading transformer internals with the logit lens},
  year   = {2026},
  url    = {https://github.com/ineedasername/tokographer}
}
```

## License

MIT — see [LICENSE](LICENSE). © 2026 James J. Davison.

Built with Claude (Anthropic) as coding collaborator. **Responsibility for the code and its design choices rests with the human author, James J. Davison.**
