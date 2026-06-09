"""Logit lens — read what each layer's residual stream "wants to say".

Projects the residual at the last position, layer by layer, through the model's
final norm + unembedding (the standard logit-lens readout). The point of this
example is the *ergonomics*: architecture is probed at load time, so there's no
per-model wiring to get `norm` and `lm_head`.

    pip install tokographer        # (or: pip install -e . from the repo)
    python examples/01_logit_lens.py
"""
import torch
from tokographer.inspect import ModelEngine
from tokographer.hook import project_to_token

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"   # tiny, ungated, CPU-friendly

engine = ModelEngine()
profile = engine.load_model(MODEL)
model, tokenizer = engine.model, engine.tokenizer
print(f"{profile.architecture} — {profile.num_layers} layers\n")

ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to(engine.device)
with torch.no_grad():
    out = model(ids, output_hidden_states=True)
norm, lm_head = profile.get_norm(), profile.get_lm_head()

print("layer | top token (logit lens, last position)")
print("------+--------------------------------------")
step = max(1, profile.num_layers // 8)
for layer in range(0, profile.num_layers + 1, step):
    h = out.hidden_states[layer][:, -1, :]
    tok, p = project_to_token(h, norm, lm_head, tokenizer)
    print(f"  {layer:2d}  | {tok!r:18} (p={p:.3f})")
