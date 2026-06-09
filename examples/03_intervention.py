"""Intervention — mute an attention head and watch the prediction move.

Zeros one head's contribution (before the output projection) via a forward-pre-hook,
then compares the model's next-token prediction with and without it. The hooks are
plain torch and architecture-agnostic: you supply a `layer_accessor` so the same
code runs on any HF model family.

    pip install tokographer
    python examples/03_intervention.py
"""
import torch
from tokographer.inspect import load_model_simple
from tokographer.intervene import HeadScaleHook

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
model, tok, device, n_layers, n_heads = load_model_simple(MODEL)
head_dim = model.config.hidden_size // model.config.num_attention_heads

ids = tok("The capital of France is", return_tensors="pt").input_ids.to(device)

def top_token(ids):
    with torch.no_grad():
        logits = model(ids).logits[0, -1].float()
    p, i = torch.softmax(logits, dim=-1).max(dim=-1)
    return tok.decode(i.item()), p.item()

# o_proj for an HF causal LM lives at model.model.layers[l].self_attn.o_proj
o_proj_of = lambda m, l: m.model.layers[l].self_attn.o_proj

print("baseline next token   :", top_token(ids))

hook = HeadScaleHook(layer=n_layers - 1, head_scales={0: 0.0}, head_dim=head_dim,
                     position=None, layer_accessor=o_proj_of)
hook.install(model)
print(f"L{n_layers-1} head 0 muted     :", top_token(ids))
hook.remove()
print("hook removed (restored):", top_token(ids))
