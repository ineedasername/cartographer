"""Tuned lens — train a per-layer affine probe with a swappable objective.

Where the logit lens (example 01) reads the *raw* residual stream, a tuned lens learns
a small per-layer correction so each layer's readout accounts for the transform the
remaining layers would apply. The objective is a strategy object: `kl` is the canonical
tuned lens (matches the model's final distribution, head-dominated); the rank/geometry
objectives (`pairwise_rank`, `pearson`, `soft_spearman`) interrogate the long tail that
KL ignores — see tokographer/rankd/tuned_lens.py for what each one *means*.

    pip install -e .                              # from the repo root
    python examples/04_tuned_lens.py             # quick smoke check (demo data, 270m)
    # real run (needs `datasets`; streams FineWeb-Edu):
    python -m tokographer.rankd.tuned_lens --loss kl   --data fineweb --steps 2000 --out kl_lens.pt
    python -m tokographer.rankd.tuned_lens --loss pairwise_rank --data fineweb --steps 2000 --out rank_lens.pt

The A/B that motivates the whole thing: train a `kl` lens and a `pairwise_rank` lens on
the same model/layers/data, then compare per-layer Spearman(lens, teacher) over the full
262k vocab. The rank lens should preserve tail ordering the KL lens throws away.
"""
import logging

from tokographer.rankd.tuned_lens import TrainConfig, TunedLensTrainer, make_loss, demo_stream, _eval_layer_sample

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Quick, dependency-free demonstration on the smallest Gemma 3 (already cached locally).
cfg = TrainConfig(model_id="google/gemma-3-270m-it", loss="kl", steps=60, slice_size=128)
trainer = TunedLensTrainer(cfg)
loss_fn = make_loss(cfg.loss)

probe_layers = _eval_layer_sample(trainer.layers)
before = trainer.eval_rank(layers=probe_layers)
print("\nstep 0 (logit lens) — Spearman(lens, teacher) over full vocab:")
print("  " + "  ".join(f"L{r['layer']}={r['spearman']:.3f}" for r in before))

stream = demo_stream(trainer.tokenizer, trainer.device, cfg.max_length)
for step, ids in enumerate(stream):
    m = trainer.step(ids, loss_fn)
    if step % 10 == 0:
        print(f"step {step:3d} | {cfg.loss} {m['loss']:.5f} | grad {m['grad_norm']:.3f} | drift {trainer.lens.translator_drift():.4f}")
    if step >= cfg.steps:
        break

after = trainer.eval_rank(layers=probe_layers)
print("\nafter training — Spearman(lens, teacher) over full vocab:")
print("  " + "  ".join(f"L{r['layer']}={r['spearman']:.3f}" for r in after))
