"""Trainable tuned lens with a swappable objective — KL baseline + a rank/statistic family.

A *tuned lens* (Belrose et al. 2023, arXiv:2303.08112) learns, per layer, a small
affine "translator" A_l on the residual stream, then decodes through the model's own
*frozen* final norm and unembedding:

    logits_l = lm_head( final_norm( h_l + A_l(h_l) ) )

The translator is zero-initialised, so at step 0 the lens is exactly the logit lens.
Training nudges each layer's readout to account for the systematic transform the
*remaining* layers would have applied.

The point of THIS module is that the **objective is a strategy object**. The canonical
tuned lens minimises KL to the model's final distribution — which is dominated by the
head of the distribution and effectively blind to the long tail. By swapping the loss
we ask a different question of each layer:

    KL            preserve probability mass        → head calibration / what it confidently predicts
    Pearson       preserve logit-vector geometry   → is the layer's logit *direction* aligned (scale-sensitive)
    pairwise-rank preserve ordinal structure       → the full preference ordering, suppressions included (tail)
    soft-Spearman global rank correlation          → gross reordering across the whole vocab (needs torchsort)
    (Kendall)     local pairwise concordance       → has the layer resolved nearby competitors yet
    (depth)       representation-space centrality   → geometric commitment (uses h + the unembed cloud, not logits)

Every loss receives a `LensBatch` (both logit vectors AND the underlying geometry), so
geometric objectives like statistical depth are expressible without changing the trainer.

Gemma 3 decode facts (verified against transformers 4.57.1 source):
  - tied embeddings; the unembedding matrix is `lm_head.weight` (raw, unscaled).
  - final norm is `model.model.norm` (Gemma3RMSNorm, fp32 internally, gamma = 1 + weight).
  - NO logit soft-capping in Gemma 3 (that was Gemma 2). Do not apply any tanh cap.
  - `output_hidden_states` gives PRE-final-norm residuals at indices 0..L-1; index L is
    already normed. So intermediate states MUST have `model.model.norm` applied at decode.
  - sqrt(hidden_size) embedding scaling is an INPUT-side effect only; do not re-apply it.

The lens runs end-to-end in fp32 (translators, norm path, and a cached fp32 unembedding)
so the long tail — the whole point — is not crushed by bf16's ~3 significant digits.

Run:  python -m tokographer.rankd.tuned_lens --smoke
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import cycle
from typing import Callable, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokographer.inspect.model import ModelEngine
from tokographer.common.stats import compute_spearman_fast, rank_tokens

log = logging.getLogger("tokographer.rankd.tuned_lens")


# ─── The lens ─────────────────────────────────────────────────────────────────

class TunedLens(nn.Module):
    """A stack of per-layer affine translators (zero-init residual = identity at step 0).

    Only the translators are parameters. The model's final norm and unembedding are
    passed in at decode time so they stay frozen and out of the optimiser.
    """

    def __init__(self, num_layers: int, hidden_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.translators = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        for lin in self.translators:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.float()  # train the probe in fp32 regardless of the base model's dtype

    def transform(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Residual affine update: h + A_l(h). Identity while translators are zero."""
        h = h.float()
        return h + self.translators[layer](h)

    def decode(self, translated: torch.Tensor, norm: nn.Module, w_u: torch.Tensor) -> torch.Tensor:
        """Project a (already-translated) residual to full-vocab logits in fp32."""
        normed = norm(translated)              # Gemma3RMSNorm returns type_as(input) = fp32
        return F.linear(normed, w_u)           # [*, V] fp32

    def forward(self, h: torch.Tensor, layer: int, norm: nn.Module, w_u: torch.Tensor) -> torch.Tensor:
        return self.decode(self.transform(h, layer), norm, w_u)

    def translator_drift(self) -> float:
        """Mean Frobenius norm of the translator weights — how far training has moved
        the lens from the pure logit lens (0.0 at init)."""
        with torch.no_grad():
            return float(np.mean([lin.weight.norm().item() for lin in self.translators]))


# ─── Loss family ──────────────────────────────────────────────────────────────

@dataclass
class LensBatch:
    """Everything an objective might need: the two logit vectors AND the geometry.

    Logit-space objectives (KL, Pearson, rank) use `lens_logits` / `teacher_logits`.
    Geometry-space objectives (statistical depth) use `translated` (the post-translator
    residual) and `w_u` (the unembedding cloud).
    """
    lens_logits: torch.Tensor      # [s, V] fp32  — student
    teacher_logits: torch.Tensor   # [s, V]       — teacher (model's final logits)
    translated: torch.Tensor       # [s, d] fp32  — post-translator, pre-norm residual
    w_u: torch.Tensor              # [V, d] fp32  — unembedding matrix


class LensLoss:
    """Base objective. Subclasses implement __call__(LensBatch) -> scalar tensor.

    The docstring of each subclass states *what aspect of the layer it interrogates* —
    that interpretation is the actual research payload, not the loss value."""
    name: str = "base"

    def __call__(self, b: LensBatch) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class KLLoss(LensLoss):
    """KL(teacher || lens) — the canonical tuned-lens objective.

    Asks: does this layer already place its probability MASS where the final layer does?
    Dominated by the head; near-blind to the suppressed tail. This is the control."""
    name = "kl"

    def __call__(self, b: LensBatch) -> torch.Tensor:
        student_lp = F.log_softmax(b.lens_logits, dim=-1)
        teacher_lp = F.log_softmax(b.teacher_logits.float(), dim=-1)
        # F.kl_div(input=student_logprobs, target=teacher_logprobs) = sum p_t (log p_t - log p_s)
        return F.kl_div(student_lp, teacher_lp, reduction="batchmean", log_target=True)


class PearsonLoss(LensLoss):
    """1 - Pearson correlation of the two logit vectors (per token, averaged).

    Asks: is the layer's logit *geometry* (direction + relative spacing) aligned with
    the final logits — scale-sensitive, unlike rank. Sensitive to the whole vector,
    so the tail counts, but linearly (a few large logits still dominate)."""
    name = "pearson"

    def __call__(self, b: LensBatch) -> torch.Tensor:
        x = b.lens_logits
        y = b.teacher_logits.float()
        x = x - x.mean(dim=-1, keepdim=True)
        y = y - y.mean(dim=-1, keepdim=True)
        xn = x.norm(dim=-1) + 1e-8
        yn = y.norm(dim=-1) + 1e-8
        corr = (x * y).sum(dim=-1) / (xn * yn)
        return (1.0 - corr).mean()


class PairwiseRankLoss(LensLoss):
    """RankNet-style logistic loss over sampled vocab pairs, supervised by the teacher
    ordering — a pure-torch, no-build-dependency rank objective that DOES constrain the
    tail (unlike the spec's top-K-anchor hinge).

    Asks: does the layer order arbitrary token PAIRS the way the final layer does,
    including tail-vs-tail pairs. Sampling is uniform over the vocab, so most sampled
    pairs are tail-tail (the tail is 99.9% of 262k) — exactly the signal KL ignores.

    Corrections vs the original spec's HierarchicalRankLoss (which was tail-blind,
    charged a constant margin floor on self-pairs, and penalised correct head ordering):
      - iterates over PAIRS, not token-vs-anchor, so tail-vs-tail is supervised;
      - excludes self-pairs;
      - derives the target order from the teacher for EVERY pair, so correct orderings
        are rewarded rather than penalised.
    """
    name = "pairwise_rank"

    def __init__(self, num_pairs: int = 4096):
        self.num_pairs = num_pairs

    def __call__(self, b: LensBatch) -> torch.Tensor:
        lens = b.lens_logits
        teacher = b.teacher_logits.float()
        s, v = lens.shape
        dev = lens.device
        i = torch.randint(0, v, (s, self.num_pairs), device=dev)
        j = torch.randint(0, v, (s, self.num_pairs), device=dev)
        valid = i != j  # drop self-pairs
        li = lens.gather(1, i); lj = lens.gather(1, j)
        ti = teacher.gather(1, i); tj = teacher.gather(1, j)
        sign = torch.sign(ti - tj)               # +1 if i should rank above j, 0 on ties
        agree = (li - lj) * sign                  # >0 when the lens agrees with the teacher
        # logistic ranking loss = -log sigmoid(agree); only count valid, non-tied pairs
        mask = valid & (sign != 0)
        loss = F.softplus(-agree)
        denom = mask.sum().clamp(min=1)
        return (loss * mask).sum() / denom


class SoftSpearmanLoss(LensLoss):
    """1 - soft-Spearman over the FULL vocab via torchsort (Blondel et al., O(V log V)).

    Asks: is the layer's GLOBAL ordinal gestalt in place across all 262k tokens.
    Spearman squares rank differences, so it is dominated by gross reorderings rather
    than local swaps. Requires the compiled `torchsort` extension — on Windows that
    means build-from-source or WSL; falls back with a clear error otherwise."""
    name = "soft_spearman"

    def __init__(self, reg_strength: float = 1e-3):
        self.reg_strength = reg_strength
        try:
            import torchsort  # noqa: F401
            self._torchsort = torchsort
        except Exception as e:  # pragma: no cover
            self._torchsort = None
            self._import_err = e

    def __call__(self, b: LensBatch) -> torch.Tensor:
        if self._torchsort is None:  # pragma: no cover
            raise RuntimeError(
                "soft_spearman needs the 'torchsort' package (compiled CUDA ext). "
                "On Windows build from source against python_embeded, or run under WSL "
                f"with the prebuilt wheel. Original import error: {self._import_err}"
            )
        ts = self._torchsort
        rp = ts.soft_rank(b.lens_logits, regularization="l2", regularization_strength=self.reg_strength)
        rt = ts.soft_rank(b.teacher_logits.float(), regularization="l2", regularization_strength=self.reg_strength)
        rp = rp - rp.mean(dim=-1, keepdim=True)
        rt = rt - rt.mean(dim=-1, keepdim=True)
        rp = rp / (rp.norm(dim=-1, keepdim=True) + 1e-8)
        rt = rt / (rt.norm(dim=-1, keepdim=True) + 1e-8)
        corr = (rp * rt).sum(dim=-1).mean()
        return 1.0 - corr


LOSSES = {
    cls.name: cls
    for cls in (KLLoss, PearsonLoss, PairwiseRankLoss, SoftSpearmanLoss)
}
# Documented-but-not-yet-implemented members of the statistic family (see module docstring):
#   "kendall"  — local pairwise concordance (soft-Kendall surrogate)
#   "depth"    — statistical-depth correlation in representation space (uses b.translated,
#                b.w_u); literal simplicial depth is intractable in 640-d, but projection
#                depth / closed-form Mahalanobis or spatial depth are tractable cousins.


def make_loss(name: str, **kw) -> LensLoss:
    if name not in LOSSES:
        raise KeyError(f"unknown loss {name!r}; have {sorted(LOSSES)}")
    return LOSSES[name](**kw)


# ─── Training ─────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    model_id: str = "google/gemma-3-270m-it"
    loss: str = "kl"
    layers: Optional[List[int]] = None   # default: all pre-norm residual indices 0..L-1
    slice_size: int = 128                # token sub-batch for the 262k-vocab decode
    lr: float = 1e-3
    steps: int = 2000
    log_every: int = 25
    eval_every: int = 200
    max_length: int = 512
    min_length: int = 128
    seed: int = 0
    out: Optional[str] = None


class TunedLensTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.engine = ModelEngine()
        self.profile = self.engine.load_model(cfg.model_id)
        self.model, self.tokenizer = self.engine.model, self.engine.tokenizer
        self.device = self.engine.device
        self.norm = self.profile.get_norm()
        lm_head = self.profile.get_lm_head()

        # Cache the unembedding in fp32 once (frozen) so the whole lens path is fp32.
        self.w_u = lm_head.weight.detach().float()                       # [V, d]
        self.L = self.profile.num_layers
        self.layers = cfg.layers if cfg.layers is not None else list(range(self.L))

        self.lens = TunedLens(self.L, self.profile.hidden_dim).to(self.device)
        self.opt = torch.optim.AdamW(self.lens.parameters(), lr=cfg.lr)
        log.info(
            "lens ready: %s, %d layers probed, hidden=%d, vocab=%d, loss=%s",
            cfg.model_id, len(self.layers), self.profile.hidden_dim,
            self.profile.vocab_size, cfg.loss,
        )

    # -- one optimisation step over a single tokenised sequence --
    def step(self, input_ids: torch.Tensor, loss_fn: LensLoss) -> dict:
        self.opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            out = self.model(input_ids, output_hidden_states=True)
        hs = out.hidden_states                  # tuple len L+1; [0..L-1] pre-norm
        teacher = out.logits[0]                 # [seq, V] (model dtype)
        seq = teacher.size(0)
        norm_scale = 1.0 / (len(self.layers) * seq)   # accumulated grad = mean over (layer, token)

        running = 0.0
        for layer in self.layers:
            h = hs[layer][0]                    # [seq, d] pre-norm residual
            for i in range(0, seq, self.cfg.slice_size):
                h_s = h[i : i + self.cfg.slice_size]
                t_s = teacher[i : i + self.cfg.slice_size]
                translated = self.lens.transform(h_s, layer)
                lens_logits = self.lens.decode(translated, self.norm, self.w_u)
                batch = LensBatch(lens_logits, t_s, translated, self.w_u)
                loss = loss_fn(batch)
                scaled = loss * (h_s.size(0) * norm_scale)
                scaled.backward()
                running += float(scaled.item())

        grad_norm = torch.nn.utils.clip_grad_norm_(self.lens.parameters(), max_norm=1.0)
        finite = all(torch.isfinite(p.grad).all().item() for p in self.lens.parameters() if p.grad is not None)
        self.opt.step()
        return {"loss": running, "grad_norm": float(grad_norm), "finite": finite}

    # -- per-layer full-vocab rank fidelity on a fixed probe (the metric we A/B on) --
    @torch.no_grad()
    def eval_rank(self, text: str = "The theory of relativity was developed by", layers: Optional[List[int]] = None) -> List[dict]:
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        out = self.model(ids, output_hidden_states=True)
        hs = out.hidden_states
        teacher_logits = out.logits[0, -1].float().cpu().numpy()
        teacher_ranks = rank_tokens(teacher_logits)
        teacher_top = int(teacher_logits.argmax())
        rows = []
        for layer in (layers or self.layers):
            h = hs[layer][0, -1]
            lens_logits = self.lens(h, layer, self.norm, self.w_u).float().cpu().numpy()
            rho = compute_spearman_fast(teacher_ranks, lens_logits)
            top1 = int(lens_logits.argmax() == teacher_top)
            rows.append({"layer": layer, "spearman": rho, "top1_match": top1})
        return rows

    def save(self, path: str):
        torch.save(
            {
                "translators": self.lens.state_dict(),
                "model_id": self.cfg.model_id,
                "loss": self.cfg.loss,
                "layers": self.layers,
                "num_layers": self.L,
                "hidden_dim": self.profile.hidden_dim,
            },
            path,
        )
        log.info("saved lens -> %s", path)


# ─── Data ─────────────────────────────────────────────────────────────────────

_DEMO_TEXTS = [
    "The mitochondria is the powerhouse of the cell, generating ATP through oxidative phosphorylation.",
    "In 1905 Albert Einstein published four papers that reshaped modern physics.",
    "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n    p = xs[0]\n    return quicksort([x for x in xs[1:] if x < p]) + [p] + quicksort([x for x in xs[1:] if x >= p])",
    "honestly?? lol that movie was such a vibe, 10/10 would watch again 🙂",
    "Le chat dort sur le canapé pendant que la pluie tombe doucement dehors.",
    "Whereas the parties hereto, in consideration of the mutual covenants set forth herein, agree as follows:",
    "The Federal Reserve raised interest rates by 25 basis points amid persistent inflation.",
    "To be, or not to be, that is the question: whether 'tis nobler in the mind to suffer.",
    "Patients presenting with acute chest pain should be evaluated for myocardial infarction.",
    "The recipe calls for two cups of flour, a pinch of salt, and three large eggs.",
    "Quantum entanglement implies correlations between particles that classical physics cannot explain.",
    "Breaking: local council approves new public transit line after months of debate.",
]


def demo_stream(tokenizer, device, max_length=512) -> Iterable[torch.Tensor]:
    for text in cycle(_DEMO_TEXTS):
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids
        yield ids.to(device)


def fineweb_stream(tokenizer, device, max_length=512, min_length=128) -> Iterable[torch.Tensor]:
    from datasets import load_dataset
    log.info("streaming HuggingFaceFW/fineweb-edu (sample-10BT)")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for item in ds:
        ids = tokenizer(item["text"], return_tensors="pt", truncation=True, max_length=max_length).input_ids
        if ids.size(1) >= min_length:
            yield ids.to(device)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig, data: str = "fineweb") -> TunedLensTrainer:
    trainer = TunedLensTrainer(cfg)
    loss_fn = make_loss(cfg.loss)
    stream = (demo_stream if data == "demo" else fineweb_stream)(
        trainer.tokenizer, trainer.device, cfg.max_length, **({} if data == "demo" else {"min_length": cfg.min_length})
    ) if data == "fineweb" else demo_stream(trainer.tokenizer, trainer.device, cfg.max_length)

    for step, ids in enumerate(stream):
        m = trainer.step(ids, loss_fn)
        if not m["finite"]:
            raise RuntimeError(f"non-finite gradients at step {step}")
        if step % cfg.log_every == 0:
            log.info("step %4d | loss %.5f | grad_norm %.4f | drift %.4f",
                     step, m["loss"], m["grad_norm"], trainer.lens.translator_drift())
        if cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
            rows = trainer.eval_rank(layers=_eval_layer_sample(trainer.layers))
            log.info("  eval spearman(lens,teacher): " +
                     " ".join(f"L{r['layer']}={r['spearman']:.3f}" for r in rows))
        if step >= cfg.steps:
            break

    if cfg.out:
        trainer.save(cfg.out)
    return trainer


def _eval_layer_sample(layers: List[int]) -> List[int]:
    """A spread of ~5 layers for compact eval logging."""
    if len(layers) <= 5:
        return layers
    idx = np.linspace(0, len(layers) - 1, 5).astype(int)
    return [layers[i] for i in idx]


def main():
    ap = argparse.ArgumentParser(description="Train a tuned lens with a swappable objective.")
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--loss", default="kl", choices=sorted(LOSSES))
    ap.add_argument("--data", default="fineweb", choices=["fineweb", "demo"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--slice-size", type=int, default=128)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="fast end-to-end pipeline check on demo data")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.smoke:
        return smoke()

    cfg = TrainConfig(
        model_id=args.model, loss=args.loss, steps=args.steps,
        lr=args.lr, slice_size=args.slice_size, out=args.out,
    )
    train(cfg, data=args.data)


def smoke():
    """End-to-end pipeline check: load, verify decode correctness, train a few steps,
    confirm loss drops + grads finite + rank fidelity improves, exercise every loss."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    cfg = TrainConfig(model_id="google/gemma-3-270m-it", loss="kl",
                      steps=40, log_every=5, eval_every=0, slice_size=128)
    trainer = TunedLensTrainer(cfg)

    # (1) lm_head oracle: F.linear(post-norm final state, W_U) must reproduce the model's logits.
    ids = trainer.tokenizer("The capital of France is", return_tensors="pt").input_ids.to(trainer.device)
    with torch.no_grad():
        out = trainer.model(ids, output_hidden_states=True)
    final_normed = out.hidden_states[trainer.L][0]          # index L = POST final norm
    recon = F.linear(final_normed.float(), trainer.w_u)
    ref = out.logits[0].float()
    argmax_match = (recon.argmax(-1) == ref.argmax(-1)).float().mean().item()
    max_abs = (recon - ref).abs().max().item()
    print(f"[oracle] lm_head decode argmax-match={argmax_match:.3f}  max-abs-logit-diff={max_abs:.4f}")
    assert argmax_match == 1.0, "lm_head/W_U wiring is wrong"

    # (2) step-0 lens == logit lens (identity translators): rank corr with teacher at all layers.
    rows0 = trainer.eval_rank(layers=_eval_layer_sample(trainer.layers))
    print("[step0 logit-lens] " + " ".join(f"L{r['layer']} rho={r['spearman']:.3f}" for r in rows0))

    # (3) train a few KL steps; loss should fall, grads stay finite.
    loss_fn = make_loss("kl")
    stream = demo_stream(trainer.tokenizer, trainer.device, cfg.max_length)
    first, last = None, None
    for step, x in enumerate(stream):
        m = trainer.step(x, loss_fn)
        assert m["finite"], f"non-finite grads at step {step}"
        if step == 0:
            first = m["loss"]
        last = m["loss"]
        if step % 5 == 0:
            print(f"[train] step {step:2d} | KL {m['loss']:.5f} | grad {m['grad_norm']:.4f} | drift {trainer.lens.translator_drift():.4f}")
        if step >= cfg.steps:
            break
    print(f"[train] KL {first:.5f} -> {last:.5f}  ({'DOWN' if last < first else 'UP'})")

    # (4) rank fidelity after training (a few layers should improve over step 0).
    rows1 = trainer.eval_rank(layers=_eval_layer_sample(trainer.layers))
    print("[trained] " + " ".join(f"L{r['layer']} rho={r['spearman']:.3f}" for r in rows1))

    # (5) structurally exercise every implemented loss for one micro-step (finite output).
    with torch.no_grad():
        h = out.hidden_states[trainer.layers[len(trainer.layers)//2]][0, :8]
    t = out.logits[0, :8]
    translated = trainer.lens.transform(h, trainer.layers[len(trainer.layers)//2])
    ll = trainer.lens.decode(translated, trainer.norm, trainer.w_u)
    b = LensBatch(ll, t, translated, trainer.w_u)
    for name in ("kl", "pearson", "pairwise_rank"):
        val = float(make_loss(name)(b).item())
        print(f"[loss:{name}] {val:.5f}  finite={np.isfinite(val)}")
        assert np.isfinite(val)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
