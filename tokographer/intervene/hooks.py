"""
Reusable intervention hooks for attention heads, MLPs, and logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Callable


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class InterventionHook:
    """Base class for all intervention hooks."""

    def __init__(self):
        self._handles: List = []

    def install(self, model, profile=None):
        """Register hook(s) on the model. Subclasses must implement."""
        raise NotImplementedError

    def remove(self):
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False


# ---------------------------------------------------------------------------
# HeadScaleHook — scale specific heads' contributions before o_proj
# ---------------------------------------------------------------------------

class HeadScaleHook(InterventionHook):
    """Scale attention head outputs before the output projection.

    Args:
        layer: Layer index.
        head_scales: Dict mapping head index -> scale factor.
        head_dim: Dimension per head.
        position: Which sequence position to modify (-1 = last, None = all).
        layer_accessor: Callable(model, layer_idx) -> layer module.
            Default assumes model.layers[layer].self_attn.o_proj.
    """

    def __init__(self, layer: int, head_scales: Dict[int, float],
                 head_dim: int, position: Optional[int] = -1,
                 layer_accessor: Optional[Callable] = None):
        super().__init__()
        self.layer = layer
        self.head_scales = head_scales
        self.head_dim = head_dim
        self.position = position
        self.layer_accessor = layer_accessor

    def install(self, model, profile=None):
        scales = self.head_scales
        head_dim = self.head_dim
        position = self.position

        def hook_fn(module, args):
            x = args[0]
            for h, scale in scales.items():
                start = h * head_dim
                end = start + head_dim
                if position is not None:
                    x[:, position, start:end] = x[:, position, start:end] * scale
                else:
                    x[:, :, start:end] = x[:, :, start:end] * scale
            return (x,) + args[1:]

        if self.layer_accessor:
            target = self.layer_accessor(model, self.layer)
        else:
            target = model.layers[self.layer].self_attn.o_proj

        self._handles.append(target.register_forward_pre_hook(hook_fn))
        return self


# ---------------------------------------------------------------------------
# HeadZeroHook — zero out specific heads before o_proj
# ---------------------------------------------------------------------------

class HeadZeroHook(InterventionHook):
    """Zero out specific attention heads' slots before the output projection.

    Args:
        layer: Layer index.
        heads: Set/list of head indices to zero.
        head_dim: Dimension per head.
        layer_accessor: Callable(model, layer_idx) -> target module.
    """

    def __init__(self, layer: int, heads: Set[int], head_dim: int,
                 layer_accessor: Optional[Callable] = None):
        super().__init__()
        self.layer = layer
        self.heads = set(heads)
        self.head_dim = head_dim
        self.layer_accessor = layer_accessor

    def install(self, model, profile=None):
        suppress_set = self.heads
        head_dim = self.head_dim

        def hook_fn(module, args):
            if not suppress_set:
                return args
            hidden = args[0].clone()
            for h_idx in suppress_set:
                start = h_idx * head_dim
                hidden[:, :, start:start + head_dim] = 0.0
            return (hidden,) + args[1:]

        if self.layer_accessor:
            target = self.layer_accessor(model, self.layer)
        else:
            target = model.layers[self.layer].self_attn.o_proj

        self._handles.append(target.register_forward_pre_hook(hook_fn))
        return self


# ---------------------------------------------------------------------------
# HeadSubtractHook — subtract one head's projected contribution after o_proj
# ---------------------------------------------------------------------------

class HeadSubtractHook(InterventionHook):
    """Subtract a single head's contribution after the output projection.

    Computes o_proj(zero_except_head) and subtracts from full output.

    Args:
        layer: Layer index.
        head: Head index to subtract.
        head_dim: Dimension per head.
        o_proj_weight: The [hidden, hidden] weight of o_proj. If None,
            extracted from model during install().
        layer_accessor: Callable(model, layer_idx) -> (target_module, o_weight).
    """

    def __init__(self, layer: int, head: int, head_dim: int,
                 o_proj_weight: Optional[torch.Tensor] = None,
                 layer_accessor: Optional[Callable] = None):
        super().__init__()
        self.layer = layer
        self.head = head
        self.head_dim = head_dim
        self.o_proj_weight = o_proj_weight
        self.layer_accessor = layer_accessor

    def install(self, model, profile=None):
        head = self.head
        head_dim = self.head_dim

        if self.layer_accessor:
            target, o_weight = self.layer_accessor(model, self.layer)
        else:
            target = model.layers[self.layer].self_attn.o_proj
            o_weight = target.weight.detach()

        o_weight = self.o_proj_weight if self.o_proj_weight is not None else o_weight

        def hook_fn(module, args, output):
            h_input = torch.zeros_like(args[0])
            h_input[:, :, head * head_dim:(head + 1) * head_dim] = \
                args[0][:, :, head * head_dim:(head + 1) * head_dim]
            contribution = F.linear(h_input, o_weight)
            return output - contribution

        self._handles.append(target.register_forward_hook(hook_fn))
        return self


# ---------------------------------------------------------------------------
# DirectionalNudgeHook — nudge a head's slot in a computed direction
# ---------------------------------------------------------------------------

class DirectionalNudgeHook(InterventionHook):
    """Add a directional nudge to a specific head's slot before o_proj.

    Args:
        layer: Layer index.
        head: Head index.
        head_dim: Dimension per head.
        direction: [head_dim] unit vector (the nudge direction).
        strength: Scalar multiplier.
        position: Sequence position to nudge (0 = CLS, -1 = last, None = all).
        layer_accessor: Callable(model, layer_idx) -> target module.
    """

    def __init__(self, layer: int, head: int, head_dim: int,
                 direction: torch.Tensor, strength: float = 1.0,
                 position: int = 0,
                 layer_accessor: Optional[Callable] = None):
        super().__init__()
        self.layer = layer
        self.head = head
        self.head_dim = head_dim
        self.direction = direction
        self.strength = strength
        self.position = position
        self.layer_accessor = layer_accessor

    def install(self, model, profile=None):
        h_start = self.head * self.head_dim
        h_end = h_start + self.head_dim
        direction = self.direction
        strength = self.strength
        position = self.position

        def hook_fn(module, args):
            if strength == 0:
                return args
            hidden = args[0].clone()
            if position is not None:
                hidden[:, position, h_start:h_end] += strength * direction
            else:
                hidden[:, :, h_start:h_end] += strength * direction
            return (hidden,) + args[1:]

        if self.layer_accessor:
            target = self.layer_accessor(model, self.layer)
        else:
            target = model.layers[self.layer].self_attn.o_proj

        self._handles.append(target.register_forward_pre_hook(hook_fn))
        return self


# ---------------------------------------------------------------------------
# MLPNudgeHook — nudge hidden state after MLP output
# ---------------------------------------------------------------------------

class MLPNudgeHook(InterventionHook):
    """Add a directional nudge to the MLP/LayerNorm output.

    Args:
        layer: Layer index.
        direction: [hidden_dim] unit vector.
        strength: Scalar multiplier.
        position: Sequence position to nudge (0 = CLS).
        layer_accessor: Callable(model, layer_idx) -> target module
            (should return the post-MLP LayerNorm or equivalent).
    """

    def __init__(self, layer: int, direction: torch.Tensor,
                 strength: float = 1.0, position: int = 0,
                 layer_accessor: Optional[Callable] = None):
        super().__init__()
        self.layer = layer
        self.direction = direction
        self.strength = strength
        self.position = position
        self.layer_accessor = layer_accessor

    def install(self, model, profile=None):
        direction = self.direction
        strength = self.strength
        position = self.position

        def hook_fn(module, args, output):
            if strength == 0:
                return output
            modified = output.clone()
            if position is not None:
                modified[:, position, :] += strength * direction
            else:
                modified += strength * direction
            return modified

        if self.layer_accessor:
            target = self.layer_accessor(model, self.layer)
        else:
            target = model.layers[self.layer].output.LayerNorm

        self._handles.append(target.register_forward_hook(hook_fn))
        return self


# ---------------------------------------------------------------------------
# LogitBiasHook — modify final logits (four modes)
# ---------------------------------------------------------------------------

class LogitBiasHook(InterventionHook):
    """Apply bias to output logits. Four modes:

    - 'flat': boost 2nd-highest among target tokens.
    - 'suppress_top': subtract bias from top target token.
    - 'boost_set': add bias to a specified set of token ids.
    - 'head_guided': boost a token selected by a callback.

    Args:
        target_token_ids: Dict[label, token_id] for the answer tokens.
        bias: Bias magnitude.
        mode: One of 'flat', 'suppress_top', 'boost_set', 'head_guided'.
        boost_ids: Token ids to boost (for 'boost_set' mode).
        vote_fn: Callable(input_ids) -> label (for 'head_guided' mode).
        logit_accessor: Callable(model) -> module whose output is logits.
    """

    def __init__(self, target_token_ids: Dict[str, int], bias: float = 1.0,
                 mode: str = 'flat', boost_ids: Optional[List[int]] = None,
                 vote_fn: Optional[Callable] = None,
                 logit_accessor: Optional[Callable] = None):
        super().__init__()
        self.target_token_ids = target_token_ids
        self.bias = bias
        self.mode = mode
        self.boost_ids = boost_ids or []
        self.vote_fn = vote_fn
        self.logit_accessor = logit_accessor

    def install(self, model, profile=None):
        target_ids = self.target_token_ids
        bias = self.bias
        mode = self.mode
        boost_ids = self.boost_ids
        vote_fn = self.vote_fn

        def hook_fn(module, args, output):
            logits = output.clone() if not isinstance(output, tuple) else output[0].clone()

            if mode == 'flat':
                sorted_answers = sorted(target_ids.items(),
                                        key=lambda x: logits[0, -1, x[1]].item(), reverse=True)
                logits[0, -1, sorted_answers[1][1]] += bias

            elif mode == 'suppress_top':
                sorted_answers = sorted(target_ids.items(),
                                        key=lambda x: logits[0, -1, x[1]].item(), reverse=True)
                logits[0, -1, sorted_answers[0][1]] -= bias

            elif mode == 'boost_set':
                for tid in boost_ids:
                    logits[0, -1, tid] += bias

            elif mode == 'head_guided':
                if vote_fn is not None:
                    input_ids = args[0] if len(args) > 0 else None
                    favored = vote_fn(input_ids)
                    logits[0, -1, target_ids[favored]] += bias

            if isinstance(output, tuple):
                return (logits,) + output[1:]
            return logits

        if self.logit_accessor:
            target = self.logit_accessor(model)
        else:
            target = model.lm_head

        self._handles.append(target.register_forward_hook(hook_fn))
        return self


# ---------------------------------------------------------------------------
# Standalone direction-computation functions
# ---------------------------------------------------------------------------

def compute_nudge_direction(model, target_head: int, target_layer: int,
                            from_class: int, to_class: int,
                            dense_accessor: Optional[Callable] = None,
                            classifier_weight: Optional[torch.Tensor] = None,
                            pooler_weight: Optional[torch.Tensor] = None,
                            head_dim: int = 64) -> torch.Tensor:
    """Compute a unit direction in head-space that pushes logits from one class toward another.

    Linear approximation: direction = (cls_to @ pooler @ dense[:, head_slice])
                                    - (cls_from @ pooler @ dense[:, head_slice])

    Args:
        model: The model (used to extract weights if accessors not provided).
        target_head: Head index.
        target_layer: Layer index.
        from_class: Class index to push away from.
        to_class: Class index to push toward.
        dense_accessor: Callable(model, layer) -> dense weight [hidden, hidden].
        classifier_weight: [num_classes, hidden] classifier weight.
        pooler_weight: [hidden, hidden] pooler weight.
        head_dim: Dimension per head.

    Returns:
        Unit [head_dim] direction tensor on model's device.
    """
    if dense_accessor:
        dense_w = dense_accessor(model, target_layer)
    else:
        layer = model.deberta.encoder.layer[target_layer]
        dense_w = layer.attention.output.dense.weight.detach()

    h_start = target_head * head_dim
    h2_to_hidden = dense_w[:, h_start:h_start + head_dim]

    cls_w = classifier_weight if classifier_weight is not None else model.classifier.weight.detach()
    pool_w = pooler_weight if pooler_weight is not None else model.pooler.dense.weight.detach()

    direction_from = cls_w[from_class] @ pool_w @ h2_to_hidden
    direction_to = cls_w[to_class] @ pool_w @ h2_to_hidden

    nudge = direction_to - direction_from
    nudge = nudge / (torch.norm(nudge) + 1e-8)
    return nudge.to(next(model.parameters()).device)


def compute_mlp_nudge_direction(model, from_class: int, to_class: int,
                                classifier_weight: Optional[torch.Tensor] = None,
                                pooler_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute a unit direction in hidden-space for MLP-level nudging.

    Simpler than head-space: cls @ pooler gives [hidden] direction directly.

    Returns:
        Unit [hidden_dim] direction tensor on model's device.
    """
    cls_w = classifier_weight if classifier_weight is not None else model.classifier.weight.detach()
    pool_w = pooler_weight if pooler_weight is not None else model.pooler.dense.weight.detach()

    direction_from = cls_w[from_class] @ pool_w
    direction_to = cls_w[to_class] @ pool_w

    nudge = direction_to - direction_from
    nudge = nudge / (torch.norm(nudge) + 1e-8)
    return nudge.to(next(model.parameters()).device)
