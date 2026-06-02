"""
Conditional intervention wrappers — gate interventions on runtime signals.

Source attribution:
  - ConditionalGate: negation/intervene_h2.py L126-143 (L07-H4 monitor pattern)
  - GoldenVectorInjection: mmfld/intervene_golden.py L224-242
"""

from __future__ import annotations

import torch
from typing import Callable, Dict, List, Optional, Set

from .hooks import InterventionHook


# ---------------------------------------------------------------------------
# ConditionalGate — wraps any hook with a detector that gates it
# Source: negation/intervene_h2.py L126-143
# ---------------------------------------------------------------------------

class ConditionalGate(InterventionHook):
    """Wraps an intervention hook so it only fires when a monitor condition is met.

    The monitor hook runs at an earlier layer and sets a gate signal.
    The wrapped intervention checks the gate before applying.

    Args:
        intervention: The InterventionHook to gate.
        monitor_layer: Layer index for the monitor hook.
        monitor_head: Head index to monitor.
        head_dim: Dimension per head.
        threshold: If provided, gate fires when monitor head's norm exceeds this.
            If None, gate always fires (useful for subclassing).
        monitor_accessor: Callable(model, layer_idx) -> module to hook.
        monitor_position: Sequence position to read (0 = CLS, -1 = last).
        invert: If True, fire when norm is BELOW threshold (head is quiet).
    """

    def __init__(self, intervention: InterventionHook,
                 monitor_layer: int, monitor_head: int, head_dim: int,
                 threshold: Optional[float] = None,
                 monitor_accessor: Optional[Callable] = None,
                 monitor_position: int = 0,
                 invert: bool = False):
        super().__init__()
        self.intervention = intervention
        self.monitor_layer = monitor_layer
        self.monitor_head = monitor_head
        self.head_dim = head_dim
        self.threshold = threshold
        self.monitor_accessor = monitor_accessor
        self.monitor_position = monitor_position
        self.invert = invert
        self._gate_signal: Dict[str, object] = {}

    def install(self, model, profile=None):
        gate = self._gate_signal
        head = self.monitor_head
        head_dim = self.head_dim
        position = self.monitor_position
        threshold = self.threshold
        invert = self.invert

        def monitor_hook(module, args, output):
            ctx = args[0].detach()
            h_start = head * head_dim
            h_vec = ctx[:, position, h_start:h_start + head_dim]
            norm = torch.norm(h_vec).item()
            gate['norm'] = norm

            if threshold is not None:
                fired = norm < threshold if invert else norm > threshold
            else:
                fired = True
            gate['fired'] = fired

        if self.monitor_accessor:
            monitor_target = self.monitor_accessor(model, self.monitor_layer)
        else:
            monitor_target = model.layers[self.monitor_layer].self_attn.o_proj

        self._handles.append(
            monitor_target.register_forward_hook(monitor_hook)
        )

        # Wrap the intervention's install — we monkey-patch the hook_fn
        # to check our gate. The simplest approach: install normally,
        # then wrap each handle's hook.
        #
        # Actually, we use a simpler pattern: store gate ref on intervention
        # and install it. The intervention checks gate['fired'] if present.
        self.intervention._gate_signal = gate
        self.intervention.install(model, profile)
        self._handles.extend(self.intervention._handles)
        self.intervention._handles = []  # we own them now

        return self

    def remove(self):
        super().remove()
        # Don't double-remove; we already took ownership
        self.intervention._handles.clear()


class _GatedDirectionalNudgeHook(InterventionHook):
    """DirectionalNudgeHook variant that checks a gate signal before firing.

    This is the pattern from negation/intervene_h2.py where the nudge hook
    checks monitor_signal['fired'] before applying.

    Prefer using ConditionalGate wrapping a DirectionalNudgeHook for new code.
    This exists for back-compat with the exact original pattern.
    """

    def __init__(self, layer, head, head_dim, direction, strength=1.0,
                 position=0, gate_signal=None, layer_accessor=None):
        super().__init__()
        self.layer = layer
        self.head = head
        self.head_dim = head_dim
        self.direction = direction
        self.strength = strength
        self.position = position
        self._gate_signal = gate_signal or {}
        self.layer_accessor = layer_accessor

    def install(self, model, profile=None):
        h_start = self.head * self.head_dim
        h_end = h_start + self.head_dim
        direction = self.direction
        strength = self.strength
        position = self.position
        gate = self._gate_signal

        def hook_fn(module, args):
            if strength == 0:
                return args
            if not gate.get('fired', True):
                return args
            # Reset for next example
            gate['fired'] = False

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
# GoldenVectorInjection — inject golden vectors when heads are quiet
# Source: mmfld/intervene_golden.py L224-242
# ---------------------------------------------------------------------------

class GoldenVectorInjection(InterventionHook):
    """Inject precomputed 'golden' activation vectors when target heads are quiet.

    When ALL target heads' norms at the monitored position fall below the
    threshold, inject scaled golden vectors into those heads' slots.

    Args:
        layer: Layer index.
        target_heads: List of head indices to monitor and inject into.
        head_dim: Dimension per head.
        golden_vectors: Dict[head_idx, Tensor[head_dim]] — the golden activations.
        threshold: Norm threshold; inject when all heads are below this.
        inject_scale: Scale factor for the injected golden vectors.
        position: Sequence position to check/inject (-1 = last).
        layer_accessor: Callable(model, layer_idx) -> target module.
    """

    def __init__(self, layer: int, target_heads: List[int], head_dim: int,
                 golden_vectors: Dict[int, torch.Tensor],
                 threshold: float = 1.0, inject_scale: float = 1.0,
                 position: int = -1,
                 layer_accessor: Optional[Callable] = None):
        super().__init__()
        self.layer = layer
        self.target_heads = target_heads
        self.head_dim = head_dim
        self.golden_vectors = golden_vectors
        self.threshold = threshold
        self.inject_scale = inject_scale
        self.position = position
        self.layer_accessor = layer_accessor
        self.injection_count = 0

    def install(self, model, profile=None):
        heads = self.target_heads
        head_dim = self.head_dim
        golden = {h: v.to(next(model.parameters()).device)
                  for h, v in self.golden_vectors.items()}
        threshold = self.threshold
        scale = self.inject_scale
        pos = self.position

        # Use list for mutability in closure
        count = [0]
        self._injection_counter = count

        def hook_fn(module, args):
            modified = args[0].clone()
            # Check if all target heads are quiet
            quiet = True
            for h in heads:
                h_vec = modified[0, pos, h * head_dim:(h + 1) * head_dim]
                if torch.norm(h_vec).item() > threshold:
                    quiet = False
                    break

            if quiet:
                count[0] += 1
                for h in heads:
                    modified[0, pos, h * head_dim:(h + 1) * head_dim] += scale * golden[h]

            return (modified,) + args[1:]

        if self.layer_accessor:
            target = self.layer_accessor(model, self.layer)
        else:
            target = model.layers[self.layer].self_attn.o_proj

        self._handles.append(target.register_forward_pre_hook(hook_fn))
        return self

    @property
    def injection_count(self):
        if hasattr(self, '_injection_counter'):
            return self._injection_counter[0]
        return self._injection_count

    @injection_count.setter
    def injection_count(self, val):
        self._injection_count = val
        if hasattr(self, '_injection_counter'):
            self._injection_counter[0] = val
