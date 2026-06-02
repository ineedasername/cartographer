"""Hook lifecycle management for capturing model internals.

Consolidates the hook registration/removal pattern found in ~8 files:
  - mental_state.py L56-68, L122-124
  - mri.py L81-90, L139-140
  - rank_scan.py L186-195, L276-286, L335-344
  - refusal_scan.py L72-88
  - medgemma/capture_registers.py L103-118
  - mmfld/capture_errors.py L94-122
  - negation/deberta_cpa.py L129-150
  - negation/capture_sample.py L93-112
"""


class HookManager:
    """Context manager for model hook lifecycle.

    Usage:
        with HookManager() as hm:
            hm.register_o_proj_hooks(model, profile, layers=range(26))
            # ... forward pass ...
            captured = hm.captured  # {layer_idx: tensor}

    Or manually:
        hm = HookManager()
        hm.register_o_proj_hooks(model, profile, layers=range(26))
        # ... forward pass ...
        captured = hm.captured
        hm.remove_all()
    """

    def __init__(self):
        self.handles = []
        self.captured = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove_all()
        return False

    def register_o_proj_hooks(self, model, profile, layers=None):
        """Register forward hooks on o_proj for specified layers.

        Captures the INPUT to o_proj (pre-projection per-head concatenated context)
        at the last position. This is the standard capture pattern.

        Args:
            model: the loaded model
            profile: ModelProfile (or None to use model.model.layers[l].self_attn.o_proj)
            layers: iterable of layer indices (default: all layers)
        """
        if layers is None:
            num_layers = profile.num_layers if profile else model.config.num_hidden_layers
            layers = range(num_layers)

        for l in layers:
            if profile is not None:
                o_proj = profile.get_o_proj(l)
            else:
                o_proj = model.model.layers[l].self_attn.o_proj

            h = o_proj.register_forward_hook(make_o_proj_hook(l, self.captured))
            self.handles.append(h)

    def register_custom_hook(self, module, hook_fn):
        """Register an arbitrary hook on a module.

        Args:
            module: nn.Module to hook
            hook_fn: callable(module, args, output) -> None
        """
        h = module.register_forward_hook(hook_fn)
        self.handles.append(h)
        return h

    def remove_all(self):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def clear_captured(self):
        """Clear captured tensors (between generation steps)."""
        self.captured.clear()


def make_o_proj_hook(layer_idx, captured_dict):
    """Create a forward hook that captures o_proj input for a given layer.

    Source: mental_state.py L58-61

    This is the universal hook factory used across the codebase.
    The hook captures args[0] (the input to o_proj) which contains
    the concatenated per-head attention output before projection.

    Args:
        layer_idx: layer index (used as key in captured_dict)
        captured_dict: dict to store captured tensors

    Returns:
        hook function compatible with register_forward_hook
    """
    def hook_fn(module, args, output):
        captured_dict[layer_idx] = args[0].detach()
    return hook_fn
