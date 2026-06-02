"""Model loading and architecture probing.

ModelProfile discovers a model's internal structure at load time by probing
the model object for known patterns. This replaces hardcoded references like
model.model.layers[i].self_attn.o_proj across all tools.

New model families require adding patterns to the probe — not changing tool code.

Lifted from:
  - python/engine/model.py L22-324 (ARCHITECTURE_PATTERNS, ModelProfile, ModelEngine)
  - projection_map.py L96-119 (load_model_simple — bf16 with fallback)
  - rank_scan.py L57-71 (resolve_inner_model — multimodal wrapper unwrapping)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

log = logging.getLogger("cartographer.inspect.model")


# ─── Architecture patterns ────────────────────────────────────────────────────
# Source: python/engine/model.py L22-95

ARCHITECTURE_PATTERNS = [
    {
        "name": "gemma",
        "families": ["gemma", "gemma2", "gemma3"],
        "layer_path": "model.layers",
        "norm_path": "model.norm",
        "lm_head_path": "lm_head",
        "attn_attr": "self_attn",
        "o_proj_attr": "o_proj",
    },
    {
        "name": "llama",
        "families": ["llama", "llama2", "llama3", "codellama", "tinyllama"],
        "layer_path": "model.layers",
        "norm_path": "model.norm",
        "lm_head_path": "lm_head",
        "attn_attr": "self_attn",
        "o_proj_attr": "o_proj",
    },
    {
        "name": "mistral",
        "families": ["mistral", "mixtral"],
        "layer_path": "model.layers",
        "norm_path": "model.norm",
        "lm_head_path": "lm_head",
        "attn_attr": "self_attn",
        "o_proj_attr": "o_proj",
    },
    {
        "name": "phi",
        "families": ["phi", "phi3", "phi-3"],
        "layer_path": "model.layers",
        "norm_path": "model.final_layernorm",
        "lm_head_path": "lm_head",
        "attn_attr": "self_attn",
        "o_proj_attr": "dense",
    },
    {
        "name": "gpt_neox",
        "families": ["gpt_neox", "pythia", "stablelm"],
        "layer_path": "gpt_neox.layers",
        "norm_path": "gpt_neox.final_layer_norm",
        "lm_head_path": "embed_out",
        "attn_attr": "attention",
        "o_proj_attr": "dense",
    },
    {
        "name": "falcon",
        "families": ["falcon", "refinedweb"],
        "layer_path": "transformer.h",
        "norm_path": "transformer.ln_f",
        "lm_head_path": "lm_head",
        "attn_attr": "self_attention",
        "o_proj_attr": "dense",
    },
    {
        "name": "qwen2",
        "families": ["qwen2", "qwen", "qwen3"],
        "layer_path": "model.layers",
        "norm_path": "model.norm",
        "lm_head_path": "lm_head",
        "attn_attr": "self_attn",
        "o_proj_attr": "o_proj",
    },
    {
        "name": "starcoder",
        "families": ["starcoder", "gpt_bigcode"],
        "layer_path": "transformer.h",
        "norm_path": "transformer.ln_f",
        "lm_head_path": "lm_head",
        "attn_attr": "attn",
        "o_proj_attr": "c_proj",
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_attr(obj: Any, dotted_path: str) -> Optional[Any]:
    """Resolve a dotted attribute path like 'model.layers' on an object."""
    parts = dotted_path.split(".")
    current = obj
    for part in parts:
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


# ─── ModelProfile ─────────────────────────────────────────────────────────────
# Source: python/engine/model.py L109-266

@dataclass
class ModelProfile:
    """Discovered model architecture profile.

    Built at load time by probing the model object. Provides uniform access
    to architecture internals regardless of model family.
    """

    model_id: str
    num_layers: int
    num_heads: int
    num_kv_heads: int
    hidden_dim: int
    vocab_size: int
    head_dim: int
    architecture: str  # "gemma", "llama", "mistral", "phi", etc.

    # Non-serialized references (used internally)
    _norm_module: Optional[nn.Module] = field(default=None, repr=False)
    _lm_head: Optional[nn.Module] = field(default=None, repr=False)
    _layers: Optional[Any] = field(default=None, repr=False)
    _pattern: Optional[Dict] = field(default=None, repr=False)

    def get_norm(self) -> nn.Module:
        """Get the final layer norm module."""
        if self._norm_module is None:
            raise RuntimeError("Model profile not fully initialized (no norm module)")
        return self._norm_module

    def get_lm_head(self) -> nn.Module:
        """Get the language model head (output projection)."""
        if self._lm_head is None:
            raise RuntimeError("Model profile not fully initialized (no lm_head)")
        return self._lm_head

    def get_layer(self, index: int) -> nn.Module:
        """Get a specific transformer layer by index."""
        if self._layers is None:
            raise RuntimeError("Model profile not fully initialized (no layers)")
        return self._layers[index]

    def get_attn(self, layer_index: int) -> nn.Module:
        """Get the attention module for a given layer."""
        layer = self.get_layer(layer_index)
        attn_attr = self._pattern["attn_attr"] if self._pattern else "self_attn"
        return getattr(layer, attn_attr)

    def get_o_proj(self, layer_index: int) -> nn.Module:
        """Get the output projection for a given layer's attention."""
        attn = self.get_attn(layer_index)
        o_proj_attr = self._pattern["o_proj_attr"] if self._pattern else "o_proj"
        return getattr(attn, o_proj_attr)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON transport (excludes module references)."""
        return {
            "model_id": self.model_id,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "hidden_dim": self.hidden_dim,
            "vocab_size": self.vocab_size,
            "head_dim": self.head_dim,
            "architecture": self.architecture,
        }

    @classmethod
    def from_model(cls, model: nn.Module, model_id: str) -> "ModelProfile":
        """Probe a loaded model to discover its architecture.

        Tries known patterns for different model families. Falls back to
        config-based discovery if no pattern matches.
        """
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError("Model has no config attribute — cannot probe architecture")

        # Extract basic dimensions from config
        num_layers = getattr(config, "num_hidden_layers", 0)
        num_heads = getattr(config, "num_attention_heads", 0)
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        hidden_dim = getattr(config, "hidden_size", 0)
        vocab_size = getattr(config, "vocab_size", 0)
        # Prefer explicit head_dim from config (Gemma 3 uses 256, not 1152/4=288)
        head_dim = getattr(config, "head_dim", hidden_dim // num_heads if num_heads > 0 else 0)

        # Identify architecture family from config
        model_type = getattr(config, "model_type", "").lower()
        architectures = getattr(config, "architectures", [])
        arch_str = architectures[0].lower() if architectures else model_type

        # Try to match a known pattern
        matched_pattern = None
        matched_name = "unknown"

        for pattern in ARCHITECTURE_PATTERNS:
            for family in pattern["families"]:
                if family in model_type or family in arch_str:
                    layers = _resolve_attr(model, pattern["layer_path"])
                    norm = _resolve_attr(model, pattern["norm_path"])
                    lm_head = _resolve_attr(model, pattern["lm_head_path"])

                    if layers is not None and norm is not None and lm_head is not None:
                        matched_pattern = pattern
                        matched_name = pattern["name"]
                        break
            if matched_pattern:
                break

        # Fallback: try common patterns in order
        if matched_pattern is None:
            log.warning("No known pattern matched for model_type=%s. Trying fallback.", model_type)
            for pattern in ARCHITECTURE_PATTERNS:
                layers = _resolve_attr(model, pattern["layer_path"])
                norm = _resolve_attr(model, pattern["norm_path"])
                lm_head = _resolve_attr(model, pattern["lm_head_path"])
                if layers is not None and norm is not None and lm_head is not None:
                    matched_pattern = pattern
                    matched_name = pattern["name"] + "_compat"
                    log.info("Fallback matched pattern: %s", matched_name)
                    break

        if matched_pattern is None:
            raise ValueError(
                f"Could not probe architecture for model_type={model_type}. "
                f"None of the known patterns matched. "
                f"Add a new pattern to ARCHITECTURE_PATTERNS."
            )

        # Resolve references
        layers = _resolve_attr(model, matched_pattern["layer_path"])
        norm = _resolve_attr(model, matched_pattern["norm_path"])
        lm_head = _resolve_attr(model, matched_pattern["lm_head_path"])

        actual_layers = len(layers) if hasattr(layers, "__len__") else num_layers

        log.info(
            "Probed %s: %d layers, %d heads (%d kv), hidden=%d, vocab=%d",
            matched_name, actual_layers, num_heads, num_kv_heads, hidden_dim, vocab_size,
        )

        return cls(
            model_id=model_id,
            num_layers=actual_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            head_dim=head_dim,
            architecture=matched_name,
            _norm_module=norm,
            _lm_head=lm_head,
            _layers=layers,
            _pattern=matched_pattern,
        )


# ─── ModelEngine ──────────────────────────────────────────────────────────────
# Source: python/engine/model.py L269-324

class ModelEngine:
    """Manages model lifecycle: loading, unloading, device placement.

    Holds the model, tokenizer, and ModelProfile.
    """

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.profile: Optional[ModelProfile] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load_model(self, model_id: str, dtype=None) -> ModelProfile:
        """Load a model from HuggingFace and probe its architecture.

        Uses bfloat16 by default (required for Gemma 3 RMSNorm — float16 produces NaN).
        Falls back to float32 if bfloat16 is unavailable.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Determine dtype
        if dtype is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float32

        log.info("Loading model: %s (device=%s, dtype=%s)", model_id, self.device, dtype)

        self.unload()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", dtype=dtype
        )
        self.model.eval()

        self.profile = ModelProfile.from_model(self.model, model_id)
        log.info("Model loaded: %s (%s)", model_id, self.profile.architecture)
        return self.profile

    def unload(self) -> None:
        """Unload the current model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.profile = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("Model unloaded.")


# ─── Simple loader (backward compat) ─────────────────────────────────────────
# Source: projection_map.py L96-119

def load_model_simple(model_id="google/gemma-3-1b-it"):
    """Load a model with bf16 dtype and return (model, tokenizer, device, num_layers, num_heads).

    This is the original projection_map.py load_model() — a lightweight loader
    for scripts that don't need the full ModelEngine/ModelProfile.

    IMPORTANT: Uses bfloat16 by default. Gemma 3's RMSNorm produces NaN with float16.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", dtype=dtype
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    return model, tokenizer, device, num_layers, num_heads


# ─── Inner model resolution ──────────────────────────────────────────────────
# Source: rank_scan.py L57-71

def resolve_inner_model(model, num_layers=None, num_heads=None):
    """Resolve inner model path, norm, lm_head, head_dim.

    Handles multimodal wrappers (e.g., MedGemma) where the text model is
    nested inside model.model.language_model.

    Returns:
        (inner_model, final_norm, lm_head, head_dim)
    """
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        inner = model.model.language_model
    elif hasattr(model, 'model') and hasattr(model.model, 'norm'):
        inner = model.model
    else:
        inner = model.model

    cfg = model.config
    if hasattr(cfg, 'text_config'):
        cfg = cfg.text_config
    head_dim = cfg.head_dim if hasattr(cfg, 'head_dim') else cfg.hidden_size // cfg.num_attention_heads

    return inner, inner.norm, model.lm_head, head_dim
