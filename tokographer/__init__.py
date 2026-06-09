"""
Tokographer — a library for inspecting transformer internals.

Projects hidden states and per-head attention outputs through the model's
final norm and unembedding (logit-lens style), then measures how the resulting
token rankings shift across layers and heads. Architecture-agnostic: model
structure is discovered at load time rather than hardcoded.

Consolidated from a set of standalone interpretability tools. Provides model
inspection, hooking, projection, rank-displacement analysis, decomposition,
intervention, evaluation, and visualization.
"""

__version__ = "0.1.0"
