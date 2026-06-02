"""
cartographer.intervene — Reusable intervention hooks for neural network analysis.

Source projects: medgemma/, negation/, mmfld/
"""

from .hooks import (
    InterventionHook,
    HeadScaleHook,
    HeadZeroHook,
    HeadSubtractHook,
    DirectionalNudgeHook,
    MLPNudgeHook,
    LogitBiasHook,
    compute_nudge_direction,
    compute_mlp_nudge_direction,
)

from .conditional import (
    ConditionalGate,
    GoldenVectorInjection,
)

from .sweep import (
    SweepResult,
    SweepReport,
    run_sweep,
    make_strength_range,
)

__all__ = [
    'InterventionHook',
    'HeadScaleHook',
    'HeadZeroHook',
    'HeadSubtractHook',
    'DirectionalNudgeHook',
    'MLPNudgeHook',
    'LogitBiasHook',
    'compute_nudge_direction',
    'compute_mlp_nudge_direction',
    'ConditionalGate',
    'GoldenVectorInjection',
    'SweepResult',
    'SweepReport',
    'run_sweep',
    'make_strength_range',
]
