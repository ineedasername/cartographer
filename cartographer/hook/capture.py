"""Capture profiles and data structures for hook-based model inspection.

Defines what to record from a forward pass and how to store it.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class CaptureProfile(Enum):
    """What to capture during a forward pass."""
    ARGMAX = auto()          # Top-1 token per head per layer
    RICH = auto()            # Top-k tokens + entropy + mass per cell
    FULL_SOFTMAX = auto()    # Complete 262k probability distribution
    TELESCOPE = auto()       # 23 capture points per layer (full attention internals)


@dataclass
class StepCapture:
    """Captured data for a single generation step."""
    step: int
    token: str
    token_id: int
    prob: float
    context_len: int
    hidden_states: Optional[Dict[int, Any]] = field(default=None, repr=False)
    attn_outputs: Optional[Dict[int, Any]] = field(default=None, repr=False)
