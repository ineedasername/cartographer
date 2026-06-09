"""Hook management, capture profiles, and projection functions."""

from tokographer.hook.manager import HookManager
from tokographer.hook.project import (
    project_to_token,
    project_to_token_rich,
    project_to_softmax,
    project_full,
    project_batch_topk,
    project_hidden_topk,
    project_heads_batch,
    project_head_topk,
    project_kv_topk,
    project_all_heads_all_positions,
    project_to_constrained_vocab,
    project_to_constrained_topk,
)
