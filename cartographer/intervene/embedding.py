"""Embedding vector injection and restoration at runtime.

Inject a custom vector into a model's embedding matrix for a specific token,
then optionally restore the original. Used for perturbation experiments where
a single token's embedding is trained or modified while all other weights
remain frozen.

Lifted from:
  - reference/unused_token_training/tc_infer.py L29-38 (injection pattern)

The core operation is trivial:
    embeddings.weight[target_id] = custom_vector

But wrapping it properly matters: save the original, handle string→id conversion,
support context manager for automatic restore, type/device matching.
"""

import torch


def inject_embedding_vector(model, token_id_or_str, custom_vector, tokenizer=None):
    """Replace a token's embedding vector at runtime.

    Args:
        model: HuggingFace model
        token_id_or_str: int token_id, or str token (e.g., '<unused42>')
        custom_vector: tensor [embedding_dim] — the replacement vector
        tokenizer: required if token_id_or_str is a string

    Returns:
        (token_id, original_vector) — save this for restore_embedding()
    """
    # Resolve token ID
    if isinstance(token_id_or_str, str):
        if tokenizer is None:
            raise ValueError("tokenizer required when token_id_or_str is a string")
        token_id = tokenizer.convert_tokens_to_ids(token_id_or_str)
        if token_id is None:
            raise ValueError(f"Token '{token_id_or_str}' not found in tokenizer")
    else:
        token_id = int(token_id_or_str)

    embeddings = model.get_input_embeddings()

    # Save original
    with torch.no_grad():
        original = embeddings.weight[token_id].clone().cpu()

    # Inject — match device and dtype
    vec = custom_vector.to(
        device=embeddings.weight.device,
        dtype=embeddings.weight.dtype,
    )
    with torch.no_grad():
        embeddings.weight[token_id] = vec

    return token_id, original


def restore_embedding(model, token_id, original_vector):
    """Restore a token's original embedding vector.

    Args:
        model: HuggingFace model
        token_id: int — which token to restore
        original_vector: tensor — the saved original from inject_embedding_vector()
    """
    embeddings = model.get_input_embeddings()
    vec = original_vector.to(
        device=embeddings.weight.device,
        dtype=embeddings.weight.dtype,
    )
    with torch.no_grad():
        embeddings.weight[token_id] = vec


class EmbeddingPerturbation:
    """Context manager for temporary embedding injection.

    Usage:
        vector = torch.load("thinking_cap_vector.pt")
        with EmbeddingPerturbation(model, "<unused42>", vector, tokenizer):
            # model now has the custom embedding
            output = model.generate(...)
        # original embedding automatically restored
    """

    def __init__(self, model, token_id_or_str, custom_vector, tokenizer=None):
        self.model = model
        self.token_id_or_str = token_id_or_str
        self.custom_vector = custom_vector
        self.tokenizer = tokenizer
        self.token_id = None
        self.original = None

    def __enter__(self):
        self.token_id, self.original = inject_embedding_vector(
            self.model, self.token_id_or_str, self.custom_vector, self.tokenizer
        )
        return self

    def __exit__(self, *exc):
        if self.token_id is not None and self.original is not None:
            restore_embedding(self.model, self.token_id, self.original)
        return False
