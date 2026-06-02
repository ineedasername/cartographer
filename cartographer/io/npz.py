"""NPZ scan file I/O for MRI and telescope captures.

Lifted from:
  - mri.py L252-259 (load_scan)
  - mri.py L217-236 (save pattern)
"""

import json

import numpy as np


def load_scan(filepath):
    """Load an MRI scan from .npz file.

    Source: mri.py L252-259

    Returns:
        (scan, meta, vectors) where:
        - scan: ndarray (steps, layers, cols, vocab_size) float16
        - meta: dict with prompt, output, model, steps, col_labels, etc.
        - vectors: ndarray (steps, layers, cols, hidden_dim) float32 or None
    """
    data = np.load(filepath, allow_pickle=False)
    scan = data['scan']
    meta = json.loads(str(data['meta']))
    vectors = data['vectors'] if 'vectors' in data else None
    return scan, meta, vectors


def save_scan(filepath, scan, meta, vectors=None):
    """Save an MRI scan to compressed .npz file.

    Source: mri.py L231-236

    Args:
        filepath: output path (e.g., 'scan.npz')
        scan: ndarray (steps, layers, cols, vocab_size) float16
        meta: dict — will be JSON-serialized
        vectors: optional ndarray (steps, layers, cols, hidden_dim) float32
    """
    kwargs = {
        'scan': scan,
        'meta': json.dumps(meta),
    }
    if vectors is not None:
        kwargs['vectors'] = vectors
    np.savez_compressed(filepath, **kwargs)


def load_tokenizer_for_scan(meta):
    """Load the tokenizer used for a scan (for decoding token IDs).

    Source: mri.py L262-266

    Args:
        meta: scan metadata dict (must contain 'model' key)

    Returns:
        tokenizer instance
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(meta['model'])
