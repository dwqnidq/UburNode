"""向量后处理：与 sentence-transformers Pooling + Normalize 对齐。"""

from __future__ import annotations

import numpy as np


def cls_pool(token_embeddings: np.ndarray) -> np.ndarray:
    """取 [CLS] 位（index 0）作为句向量（bge-small-zh-v1.5 使用 cls pooling）。"""
    return token_embeddings[:, 0, :]


def mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """对 last_hidden_state 做 attention mask 加权平均池化。"""
    mask = attention_mask.astype(np.float32)
    expanded = np.expand_dims(mask, axis=-1)
    summed = np.sum(token_embeddings * expanded, axis=1)
    counts = np.clip(expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-9, a_max=None)
    return vectors / norms
