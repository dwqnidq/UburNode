"""向量池化与归一化单元测试。"""

from __future__ import annotations

import numpy as np

from app.embedding.pooling import cls_pool, l2_normalize, mean_pool


def test_cls_pool_takes_first_token() -> None:
    token_embeddings = np.array(
        [[[1.0, 0.0], [9.0, 9.0], [2.0, 0.0]]],
        dtype=np.float32,
    )
    pooled = cls_pool(token_embeddings)
    np.testing.assert_allclose(pooled[0], [1.0, 0.0], rtol=1e-5)



def test_mean_pool_respects_attention_mask() -> None:
    token_embeddings = np.array(
        [
            [[1.0, 0.0], [2.0, 0.0], [9.0, 9.0]],
            [[4.0, 0.0], [6.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    attention_mask = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.int64)

    pooled = mean_pool(token_embeddings, attention_mask)

    np.testing.assert_allclose(pooled[0], [1.5, 0.0], rtol=1e-5)
    np.testing.assert_allclose(pooled[1], [5.0, 0.0], rtol=1e-5)


def test_l2_normalize_produces_unit_vectors() -> None:
    vectors = np.array([[3.0, 4.0], [0.0, 7.0]], dtype=np.float32)
    normalized = l2_normalize(vectors)

    np.testing.assert_allclose(np.linalg.norm(normalized[0]), 1.0, rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(normalized[1]), 1.0, rtol=1e-5)
