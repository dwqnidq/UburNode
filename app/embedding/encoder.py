"""BGE-M3 文本向量化。

sentence-transformers 推理为 CPU 阻塞调用，必须用 anyio.to_thread 避免卡住事件循环（规范 §八）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
from loguru import logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from app.core.config import Settings


class Encoder:
    """文本 → 1024 维向量（BAAI/bge-m3，normalize_embeddings=True 便于余弦比较）。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("正在加载向量模型：{}", self._settings.embedding_model)
        self._model = SentenceTransformer(self._settings.embedding_model)
        logger.info("向量模型加载完成")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    async def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            raise RuntimeError("向量编码器未加载，请先在 lifespan 中调用 load()")

        return await anyio.to_thread.run_sync(self._encode_sync, texts)

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        assert self._model is not None
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    async def encode_one(self, text: str) -> list[float]:
        results = await self.encode([text])
        return results[0]
