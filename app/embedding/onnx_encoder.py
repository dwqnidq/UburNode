"""ONNX Runtime 向量编码（生产默认，无 PyTorch 依赖）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import numpy as np
import onnxruntime as ort
from loguru import logger
from transformers import AutoTokenizer

from app.core.exceptions import EncoderNotReadyError
from app.embedding.encoder import EncoderBase
from app.embedding.pooling import cls_pool, l2_normalize

if TYPE_CHECKING:
    from app.core.config import Settings


class OnnxEncoder(EncoderBase):
    """文本 → 512 维向量（ONNX 输出 last_hidden_state，Python 侧 CLS pool + L2）。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session: ort.InferenceSession | None = None
        self._tokenizer: AutoTokenizer | None = None

    def load(self) -> None:
        onnx_path = self._settings.embedding_onnx_path
        tokenizer_dir = self._settings.embedding_tokenizer_dir
        if not onnx_path.is_file():
            msg = f"ONNX 模型不存在：{onnx_path}，请先运行 scripts/export_onnx_model.py"
            raise FileNotFoundError(msg)

        logger.info("正在加载 ONNX 向量模型：{}", onnx_path)
        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        logger.info("ONNX 向量模型加载完成")

    @property
    def is_loaded(self) -> bool:
        return self._session is not None and self._tokenizer is not None

    async def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.is_loaded:
            raise EncoderNotReadyError()
        return await anyio.to_thread.run_sync(self._encode_sync, texts)

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        assert self._tokenizer is not None and self._session is not None
        batch = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        hidden = self._session.run(
            None,
            {
                "input_ids": batch["input_ids"].astype(np.int64),
                "attention_mask": batch["attention_mask"].astype(np.int64),
            },
        )[0]
        pooled = cls_pool(hidden)
        normalized = l2_normalize(pooled)
        return normalized.tolist()

    async def encode_one(self, text: str) -> list[float]:
        results = await self.encode([text])
        return results[0]
