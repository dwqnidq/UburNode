#!/usr/bin/env python3
"""对比 PyTorch 与 ONNX 向量编码一致性，并模拟检索粗排顺序差异。

用法（需 dev 依赖 + 已导出 ONNX）:
  .venv/bin/python scripts/export_onnx_model.py
  .venv/bin/python scripts/compare_embedding_backends.py

判定建议:
  - 向量余弦相似度均值 ≥ 0.999 且最小值 ≥ 0.995 → 检索效果通常等价
  - 粗排顺序一致率 ≥ 95% → 可考虑切换 ONNX 生产部署
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.core.config import Settings
from app.embedding.onnx_encoder import OnnxEncoder
from app.embedding.torch_encoder import TorchEncoder

# 覆盖检索评测中的典型中文标签
SAMPLE_TAGS = [
    "雨声",
    "森林",
    "白噪音",
    "溪流",
    "篝火",
    "海浪",
    "摇篮曲",
    "哼唱",
    "鸟鸣",
    "自然",
    "日光冥想",
    "播客",
    "咖啡店",
    "嘈杂",
    "放松",
    "入睡",
    "清醒",
    "守护",
    "完全不存在的标签xyz",
]


@dataclass
class VectorReport:
    tag: str
    cosine: float
    l2_diff: float


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _l2_diff(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _coarse_rank(query_vecs: list[list[float]], doc_vecs: list[list[float]]) -> list[int]:
    """按与多查询标签的最大余弦相似度粗排文档（越高越靠前）。"""
    scores: list[tuple[int, float]] = []
    for doc_idx, doc_vec in enumerate(doc_vecs):
        best = max(_cosine(q, doc_vec) for q in query_vecs)
        scores.append((doc_idx, best))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [idx for idx, _ in scores]


async def _load_both(settings: Settings) -> tuple[TorchEncoder, OnnxEncoder]:
    torch_enc = TorchEncoder(settings)
    onnx_enc = OnnxEncoder(settings)
    torch_enc.load()
    onnx_enc.load()
    return torch_enc, onnx_enc


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=_ROOT / "models/onnx/bge-small-zh-v1.5",
    )
    args = parser.parse_args()

    settings = Settings(
        embedding_backend="onnx",
        embedding_onnx_dir=str(args.onnx_dir),
    )
    torch_enc, onnx_enc = await _load_both(settings)

    print("=" * 72)
    print("一、逐标签向量一致性（PyTorch vs ONNX）")
    print("=" * 72)

    reports: list[VectorReport] = []
    for tag in SAMPLE_TAGS:
        torch_vec = await torch_enc.encode_one(tag)
        onnx_vec = await onnx_enc.encode_one(tag)
        reports.append(
            VectorReport(
                tag=tag,
                cosine=_cosine(torch_vec, onnx_vec),
                l2_diff=_l2_diff(torch_vec, onnx_vec),
            )
        )

    cosines = [r.cosine for r in reports]
    mean_cos = sum(cosines) / len(cosines)
    min_cos = min(cosines)
    worst = min(reports, key=lambda r: r.cosine)

    for row in reports:
        flag = "OK" if row.cosine >= 0.995 else "WARN"
        print(f"[{flag}] {row.tag:16s} cosine={row.cosine:.6f}  l2_diff={row.l2_diff:.6f}")
    print(f"\n均值 cosine={mean_cos:.6f}  最小 cosine={min_cos:.6f}  最差标签={worst.tag}")

    print("\n" + "=" * 72)
    print("二、检索粗排顺序模拟（多标签查询）")
    print("=" * 72)

    query_sets = [
        (["雨声", "森林", "白噪音"], "放松·3标签"),
        (["雨声", "森林", "白噪音", "溪流"], "放松·4标签"),
        (["海浪", "雨声", "摇篮曲"], "入睡·3标签"),
        (["鸟鸣", "自然", "日光冥想"], "清醒·3标签"),
    ]
    doc_tags = ["雨声", "森林", "白噪音", "溪流", "海浪", "摇篮曲", "鸟鸣", "自然"]

    torch_docs = await torch_enc.encode(doc_tags)
    onnx_docs = await onnx_enc.encode(doc_tags)

    agree = 0
    for query_tags, name in query_sets:
        torch_q = await torch_enc.encode(query_tags)
        onnx_q = await onnx_enc.encode(query_tags)
        torch_rank = _coarse_rank(torch_q, torch_docs)
        onnx_rank = _coarse_rank(onnx_q, onnx_docs)
        same = torch_rank == onnx_rank
        agree += int(same)
        print(f"\n[{'OK' if same else 'DIFF'}] {name}")
        print(f"  查询: {query_tags}")
        print(f"  PyTorch 粗排索引: {torch_rank} -> {[doc_tags[i] for i in torch_rank]}")
        print(f"  ONNX    粗排索引: {onnx_rank} -> {[doc_tags[i] for i in onnx_rank]}")

    order_rate = agree / len(query_sets) * 100
    print(f"\n粗排顺序一致率: {agree}/{len(query_sets)} = {order_rate:.1f}%")

    print("\n" + "=" * 72)
    print("三、切换建议")
    print("=" * 72)
    if mean_cos >= 0.999 and min_cos >= 0.995:
        print("建议：向量与 PyTorch 等价，可切换 ONNX 生产部署以缩小镜像。")
        print("      若 ES 中已有 PyTorch 时期写入的向量，召回应保持一致（同模型同维度）。")
    elif mean_cos >= 0.995:
        print("建议：向量接近，建议再跑 eval_search_recall.py 对比真实 ES 召回后决定。")
    else:
        print("建议：差异偏大，暂保留 PyTorch 或检查 ONNX 导出。")
    print("\n完整召回对比：分别设置 EMBEDDING_BACKEND=torch/onnx 启动服务后运行")
    print("  .venv/bin/python scripts/eval_search_recall.py")


if __name__ == "__main__":
    asyncio.run(main())
