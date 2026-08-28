from __future__ import annotations

import numpy as np
import pandas as pd

from research_v15.features import load_event_documents

from .config import V18Settings


def build_embeddings(
    event_path: str = "data/event_documents_pit_v15.csv",
    settings: V18Settings | None = None,
) -> np.ndarray:
    settings = settings or V18Settings()
    if settings.embed_cache.exists():
        matrix = np.load(settings.embed_cache)
        print(f"V18 embed: 使用缓存 {settings.embed_cache} shape={matrix.shape}", flush=True)
        return matrix

    from sentence_transformers import SentenceTransformer

    events = load_event_documents(event_path)
    documents = events["document"].fillna("").astype(str).tolist()
    print(f"V18 embed: 编码 {len(documents)} 条事件文本", flush=True)
    model = SentenceTransformer(settings.embedding_model)
    matrix = model.encode(
        documents,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(settings.embed_cache, matrix)
    print(f"V18 embed: 已缓存 shape={matrix.shape}", flush=True)
    return matrix


if __name__ == "__main__":
    settings = V18Settings()
    settings.ensure_dirs()
    matrix = build_embeddings(settings=settings)
    print("完成, 嵌入维度:", matrix.shape)
