import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from src.loader import load_pdfs
except ModuleNotFoundError:
    from loader import load_pdfs

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "faiss_index"

_model: SentenceTransformer | None = None
_cached_index_dir: Path | None = None
_cached_index: faiss.Index | None = None
_cached_chunks: list[dict] | None = None


def get_model() -> SentenceTransformer:
    """Load and cache the sentence-transformers embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def build_index(
    chunks: list[dict] | None = None,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> tuple[faiss.Index, list[dict]]:
    """Embed chunks, build a normalized FAISS IndexFlatIP index, and save to disk."""
    global _cached_index_dir, _cached_index, _cached_chunks

    if chunks is None:
        chunks = load_pdfs()
    if not chunks:
        raise ValueError("No chunks found to index. Add PDF files to data/ first.")

    model = get_model()
    texts = [chunk["text"] for chunk in chunks]

    # normalize_embeddings=True makes inner product (IndexFlatIP) equal cosine similarity
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    target_dir = Path(index_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(target_dir / "index.faiss"))
    (target_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _cached_index_dir = target_dir.resolve()
    _cached_index = index
    _cached_chunks = chunks
    return index, chunks


def load_or_build_index(
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> tuple[faiss.Index, list[dict]]:
    """Load the saved FAISS index and chunk metadata if present, else build them."""
    global _cached_index_dir, _cached_index, _cached_chunks

    target_dir = Path(index_dir).resolve()
    if (
        _cached_index is not None
        and _cached_chunks is not None
        and _cached_index_dir == target_dir
    ):
        return _cached_index, _cached_chunks

    index_path = target_dir / "index.faiss"
    meta_path = target_dir / "chunks.json"

    if index_path.exists() and meta_path.exists():
        index = faiss.read_index(str(index_path))
        chunks = json.loads(meta_path.read_text(encoding="utf-8"))
        _cached_index_dir = target_dir
        _cached_index = index
        _cached_chunks = chunks
        return index, chunks

    return build_index(index_dir=index_dir)


def clear_index(index_dir: str | Path = DEFAULT_INDEX_DIR) -> None:
    """Clear both the in-memory index cache and persisted index files on disk."""
    global _cached_index_dir, _cached_index, _cached_chunks
    _cached_index_dir = None
    _cached_index = None
    _cached_chunks = None

    target_dir = Path(index_dir)
    for filename in ("index.faiss", "chunks.json"):
        target = target_dir / filename
        if target.exists():
            target.unlink()


def search(
    query: str,
    k: int = 3,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
) -> list[dict]:
    """Return the top-k matching chunks with cosine similarity scores, source, and page."""
    index, chunks = load_or_build_index(index_dir=index_dir)
    if not chunks or k <= 0:
        return []

    model = get_model()
    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_vec = np.asarray(query_vec, dtype="float32")

    top_k = min(k, len(chunks))
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        results.append(
            {
                "text": chunk["text"],
                "score": float(score),
                "source": chunk["source"],
                "page": chunk["page"],
            }
        )
    return results
