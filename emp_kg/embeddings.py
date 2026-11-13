"""Optional local embedding runtime for semantic context retrieval."""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingUnavailableError(RuntimeError):
    """Local embedding runtime or current vector index is unavailable."""


class Embedder(Protocol):
    model_id: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Lazy adapter around optional sentence-transformers dependency."""

    def __init__(self, model_id: str = DEFAULT_MODEL, *, cache_dir: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingUnavailableError(
                "local embeddings require `pip install -e '.[semantic]'`"
            ) from exc
        self.model_id = model_id
        try:
            self._model = SentenceTransformer(model_id, cache_folder=cache_dir)
        except Exception as exc:
            raise EmbeddingUnavailableError(
                f"could not load local embedding model {model_id!r}: {exc}"
            ) from exc

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]


def content_hash(text: str) -> str:
    """Return a deterministic source-content identity."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedding_content_hashes(documents: list[dict[str, str]]) -> list[dict[str, str]]:
    """Attach deterministic source hashes to FTS documents."""
    return [
        {**item, "content_hash": content_hash(item["content"])}
        for item in documents
    ]


def corpus_digest(documents: list[dict[str, str]]) -> str:
    """Return stable digest for node/content-hash pairs."""
    payload = "\n".join(
        f"{item['node_id']}:{item['content_hash']}"
        for item in sorted(documents, key=lambda item: item["node_id"])
    )
    return content_hash(payload)


def validate_vectors(vectors: list[list[float]], expected_count: int) -> int:
    """Validate encoder cardinality, common dimension, and finite values."""
    if len(vectors) != expected_count:
        raise EmbeddingUnavailableError(
            f"embedding encoder returned {len(vectors)} vectors for {expected_count} documents"
        )
    if not vectors:
        return 0
    dimension = len(vectors[0])
    if dimension == 0:
        raise EmbeddingUnavailableError("embedding encoder returned empty vectors")
    for vector in vectors:
        if len(vector) != dimension or any(
            not math.isfinite(float(value)) for value in vector
        ):
            raise EmbeddingUnavailableError(
                "embedding encoder returned invalid vector dimensions or values"
            )
    return dimension


def normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0.0:
        return [0.0 for _ in vector]
    return [value / magnitude for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    normalized_left = normalize(left)
    normalized_right = normalize(right)
    return sum(a * b for a, b in zip(normalized_left, normalized_right))


def rank_embeddings(
    query_vector: list[float],
    records: list[dict[str, object]],
    *,
    limit: int,
    min_similarity: float = 0.2,
) -> list[tuple[str, float]]:
    """Return deterministic cosine-ranked node IDs above a minimum score."""
    ranked = []
    for record in records:
        node_id = record.get("node_id")
        vector = record.get("vector")
        if not isinstance(node_id, str) or not isinstance(vector, list):
            continue
        similarity = cosine_similarity(query_vector, [float(value) for value in vector])
        if similarity >= min_similarity:
            ranked.append((node_id, similarity))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:limit]


def semantic_seeds(
    conn,
    task_description: str,
    model_id: str,
    limit: int,
) -> list[dict[str, object]]:
    """Return semantic seeds with model and source-content provenance."""
    from emp_kg import store

    records = store.load_fresh_node_embeddings(conn, model_id)
    if not records:
        raise EmbeddingUnavailableError(
            f"no fresh embeddings for {model_id!r}; rebuild with --embeddings local"
        )
    embedder = SentenceTransformerEmbedder(model_id)
    query = embedder.encode([task_description])[0]
    query_dimension = validate_vectors([query], 1)
    if any(int(record["dimension"]) != query_dimension for record in records):
        raise EmbeddingUnavailableError(
            f"query vector dimension disagrees with fresh {model_id!r} index; rebuild with --embeddings local"
        )
    hashes = {
        str(record["node_id"]): str(record["content_hash"])
        for record in records
    }
    return [
        {
            "node_id": node_id,
            "cosine": similarity,
            "content_hash": hashes[node_id],
            "model_id": model_id,
        }
        for node_id, similarity in rank_embeddings(query, records, limit=limit)
    ]
