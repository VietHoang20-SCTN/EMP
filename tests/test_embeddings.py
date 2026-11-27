"""Tests for optional local semantic retrieval helpers."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from emp_kg import embeddings


class TestEmbeddings(unittest.TestCase):
    def test_content_hash_is_deterministic(self) -> None:
        self.assertEqual(embeddings.content_hash("same"), embeddings.content_hash("same"))
        self.assertNotEqual(embeddings.content_hash("one"), embeddings.content_hash("two"))

    def test_cosine_similarity_and_order_are_deterministic(self) -> None:
        records = [
            {"node_id": "node:b", "vector": [0.0, 1.0]},
            {"node_id": "node:a", "vector": [1.0, 0.0]},
            {"node_id": "node:c", "vector": [0.8, 0.2]},
        ]
        self.assertEqual(
            embeddings.rank_embeddings([1.0, 0.0], records, limit=3),
            [("node:a", 1.0), ("node:c", 0.9701425001453318)],
        )

    def test_dimension_mismatch_is_not_ranked(self) -> None:
        self.assertEqual(
            embeddings.rank_embeddings(
                [1.0, 0.0], [{"node_id": "bad", "vector": [1.0]}], limit=5
            ),
            [],
        )

    def test_validate_vectors_rejects_cardinality_and_dimensions(self) -> None:
        with self.assertRaises(embeddings.EmbeddingUnavailableError):
            embeddings.validate_vectors([[1.0]], 2)
        with self.assertRaises(embeddings.EmbeddingUnavailableError):
            embeddings.validate_vectors([[1.0], [1.0, 2.0]], 2)
        with self.assertRaises(embeddings.EmbeddingUnavailableError):
            embeddings.validate_vectors([[float("nan")]], 1)

    def test_semantic_seeds_rejects_query_dimension_mismatch(self) -> None:
        class FakeEmbedder:
            def __init__(self, _model_id: str) -> None:
                pass

            def encode(self, _texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0]]

        with patch.object(
            embeddings, "SentenceTransformerEmbedder", FakeEmbedder
        ), patch(
            "emp_kg.store.load_fresh_node_embeddings",
            return_value=[{
                "node_id": "function:x", "dimension": 1,
                "vector": [1.0], "content_hash": "hash",
            }],
        ):
            with self.assertRaises(embeddings.EmbeddingUnavailableError):
                embeddings.semantic_seeds(object(), "query", "fake-model", 5)

    def test_missing_sentence_transformers_has_clear_error(self) -> None:
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with self.assertRaises(embeddings.EmbeddingUnavailableError):
                embeddings.SentenceTransformerEmbedder("fake-model")


if __name__ == "__main__":
    unittest.main()
