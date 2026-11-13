"""Tests for AI context compression."""
from __future__ import annotations

import unittest

from emp_kg.compression import CompressionConfig, compress_context
from emp_kg.context import ContextFact, ContextResult


class TestCompression(unittest.TestCase):
    def _result(self) -> ContextResult:
        facts = [
            ContextFact("code", "function:a", "alpha", "function", "alpha body", 1.0),
            ContextFact("code", "function:a", "alpha", "function", "alpha body", 0.9),
            ContextFact("code", "function:b", "beta", "function", "beta body", 0.2),
        ]
        return ContextResult(
            "task", "normal", "test", [], 5000, {"selected_facts": 3},
            facts, [], 100,
        )

    def test_compression_deduplicates_and_summarizes(self) -> None:
        compressed = compress_context(self._result(), CompressionConfig(
            target_tokens=1000, summary_score_threshold=0.3, similarity_threshold=0.85,
        ))
        self.assertLess(compressed.compressed_tokens, compressed.original_tokens)
        self.assertEqual(compressed.compression_ratio, compressed.compressed_tokens / 100)
        self.assertIn("function:a", compressed.dropped_fact_ids)
        self.assertTrue(compressed.summary_facts)

    def test_compression_is_deterministic(self) -> None:
        config = CompressionConfig(target_tokens=10)
        first = compress_context(self._result(), config)
        second = compress_context(self._result(), config)
        self.assertEqual([f.node_id for f in first.retained_facts], [f.node_id for f in second.retained_facts])
        self.assertEqual(first.dropped_fact_ids, second.dropped_fact_ids)


if __name__ == "__main__":
    unittest.main()
