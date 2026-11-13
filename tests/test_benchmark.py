"""Tests for deterministic Context Builder quality benchmarks."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from emp_kg import benchmark, context, store


class TestBenchmark(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "graph.db"
        conn = store.init_db(self.db_path)
        store.save_structural_nodes(conn, [
            {"id": "function:login", "kind": "function", "name": "login"},
            {"id": "function:unrelated", "kind": "function", "name": "unrelated"},
        ])
        store.save_decision_nodes(conn, [{
            "id": "0001", "title": "Login architecture",
            "source_path": "docs/decisions/0001-login.md",
        }])
        store.save_edges(conn, [
            {"source": "decision_ref:0001", "target": "function:login", "kind": "AFFECTS"},
        ])
        store.rebuild_fts(conn)
        conn.commit()
        conn.close()
        self.cases = [
            {
                "id": "direct-login",
                "task": "login",
                "expected_node_ids": ["function:login"],
                "lane": "tiny",
                "affected_files": [],
                "k": 1,
            },
            {
                "id": "synonym-login",
                "task": "signin",
                "expected_node_ids": ["function:login"],
                "lane": "tiny",
                "affected_files": [],
                "k": 1,
            },
        ]

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_bm25_and_synonym_modes_produce_measured_difference(self) -> None:
        bm25 = benchmark.run_benchmark(self.db_path, self.cases, mode="bm25")
        synonyms = benchmark.run_benchmark(self.db_path, self.cases, mode="synonyms")

        self.assertEqual(bm25["mode"], "bm25")
        self.assertEqual(bm25["cases"][0]["recall_at_k"], 1.0)
        self.assertEqual(bm25["cases"][1]["recall_at_k"], 0.0)
        self.assertEqual(synonyms["cases"][1]["recall_at_k"], 1.0)
        self.assertGreater(synonyms["aggregate"]["recall_at_k"], bm25["aggregate"]["recall_at_k"])

    def test_metrics_use_ranked_top_k(self) -> None:
        metrics = benchmark._metrics(
            ["function:unrelated", "function:login"],
            ["function:login"],
            2,
        )
        self.assertEqual(metrics["hit_count"], 1)
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["precision_at_k"], 0.5)
        self.assertEqual(metrics["mrr"], 0.5)

    def test_local_mode_requests_local_semantics(self) -> None:
        captured = []

        def fake_context(request, _db_path):
            captured.append(request)
            return SimpleNamespace(facts=[
                context.ContextFact(
                    fact_type="code", node_id="function:login", name="login",
                    kind="function", snippet="", relevance_score=1.0,
                ),
            ])

        with patch.object(benchmark, "build_context", side_effect=fake_context):
            result = benchmark.run_benchmark(self.db_path, self.cases[:1], mode="local")

        self.assertEqual(result["aggregate"]["recall_at_k"], 1.0)
        self.assertEqual(captured[0].semantic_mode, "local")
        self.assertEqual(captured[0].synonym_mode, "auto")

    def test_load_cases_validates_schema_and_order(self) -> None:
        path = Path(self.tmpdir.name) / "cases.json"
        path.write_text(json.dumps({"version": 1, "cases": self.cases}), encoding="utf-8")
        self.assertEqual([case["id"] for case in benchmark.load_cases(path)], [
            "direct-login", "synonym-login",
        ])
        path.write_text(json.dumps({"version": 2, "cases": []}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "version 1"):
            benchmark.load_cases(path)
        path.write_text(json.dumps({
            "version": 1,
            "cases": [{
                "id": "invalid", "task": "task", "expected_node_ids": ["function:x"],
                "lane": "invalid", "affected_files": [1],
            }],
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lane is invalid"):
            benchmark.load_cases(path)

    def test_default_k_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "default_k"):
            benchmark.run_benchmark(self.db_path, self.cases, mode="bm25", default_k=0)
        with self.assertRaisesRegex(ValueError, "default_k"):
            benchmark.run_benchmark(self.db_path, self.cases, mode="bm25", default_k=True)


if __name__ == "__main__":
    unittest.main()
