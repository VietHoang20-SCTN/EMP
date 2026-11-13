"""Tests for deterministic final-week evidence runner."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from emp_kg import store
from scripts.final_validation import run_final_validation


class TestFinalValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "graph.db"
        conn = store.init_db(self.db_path)
        store.save_structural_nodes(conn, [
            {"id": "function:login", "kind": "function", "name": "login"},
        ])
        store.save_decision_nodes(conn, [{
            "id": "0001", "title": "Login architecture",
            "source_path": "docs/decisions/0001-login.md",
        }])
        store.save_edges(conn, [
            {"source": "decision_ref:0001", "target": "function:login", "kind": "AFFECTS"},
        ])
        store.rebuild_fts(conn)
        store.save_metadata(conn, schema_version="test-version", project_path=Path(self.tmpdir.name), git_sha="abc123")
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_final_validation_missing_db_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            run_final_validation(db_path=Path(self.tmpdir.name) / "missing.db")

    def test_final_validation_rejects_empty_benchmark_cases(self) -> None:
        with self.assertRaises(ValueError):
            run_final_validation(db_path=self.db_path, benchmark_cases=[])

    def test_final_validation_returns_deterministic_shape(self) -> None:
        payload = run_final_validation(
            db_path=self.db_path,
            benchmark_cases_path=Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "context_benchmark.json",
            benchmark_cases=[{
                "id": "direct-login",
                "task": "login",
                "expected_node_ids": ["function:login"],
                "lane": "tiny",
                "k": 5,
            }],
            latency_repeat=2,
        )
        self.assertEqual(payload["artifact"], "final_validation")
        self.assertEqual(payload["schema_version"], "2.5.0")
        self.assertIn("runtime", payload)
        self.assertEqual(payload["graph"]["stats"]["nodes"], 2)
        self.assertIn("mcp_smoke", payload)
        self.assertIn("retrieval_benchmark", payload)
        self.assertEqual(payload["retrieval_benchmark"]["case_count"], 1)
        self.assertEqual(payload["retrieval_benchmark"]["modes"]["bm25"]["recall_at_k"], 1.0)
        self.assertEqual(payload["retrieval_benchmark"]["modes"]["synonyms"]["recall_at_k"], 1.0)
        self.assertTrue(payload["context_retrieval"]["aggregate_ms"] >= 0)
        self.assertTrue(payload["readiness_flags"]["offline_evidence_verified"])
        self.assertFalse(payload["readiness_flags"]["full_suite_verified"])
        self.assertEqual(payload["readiness_flags"]["live_client_evidence"], "available_partial")
        self.assertEqual(
            payload["readiness_flags"]["live_client_tool_coverage"],
            {
                "claude_code": ["emp_kg_metadata"],
                "cline": ["emp_kg_metadata", "emp_kg_context"],
                "codex_cli": ["emp_kg_metadata"],
            },
        )
        self.assertEqual(payload["readiness_flags"]["live_provider_evidence"], "pending_optional_credentialed_run")
        verified = run_final_validation(
            db_path=self.db_path,
            benchmark_cases=[{
                "id": "direct-login",
                "task": "login",
                "expected_node_ids": ["function:login"],
                "lane": "tiny",
                "k": 5,
            }],
            latency_repeat=1,
            full_suite_verified=True,
        )
        self.assertTrue(verified["readiness_flags"]["full_suite_verified"])
        serialized = json.loads(json.dumps(payload))
        self.assertEqual(serialized["artifact"], "final_validation")


if __name__ == "__main__":
    unittest.main()
