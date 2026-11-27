"""Tests for sanitized provider evidence capture."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from emp_kg import store
from emp_kg.reasoning import ReasoningFact, ReasoningResult
from scripts.provider_evidence import DEFAULT_MODEL, TOKEN_CAP, TIMEOUT_CAP, capture_provider_evidence


class TestProviderEvidence(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db = Path(self.tmpdir.name) / "graph.db"
        conn = store.init_db(self.db)
        store.save_structural_nodes(conn, [{"id": "function:login", "kind": "function", "name": "login"}])
        store.rebuild_fts(conn)
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_dry_run_produces_sanitized_envelope(self) -> None:
        result = capture_provider_evidence(self.db, task="login")
        self.assertEqual(result["mode"], "dry-run")
        self.assertEqual(result["response"]["provider"], "none")
        self.assertEqual(result["response"]["model"], "none")
        self.assertTrue(result["db_unchanged"])
        self.assertTrue(result["sanitized"]["prompt_content_replaced_with_hash"])
        self.assertNotIn("Context JSON", json.dumps(result))

    @patch("scripts.provider_evidence.AnthropicReasoner")
    def test_execute_mode_uses_caps_and_captures_usage(self, reasoner_class) -> None:
        reasoner_class._prompt.return_value = "test provider prompt"
        reasoner_class.return_value.reason.return_value = ReasoningResult(
            ok=True, correlation_id="abc123", elapsed_ms=12.5,
            text="Login cites fact_id:function:login.",
            citation_facts=[ReasoningFact("function:login", "function", "login", "", [])],
            model=DEFAULT_MODEL, provider="anthropic", citation_valid=True,
            metadata={
                "fallback_occurred": False,
                "input_tokens": 100,
                "output_tokens": 20,
                "served_model": DEFAULT_MODEL,
            },
        )
        result = capture_provider_evidence(self.db, task="login", execute=True)
        reasoner_class.assert_called_once_with(
            model=DEFAULT_MODEL, max_tokens=TOKEN_CAP, timeout=TIMEOUT_CAP, max_retries=0,
        )
        self.assertEqual(reasoner_class.return_value.thinking_mode, "adaptive")
        self.assertEqual(result["mode"], "execute")
        self.assertEqual(result["response"]["input_tokens"], 100)
        self.assertEqual(result["response"]["output_tokens"], 20)
        self.assertTrue(result["db_unchanged"])

    @patch("scripts.provider_evidence.AnthropicReasoner")
    def test_disabled_thinking_passed_through(self, reasoner_class) -> None:
        reasoner_class._prompt.return_value = "test prompt"
        reasoner_class.return_value.reason.return_value = ReasoningResult(
            ok=True, correlation_id="x1", elapsed_ms=5.0,
            text="Coverage uses function:login", model=DEFAULT_MODEL, provider="anthropic",
            metadata={"fallback_occurred": False, "input_tokens": 50, "output_tokens": 10, "served_model": "test"},
        )
        result = capture_provider_evidence(self.db, thinking="disabled", max_tokens=256, execute=True)
        self.assertEqual(reasoner_class.return_value.thinking_mode, "disabled")
        self.assertEqual(reasoner_class.call_args.kwargs["max_tokens"], 256)
        self.assertEqual(result["request"]["thinking"], {"type": "disabled"})
        self.assertEqual(result["request"]["max_tokens"], 256)

    def test_missing_db_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            capture_provider_evidence(self.db.with_name("missing.db"))

    def test_json_serializable(self) -> None:
        result = capture_provider_evidence(self.db)
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
