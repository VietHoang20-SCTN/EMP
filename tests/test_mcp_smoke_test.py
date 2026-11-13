"""Tests for client-neutral MCP smoke test."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from emp_kg import store
from scripts.mcp_smoke_test import run_smoke_test


class TestMCPSmokeTest(unittest.TestCase):
    def test_smoke_test_passes_on_temporary_db(self) -> None:
        result = run_smoke_test()
        self.assertTrue(result["initialize_ok"])
        self.assertGreaterEqual(result["tool_count"], 5)
        self.assertTrue(result["context_ok"])
        self.assertTrue(result["reason_ok"])
        self.assertEqual(result["reason_provider"], "none")
        self.assertEqual(result["reason_model"], "none")
        self.assertTrue(result["reason_correlation_id"])
        self.assertTrue(result["unknown_tool_rejected"])
        self.assertTrue(result["db_unchanged"])


if __name__ == "__main__":
    unittest.main()
