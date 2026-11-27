"""Tests for MCP-compatible context protocol adapter."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from emp_kg import mcp_context, store


class TestMCPContext(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db = Path(self.tmpdir.name) / "graph.db"
        conn = store.init_db(self.db)
        store.save_structural_nodes(conn, [{
            "id": "function:login", "kind": "function", "name": "login",
        }])
        store.rebuild_fts(conn)
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_context_tool_returns_structured_packet(self) -> None:
        response = mcp_context.handle_context(mcp_context.MCPContextRequest(
            task="login", db_path=str(self.db), lane="tiny",
        ))
        self.assertTrue(response.ok)
        self.assertEqual(response.data["lane"], "tiny")
        self.assertEqual(response.data["facts"][0]["node_id"], "function:login")
        self.assertTrue(response.correlation_id)
        self.assertIn("correlation_id", response.data)
        self.assertIn("elapsed_ms", response.data)
        self.assertIsInstance(response.data["elapsed_ms"], float)

    def test_context_tool_rejects_missing_db_and_empty_task(self) -> None:
        missing = mcp_context.handle_context(mcp_context.MCPContextRequest(
            task="login", db_path=str(self.db.with_name("missing.db")),
        ))
        self.assertFalse(missing.ok)
        empty = mcp_context.handle_context(mcp_context.MCPContextRequest(
            task="  ", db_path=str(self.db),
        ))
        self.assertFalse(empty.ok)

    def test_reason_tool_defaults_to_noop(self) -> None:
        response = mcp_context.handle_reason(mcp_context.MCPContextRequest(
            task="login", db_path=str(self.db), lane="tiny",
        ))
        self.assertTrue(response.ok)
        self.assertEqual(response.data["reasoning"]["provider"], "none")
        self.assertEqual(response.provenance["provider"], "none")
        self.assertEqual(response.data["provider"], "none")
        self.assertEqual(response.data["model"], "none")
        self.assertIn("correlation_id", response.data)
        self.assertIn("elapsed_ms", response.data)
        self.assertIn("fallback_occurred", response.data)
        self.assertFalse(response.data["fallback_occurred"])

    def test_metadata_tool_is_read_only(self) -> None:
        response = mcp_context.handle_metadata(mcp_context.MCPMetadataRequest(
            db_path=str(self.db),
        ))
        self.assertTrue(response.ok)
        self.assertEqual(response.data["graph"]["nodes"], 1)
        conn = store.open_readonly(self.db)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0], 1)
        conn.close()

    def test_tool_descriptors_have_required_schemas(self) -> None:
        names = {tool["name"] for tool in mcp_context.TOOLS}
        self.assertEqual(names, {"emp_kg_context", "emp_kg_impact", "emp_kg_reason", "emp_kg_metadata", "emp_kg_benchmark"})
        for tool in mcp_context.TOOLS:
            self.assertEqual(tool["inputSchema"]["type"], "object")
            self.assertTrue(tool["inputSchema"]["required"])
            self.assertNotIn("input_schema", tool)


if __name__ == "__main__":
    unittest.main()
