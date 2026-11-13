"""Tests for MCP stdio JSON-RPC transport."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from emp_kg import mcp_server, store


class TestMCPServer(unittest.TestCase):
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

    def _run(self, messages: list[dict] | list[str]) -> list[dict]:
        lines = [item if isinstance(item, str) else json.dumps(item) for item in messages]
        output = io.StringIO()
        mcp_server.run_stdio(io.StringIO("\n".join(lines) + "\n"), output)
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_initialize_and_list_tools(self) -> None:
        responses = self._run([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ])
        self.assertEqual(responses[0]["result"]["serverInfo"]["name"], "emp-kg")
        tools = responses[1]["result"]["tools"]
        names = {tool["name"] for tool in tools}
        self.assertIn("emp_kg_context", names)
        self.assertIn("emp_kg_reason", names)
        self.assertTrue(all(tool["inputSchema"]["type"] == "object" for tool in tools))
        self.assertTrue(all("input_schema" not in tool for tool in tools))

    def test_context_and_reason_tools(self) -> None:
        responses = self._run([
            {
                "jsonrpc": "2.0", "id": "context", "method": "tools/call",
                "params": {"name": "emp_kg_context", "arguments": {"task": "login", "db_path": str(self.db), "lane": "tiny"}},
            },
            {
                "jsonrpc": "2.0", "id": "reason", "method": "tools/call",
                "params": {"name": "emp_kg_reason", "arguments": {"task": "login", "db_path": str(self.db), "lane": "tiny"}},
            },
        ])
        context_data = json.loads(responses[0]["result"]["content"][0]["text"])
        reason_data = json.loads(responses[1]["result"]["content"][0]["text"])
        self.assertEqual(context_data["facts"][0]["node_id"], "function:login")
        self.assertIn("correlation_id", context_data)
        self.assertIn("elapsed_ms", context_data)
        self.assertEqual(context_data["provider"], "none")
        self.assertEqual(reason_data["reasoning"]["provider"], "none")
        self.assertIn("correlation_id", reason_data)
        self.assertEqual(reason_data["provider"], "none")
        self.assertEqual(reason_data["model"], "none")
        self.assertIn("elapsed_ms", reason_data)
        self.assertFalse(reason_data["fallback_occurred"])

    def test_parse_method_and_tool_errors(self) -> None:
        responses = self._run([
            "{not json}",
            {"jsonrpc": "2.0", "id": 1, "method": "unknown", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "unknown", "arguments": {}}},
        ])
        self.assertEqual(responses[0]["error"]["code"], -32700)
        self.assertEqual(responses[1]["error"]["code"], -32601)
        self.assertTrue(responses[2]["result"]["isError"])

    def test_notification_has_no_response_and_db_stays_readonly(self) -> None:
        conn = store.open_readonly(self.db)
        before = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        conn.close()
        responses = self._run([
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        ])
        conn = store.open_readonly(self.db)
        after = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        conn.close()
        self.assertEqual(responses, [])
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
