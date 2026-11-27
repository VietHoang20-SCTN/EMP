"""Client-neutral smoke test for the emp-kg MCP stdio server."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _send(proc: subprocess.Popen, message: dict) -> dict:
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("MCP server closed without response")
    return json.loads(line)


def run_smoke_test(db_path: Path | None = None) -> dict:
    if db_path is None:
        import tempfile
        from emp_kg import store
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "graph.db"
        conn = store.init_db(db_path)
        store.save_structural_nodes(conn, [{"id": "function:login", "kind": "function", "name": "login"}])
        store.rebuild_fts(conn)
        conn.commit()
        conn.close()
    before = db_path.stat().st_size
    env = os.environ.copy()
    current_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_PROJECT_ROOT) + (os.pathsep + current_path if current_path else "")
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "emp_kg.mcp_server"],
        cwd=str(_PROJECT_ROOT),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        initialize = _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        tools = _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        metadata = _send(proc, {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "emp_kg_metadata", "arguments": {"db_path": str(db_path)}},
        })
        context = _send(proc, {
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "emp_kg_context", "arguments": {"task": "login", "db_path": str(db_path), "lane": "tiny"}},
        })
        reason = _send(proc, {
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "emp_kg_reason", "arguments": {"task": "login", "db_path": str(db_path), "lane": "tiny"}},
        })
        reason_data = json.loads(reason["result"]["content"][0]["text"])
        unknown = _send(proc, {
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": {"name": "unknown", "arguments": {}},
        })
        return {
            "initialize_ok": initialize.get("result", {}).get("serverInfo", {}).get("name") == "emp-kg",
            "tool_count": len(tools.get("result", {}).get("tools", [])),
            "metadata_ok": not metadata.get("result", {}).get("isError", True),
            "context_ok": not context.get("result", {}).get("isError", True),
            "reason_ok": not reason.get("result", {}).get("isError", True),
            "reason_provider": reason_data.get("provider"),
            "reason_model": reason_data.get("model"),
            "reason_correlation_id": bool(reason_data.get("correlation_id")),
            "unknown_tool_rejected": unknown.get("result", {}).get("isError") is True,
            "db_unchanged": db_path.stat().st_size == before,
        }
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, type=Path)
    args = parser.parse_args()
    result = run_smoke_test(args.db)
    print(json.dumps(result, indent=2, sort_keys=True))
    required_checks = (
        "initialize_ok", "context_ok", "reason_ok", "reason_correlation_id",
        "unknown_tool_rejected", "db_unchanged",
    )
    if (
        not all(result.get(key) is True for key in required_checks)
        or result.get("tool_count", 0) <= 0
        or result.get("reason_provider") != "none"
        or result.get("reason_model") != "none"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
