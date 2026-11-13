"""Capture one MCP stdio transport transcript with graph immutability proof."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(_PROJECT_ROOT),
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _checksum(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), path.stat().st_size


def _send(proc: subprocess.Popen[str], message: dict[str, Any]) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("MCP server closed without response")
    return json.loads(line), round((time.perf_counter() - started) * 1000, 3)


def _extract_correlation(response: dict[str, Any]) -> str | None:
    try:
        text = response["result"]["content"][0]["text"]
        payload = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None
    value = payload.get("correlation_id")
    return value if isinstance(value, str) else None


def capture_transcript(
    db_path: Path,
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    client_label: str = "client_neutral_transport",
    timeout: float = 30.0,
) -> dict[str, Any]:
    db_path = db_path.resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Graph database not found: {db_path}")
    before_hash, before_size = _checksum(db_path)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "emp_kg.mcp_server"],
        cwd=str(_PROJECT_ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    initialize_request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    tool_request = {
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": tool_name, "arguments": tool_args},
    }
    try:
        initialize_response, initialize_ms = _send(proc, initialize_request)
        tool_response, tool_ms = _send(proc, tool_request)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
    after_hash, after_size = _checksum(db_path)
    return {
        "artifact": "mcp_transport_transcript",
        "client": client_label,
        "emp_kg_commit": _git_sha(),
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "db_path": str(db_path),
        "db_sha256_before": before_hash,
        "db_size_bytes_before": before_size,
        "initialize": {
            "request": initialize_request,
            "response": initialize_response,
            "latency_ms": initialize_ms,
        },
        "tool_call": {
            "request": tool_request,
            "response": tool_response,
            "latency_ms": tool_ms,
            "correlation_id": _extract_correlation(tool_response),
        },
        "db_sha256_after": after_hash,
        "db_size_bytes_after": after_size,
        "db_unchanged": before_hash == after_hash and before_size == after_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one emp-kg MCP transport transcript.")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--args", required=True, help="JSON object of tool arguments")
    parser.add_argument("--client", default="client_neutral_transport")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tool_args = json.loads(args.args)
    transcript = capture_transcript(args.db, args.tool, tool_args, client_label=args.client)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(transcript, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(transcript, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
