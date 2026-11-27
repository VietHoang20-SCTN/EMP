"""MCP stdio JSON-RPC 2.0 server for emp-kg."""
from __future__ import annotations

import json
import sys
from typing import Any, TextIO
from dataclasses import asdict

from emp_kg import mcp_context


SERVER_INFO = {"name": "emp-kg", "version": "0.1.0"}
PROTOCOL_VERSION = "2024-11-05"


def _jsonrpc_response(
    id: Any,
    result: Any | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {"jsonrpc": "2.0", "id": id}
    if error is not None:
        response["error"] = error
    else:
        response["result"] = result
    return response


def _jsonrpc_error(code: int, message: str) -> dict[str, Any]:
    return {"code": code, "message": message}


def _handle_initialize(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "serverInfo": SERVER_INFO,
        "capabilities": {"tools": {}},
    }


def _handle_tools_list() -> dict[str, Any]:
    return {"tools": mcp_context.TOOLS}


def _handle_tools_call(params: dict[str, Any]) -> dict[str, Any]:
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})
    handler_map = {
        "emp_kg_context": lambda args: mcp_context.handle_context(mcp_context.MCPContextRequest(**args)),
        "emp_kg_impact": lambda args: mcp_context.handle_impact(mcp_context.MCPContextRequest(task=args["task"], db_path=args["db_path"], direction=args.get("direction", "both"))),
        "emp_kg_reason": lambda args: mcp_context.handle_reason(mcp_context.MCPContextRequest(**args)),
        "emp_kg_metadata": lambda args: mcp_context.handle_metadata(mcp_context.MCPMetadataRequest(**args)),
        "emp_kg_benchmark": lambda args: mcp_context.handle_benchmark(mcp_context.MCPBenchmarkRequest(**args)),
    }
    handler = handler_map.get(tool_name)
    if handler is None:
        return {"content": [{"type": "text", "text": f"unknown tool: {tool_name}"}], "isError": True}
    response = handler(arguments)
    content = [{"type": "text", "text": json.dumps(response.data if response.ok else {"error": response.error}, default=str)}]
    return {"content": content, "isError": not response.ok}


def _handle_request(method: str, params: dict[str, Any]) -> dict[str, Any] | None:
    if method == "initialize":
        return _handle_initialize(params)
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return _handle_tools_list()
    if method == "tools/call":
        return _handle_tools_call(params)
    return None


def _process_message(line: str, out: TextIO) -> None:
    line = line.strip()
    if not line:
        return
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        out.write(json.dumps(_jsonrpc_response(None, error=_jsonrpc_error(-32700, "parse error"))) + "\n")
        out.flush()
        return

    request_id = message.get("id")
    method = message.get("method", "")
    params = message.get("params", {})

    if method == "notifications/initialized" or request_id is None:
        return

    result = _handle_request(method, params)
    if result is None:
        out.write(json.dumps(_jsonrpc_response(request_id, error=_jsonrpc_error(-32601, "method not found"))) + "\n")
    else:
        out.write(json.dumps(_jsonrpc_response(request_id, result=result)) + "\n")
    out.flush()


def run_stdio(stdin: TextIO = sys.stdin, out: TextIO = sys.stdout) -> None:
    """Process newline-delimited JSON-RPC from stdin, write responses to stdout."""
    for line in stdin:
        _process_message(line, out)


def main() -> None:
    run_stdio()


if __name__ == "__main__":
    main()
