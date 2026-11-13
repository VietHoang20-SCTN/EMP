"""Capture sanitized offline or one-shot Anthropic provider evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from emp_kg import context
from emp_kg.reasoning import AnthropicReasoner, NoOpReasoner, ReasoningRequest

DEFAULT_MODEL = "claude-opus-5"
TOKEN_CAP = 512
TIMEOUT_CAP = 30.0
DEFAULT_THINKING = "adaptive"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(_PROJECT_ROOT), capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _db_checksum(db_path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with db_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), db_path.stat().st_size


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _gateway_host() -> str | None:
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if not base_url:
        return None
    return urlparse(base_url).hostname


def _response_payload(result: Any, mode: str) -> dict[str, Any]:
    metadata = result.metadata or {}
    return {
        "ok": result.ok,
        "provider": result.provider,
        "model": result.model,
        "served_model": metadata.get("served_model"),
        "stop_reason": metadata.get("stop_reason"),
        "input_tokens": metadata.get("input_tokens"),
        "output_tokens": metadata.get("output_tokens"),
        "cache_creation_input_tokens": metadata.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": metadata.get("cache_read_input_tokens"),
        "latency_ms": result.elapsed_ms,
        "correlation_id": result.correlation_id,
        "citation_valid": result.citation_valid,
        "citation_fact_ids": [fact.fact_id for fact in result.citation_facts],
        "unknown_citation_ids": result.unknown_citation_ids,
        "error": result.error,
        "error_code": result.error_code,
        "retryable": result.retryable,
        "fallback_occurred": bool(metadata.get("fallback_occurred", False)),
        "refusal": result.refusal,
        "text": result.text,
        "mode": mode,
    }


def capture_provider_evidence(
    db_path: Path,
    task: str = "coverage",
    *,
    model: str = DEFAULT_MODEL,
    thinking: str = DEFAULT_THINKING,
    max_tokens: int = TOKEN_CAP,
    execute: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    db_path = db_path.resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"Graph database not found: {db_path}")
    if thinking not in {"adaptive", "disabled"}:
        raise ValueError(f"unsupported thinking mode: {thinking}")
    if max_tokens <= 0 or max_tokens > 1024:
        raise ValueError("max_tokens must be between 1 and 1024")
    before_hash, before_size = _db_checksum(db_path)
    build_started = time.perf_counter()
    ctx = context.build_context(context.ContextRequest(task), db_path)
    context_elapsed = round((time.perf_counter() - build_started) * 1000, 3)
    request = ReasoningRequest(task=task, context=ctx, output_size=max_tokens, timeout_seconds=TIMEOUT_CAP)
    prompt = AnthropicReasoner._prompt(request)
    if execute:
        reasoner = AnthropicReasoner(
            model=model, max_tokens=max_tokens, timeout=TIMEOUT_CAP,
            max_retries=0,
        )
        reasoner.thinking_mode = thinking
        mode = "execute"
    else:
        reasoner = NoOpReasoner()
        mode = "dry-run"
    result = reasoner.reason(request)
    after_hash, after_size = _db_checksum(db_path)
    envelope = {
        "artifact": "provider_evidence",
        "mode": mode,
        "emp_kg_commit": _git_sha(),
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
        "gateway_host": _gateway_host(),
        "model_requested": model,
        "token_cap": max_tokens,
        "timeout_cap_s": TIMEOUT_CAP,
        "task": task,
        "db_path": str(db_path),
        "db_sha256_before": before_hash,
        "db_size_bytes_before": before_size,
        "request": {
            "max_tokens": max_tokens,
            "model": model,
            "thinking": {"type": thinking},
            "user_prompt_hash": _sha256_text(prompt),
        },
        "context_build": {
            "lane": ctx.lane,
            "selected_facts": len(ctx.facts),
            "total_tokens": ctx.total_tokens,
            "elapsed_ms": context_elapsed,
        },
        "response": _response_payload(result, mode),
        "db_sha256_after": after_hash,
        "db_size_bytes_after": after_size,
        "db_unchanged": before_hash == after_hash and before_size == after_size,
        "sanitized": {
            "api_key_redacted": True,
            "auth_token_redacted": True,
            "base_url_redacted": True,
            "prompt_content_replaced_with_hash": True,
        },
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return envelope


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture sanitized emp-kg provider evidence.")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--task", default="coverage")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--thinking", choices=("adaptive", "disabled"), default=DEFAULT_THINKING)
    parser.add_argument("--max-tokens", type=int, default=TOKEN_CAP)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Make exactly one capped provider request.")
    args = parser.parse_args()
    envelope = capture_provider_evidence(
        args.db, args.task, model=args.model, thinking=args.thinking,
        max_tokens=args.max_tokens, execute=args.execute, output_path=args.output,
    )
    print(json.dumps(envelope, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
