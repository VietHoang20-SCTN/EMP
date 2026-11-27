"""Deterministic final-week evidence runner for emp-kg.

Captures one reproducible JSON evidence document containing graph
identity, offline MCP smoke proof, retrieval benchmark quality,
context latency and token utilization, and explicit pending markers
for manual client/provider evidence.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from emp_kg import benchmark, context, store
from emp_kg.schema import load_schema
from scripts.mcp_smoke_test import run_smoke_test

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EVIDENCE_PATH = _PROJECT_ROOT / "docs" / "evidence" / "final_validation.json"


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except OSError:
        return "unavailable"


def run_final_validation(
    *,
    db_path: Path | None = None,
    benchmark_cases: list[dict[str, Any]] | None = None,
    benchmark_cases_path: Path | None = None,
    case_ids: list[str] | None = None,
    case_k: int | None = None,
    latency_repeat: int = 2,
    full_suite_verified: bool = False,
) -> dict[str, Any]:
    if db_path is None:
        db_path = _PROJECT_ROOT / "emp-kg.db"

    if not db_path.is_file():
        raise FileNotFoundError(f"Graph database not found: {db_path}")

    graph_db_size_bytes = db_path.stat().st_size
    conn_ro = store.open_readonly(db_path)
    try:
        metadata = store.get_metadata(conn_ro)
        graph = store.graph_stats(conn_ro)
    finally:
        conn_ro.close()

    if benchmark_cases_path is None:
        benchmark_cases_path = _PROJECT_ROOT / "tests" / "fixtures" / "context_benchmark.json"
    if benchmark_cases is None:
        benchmark_cases = benchmark.load_cases(benchmark_cases_path)
    if case_ids:
        benchmark_cases = [case for case in benchmark_cases if case["id"] in case_ids]
    if not benchmark_cases:
        raise ValueError("benchmark cases empty after filtering")
    benchmark_cases = [
        {
            **case,
            "lane": case.get("lane", "tiny"),
            "affected_files": list(case.get("affected_files", [])),
            "k": case.get("k", 10),
        }
        for case in benchmark_cases
    ]

    effective_k = case_k
    bm25 = benchmark.run_benchmark(db_path, benchmark_cases, mode="bm25", default_k=effective_k)
    synonyms = benchmark.run_benchmark(db_path, benchmark_cases, mode="synonyms", default_k=effective_k)

    retrieval_timings: list[dict[str, Any]] = []
    for case in benchmark_cases:
        timings: list[float] = []
        selected_facts_samples: list[int] = []
        total_tokens_samples: list[int] = []
        for _ in range(max(1, latency_repeat)):
            start = time.perf_counter()
            result = context.build_context(
                context.ContextRequest(
                    task_description=case["task"],
                    lane=case.get("lane"),
                    affected_files=list(case.get("affected_files", [])),
                ),
                db_path,
            )
            timings.append((time.perf_counter() - start) * 1000)
            selected_facts_samples.append(len(result.facts))
            total_tokens_samples.append(result.total_tokens)
        retrieval_timings.append({
            "case_id": case["id"],
            "median_ms": sorted(timings)[len(timings) // 2],
            "selected_facts_avg": sum(selected_facts_samples) / len(selected_facts_samples),
            "total_tokens_avg": sum(total_tokens_samples) / len(total_tokens_samples),
        })
    aggregate_latency_ms = sum(item["median_ms"] for item in retrieval_timings) / len(retrieval_timings)
    aggregate_total_tokens = sum(item["total_tokens_avg"] for item in retrieval_timings) / len(retrieval_timings)
    aggregate_selected_facts = sum(item["selected_facts_avg"] for item in retrieval_timings) / len(retrieval_timings)

    try:
        smoke = run_smoke_test(db_path)
    except Exception as exc:
        smoke = {"ok": False, "error": str(exc)}

    payload: dict[str, Any] = {
        "artifact": "final_validation",
        "repository": str(_PROJECT_ROOT),
        "emp_kg_commit": _git_sha(),
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
        "schema_version": load_schema().get("version"),
        "graph": {
            "db_path": str(db_path),
            "db_size_bytes": graph_db_size_bytes,
            "metadata": metadata,
            "stats": graph,
        },
        "retrieval_benchmark": {
            "fixture": str(benchmark_cases_path),
            "case_count": len(benchmark_cases),
            "modes": {
                "bm25": bm25["aggregate"],
                "synonyms": synonyms["aggregate"],
            },
        },
        "context_retrieval": {
            "repeat_count": max(1, latency_repeat),
            "aggregate_ms": aggregate_latency_ms,
            "aggregate_selected_facts": aggregate_selected_facts,
            "aggregate_total_tokens": aggregate_total_tokens,
            "cases": retrieval_timings,
        },
        "mcp_smoke": smoke,
        "readiness_flags": {
            "full_suite_verified": full_suite_verified,
            "offline_evidence_verified": False,
            "live_client_evidence": "available_partial",
            "live_client_tool_coverage": {
                "claude_code": ["emp_kg_metadata"],
                "cline": ["emp_kg_metadata", "emp_kg_context"],
                "codex_cli": ["emp_kg_metadata"],
            },
            "live_provider_evidence": "pending_optional_credentialed_run",
            "optimization_change_applied": False,
        },
    }

    if isinstance(smoke, dict) and smoke.get("initialize_ok") and smoke.get("context_ok") and smoke.get("reason_ok") and smoke.get("db_unchanged"):
        payload["readiness_flags"]["offline_evidence_verified"] = True

    return payload


def save_validation(
    payload: dict[str, Any] | None = None,
    *,
    output_path: Path = _EVIDENCE_PATH,
    db_path: Path | None = None,
    benchmark_cases_path: Path | None = None,
    latency_repeat: int = 2,
    full_suite_verified: bool = False,
) -> dict[str, Any]:
    if payload is None:
        payload = run_final_validation(
            db_path=db_path,
            benchmark_cases_path=benchmark_cases_path,
            latency_repeat=latency_repeat,
            full_suite_verified=full_suite_verified,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture final emp-kg validation evidence.")
    parser.add_argument("--db", type=Path, default=_PROJECT_ROOT / "emp-kg.db")
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=_EVIDENCE_PATH)
    parser.add_argument("--latency-repeat", type=int, default=2)
    parser.add_argument("--full-suite-verified", action="store_true")
    args = parser.parse_args()
    if args.latency_repeat <= 0:
        parser.error("--latency-repeat must be positive")
    payload = save_validation(
        output_path=args.output,
        db_path=args.db,
        benchmark_cases_path=args.cases,
        latency_repeat=args.latency_repeat,
        full_suite_verified=args.full_suite_verified,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
