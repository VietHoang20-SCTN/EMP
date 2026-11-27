"""Deterministic quality benchmarks for Context Builder retrieval."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from emp_kg.context import ContextRequest, build_context

_MODES = {"bm25", "synonyms", "local"}


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load and validate a versioned benchmark case file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("benchmark fixture must be an object with version 1")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark fixture must contain nonempty cases")
    validated = []
    seen_ids: set[str] = set()
    for raw in cases:
        if not isinstance(raw, dict):
            raise ValueError("benchmark case must be an object")
        case_id = raw.get("id")
        task = raw.get("task")
        expected = raw.get("expected_node_ids")
        k = raw.get("k", 10)
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_ids:
            raise ValueError("benchmark case id must be nonempty and unique")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"benchmark case {case_id!r} task must be nonempty")
        if not isinstance(expected, list) or not expected or any(
            not isinstance(node_id, str) or not node_id for node_id in expected
        ):
            raise ValueError(f"benchmark case {case_id!r} expected_node_ids must be nonempty strings")
        lane = raw.get("lane", "tiny")
        affected_files = raw.get("affected_files", [])
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError(f"benchmark case {case_id!r} k must be positive")
        if lane not in {"tiny", "normal", "high-risk"}:
            raise ValueError(f"benchmark case {case_id!r} lane is invalid")
        if not isinstance(affected_files, list) or any(
            not isinstance(path, str) or not path for path in affected_files
        ):
            raise ValueError(f"benchmark case {case_id!r} affected_files must be strings")
        seen_ids.add(case_id)
        validated.append({
            "id": case_id,
            "task": task,
            "expected_node_ids": list(dict.fromkeys(expected)),
            "lane": lane,
            "affected_files": affected_files,
            "k": k,
        })
    return validated


def _metrics(returned: list[str], expected: list[str], k: int) -> dict[str, float | int]:
    ranked = list(dict.fromkeys(returned))[:k]
    expected_set = set(expected)
    hits = [node_id for node_id in ranked if node_id in expected_set]
    first_rank = next(
        (index for index, node_id in enumerate(ranked, start=1) if node_id in expected_set),
        None,
    )
    return {
        "k": k,
        "returned_count": len(ranked),
        "hit_count": len(hits),
        "recall_at_k": len(hits) / len(expected_set),
        "precision_at_k": len(hits) / k,
        "mrr": 0.0 if first_rank is None else 1.0 / first_rank,
    }


def run_benchmark(
    db_path: Path,
    cases: list[dict[str, Any]],
    *,
    mode: str,
    default_k: int | None = None,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Run labeled tasks and return per-case plus macro retrieval metrics."""
    if mode not in _MODES:
        raise ValueError(f"invalid benchmark mode: {mode}")
    if default_k is not None and (isinstance(default_k, bool) or default_k <= 0):
        raise ValueError("default_k must be positive")
    results = []
    for case in cases:
        k = default_k if default_k is not None else case["k"]
        request = ContextRequest(
            task_description=case["task"],
            lane=case["lane"],
            affected_files=list(case["affected_files"]),
            semantic_mode="local" if mode == "local" else "off",
            synonym_mode="auto" if mode in {"synonyms", "local"} else "off",
            embedding_model=embedding_model,
        )
        result = build_context(request, db_path)
        returned = [fact.node_id for fact in result.facts]
        metrics = _metrics(returned, case["expected_node_ids"], k)
        results.append({
            "id": case["id"],
            "task": case["task"],
            "expected_node_ids": case["expected_node_ids"],
            "returned_node_ids": returned[:k],
            **metrics,
        })
    count = len(results)
    aggregate = {
        "case_count": count,
        "recall_at_k": sum(item["recall_at_k"] for item in results) / count,
        "precision_at_k": sum(item["precision_at_k"] for item in results) / count,
        "mrr": sum(item["mrr"] for item in results) / count,
    }
    return {"version": 1, "mode": mode, "aggregate": aggregate, "cases": results}
