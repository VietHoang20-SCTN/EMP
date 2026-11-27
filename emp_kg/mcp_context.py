"""MCP-compatible context protocol adapter for emp-kg."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from emp_kg import benchmark, context, impact, store, system
from emp_kg.context import ContextRetrievalError, ContextResult
from emp_kg.reasoning import AnthropicUnavailableError


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MCPContextRequest:
    task: str
    db_path: str
    lane: Literal["auto", "tiny", "normal", "high-risk"] = "auto"
    affected_files: list[str] = field(default_factory=list)
    direction: Literal["in", "out", "both"] = "both"
    max_tokens: int | None = None
    encoding: str | None = None
    semantic_mode: Literal["off", "local"] = "off"
    embedding_model: str = "all-MiniLM-L6-v2"
    semantic_weight: float = 0.85
    output_format: Literal["json", "markdown", "text"] = "json"
    reasoning_provider: Literal["noop", "anthropic"] = "noop"
    reasoning_model: str = "claude-opus-5"
    reasoning_max_tokens: int = 4096
    reasoning_timeout: float = 30.0


@dataclass(frozen=True)
class MCPMetadataRequest:
    db_path: str


@dataclass(frozen=True)
class MCPBenchmarkRequest:
    db_path: str
    cases_path: str
    mode: Literal["bm25", "synonyms", "local"] = "bm25"
    k: int | None = None
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class MCPResponse:
    ok: bool
    correlation_id: str
    elapsed_ms: float
    data: dict[str, Any] | None = None
    error: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_file():
        raise ValueError(f"database not found: {path_str}")
    return p.resolve()


def _validate_task(task: str) -> str:
    task = task.strip()
    if not task:
        raise ValueError("task must not be empty")
    if len(task) > 8192:
        raise ValueError("task too long (max 8192 chars)")
    return task


# ---------------------------------------------------------------------------
# Tool descriptors (for MCP schema exposure)
# ---------------------------------------------------------------------------

def handle_impact(request: MCPContextRequest) -> MCPResponse:
    correlation_id = uuid.uuid4().hex[:16]
    start = time.perf_counter()
    try:
        validated_path = _validate_path(request.db_path)
        report = impact.analyze_impact(
            validated_path,
            impact.ImpactRequest(
                target_id=request.task,
                change_type="modified",
                depth=3,
                direction=request.direction,
            ),
        )
        from dataclasses import asdict as dataclass_asdict
        elapsed = (time.perf_counter() - start) * 1000
        return MCPResponse(
            ok=True,
            correlation_id=correlation_id,
            elapsed_ms=round(elapsed, 3),
            data=dataclass_asdict(report),
            provenance={"target": report.target_id, "risk_level": report.risk_level},
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return MCPResponse(ok=False, correlation_id=correlation_id, elapsed_ms=round(elapsed, 3), error=str(exc))


def handle_reason(request: MCPContextRequest) -> MCPResponse:
    correlation_id = uuid.uuid4().hex[:16]
    start = time.perf_counter()
    try:
        validated_path = _validate_path(request.db_path)
        validated_task = _validate_task(request.task)
        lane_override = None if request.lane == "auto" else request.lane
        system_result = system.reason_over_context(
            context.ContextRequest(
                task_description=validated_task,
                lane=lane_override,
                affected_files=list(request.affected_files),
                max_tokens=request.max_tokens,
                direction=request.direction,
                semantic_mode=request.semantic_mode,
                embedding_model=request.embedding_model,
                semantic_weight=request.semantic_weight,
                encoding=request.encoding,
            ),
            validated_path,
            provider=request.reasoning_provider,
            model=request.reasoning_model,
            max_tokens=request.reasoning_max_tokens,
            timeout=request.reasoning_timeout,
        )
        elapsed = (time.perf_counter() - start) * 1000
        reasoning_payload = asdict(system_result.reasoning)
        data = {
            "context": system_result.context.to_dict(),
            "reasoning": reasoning_payload,
            "correlation_id": correlation_id,
            "provider": system_result.reasoning.provider,
            "model": system_result.reasoning.model,
            "elapsed_ms": round(elapsed, 3),
        }
        for key in ("fallback_occurred", "served_model"):
            if key in system_result.reasoning.metadata:
                data[key] = system_result.reasoning.metadata[key]
        return MCPResponse(
            ok=True,
            correlation_id=correlation_id,
            elapsed_ms=round(elapsed, 3),
            data=data,
            provenance={
                "provider": system_result.reasoning.provider,
                "model": system_result.reasoning.model,
                "selected_facts": len(system_result.context.facts),
            },
        )
    except (ContextRetrievalError, RuntimeError, ValueError) as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return MCPResponse(ok=False, correlation_id=correlation_id, elapsed_ms=round(elapsed, 3), error=str(exc))
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return MCPResponse(ok=False, correlation_id=correlation_id, elapsed_ms=round(elapsed, 3), error=f"internal: {exc}")


TOOLS: list[dict[str, Any]] = [
    {
        "name": "emp_kg_context",
        "description": "Build bounded task-specific context from a persisted emp-kg graph database.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task description to retrieve context for."},
                "db_path": {"type": "string", "description": "Path to the emp-kg graph database file."},
                "lane": {"type": "string", "enum": ["auto", "tiny", "normal", "high-risk"], "default": "auto"},
                "affected_files": {"type": "array", "items": {"type": "string"}},
                "direction": {"type": "string", "enum": ["in", "out", "both"], "default": "both"},
                "max_tokens": {"type": "integer"},
                "encoding": {"type": "string"},
                "semantic_mode": {"type": "string", "enum": ["off", "local"], "default": "off"},
                "embedding_model": {"type": "string", "default": "all-MiniLM-L6-v2"},
                "semantic_weight": {"type": "number", "default": 0.85},
                "output_format": {"type": "string", "enum": ["json", "markdown", "text"], "default": "json"},
            },
            "required": ["task", "db_path"],
        },
    },
    {
        "name": "emp_kg_impact",
        "description": "Analyze change impact blast radius for a graph entity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Target node ID to analyze."},
                "db_path": {"type": "string"},
                "direction": {"type": "string", "enum": ["in", "out", "both"], "default": "both"},
            },
            "required": ["task", "db_path"],
        },
    },
    {
        "name": "emp_kg_reason",
        "description": "Build context then run provider-selected AI reasoning.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "db_path": {"type": "string"},
                "reasoning_provider": {"type": "string", "enum": ["noop", "anthropic"], "default": "noop"},
                "reasoning_model": {"type": "string", "default": "claude-opus-5"},
                "reasoning_max_tokens": {"type": "integer", "default": 4096},
                "reasoning_timeout": {"type": "number", "default": 30.0},
            },
            "required": ["task", "db_path"],
        },
    },
    {
        "name": "emp_kg_metadata",
        "description": "Return persisted graph metadata for a database.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "db_path": {"type": "string", "description": "Path to the emp-kg graph database file."},
            },
            "required": ["db_path"],
        },
    },
    {
        "name": "emp_kg_benchmark",
        "description": "Run labeled retrieval quality benchmarks against a database.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "db_path": {"type": "string", "description": "Path to the emp-kg graph database file."},
                "cases_path": {"type": "string", "description": "Path to benchmark JSON fixture."},
                "mode": {"type": "string", "enum": ["bm25", "synonyms", "local"], "default": "bm25"},
                "k": {"type": "integer"},
                "embedding_model": {"type": "string", "default": "all-MiniLM-L6-v2"},
            },
            "required": ["db_path", "cases_path"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def handle_context(request: MCPContextRequest) -> MCPResponse:
    """Handle a context build request."""
    correlation_id = uuid.uuid4().hex[:16]
    start = time.perf_counter()
    try:
        validated_path = _validate_path(request.db_path)
        validated_task = _validate_task(request.task)
        lane_override = None if request.lane == "auto" else request.lane
        ctx_result = context.build_context(
            context.ContextRequest(
                task_description=validated_task,
                lane=lane_override,
                affected_files=list(request.affected_files),
                max_tokens=request.max_tokens,
                direction=request.direction,
                semantic_mode=request.semantic_mode,
                embedding_model=request.embedding_model,
                semantic_weight=request.semantic_weight,
                encoding=request.encoding,
            ),
            validated_path,
        )
        elapsed = (time.perf_counter() - start) * 1000
        data = ctx_result.to_dict()
        data.update({
            "correlation_id": correlation_id,
            "provider": "none",
            "model": "none",
            "elapsed_ms": round(elapsed, 3),
        })
        return MCPResponse(
            ok=True,
            correlation_id=correlation_id,
            elapsed_ms=round(elapsed, 3),
            data=data,
            provenance={"selected_facts": len(ctx_result.facts), "lane": ctx_result.lane},
        )
    except ContextRetrievalError as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return MCPResponse(ok=False, correlation_id=correlation_id, elapsed_ms=round(elapsed, 3), error=str(exc))
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return MCPResponse(ok=False, correlation_id=correlation_id, elapsed_ms=round(elapsed, 3), error=f"internal: {exc}")


def handle_metadata(request: MCPMetadataRequest) -> MCPResponse:
    correlation_id = uuid.uuid4().hex[:16]
    start = time.perf_counter()
    try:
        validated_path = _validate_path(request.db_path)
        conn = store.open_readonly(validated_path)
        meta = store.get_metadata(conn)
        graph = store.graph_stats(conn)
        conn.close()
        elapsed = (time.perf_counter() - start) * 1000
        return MCPResponse(ok=True, correlation_id=correlation_id, elapsed_ms=round(elapsed, 3), data={"metadata": meta, "graph": graph})
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return MCPResponse(ok=False, correlation_id=correlation_id, elapsed_ms=round(elapsed, 3), error=str(exc))


def handle_benchmark(request: MCPBenchmarkRequest) -> MCPResponse:
    correlation_id = uuid.uuid4().hex[:16]
    start = time.perf_counter()
    try:
        validated_path = _validate_path(request.db_path)
        cases_path = Path(request.cases_path)
        cases = benchmark.load_cases(cases_path)
        result = benchmark.run_benchmark(
            validated_path, cases,
            mode=request.mode,
            default_k=request.k,
            embedding_model=request.embedding_model,
        )
        elapsed = (time.perf_counter() - start) * 1000
        return MCPResponse(ok=True, correlation_id=correlation_id, elapsed_ms=round(elapsed, 3), data=result)
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return MCPResponse(ok=False, correlation_id=correlation_id, elapsed_ms=round(elapsed, 3), error=str(exc))
