"""Impact Analysis Engine — compute blast radius from a changed graph entity."""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from emp_kg import store


_HIGH_RISK_TRIGGERS = frozenset({
    "auth", "authentication", "authorization", "security", "audit",
    "migration", "provider", "payment", "schema", "delete", "data",
})


@dataclass(frozen=True)
class ImpactRequest:
    target_id: str
    change_type: str = "modified"
    depth: int = 3
    direction: str = "both"

    def __post_init__(self) -> None:
        if self.change_type not in {"modified", "deleted", "renamed"}:
            raise ValueError(f"invalid change_type: {self.change_type}")
        if not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError("depth must be non-negative")
        if self.direction not in {"in", "out", "both"}:
            raise ValueError(f"invalid direction: {self.direction}")


@dataclass
class ImpactReport:
    target_id: str
    change_type: str
    directly_affected: list[dict[str, Any]]
    transitive_affected: list[dict[str, Any]]
    edge_summary: dict[str, int]
    risk_level: str
    risk_reasons: list[str]
    affected_decisions: list[str]
    affected_stories: list[str]
    recommended_tests: list[str]
    elapsed_ms: float


def _classify_risk(
    directly_affected: list[dict[str, Any]],
    transitive_affected: list[dict[str, Any]],
    affected_decisions: list[str],
    affected_stories: list[str],
    edge_summary: dict[str, int],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    total = len(directly_affected) + len(transitive_affected)
    if affected_decisions or affected_stories:
        reasons.append(f"affected {len(affected_decisions)} decisions, {len(affected_stories)} stories")
    if edge_summary.get("AFFECTS", 0) > 0:
        reasons.append("AFFECTS edges traversed")
    if any(kind in _HIGH_RISK_TRIGGERS for item in directly_affected for kind in item.get("kind", "").split()):
        reasons.append("high-risk node type affected")
    if any(kind in _HIGH_RISK_TRIGGERS for item in transitive_affected for kind in item.get("kind", "").split()):
        reasons.append("high-risk transitive dependency")
    if total >= 10:
        reasons.append(f"large blast radius ({total} nodes)")
    if not reasons:
        return "low", ["minimal impact"]
    if len(reasons) >= 2 or any("high-risk" in r for r in reasons):
        return "high", reasons
    return "medium", reasons


def analyze_impact(db_path: Path, request: ImpactRequest) -> ImpactReport:
    start = time.perf_counter()
    conn = store.open_readonly(db_path)
    try:
        target_nodes = store.query_by_ids(conn, [request.target_id])
        directly_affected: list[dict[str, Any]] = []
        transitive_affected: list[dict[str, Any]] = []
        seen = {request.target_id}
        edge_counter: Counter = Counter()
        frontier = [request.target_id]

        for depth in range(1, request.depth + 1):
            next_frontier: list[str] = []
            for node_id in frontier:
                neighbors = store.neighbors(conn, node_id, request.direction)
                for record in neighbors:
                    edge = record["edge"]
                    neighbor_id = edge["target"] if record["direction"] == "out" else edge["source"]
                    if neighbor_id in seen:
                        continue
                    seen.add(neighbor_id)
                    edge_counter[edge["kind"]] += 1
                    hydrated = store.query_by_ids(conn, [neighbor_id])
                    entry = hydrated[0] if hydrated else {"id": neighbor_id, "kind": "unknown", "name": neighbor_id}
                    path = store.shortest_path(conn, request.target_id, neighbor_id, max_depth=depth, direction=request.direction)
                    enriched = {**entry, "edge_kind": edge["kind"], "distance": depth, "evidence_path": [n.get("id", n) for n in path["nodes"]]}
                    if depth == 1:
                        directly_affected.append(enriched)
                    else:
                        transitive_affected.append(enriched)
                    next_frontier.append(neighbor_id)
            frontier = next_frontier
            if not frontier:
                break

        affected_decisions = [n["id"] for n in directly_affected + transitive_affected if n.get("kind") == "decision_ref"]
        affected_stories = [n["id"] for n in directly_affected + transitive_affected if n.get("kind") == "story"]
        recommended_tests = [
            n["id"] for n in directly_affected + transitive_affected
            if isinstance(n.get("name"), str) and n["name"].startswith("test_")
        ]

        risk_level, risk_reasons = _classify_risk(
            directly_affected, transitive_affected,
            affected_decisions, affected_stories, dict(edge_counter),
        )

        return ImpactReport(
            target_id=request.target_id,
            change_type=request.change_type,
            directly_affected=directly_affected,
            transitive_affected=transitive_affected,
            edge_summary=dict(edge_counter),
            risk_level=risk_level,
            risk_reasons=risk_reasons,
            affected_decisions=affected_decisions,
            affected_stories=affected_stories,
            recommended_tests=recommended_tests,
            elapsed_ms=round((time.perf_counter() - start) * 1000, 3),
        )
    finally:
        conn.close()
