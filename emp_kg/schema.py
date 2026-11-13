"""Loads and validates against emp-kg's bundled EMP graph schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.json"
_cache: dict[str, Any] | None = None


def load_schema() -> dict[str, Any]:
    """Lazy-load and cache the bundled schema."""
    global _cache
    if _cache is None:
        with open(_SCHEMA_PATH, encoding="utf-8") as f:
            _cache = json.load(f)
    return _cache


def node_kinds() -> set[str]:
    """Every valid node kind name declared in the schema."""
    return {nk["kind"] for nk in load_schema()["knowledge_graph"]["node_kinds"]}


def edge_types() -> dict[str, dict[str, str]]:
    """Edge type name -> {source, target, meaning}."""
    return {et["name"]: et for et in load_schema()["knowledge_graph"]["edge_types"]}


def implementation_scope() -> dict[str, Any]:
    """Machine-readable capabilities implemented by the current release."""
    return load_schema()["knowledge_graph"]["implementation_scope"]


def supported_structural_node_kinds() -> set[str]:
    return set(implementation_scope()["supported_structural_node_kinds"])


def supported_semantic_node_kinds() -> set[str]:
    return set(implementation_scope()["supported_semantic_node_kinds"])


def supported_edge_types() -> set[str]:
    scope = implementation_scope()
    return set(scope["supported_edge_types"]) | set(scope["partially_supported_edge_types"])


def supported_edge_sources(edge_name: str) -> set[str]:
    partial = implementation_scope()["partially_supported_edge_types"].get(edge_name)
    if partial is None:
        return set()
    return set(partial["supported_sources"])


def is_valid_node_kind(kind: str) -> bool:
    return kind in node_kinds()


def is_valid_edge(edge_name: str, source_kind: str, target_kind: str) -> bool:
    """True if edge_name exists and source_kind/target_kind are allowed for it."""
    types = edge_types()
    if edge_name not in types:
        return False
    allowed_sources = {s.strip() for s in types[edge_name]["source"].split("|")}
    allowed_targets = {t.strip() for t in types[edge_name]["target"].split("|")}
    return source_kind in allowed_sources and target_kind in allowed_targets
