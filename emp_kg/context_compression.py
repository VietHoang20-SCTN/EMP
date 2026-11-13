"""AI Context Compression Layer — deterministic fact-deduplication and compression."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from emp_kg.context import (
    ContextFact,
    ContextProvenance,
    ContextResult,
    _estimate_tokens,
    _provenance_text,
)


@dataclass(frozen=True)
class CompressionConfig:
    """Configurable compression parameters."""
    target_tokens: int = 2000
    max_facts: int = 50
    similarity_threshold: float = 0.95
    summary_score_threshold: float = 0.3
    preserve_full_provenance: bool = True


@dataclass(frozen=True)
class CompressedContext:
    """Result of context compression."""
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    retained_facts: list[ContextFact]
    dropped_fact_ids: list[str]
    merged_evidence: list[dict[str, Any]]
    summary_facts: list[ContextFact]
    config_used: CompressionConfig


def _fact_token_cost(fact: ContextFact, config: CompressionConfig) -> int:
    if config.preserve_full_provenance:
        return _estimate_tokens(
            f"{fact.name} {fact.kind} {fact.snippet} {fact.source_path or ''} "
            f"{json.dumps(fact.score_components, sort_keys=True)}"
        )
    return _estimate_tokens(f"{fact.name} {fact.kind} {fact.snippet}")


def _are_similar(a: ContextFact, b: ContextFact, threshold: float) -> bool:
    if a.node_id != b.node_id:
        return False
    if abs(a.relevance_score - b.relevance_score) > 1.0 - threshold:
        return False
    a_text = f"{a.name} {a.snippet}"
    b_text = f"{b.name} {b.snippet}"
    return a_text == b_text


def _merge_provenance_facts(facts: list[ContextFact]) -> list[ContextFact]:
    """Collapse facts with identical node_id into strongest-score fact."""
    by_node: dict[str, list[ContextFact]] = {}
    for fact in facts:
        by_node.setdefault(fact.node_id, []).append(fact)
    result = []
    for node_id, group in by_node.items():
        best = max(group, key=lambda f: f.relevance_score)
        if len(group) > 1:
            result.append(best)
        else:
            result.append(best)
    return result


def _deduplicate(facts: list[ContextFact], config: CompressionConfig) -> tuple[list[ContextFact], list[dict[str, Any]]]:
    seen: list[ContextFact] = []
    dropped: list[dict[str, Any]] = []
    for fact in facts:
        if any(_are_similar(fact, existing, config.similarity_threshold) for existing in seen):
            dropped.append({"node_id": fact.node_id, "reason": "duplicate"})
            continue
        seen.append(fact)
    return seen, dropped


def _merge_evidence(deduped: list[ContextFact]) -> tuple[list[ContextFact], list[dict[str, Any]]]:
    """Merge facts with identical primary provenance path."""
    seen_paths: dict[tuple[Any, ...], str] = {}
    result: list[ContextFact] = []
    merged: list[dict[str, Any]] = []
    for fact in deduped:
        path_key = (
            fact.provenance.seed_id,
            tuple((s.kind, s.source_id, s.target_id) for s in fact.provenance.path),
        ) if fact.provenance else None
        if path_key is not None:
            if path_key in seen_paths:
                merged.append({
                    "node_id": fact.node_id,
                    "merged_into": seen_paths[path_key],
                    "path_key": list(path_key),
                })
                continue
            seen_paths[path_key] = fact.node_id
        result.append(fact)
    return result, merged


def _summarize(facts: list[ContextFact], config: CompressionConfig) -> tuple[list[ContextFact], list[ContextFact]]:
    retained: list[ContextFact] = []
    summarized: list[ContextFact] = []
    for fact in facts:
        if fact.relevance_score < config.summary_score_threshold:
            summary = ContextFact(
                fact_type=fact.fact_type,
                node_id=fact.node_id,
                name=fact.name,
                kind=fact.kind,
                snippet="[summarized]",
                relevance_score=fact.relevance_score,
                source_path=fact.source_path,
                provenance=fact.provenance if config.preserve_full_provenance else None,
            )
            summarized.append(summary)
        else:
            retained.append(fact)
    return retained, summarized


def compress_context(
    result: ContextResult,
    config: CompressionConfig | None = None,
) -> CompressedContext:
    """Compress a context packet by deduplication, evidence merge, and summarization."""
    config = config or CompressionConfig()
    original_tokens = result.total_tokens

    # Step 1: deduplicate
    deduped, dropped = _deduplicate(result.facts, config)

    # Step 2: merge evidence paths
    merged, merged_evidence = _merge_evidence(deduped)

    # Step 3: summarize low-score facts
    retained, summarized = _summarize(merged, config)

    # Step 4: budget enforcement
    selected: list[ContextFact] = []
    summary_slots: list[ContextFact] = []
    used_tokens = 0
    for fact in retained:
        cost = _fact_token_cost(fact, config)
        if used_tokens + cost > config.target_tokens:
            summary_slots.append(ContextFact(
                fact_type=fact.fact_type, node_id=fact.node_id,
                name=fact.name, kind=fact.kind, snippet="[summarized]",
                relevance_score=fact.relevance_score, source_path=fact.source_path,
                provenance=fact.provenance if config.preserve_full_provenance else None,
            ))
            continue
        if len(selected) >= config.max_facts:
            summary_slots.append(ContextFact(
                fact_type=fact.fact_type, node_id=fact.node_id,
                name=fact.name, kind=fact.kind, snippet="[summarized]",
                relevance_score=fact.relevance_score, source_path=fact.source_path,
                provenance=fact.provenance if config.preserve_full_provenance else None,
            ))
            continue
        selected.append(fact)
        used_tokens += cost

    compressed_tokens = used_tokens
    return CompressedContext(
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=round(compressed_tokens / original_tokens, 4) if original_tokens > 0 else 0,
        retained_facts=selected,
        dropped_fact_ids=[f["node_id"] for f in dropped],
        merged_evidence=merged_evidence,
        summary_facts=summarized + summary_slots,
        config_used=config,
    )
