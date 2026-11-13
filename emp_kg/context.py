"""Task-specific context assembly from an emp-kg graph database."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Literal

from emp_kg import store
from emp_kg.tokenizers import Tokenizer, get_tokenizer
from emp_kg.embeddings import (
    DEFAULT_MODEL,
    EmbeddingUnavailableError,
    semantic_seeds,
)

_LANES = {
    "tiny": {
        "depth": 1, "max_facts": 20, "max_tokens": 2_000,
        "search_limit": 10, "max_seeds": 20, "max_candidates": 100,
    },
    "normal": {
        "depth": 2, "max_facts": 50, "max_tokens": 5_000,
        "search_limit": 25, "max_seeds": 50, "max_candidates": 500,
    },
    "high-risk": {
        "depth": 3, "max_facts": 100, "max_tokens": 10_000,
        "search_limit": 50, "max_seeds": 100, "max_candidates": 1_000,
    },
}
_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "the", "to", "with",
}
_HIGH_RISK_TERMS = {
    "architecture", "audit", "auth", "authentication", "authorization", "cloud",
    "contract", "delete", "email", "migration", "password", "payment", "permission",
    "privacy", "provider", "queue", "retention", "role", "schema", "scope", "sdk",
    "security", "sensitive", "session", "token", "validation", "webhook",
}
_HIGH_RISK_PHRASES = {
    "data loss", "data model", "drop table", "external provider", "multi role",
    "purge data", "remove data", "remove records",
}
_SEMANTIC_TERM_GROUPS = (
    ("auth", "authentication", "login", "sign-in", "signin"),
    ("authorization", "authorize", "permission", "permissions", "access", "role", "roles"),
    ("token", "jwt", "credential", "credentials"),
    ("migration", "migrations", "schema", "upgrade", "backfill"),
    ("delete", "deletion", "remove", "removal", "purge", "erase"),
    ("audit", "auditing", "trace", "logging", "log"),
    ("provider", "providers", "integration", "webhook", "webhooks", "external"),
    ("payment", "payments", "billing", "invoice", "charge"),
    ("validation", "validate", "verification", "verify", "check"),
    ("error", "errors", "exception", "fault", "failure"),
    ("config", "configuration", "settings", "options"),
    ("database", "db", "storage", "store", "persistence"),
    ("api", "endpoint", "route", "handler"),
    ("test", "tests", "testing", "spec", "specification"),
    ("cache", "caching", "redis", "memoization"),
    ("queue", "broker", "worker", "job", "task"),
    ("logger", "logging", "trace", "telemetry"),
    ("deploy", "deployment", "release", "ship"),
    ("refactor", "restructure", "reorganize", "cleanup"),
    ("serialize", "serialization", "encode", "decode", "json"),
    ("ui", "interface", "frontend", "view", "screen"),
)
_MAX_SYNONYMS_PER_TERM = 4
_SYNONYM_SCORE_WEIGHT = 0.85
_MAX_PROVENANCE_PATHS = 3
_REQUEST_CLASSES = frozenset({
    "new_spec", "spec_slice", "change_request", "new_initiative",
    "maintenance_request", "harness_improvement",
})
_RISK_FLAGS = frozenset({
    "auth", "authorization", "data_model", "audit_security",
    "external_systems", "public_contracts", "cross_platform",
    "existing_behavior", "weak_proof", "multi_domain",
})
_HARD_GATE_FLAGS = frozenset({
    "auth", "authorization", "data_model", "audit_security",
    "external_systems",
})


@dataclass(frozen=True)
class ContextRequest:
    task_description: str
    lane: str | None = None
    affected_files: list[str] = field(default_factory=list)
    max_tokens: int | None = None
    direction: Literal["in", "out", "both"] = "both"
    request_class: str | None = None
    risk_flags: set[str] = field(default_factory=set)
    scope: Literal["behavioral", "cosmetic"] = "behavioral"
    semantic_mode: Literal["off", "local"] = "off"
    synonym_mode: Literal["off", "auto"] = "auto"
    embedding_model: str = DEFAULT_MODEL
    semantic_weight: float = 0.85
    encoding: str | None = None

    def __post_init__(self) -> None:
        if self.direction not in {"in", "out", "both"}:
            raise ValueError(f"invalid context direction: {self.direction}")
        if self.request_class is not None and self.request_class not in _REQUEST_CLASSES:
            raise ValueError(f"invalid request_class: {self.request_class}")
        unknown_flags = self.risk_flags - _RISK_FLAGS
        if unknown_flags:
            raise ValueError(f"invalid risk_flags: {sorted(unknown_flags)}")
        if self.scope not in {"behavioral", "cosmetic"}:
            raise ValueError(f"invalid scope: {self.scope}")
        if self.semantic_mode not in {"off", "local"}:
            raise ValueError(f"invalid semantic_mode: {self.semantic_mode}")
        if self.synonym_mode not in {"off", "auto"}:
            raise ValueError(f"invalid synonym_mode: {self.synonym_mode}")
        if not 0.0 <= self.semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be between 0.0 and 1.0")


class ContextRetrievalError(RuntimeError):
    """Graph retrieval failed instead of returning a valid empty result."""


@dataclass(frozen=True)
class ContextPathStep:
    """One graph edge traversed to retrieve a context fact."""

    source_id: str
    target_id: str
    kind: str
    direction: str
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ContextProvenance:
    """Deterministic primary retrieval path for one context fact."""

    seed_id: str
    seed_name: str
    seed_kind: str
    matched_terms: list[str]
    path: list[ContextPathStep]


@dataclass(frozen=True)
class ContextFact:
    fact_type: str
    node_id: str
    name: str
    kind: str
    snippet: str
    relevance_score: float
    source_path: str | None = None
    provenance: ContextProvenance | None = None
    alternate_paths: list[ContextProvenance] = field(default_factory=list)
    score_components: dict[str, float] = field(default_factory=dict)
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)
    match_location: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextResult:
    task_description: str
    lane: str
    classification_reason: str
    affected_files: list[str]
    budget_tokens: int
    graph_summary: dict[str, int]
    facts: list[ContextFact]
    sources: list[str]
    total_tokens: int
    encoding: str = "estimate"
    fact_tokens: int = 0
    rendered_tokens: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _keywords(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]*", text)
    seen: set[str] = set()
    result = []
    for word in words:
        normalized = word.casefold()
        if normalized in _STOPWORDS or len(normalized) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        result.append(word)
    return result


def _expand_semantic_terms(term: str) -> list[tuple[str, bool]]:
    """Return original plus bounded known paraphrases, deterministically."""
    normalized = term.casefold()
    for group in _SEMANTIC_TERM_GROUPS:
        if normalized in group:
            ordered = [normalized, *sorted(set(group) - {normalized})]
            return [(candidate, candidate != normalized) for candidate in ordered[: _MAX_SYNONYMS_PER_TERM + 1]]
    return [(term, False)]


def _search_matches(conn: Any, term: str, limit: int) -> list[dict[str, Any]]:
    try:
        matches = store.query_by_name_ranked(conn, f'"{term}"', limit=limit)
        chunks = store.query_source_chunks_ranked(conn, f'"{term}"', limit=limit)
    except sqlite3.OperationalError as exc:
        if "fts5" in str(exc).casefold() and "syntax" in str(exc).casefold():
            return []
        raise ContextRetrievalError(f"search failed for {term!r}") from exc
    except (sqlite3.DatabaseError, json.JSONDecodeError) as exc:
        raise ContextRetrievalError(f"search failed for {term!r}") from exc

    matched_ids = {node.get("id") for node in matches}
    best_chunks: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        best_chunks.setdefault(chunk["file_node_id"], chunk)
    for file_node_id, chunk in best_chunks.items():
        if file_node_id in matched_ids:
            continue
        hydrated = store.query_by_ids(conn, [file_node_id])
        if hydrated:
            matches.append({
                **hydrated[0],
                "_bm25_rank": chunk["bm25_rank"],
                "_chunk_match": chunk,
            })
    matches.sort(key=lambda node: (
        float(node.get("_bm25_rank", 0.0)), str(node.get("id", ""))
    ))
    return matches[:limit]


def _affected_file_matches(conn: Any, path: str, limit: int) -> tuple[str, list[dict[str, Any]]]:
    """Search exact file path first, then its basename when needed."""
    matches = _search_matches(conn, path, limit)
    if matches:
        return path, matches
    basename = Path(path).name
    if basename and basename != path:
        return basename, _search_matches(conn, basename, limit)
    return path, []


def _exact_node_ids(text: str) -> list[str]:
    """Extract known graph ID shapes before broad keyword search."""
    pattern = (
        r"(?:decision_ref|story|topic|file|function|method|class|struct|interface"
        r"|exception_type|table|route|import):[A-Za-z0-9_./:-]+"
    )
    return sorted(set(re.findall(pattern, text)))


def _classify_lane(request: ContextRequest) -> tuple[str, str]:
    """Classify retrieval lane from explicit intake signals before heuristics."""
    if request.lane is not None:
        if request.lane not in _LANES:
            raise ValueError(f"invalid context lane: {request.lane}")
        return request.lane, "explicit lane override"

    flags = request.risk_flags
    hard_gates = sorted(flags & _HARD_GATE_FLAGS)
    if hard_gates:
        return "high-risk", f"explicit hard-gate risk flags: {', '.join(hard_gates)}"
    if len(flags) >= 4:
        return "high-risk", f"explicit risk checklist: {len(flags)} flags"
    if len(flags) >= 2:
        return "normal", f"explicit risk checklist: {len(flags)} flags"

    text = request.task_description.casefold()
    tokens = {word.casefold() for word in _keywords(text)}
    matched_terms = sorted(tokens & _HIGH_RISK_TERMS)
    matched_phrases = sorted(phrase for phrase in _HIGH_RISK_PHRASES if phrase in text)
    if matched_terms or matched_phrases:
        signals = [*matched_terms, *matched_phrases]
        if request.scope == "cosmetic":
            if len(request.affected_files) <= 1 and len(tokens) <= 4:
                return "tiny", (
                    "human-narrowed cosmetic scope; heuristic terms: "
                    f"{', '.join(signals)}"
                )
            return "normal", (
                "human-narrowed cosmetic scope; heuristic terms: "
                f"{', '.join(signals)}"
            )
        return "high-risk", f"heuristic hard-gate terms: {', '.join(signals)}"
    if len(request.affected_files) <= 1 and len(tokens) <= 4:
        return "tiny", "bounded task with no risk signals"
    request_note = f" for {request.request_class}" if request.request_class else ""
    return "normal", f"default bounded task{request_note}"


def _estimate_tokens(text: str, tokenizer: Tokenizer | None = None) -> int:
    if tokenizer is not None:
        return max(1, len(tokenizer.encode(text)))
    return max(1, (len(text) + 3) // 4)

def _normalize_bm25(matches: list[dict[str, Any]]) -> dict[str, float]:
    """Normalize one FTS query's BM25 ranks to a stable 0.5--1.0 range."""
    ranked: list[tuple[str, float]] = []
    for node in matches:
        node_id = node.get("id")
        rank = node.get("_bm25_rank")
        if (
            isinstance(node_id, str)
            and isinstance(rank, (int, float))
            and not isinstance(rank, bool)
        ):
            ranked.append((node_id, float(rank)))
    if not ranked:
        return {}
    best = min(float(rank) for _, rank in ranked)
    worst = max(float(rank) for _, rank in ranked)
    if best == worst:
        return {node_id: 1.0 for node_id, _ in ranked}
    normalized = {
        node_id: 1.0 - 0.5 * ((float(rank) - best) / (worst - best))
        for node_id, rank in ranked
    }
    # Exact node-name matches remain the strongest lexical seeds even when
    # longer documents receive a better BM25 score from repeated terms.
    for node in matches:
        node_id = node.get("id")
        matched_term = node.get("_matched_term")
        if (
            isinstance(node_id, str)
            and isinstance(matched_term, str)
            and str(node.get("name", "")).casefold() == matched_term.casefold()
        ):
            normalized[node_id] = 1.0
    return normalized


def _fact_type(kind: str) -> str:
    if kind == "decision_ref":
        return "decision"
    if kind in {"story", "topic"}:
        return kind
    return "code"


def _snippet(node: dict[str, Any], limit: int = 500) -> str:
    for key in ("content", "decision", "context", "title", "signature", "name"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            clean = " ".join(value.split())
            return clean[:limit]
    return str(node.get("id", ""))[:limit]


def _chunk_snippet(text: str, term: str, limit: int = 500) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    position = clean.casefold().find(term.casefold()) if term else -1
    if position < 0:
        return clean[:limit]
    start = max(0, min(position - limit // 3, len(clean) - limit))
    end = min(len(clean), start + limit)
    return ("..." if start else "") + clean[start:end] + ("..." if end < len(clean) else "")


def _source_path(node: dict[str, Any]) -> str | None:
    for key in ("source_path", "filePath"):
        value = node.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _provenance_key(provenance: ContextProvenance) -> tuple[Any, ...]:
    """Return stable identity/order key for one provenance path."""
    return (
        provenance.seed_id,
        tuple(
            (step.kind, step.source_id, step.target_id, step.direction)
            for step in provenance.path
        ),
    )


def _provenance_nodes(provenance: ContextProvenance) -> set[str]:
    nodes = {provenance.seed_id}
    for step in provenance.path:
        nodes.update((step.source_id, step.target_id))
    return nodes


def _all_provenance(provenance: ContextProvenance | None, alternates: list[ContextProvenance] | None = None) -> list[ContextProvenance]:
    if provenance is None:
        return list(alternates or [])
    return [provenance, *(alternates or [])]


def _provenance_text(
    provenance: ContextProvenance | None,
    alternates: list[ContextProvenance] | None = None,
) -> str:
    return " ".join(
        _single_provenance_text(item)
        for item in _all_provenance(provenance, alternates)
    )


def _single_provenance_text(provenance: ContextProvenance) -> str:
    parts = [
        f"seed {provenance.seed_id} {provenance.seed_name} {provenance.seed_kind}",
        *provenance.matched_terms,
    ]
    for step in provenance.path:
        parts.extend((step.source_id, step.target_id, step.kind, step.direction))
        for evidence in step.evidence:
            lines = evidence.get("source_lines", {})
            parts.extend((
                str(evidence.get("reference", "")),
                str(evidence.get("source_path", "")),
                str(lines.get("start", "")),
                str(lines.get("end", "")),
                str(evidence.get("resolution", "")),
                str(evidence.get("confidence", "")),
            ))
    return " ".join(parts)


def _evidence_source_paths(
    provenance: ContextProvenance | None,
    alternates: list[ContextProvenance] | None = None,
) -> set[str]:
    return {
        source_path
        for item in _all_provenance(provenance, alternates)
        for step in item.path
        for evidence in step.evidence
        if isinstance((source_path := evidence.get("source_path")), str) and source_path
    }


def build_context(request: ContextRequest, db_path: Path) -> ContextResult:
    """Query the graph and assemble ranked evidence within a token budget."""
    if not request.task_description.strip():
        raise ValueError("task_description must not be empty")
    lane, classification_reason = _classify_lane(request)
    policy = _LANES[lane]
    budget = request.max_tokens if request.max_tokens is not None else policy["max_tokens"]
    if budget <= 0:
        raise ValueError("max_tokens must be positive")
    try:
        tokenizer = get_tokenizer(request.encoding)
    except (ImportError, ValueError) as exc:
        raise ContextRetrievalError(str(exc)) from exc
    encoding = request.encoding or "estimate"

    conn = store.open_readonly(db_path)
    try:
        seeds: dict[str, dict[str, Any]] = {}
        seed_terms: dict[str, set[str]] = {}
        seed_scores: dict[str, float] = {}
        score_components: dict[str, dict[str, float]] = {}
        retrieval_metadata: dict[str, dict[str, Any]] = {}
        search_limit = policy["search_limit"]
        searches = []
        for term in _keywords(request.task_description):
            original_matches = [
                {**node, "_matched_term": term}
                for node in _search_matches(conn, term, search_limit)
            ]
            if original_matches:
                searches.append((term, original_matches))
                continue
            if request.synonym_mode == "off":
                continue
            for expanded_term, is_synonym in _expand_semantic_terms(term):
                if not is_synonym:
                    continue
                matches = [
                    {
                        **node,
                        "_matched_term": expanded_term,
                        "_semantic_match_weight": _SYNONYM_SCORE_WEIGHT,
                    }
                    for node in _search_matches(conn, expanded_term, search_limit)
                ]
                searches.append((term, matches))

        if request.semantic_mode == "local":
            try:
                semantic_matches = semantic_seeds(
                    conn,
                    request.task_description,
                    request.embedding_model,
                    search_limit,
                )
            except EmbeddingUnavailableError as exc:
                raise ContextRetrievalError(str(exc)) from exc
            for semantic_match in semantic_matches:
                node_id = semantic_match["node_id"]
                if not isinstance(node_id, str):
                    continue
                similarity = semantic_match["cosine"]
                matches = store.query_by_ids(conn, [node_id])
                if matches:
                    searches.append((
                        "semantic task match",
                        [{
                            **matches[0],
                            "_semantic_score": similarity,
                            "_semantic_metadata": {
                                "mode": "local",
                                "model_id": semantic_match["model_id"],
                                "content_hash": semantic_match["content_hash"],
                            },
                        }],
                    ))

        for node_id in _exact_node_ids(request.task_description):
            matches = store.query_by_ids(conn, [node_id])
            if matches:
                searches.append((node_id, matches))


        searches.extend(
            _affected_file_matches(conn, path, search_limit)
            for path in request.affected_files
        )
        #Salar Change 
        # جمع‌آوری نتایج واژگانی و برداری به صورت جداگانه برای RRF
        bm25_results_list = []
        vector_results_list = []

        for matched_term, matches in searches:
            for node in matches:
                # ذخیره موقت کلمه کلیدی جستجو برای استفاده در provenance
                node_copy = {**node, "_matched_term": matched_term}
                if node_copy.get("_semantic_score") is not None:
                    if not any(n.get("id") == node_copy.get("id") for n in vector_results_list):
                        vector_results_list.append(node_copy)
                else:
                    if not any(n.get("id") == node_copy.get("id") for n in bm25_results_list):
                        bm25_results_list.append(node_copy)

        # اعمال RRF اگر هر دو لیست موجود باشند، در غیر این صورت استفاده از ترکیب موجود
        if bm25_results_list and vector_results_list:
            fused_matches = reciprocal_rank_fusion(bm25_results_list, vector_results_list, k=60)
        else:
            fused_matches = bm25_results_list + vector_results_list

        # محاسبه نرمال‌سازی BM25 روی لیست واژگانی
        normalized_scores = _normalize_bm25(bm25_results_list)

        # پردازش نهایی گره‌ها و امتیازها بر اساس خروجی RRF
        for node in fused_matches:
            node_id = node.get("id")
            matched_term = node.pop("_matched_term", request.task_description)
            if isinstance(node_id, str):
                match_weight = float(node.pop("_semantic_match_weight", 1.0))
                semantic_score = node.pop("_semantic_score", None)
                semantic_metadata = node.pop("_semantic_metadata", {})
                chunk_match = node.pop("_chunk_match", {})
                node.pop("_bm25_rank", None)
                seeds[node_id] = node
                seed_terms.setdefault(node_id, set()).add(matched_term)
                
                if chunk_match:
                    node["_match_location"] = {
                        "source_path": node.get("filePath"),
                        "start_line": chunk_match.get("start_line"),
                        "end_line": chunk_match.get("end_line"),
                        "chunk_ordinal": chunk_match.get("ordinal"),
                        "content_hash": chunk_match.get("content_hash"),
                    }
                    node["_chunk_snippet"] = chunk_match.get("text", "")

                # محاسبه امتیاز هیبریدی و اعمال وزن مترادف
                # محاسبه امتیاز هیبریدی و اعمال وزن مترادف (تغییر مقدار پیش‌فرض به 0.0)
                bm25_score = normalized_scores.get(node_id, 0.0) * match_weight
                if isinstance(semantic_score, (int, float)):
                    sem_weight = request.semantic_weight
                    lex_weight = 1.0 - sem_weight
                    hybrid_score = (lex_weight * bm25_score) + (sem_weight * float(semantic_score))
                    score = max(bm25_score, hybrid_score)
                else:
                    score = bm25_score

                seed_scores[node_id] = max(seed_scores.get(node_id, 0.0), score)
                components = score_components.setdefault(node_id, {})
                if normalized_scores.get(node_id, 0.0) > 0:
                    components["bm25"] = max(
                        components.get("bm25", 0.0), normalized_scores.get(node_id, 0.0)
                    )
                if match_weight < 1.0:
                    components["synonym_weight"] = match_weight
                if isinstance(semantic_score, (int, float)):
                    components["cosine"] = float(semantic_score)
                    components["semantic_weight"] = request.semantic_weight
                    components["hybrid_score"] = round(score, 3)
                    retrieval_metadata[node_id] = dict(semantic_metadata)
        for matched_term, matches in searches:
            normalized_scores = _normalize_bm25(matches)
            for node in matches:
                node_id = node.get("id")
                if isinstance(node_id, str):
                    match_weight = float(node.pop("_semantic_match_weight", 1.0))
                    semantic_score = node.pop("_semantic_score", None)
                    semantic_metadata = node.pop("_semantic_metadata", {})
                    chunk_match = node.pop("_chunk_match", {})
                    node.pop("_matched_term", None)
                    node.pop("_bm25_rank", None)
                    seeds[node_id] = node
                    seed_terms.setdefault(node_id, set()).add(matched_term)
                    if chunk_match:
                        node["_match_location"] = {
                            "source_path": node.get("filePath"),
                            "start_line": chunk_match.get("start_line"),
                            "end_line": chunk_match.get("end_line"),
                            "chunk_ordinal": chunk_match.get("ordinal"),
                            "content_hash": chunk_match.get("content_hash"),
                        }
                        node["_chunk_snippet"] = chunk_match.get("text", "")
                    
                    # Hybrid Search Scoring
                    bm25_score = normalized_scores.get(node_id, 0.0) * match_weight
                    if isinstance(semantic_score, (int, float)):
                        sem_weight = request.semantic_weight
                        lex_weight = 1.0 - sem_weight
                        hybrid_score = (lex_weight * bm25_score) + (sem_weight * float(semantic_score))
                        score = max(bm25_score, hybrid_score)
                    else:
                        score = bm25_score

                    seed_scores[node_id] = max(seed_scores.get(node_id, 0.0), score)
                    components = score_components.setdefault(node_id, {})
                    if normalized_scores.get(node_id, 0.0) > 0:
                        components["bm25"] = max(
                            components.get("bm25", 0.0), normalized_scores.get(node_id, 0.0)
                        )
                    if match_weight < 1.0:
                        components["synonym_weight"] = match_weight
                    if isinstance(semantic_score, (int, float)):
                        components["cosine"] = float(semantic_score)
                        components["semantic_weight"] = request.semantic_weight
                        components["hybrid_score"] = round(score, 3)
                        retrieval_metadata[node_id] = dict(semantic_metadata)

        scores: dict[str, float] = dict(seed_scores)
        nodes: dict[str, dict[str, Any]] = dict(seeds)
        provenance_paths: dict[str, list[tuple[float, ContextProvenance]]] = {
            node_id: [(
                scores[node_id],
                ContextProvenance(
                    seed_id=node_id,
                    seed_name=str(node.get("name") or node_id),
                    seed_kind=str(node.get("kind", "")),
                    matched_terms=sorted(seed_terms.get(node_id, set()), key=str.casefold),
                    path=[],
                ),
            )]
            for node_id, node in seeds.items()
        }
        provenance: dict[str, ContextProvenance] = {
            node_id: paths[0][1] for node_id, paths in provenance_paths.items()
        }
        evidence_cache: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        frontier = sorted(seeds)
        depth_scores = {1: 0.8, 2: 0.5, 3: 0.3}
        for depth in range(1, policy["depth"] + 1):
            next_frontier: list[str] = []
            for node_id in frontier:
                records = store.neighbors(conn, node_id, request.direction)
                records.sort(key=lambda item: (
                    item["edge"]["kind"],
                    item["edge"]["source"],
                    item["edge"]["target"],
                    item["direction"],
                ))
                for record in records:
                    edge = record["edge"]
                    neighbor_id = (
                        edge["target"] if record["direction"] == "out" else edge["source"]
                    )
                    hydrated = store.query_by_ids(conn, [neighbor_id])
                    if not hydrated:
                        continue
                    neighbor = hydrated[0]
                    edge_kind = edge["kind"]
                    source_kind = nodes.get(node_id, seeds.get(node_id, {})).get("kind")
                    score = depth_scores[depth]
                    components = score_components.setdefault(neighbor_id, {})
                    if edge_kind == "AFFECTS":
                        score = max(score, 0.9 if source_kind in {"decision_ref", "story"} else 0.7)
                        components["affects"] = max(components.get("affects", 0.0), score)
                    else:
                        components["graph_depth"] = max(
                            components.get("graph_depth", 0.0), score
                        )
                    evidence: list[dict[str, Any]] = []
                    if edge_kind == "AFFECTS":
                        edge_key = (edge["source"], edge["target"], edge_kind)
                        if edge_key not in evidence_cache:
                            evidence_cache[edge_key] = store.get_edge_evidence(
                                conn, edge["source"], edge["target"], edge_kind
                            )
                        evidence = evidence_cache[edge_key]
                    current_paths = provenance_paths[node_id]
                    for current_score, current_provenance in current_paths:
                        if neighbor_id in _provenance_nodes(current_provenance):
                            continue
                        candidate = ContextProvenance(
                            seed_id=current_provenance.seed_id,
                            seed_name=current_provenance.seed_name,
                            seed_kind=current_provenance.seed_kind,
                            matched_terms=current_provenance.matched_terms,
                            path=[
                                *current_provenance.path,
                                ContextPathStep(
                                    source_id=edge["source"],
                                    target_id=edge["target"],
                                    kind=edge_kind,
                                    direction=record["direction"],
                                    evidence=evidence,
                                ),
                            ],
                        )
                        candidate_score = min(current_score, score)
                        candidate_key = _provenance_key(candidate)
                        candidates = provenance_paths.setdefault(neighbor_id, [])
                        if any(_provenance_key(item) == candidate_key for _, item in candidates):
                            continue
                        candidates.append((candidate_score, candidate))
                        candidates.sort(key=lambda item: (-item[0], _provenance_key(item[1])))
                        del candidates[_MAX_PROVENANCE_PATHS:]
                        scores[neighbor_id] = max(item[0] for item in candidates)
                        provenance[neighbor_id] = candidates[0][1]
                    if neighbor_id not in nodes:
                        nodes[neighbor_id] = neighbor
                        if neighbor_id in provenance_paths:
                            provenance[neighbor_id] = provenance_paths[neighbor_id][0][1]
                        next_frontier.append(neighbor_id)
                    elif neighbor_id not in seeds and neighbor_id not in next_frontier:
                        next_frontier.append(neighbor_id)
            frontier = sorted(set(next_frontier))
            if not frontier:
                break

        #Salar Change
        facts = [
            ContextFact(
                fact_type=_fact_type(str(node.get("kind", ""))),
                node_id=node_id,
                name=str(node.get("name") or node_id),
                kind=str(node.get("kind", "")),
                snippet=(
                    _chunk_snippet(node["_chunk_snippet"], " ".join(seed_terms.get(node_id, set())))
                    if isinstance(node.get("_chunk_snippet"), str)
                    else _snippet(node)
                ),
                relevance_score=round(scores[node_id], 3),
                source_path=_source_path(node),
                provenance=provenance.get(node_id),
                alternate_paths=[
                    item
                    for _, item in provenance_paths.get(node_id, [])[1:]
                ],
                score_components={
                    **score_components.get(node_id, {}),
                    "final": round(scores[node_id], 3),
                },
                retrieval_metadata=retrieval_metadata.get(node_id, {}),
                match_location=node.get("_match_location", {}),
            )
            for node_id, node in nodes.items()
            if scores[node_id] >= 0.05  # Relevance filter: drop nodes with a score below the threshold
        ]
        facts.sort(key=lambda fact: (-fact.relevance_score, fact.kind, fact.name, fact.node_id))

        selected: list[ContextFact] = []
        used_tokens = 0
        for fact in facts:
            if len(selected) >= policy["max_facts"]:
                break
            cost = _estimate_tokens(
                f"{fact.name} {fact.kind} {fact.snippet} {fact.source_path or ''} "
                f"{_provenance_text(fact.provenance, fact.alternate_paths)} "
                f"{json.dumps(fact.score_components, sort_keys=True)} "
                f"{json.dumps(fact.retrieval_metadata, sort_keys=True)} "
                f"{json.dumps(fact.match_location, sort_keys=True)}",
                tokenizer,
            )
            if used_tokens + cost > budget:
                # Token-budget optimization: if the current fact doesn't fit, 
                # we can check if a smaller fact exists to optimally fill capacity
                continue
            selected.append(fact)
            used_tokens += cost

        sources = sorted({
            source
            for fact in selected
            for source in ({fact.source_path} | _evidence_source_paths(fact.provenance, fact.alternate_paths))
            if source
        })
        return ContextResult(
            task_description=request.task_description,
            lane=lane,
            classification_reason=classification_reason,
            affected_files=list(request.affected_files),
            budget_tokens=budget,
            graph_summary={
                "seed_nodes": len(seeds),
                "candidate_nodes": len(nodes),
                "selected_facts": len(selected),
            },
            facts=selected,
            sources=sources,
            total_tokens=used_tokens,
            encoding=encoding,
        )
    finally:
        conn.close()


def _markdown_cell(value: str) -> str:
    """Escape one value for a Markdown table cell."""
    return " ".join(value.split()).replace("|", "\\|")


def render_markdown(result: ContextResult) -> str:
    """Render a ContextResult as a structured prompt-ready Markdown packet."""
    sections: dict[str, tuple[str, list[ContextFact]]] = {
        "decision": ("Related Decisions", []),
        "story": ("Related Stories", []),
        "topic": ("Topics", []),
        "code": ("Code Symbols", []),
    }
    for fact in result.facts:
        key = fact.fact_type if fact.fact_type in sections else "code"
        sections[key][1].append(fact)

    lines = [
        f"# Context: {result.task_description}",
        "",
        f"- **Lane:** {result.lane}",
        f"- **Reason:** {result.classification_reason}",
        f"- **Budget:** {result.budget_tokens} tokens",
        f"- **Selected:** {len(result.facts)} facts, {result.total_tokens} estimated tokens",
        "",
        "## Affected Files",
        "",
    ]
    if result.affected_files:
        lines.extend(f"- `{_markdown_cell(path)}`" for path in result.affected_files)
    else:
        lines.append("- _None_")
    lines.append("")
    for key in ("decision", "story", "topic", "code"):
        title, facts = sections[key]
        lines.extend([f"## {title}", ""])
        if key == "code":
            lines.extend(["| Name | Kind | Score | Source |", "|---|---|---:|---|"])
            for fact in facts:
                lines.append(
                    f"| {_markdown_cell(fact.name)} | `{fact.kind}` | "
                    f"{fact.relevance_score:.2f} | {_markdown_cell(fact.source_path or '')} |"
                )
        else:
            lines.extend(["| Name | Score | Source |", "|---|---:|---|"])
            for fact in facts:
                lines.append(
                    f"| {_markdown_cell(fact.name)} | {fact.relevance_score:.2f} | "
                    f"{_markdown_cell(fact.source_path or '')} |"
                )
        if not facts:
            lines.append("| _None_ | | |" if key != "code" else "| _None_ | | | |")
        lines.append("")

    lines.extend(["## Provenance", ""])
    for fact in result.facts:
        provenance = fact.provenance
        if provenance is None:
            continue
        terms = ", ".join(f"`{_markdown_cell(term)}`" for term in provenance.matched_terms)
        lines.append(f"### {_markdown_cell(fact.name)}")
        lines.append("")
        lines.append(
            f"- Seed: `{_markdown_cell(provenance.seed_id)}` "
            f"(`{_markdown_cell(provenance.seed_kind)}`)"
        )
        lines.append(f"- Matched terms: {terms or '_None_'}")
        if fact.score_components:
            components = ", ".join(
                f"{name}={value:.3f}"
                for name, value in sorted(fact.score_components.items())
            )
            lines.append(f"- Score components: {components}")
        if fact.retrieval_metadata:
            metadata = ", ".join(
                f"{name}={value}"
                for name, value in sorted(fact.retrieval_metadata.items())
            )
            lines.append(f"- Retrieval metadata: {metadata}")
        if fact.match_location:
            loc = fact.match_location
            lines.append(f"- Match location: `{loc.get('source_path', '')}:{loc.get('start_line', '')}-{loc.get('end_line', '')}` (chunk {loc.get('chunk_ordinal', '')})")
        if not provenance.path:
            lines.append("- Path: direct search match")
        for index, step in enumerate(provenance.path, start=1):
            lines.append(
                f"- Step {index}: `{_markdown_cell(step.source_id)}` "
                f"-[`{_markdown_cell(step.kind)}`]-> "
                f"`{_markdown_cell(step.target_id)}` ({step.direction})"
            )
            for evidence in step.evidence:
                source_path = _markdown_cell(str(evidence.get("source_path", "")))
                source_lines = evidence.get("source_lines", {})
                reference = _markdown_cell(str(evidence.get("reference", "")))
                lines.append(
                    f"  - Evidence: `{source_path}:"
                    f"{source_lines.get('start', '')}-{source_lines.get('end', '')}`; "
                    f"reference `{reference}`; {evidence.get('resolution', '')}; "
                    f"confidence {evidence.get('confidence', '')}"
                )
        for alternate_index, alternate in enumerate(fact.alternate_paths, start=1):
            lines.append(f"- Alternative path {alternate_index}:")
            lines.append(
                f"  - Seed: `{_markdown_cell(alternate.seed_id)}` "
                f"(`{_markdown_cell(alternate.seed_kind)}`)"
            )
            if not alternate.path:
                lines.append("  - Path: direct search match")
            for index, step in enumerate(alternate.path, start=1):
                lines.append(
                    f"  - Step {index}: `{_markdown_cell(step.source_id)}` "
                    f"-[`{_markdown_cell(step.kind)}`]-> "
                    f"`{_markdown_cell(step.target_id)}` ({step.direction})"
                )
                for evidence in step.evidence:
                    source_path = _markdown_cell(str(evidence.get("source_path", "")))
                    source_lines = evidence.get("source_lines", {})
                    reference = _markdown_cell(str(evidence.get("reference", "")))
                    lines.append(
                        f"    - Evidence: `{source_path}:"
                        f"{source_lines.get('start', '')}-{source_lines.get('end', '')}`; "
                        f"reference `{reference}`; {evidence.get('resolution', '')}; "
                        f"confidence {evidence.get('confidence', '')}"
                    )
        lines.append("")
    if not any(fact.provenance for fact in result.facts):
        lines.extend(["- _None_", ""])

    lines.extend(["## Sources", ""])
    lines.extend(f"- `{source}`" for source in result.sources)
    if not result.sources:
        lines.append("- _None_")
    return "\n".join(lines) + "\n"
  

# Salar Change
'''  
def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    """
    Combines BM25 and vector search results using Reciprocal Rank Fusion (RRF).
    """
    rrf_scores = {}
    
    # Process BM25 rankings
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc['id']
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {'doc': doc, 'score': 0.0}
        rrf_scores[doc_id]['score'] += 1.0 / (k + rank)
        
    # Process Vector search rankings
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc['id']
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {'doc': doc, 'score': 0.0}
        rrf_scores[doc_id]['score'] += 1.0 / (k + rank)
        
    # Sort by combined RRF score descending
    sorted_results = sorted(
        rrf_scores.values(), 
        key=lambda x: x['score'], 
        reverse=True
    )
    
    return [item['doc'] for item in sorted_results]
    '''
def reciprocal_rank_fusion(bm25_results: list[dict[str, Any]], vector_results: list[dict[str, Any]], k: int = 60) -> list[dict[str, Any]]:
    """
    Combines BM25 and vector search results using Reciprocal Rank Fusion (RRF).
    """
    rrf_scores = {}
    
    # Process BM25 rankings
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc.get('id')
        if not doc_id:
            continue
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {'doc': doc, 'score': 0.0}
        rrf_scores[doc_id]['score'] += 1.0 / (k + rank)
        
    # Process Vector search rankings
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc.get('id')
        if not doc_id:
            continue
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {'doc': doc, 'score': 0.0}
        rrf_scores[doc_id]['score'] += 1.0 / (k + rank)
        
    # Sort by combined RRF score descending
    sorted_results = sorted(
        rrf_scores.values(), 
        key=lambda x: x['score'], 
        reverse=True
    )
    
    return [item['doc'] for item in sorted_results]