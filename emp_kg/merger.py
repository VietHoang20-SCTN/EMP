"""Semantic-to-code linking for emp-kg.

Resolves backtick-marked documentation references deterministically and emits
AFFECTS edges with evidence. A reference must match exactly at one resolution
stage; ambiguous or unknown references are reported, never guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import posixpath
import re
from typing import Any, Callable

from emp_kg import schema

_BACKTICK_RE = re.compile(r"`([^`]+)`")

_LANG_LABELS = {
    "bash", "powershell", "json", "text", "sql", "http", "python",
    "yaml", "toml", "markdown", "shell", "js", "ts",
}
_FILE_EXTENSION_JUNK = (
    ".md", ".json", ".sql", ".txt", ".yml", ".yaml", ".toml", ".ini",
    ".ps1", ".sh", ".csv", ".log", ".svg", ".png", ".jpg", ".db",
    ".lock", ".gitignore",
)
# Plain-English words that appear in prose backtick spans but can never be
# code symbols.  Exhaustive lists are brittle; these are the tokens observed
# in real docs (e.g. "`or`", "`and`", "`yes`", "`no`") plus common articles
# and short verbs.
_PROSE_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "if", "in",
    "is", "it", "no", "not", "of", "on", "or", "that", "the", "to", "with",
    "yes", "id", "result",
}
_SOURCE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".ets",
    ".go", ".rs", ".java", ".cs", ".php", ".rb", ".c", ".h",
    ".cpp", ".hpp", ".cc", ".m", ".mm", ".swift", ".kt", ".kts",
    ".scala", ".dart", ".svelte", ".vue", ".astro", ".lua", ".sol",
}
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")

# Plain-text candidates with a code-shaped name: snake_case, camelCase,
# dotted paths, or ALL_CAPS.  Lowercase single words ("service", "main") are
# intentionally excluded -- they are indistinguishable from prose.
_CODE_SHAPE_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*|"
    r"[A-Z][a-z0-9]+([A-Z][a-z0-9]+)+"
)
_MIXED_CASE_RE = re.compile(r"[a-z0-9][A-Z]")
_ALL_CAPS_RE = re.compile(r"^[A-Z0-9_]+$")


@dataclass
class LinkResult:
    """Resolved graph edges plus their evidence and unresolved references."""

    edges: list[dict[str, str]]
    evidence: list[dict[str, Any]]
    unresolved: list[dict[str, Any]]


def _topic_id(topic: dict[str, Any]) -> str:
    """Stable ID for a topic, deterministic across rebuilds."""
    key = f"{topic.get('source_path', '')}::{topic.get('title', '')}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"topic:{digest}"


def _line_range(value: Any) -> dict[str, int]:
    """Normalize parser-specific source range representations for evidence."""
    if isinstance(value, dict):
        start, end = value.get("start", 0), value.get("end", 0)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        start, end = value
    else:
        start = end = 0
    return {"start": int(start or 0), "end": int(end or 0)}


def _is_junk_reference(reference: str) -> bool:
    """True if a backtick span is not a plausible code symbol reference.

    Syntactic filter only: CLI flags, punctuation, prose fragments,
    language labels, non-code file extensions, and path-like strings can
    never resolve to a structural node, so they are dropped before
    resolution instead of flooding the unresolved report.
    """
    if not reference:
        return True
    if reference.startswith("-"):
        return True
    if _PUNCT_ONLY_RE.match(reference):
        return True
    # Inline fenced-code spans may include the language label and the full body.
    if reference.split(maxsplit=1)[0] in _LANG_LABELS:
        return True
    # Single-letter and bare-numeric spans ("a", "1", "42") are prose
    # fragments, never code symbols.
    if len(reference) == 1 or reference.isdigit():
        return True
    lowered = reference.casefold()
    if lowered in _PROSE_STOPWORDS:
        return True
    if lowered.endswith(_FILE_EXTENSION_JUNK):
        return True
    # A path to a real source file is a legitimate reference (resolves to a
    # file node); a path to docs/config/anything else is junk. Strip
    # file-local delimiters (`src/a.py::run`) before checking the extension.
    if "/" in reference or reference.endswith("/"):
        path_part = reference.split("::", 1)[0].split("#", 1)[0]
        suffix = Path(path_part).suffix
        return suffix not in _SOURCE_EXTENSIONS
    # Prose fragments: whitespace without a code shape (dot or underscore
    # chain) -- e.g. "query tools", "git status --short", ", and".  Multiline
    # spans are always junk: a real symbol reference never spans lines.
    if "\n" in reference or (
        " " in reference and not re.search(r"[._][A-Za-z0-9_]", reference)
    ):
        return True
    return False


def _references_from_text(
    text: str,
    source_path: str,
    source_lines: Any,
) -> list[dict[str, Any]]:
    line_range = _line_range(source_lines)
    return [
        {
            "reference": match.group(1).strip(),
            "source_path": source_path,
            "source_lines": line_range,
        }
        for match in _BACKTICK_RE.finditer(text)
        if not _is_junk_reference(match.group(1).strip())
    ]


def _code_shaped(token: str) -> bool:
    """True if token looks like a code identifier (code-shaped name).

    Code-shaped means: contains a dot, an interior capital (camelCase), or is
    ALL_CAPS.  Plain lowercase words (e.g. "service", "main") are prose and
    never treated as references.
    """
    if "." in token or "_" in token:
        return True
    if _ALL_CAPS_RE.match(token):
        return True
    return bool(_MIXED_CASE_RE.search(token))


def _bare_name_references(
    text: str,
    source_path: str,
    source_lines: Any,
    eligible_names: set[str],
) -> list[dict[str, Any]]:
    """Code-shaped identifiers in plain text that name an eligible node.

    Backtick-delimited spans are skipped -- they are already handled by
    `_references_from_text` and must not be matched twice.  Tokens that do not
    match any eligible node name are silently dropped (no unresolved entry),
    keeping plain prose out of the unresolved report.
    """
    line_range = _line_range(source_lines)
    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in _BACKTICK_RE.finditer(text):
        stripped = match.group(1).strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
    for match in _CODE_SHAPE_RE.finditer(text):
        token = match.group(0)
        if token in seen:
            continue
        if not _code_shaped(token):
            continue
        if token not in eligible_names:
            continue
        seen.add(token)
        references.append({
            "reference": token,
            "source_path": source_path,
            "source_lines": line_range,
        })
    return references


def _decision_references(
    decision: dict[str, Any],
    eligible_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Backtick references from a decision_ref's fields with provenance."""
    source_path = str(decision.get("source_path", ""))
    source_lines = decision.get("source_lines", {})
    fields: list[tuple[str, list[str]]] = [
        ("context", [decision.get("context", "")]),
        ("decision", [decision.get("decision", "")]),
        ("consequences", [
            *decision.get("consequences", {}).get("positive", []),
            *decision.get("consequences", {}).get("tradeoffs", []),
        ]),
        ("alternatives", decision.get("alternatives", [])),
        ("follow_up", decision.get("follow_up", [])),
    ]
    references: list[dict[str, Any]] = []
    for field, values in fields:
        for value in values:
            references.extend(
                _references_from_text(str(value), source_path, source_lines.get(field))
            )
            references.extend(
                _bare_name_references(
                    str(value),
                    source_path,
                    source_lines.get(field),
                    eligible_names or set(),
                )
            )
    return references


def _topic_references(
    topic: dict[str, Any],
    eligible_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Backtick and plain code-shaped references from a topic's content."""
    source_path = str(topic.get("source_path", ""))
    source_lines = topic.get("source_lines")
    content = str(topic.get("content", ""))
    return [
        *_references_from_text(content, source_path, source_lines),
        *_bare_name_references(content, source_path, source_lines, eligible_names or set()),
    ]


def _eligible_nodes(structural_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    affects_target_kinds = {
        kind.strip() for kind in schema.edge_types()["AFFECTS"]["target"].split("|")
    }
    return [node for node in structural_nodes if node.get("kind") in affects_target_kinds]


def _normalize_path(value: str) -> str:
    value = value.strip().replace("\\", "/")
    if value.startswith("file:"):
        value = value[5:]
    normalized = posixpath.normpath(value)
    return normalized.removeprefix("./").casefold()


def _normalize_qualified(value: str) -> str:
    return value.strip().replace("::", ".").casefold()


def _normalize_name(value: str) -> str:
    return value.strip().casefold()


def _node_qualified_names(node: dict[str, Any]) -> set[str]:
    simple_name = _normalize_name(str(node.get("name", "")))
    names = {
        normalized
        for field in ("qualifiedName", "fullyQualifiedName", "qualified_name")
        if node.get(field)
        for normalized in [_normalize_qualified(str(node[field]))]
        if normalized != simple_name
    }
    name = node.get("name")
    module = node.get("module") or node.get("moduleName") or node.get("namespace")
    if name and module:
        names.add(_normalize_qualified(f"{module}.{name}"))
    return names


def _file_local_parts(reference: str) -> tuple[str, str] | None:
    for separator in ("::", "#"):
        path, found, name = reference.rpartition(separator)
        if found and path.strip() and name.strip():
            return _normalize_path(path), _normalize_name(name)
    return None


def _unique_match(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    return candidates[0] if len(candidates) == 1 else None


def _resolve_reference(
    reference: str,
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    """Resolve one reference in fixed precedence order, never guessing."""
    exact_id = _unique_match([
        node for node in candidates if node.get("id") == reference
    ])
    if exact_id is not None:
        return exact_id, "exact_node_id"

    normalized_path = _normalize_path(reference)
    exact_file = _unique_match([
        node for node in candidates
        if node.get("kind") == "file"
        and node.get("filePath")
        and _normalize_path(str(node["filePath"])) == normalized_path
    ])
    if exact_file is not None:
        return exact_file, "file_path"

    normalized_qualified = _normalize_qualified(reference)
    qualified = _unique_match([
        node for node in candidates
        if node.get("kind") != "file"
        and normalized_qualified in _node_qualified_names(node)
    ])
    if qualified is not None:
        return qualified, "qualified_name"

    file_local = _file_local_parts(reference)
    if file_local is not None:
        file_path, local_name = file_local
        local_matches = [
            node for node in candidates
            if node.get("kind") != "file"
            and node.get("filePath")
            and _normalize_path(str(node["filePath"])) == file_path
            and _normalize_name(str(node.get("name", ""))) == local_name
        ]
        match = _unique_match(local_matches)
        if match is not None:
            return match, "file_local_name"
        return None, "ambiguous_file_local_name" if local_matches else "unresolved_reference"

    normalized_name = _normalize_name(reference)
    global_matches = [
        node for node in candidates
        if node.get("kind") != "file"
        and _normalize_name(str(node.get("name", ""))) == normalized_name
    ]
    match = _unique_match(global_matches)
    if match is not None:
        return match, "unique_global_name"
    return None, "ambiguous_global_name" if global_matches else "unresolved_reference"


def _link(
    semantic_nodes: list[dict[str, Any]],
    structural_nodes: list[dict[str, Any]],
    semantic_kind: str,
    node_id_fn: Callable[[dict[str, Any]], str],
    references_fn: Callable[[dict[str, Any], set[str] | None], list[dict[str, Any]]],
) -> LinkResult:
    """Resolve semantic references into AFFECTS edges, evidence, and misses."""
    if semantic_kind not in schema.supported_edge_sources("AFFECTS"):
        return LinkResult(edges=[], evidence=[], unresolved=[])

    edges: list[dict[str, str]] = []
    evidence: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    edge_keys: set[tuple[str, str, str]] = set()
    eligible = _eligible_nodes(structural_nodes)
    eligible_names: set[str] = set()
    for node in eligible:
        if node.get("name"):
            eligible_names.add(_normalize_qualified(str(node["name"])))
        eligible_names.update(_node_qualified_names(node))

    for semantic_node in semantic_nodes:
        source_id = node_id_fn(semantic_node)
        for reference_data in references_fn(semantic_node, eligible_names):
            reference = reference_data["reference"]
            target, resolution = _resolve_reference(reference, eligible)
            if target is None:
                unresolved.append({
                    "source": source_id,
                    "reference": reference,
                    "source_path": reference_data["source_path"],
                    "source_lines": reference_data["source_lines"],
                    "reason": resolution,
                })
                continue

            edge = {"source": source_id, "target": target["id"], "kind": "AFFECTS"}
            edge_key = (edge["source"], edge["target"], edge["kind"])
            if edge_key not in edge_keys:
                edge_keys.add(edge_key)
                edges.append(edge)

            edge_evidence = {
                **edge,
                "confidence": 1.0,
                "reference": reference,
                "source_path": reference_data["source_path"],
                "source_lines": reference_data["source_lines"],
                "resolution": resolution,
            }
            evidence.append(edge_evidence)

    return LinkResult(edges=edges, evidence=evidence, unresolved=unresolved)


def resolve_decisions_to_code(
    decisions: list[dict[str, Any]],
    structural_nodes: list[dict[str, Any]],
) -> LinkResult:
    """Resolve decision references into deterministic AFFECTS link results."""
    return _link(
        decisions,
        structural_nodes,
        semantic_kind="decision_ref",
        node_id_fn=lambda decision: f"decision_ref:{decision['id']}",
        references_fn=_decision_references,
    )


def link_decisions_to_code(
    decisions: list[dict[str, Any]],
    structural_nodes: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Return AFFECTS edges for callers that do not need link provenance."""
    return resolve_decisions_to_code(decisions, structural_nodes).edges


def resolve_topics_to_code(
    topics: list[dict[str, Any]],
    structural_nodes: list[dict[str, Any]],
) -> LinkResult:
    """Resolve documentation-topic references into deterministic link results."""
    return _link(
        topics,
        structural_nodes,
        semantic_kind="topic",
        node_id_fn=_topic_id,
        references_fn=_topic_references,
    )


def link_topics_to_code(
    topics: list[dict[str, Any]],
    structural_nodes: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Return AFFECTS edges for callers that do not need link provenance."""
    return resolve_topics_to_code(topics, structural_nodes).edges


def _story_id(story: dict[str, Any]) -> str:
    """Stable node ID for a story packet: story:<US-id>."""
    return f"story:{story['id']}"


def _story_references(
    story: dict[str, Any],
    eligible_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Backtick and plain code-shaped references from a story's content."""
    source_path = str(story.get("source_path", ""))
    source_lines = story.get("source_lines") or {}
    fields: list[tuple[str, list[str]]] = [
        ("acceptance_criteria", story.get("acceptance_criteria", [])),
        ("design_notes", story.get("design_notes", [])),
    ]
    references: list[dict[str, Any]] = []
    for field, values in fields:
        for value in values:
            references.extend(
                _references_from_text(str(value), source_path, source_lines.get(field))
            )
            references.extend(
                _bare_name_references(
                    str(value),
                    source_path,
                    source_lines.get(field),
                    eligible_names or set(),
                )
            )
    for doc in story.get("supporting_docs", []):
        content = str(doc.get("content", ""))
        doc_path = str(doc.get("source_path", ""))
        doc_lines = doc.get("source_lines")
        references.extend(_references_from_text(content, doc_path, doc_lines))
        references.extend(
            _bare_name_references(content, doc_path, doc_lines, eligible_names or set())
        )
    return references


def resolve_stories_to_code(
    stories: list[dict[str, Any]],
    structural_nodes: list[dict[str, Any]],
) -> LinkResult:
    """Resolve story-packet references into deterministic link results."""
    return _link(
        stories,
        structural_nodes,
        semantic_kind="story",
        node_id_fn=_story_id,
        references_fn=_story_references,
    )


def link_stories_to_code(
    stories: list[dict[str, Any]],
    structural_nodes: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Return AFFECTS edges for callers that do not need link provenance."""
    return resolve_stories_to_code(stories, structural_nodes).edges


_EXTERNAL_PATH_RE = re.compile(
    r"^(?:file:|https?://|(?:[A-Za-z]:)?[\\/]|(?:\.\.?[\\/])|[^\\/\s]+[\\/])"
)
_EXTERNAL_SYMBOL_RE = re.compile(r"^[a-z_][a-z0-9_]*\d+$|^[A-Z][A-Z0-9_]+$")
_SYSTEM_SIGNAL_RE = re.compile(r"^SIG[A-Z0-9]+$")
_NUMERIC_VALUE_RE = re.compile(r"^\d+$")
_DOTTED_DOMAIN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_WORKFLOW_REF_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*[-][A-Za-z0-9_.-]+$")
_INLINE_SQL_RE = re.compile(r"^datetime\(.*\)$|^NOT NULL|^INTEGER|^TEXT|^PRIMARY KEY")
_INLINE_JSON_LIKE_RE = re.compile(r"^\{.*\}$|^\[.*\]$|^\":\"")
_INLINE_SIGNATURE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z_][A-Za-z0-9_]*){2,}$")
_DOC_SCHEMA_FIELD_RE = re.compile(r"^(task_summary|files_changed|files_read|decisions_made|errors|notes|duration_seconds|token_estimate|actions_taken|actual_outcome|decision_recorded|outcome|created_at|checked_at|decision|story|intake|story_id|intake_id|backlog|agent|protocol_version|references|init|migrate|intervention|required|optional|audit|error|propose|resolves|trace|verify_command|harness_friction|status)$")
_INLINE_STATUS_RE = re.compile(r"^(completed|blocked|partial|failed|none|proposed|implemented)$")
_INLINE_TOOL_REF_RE = re.compile(r"^(harness|mcp|cli|binary|skill|tool|codex|rg)$")
_INLINE_JSON_ARRAY_RE = re.compile(r"^\[\"[^\"]+\"\]$")


def _classify_other_reference(reference: str) -> str:
    """Subclassify references that pass all primary filters but are not true code symbols.

    The goal is documentation/schema vocabulary or literal examples rather than missing
    indexed code.  Each predicate is conservative: when in doubt, leave the entry in
    ``other`` so missing symbols remain visible.
    """
    if reference == ".git":
        return "repo_path"
    if _INLINE_SQL_RE.match(reference):
        return "sql_literal"
    if _INLINE_JSON_LIKE_RE.match(reference) or _INLINE_SIGNATURE_RE.match(reference):
        return "json_literal"
    if _INLINE_JSON_ARRAY_RE.match(reference):
        return "json_literal"
    if _INLINE_STATUS_RE.fullmatch(reference):
        return "status_value"
    if _INLINE_TOOL_REF_RE.fullmatch(reference):
        return "tool_or_component"
    if _DOC_SCHEMA_FIELD_RE.fullmatch(reference):
        return "doc_schema_field"
    return "other"



def classify_unresolved(
    unresolved: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group unresolved references without dropping or mutating any entries.

    Categories describe likely remediation, not resolution certainty. Unknown
    lowercase symbols remain in ``other`` so missing indexed code is visible.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in unresolved:
        reference = str(item.get("reference", ""))
        if _EXTERNAL_PATH_RE.match(reference):
            category = "external_path"
        elif _SYSTEM_SIGNAL_RE.fullmatch(reference):
            category = "system_signal"
        elif _EXTERNAL_SYMBOL_RE.fullmatch(reference):
            category = "external_ref"
        elif _NUMERIC_VALUE_RE.fullmatch(reference):
            category = "numeric_value"
        elif _DOTTED_DOMAIN_RE.fullmatch(reference):
            category = "dotted_domain"
        elif _WORKFLOW_REF_RE.fullmatch(reference):
            category = "workflow_ref"
        else:
            category = _classify_other_reference(reference)
        groups.setdefault(category, []).append(item)
    return groups
