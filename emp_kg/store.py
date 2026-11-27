"""Persistence layer for emp-kg.

Stores the unified knowledge graph -- structural nodes from CodeGraph,
semantic decision_ref nodes, and the AFFECTS edges linking them -- in a
single SQLite database, matching the schema's own storage_decision
(SQLite + FTS5, the same stack CodeGraph and Engram both already use).
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from emp_kg import schema


_VIS_JS_PATH = Path(__file__).resolve().parent / "vis-network.min.js"
_RESOLUTION_KINDS = {
    "exact_node_id",
    "file_path",
    "qualified_name",
    "file_local_name",
    "unique_global_name",
}


class GraphValidationError(ValueError):
    """A record or persisted graph violates the EMP graph contract."""


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GraphValidationError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GraphValidationError(f"{label} must be a nonempty string")
    return value


def _json_data(value: dict[str, Any], label: str) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise GraphValidationError(f"{label} must be JSON-serializable") from exc


def _flatten_fts_value(value: Any) -> list[str]:
    """Return deterministic plain-text fragments from a JSON-compatible value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [str(value)]
    if isinstance(value, list):
        return [fragment for item in value for fragment in _flatten_fts_value(item)]
    if isinstance(value, dict):
        return [
            fragment
            for key in sorted(value)
            for fragment in _flatten_fts_value(value[key])
        ]
    return []


def _fts_content(kind: str, data: dict[str, Any]) -> str:
    """Build the searchable document persisted beside each graph node."""
    fields = (
        "title", "decision", "context", "consequences", "alternatives",
        "follow_up", "content", "description", "summary", "signature",
        "qualifiedName", "filePath", "acceptance_criteria", "design_notes",
        "lane", "status", "docstring", "source_body", "source_text",
    )
    fragments = [kind]
    for field in fields:
        if field in data:
            fragments.extend(_flatten_fts_value(data[field]))
    return " ".join(fragment for fragment in fragments if fragment.strip())


def _validate_node_record(
    node_id: Any,
    kind: Any,
    name: Any,
    data: Any,
    *,
    allowed_kinds: set[str],
    semantic_id: str | None = None,
    verify_data_identity: bool = True,
) -> tuple[str, str, str, str, str]:
    node_id = _require_string(node_id, "node id")
    kind = _require_string(kind, "node kind")
    if kind not in allowed_kinds:
        status = "unknown" if not schema.is_valid_node_kind(kind) else "unsupported"
        raise GraphValidationError(f"{status} node kind {kind!r}")
    if not isinstance(name, str):
        raise GraphValidationError("node name must be a string")
    data = _require_mapping(data, "node data")
    if semantic_id is None and verify_data_identity:
        if data.get("id") != node_id or data.get("kind") != kind:
            raise GraphValidationError(
                f"node data identity disagrees with {kind}:{node_id}"
            )
    elif semantic_id is not None and data.get("id") != semantic_id:
        raise GraphValidationError(
            f"semantic node data id disagrees with {kind}:{node_id}"
        )
    serialized = _json_data(data, "node data")
    return node_id, kind, name, serialized, _fts_content(kind, data)


def _validate_node_batch(records: list[tuple[str, str, str, str, str]]) -> None:
    seen: dict[str, tuple[str, str, str, str]] = {}
    for node_id, kind, name, data, fts_content in records:
        current = (kind, name, data, fts_content)
        previous = seen.get(node_id)
        if previous is not None and previous != current:
            raise GraphValidationError(f"conflicting duplicate node id {node_id!r}")
        seen[node_id] = current


def _validate_edge_record(conn: sqlite3.Connection, edge: Any) -> tuple[str, str, str]:
    edge = _require_mapping(edge, "edge")
    source = _require_string(edge.get("source"), "edge source")
    target = _require_string(edge.get("target"), "edge target")
    kind = _require_string(edge.get("kind"), "edge kind")
    if kind not in schema.edge_types():
        raise GraphValidationError(f"unknown edge kind {kind!r}")
    if kind not in schema.supported_edge_types():
        raise GraphValidationError(f"unsupported edge kind {kind!r}")
    endpoints = dict(conn.execute(
        "SELECT id, kind FROM nodes WHERE id IN (?, ?)", (source, target)
    ))
    if source not in endpoints or target not in endpoints:
        missing = source if source not in endpoints else target
        raise GraphValidationError(f"edge endpoint does not exist: {missing!r}")
    if not schema.is_valid_edge(kind, endpoints[source], endpoints[target]):
        raise GraphValidationError(
            f"invalid {kind} endpoints: {endpoints[source]} -> {endpoints[target]}"
        )
    supported_sources = schema.supported_edge_sources(kind)
    if supported_sources and endpoints[source] not in supported_sources:
        raise GraphValidationError(
            f"unsupported {kind} source kind: {endpoints[source]}"
        )
    return source, target, kind


def _validate_evidence_record(
    conn: sqlite3.Connection,
    item: Any,
) -> tuple[str, str, str, str, str, int, int, str, float]:
    item = _require_mapping(item, "edge evidence")
    source = _require_string(item.get("source"), "evidence source")
    target = _require_string(item.get("target"), "evidence target")
    kind = _require_string(item.get("kind"), "evidence kind")
    if kind != "AFFECTS":
        raise GraphValidationError("evidence kind must be 'AFFECTS'")
    if conn.execute(
        "SELECT 1 FROM edges WHERE source = ? AND target = ? AND kind = ?",
        (source, target, kind),
    ).fetchone() is None:
        raise GraphValidationError("evidence edge does not exist")
    reference = _require_string(item.get("reference"), "evidence reference")
    source_path = _require_string(item.get("source_path"), "evidence source_path")
    lines = _require_mapping(item.get("source_lines"), "evidence source_lines")
    start, end = lines.get("start"), lines.get("end")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise GraphValidationError("evidence source_lines.start must be a nonnegative integer")
    if isinstance(end, bool) or not isinstance(end, int) or end < start:
        raise GraphValidationError("evidence source_lines.end must be an integer >= start")
    resolution = _require_string(item.get("resolution"), "evidence resolution")
    if resolution not in _RESOLUTION_KINDS:
        raise GraphValidationError(f"unknown evidence resolution {resolution!r}")
    confidence = item.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise GraphValidationError("evidence confidence must be numeric")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise GraphValidationError("evidence confidence must be between 0.0 and 1.0")
    return source, target, kind, reference, source_path, start, end, resolution, confidence


def _create_edges_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE edges (
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            kind TEXT NOT NULL,
            PRIMARY KEY (source, target, kind),
            FOREIGN KEY (source) REFERENCES nodes(id) ON DELETE CASCADE,
            FOREIGN KEY (target) REFERENCES nodes(id) ON DELETE CASCADE
        )
    """)


def _create_edge_evidence_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edge_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            kind TEXT NOT NULL,
            reference TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_line_start INTEGER NOT NULL,
            source_line_end INTEGER NOT NULL,
            resolution TEXT NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY (source, target, kind)
                REFERENCES edges(source, target, kind) ON DELETE CASCADE
        )
    """)


def _ensure_fts_schema(conn: sqlite3.Connection) -> None:
    """Migrate the external-content FTS table to include searchable content."""
    node_columns = {row[1] for row in conn.execute("PRAGMA table_info(nodes)")}
    needs_backfill = "fts_content" not in node_columns
    if needs_backfill:
        conn.execute("ALTER TABLE nodes ADD COLUMN fts_content TEXT NOT NULL DEFAULT ''")
        for rowid, kind, data in conn.execute("SELECT rowid, kind, data FROM nodes"):
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                payload = {}
            conn.execute(
                "UPDATE nodes SET fts_content = ? WHERE rowid = ?",
                (_fts_content(kind, payload if isinstance(payload, dict) else {}), rowid),
            )

    fts_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'nodes_fts'"
    ).fetchone() is not None
    fts_columns = [row[1] for row in conn.execute("PRAGMA table_info(nodes_fts)")]
    needs_rebuild = needs_backfill or not fts_exists
    if fts_exists and fts_columns != ["id", "name", "fts_content"]:
        conn.execute("DROP TABLE nodes_fts")
        needs_rebuild = True
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
            id UNINDEXED, name, fts_content,
            content='nodes', content_rowid='rowid'
        )
    """)
    if needs_rebuild:
        conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES ('rebuild')")


def _ensure_edges_schema(conn: sqlite3.Connection) -> None:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'edges'"
    ).fetchone()
    if not exists:
        _create_edges_table(conn)
        return

    if conn.execute("PRAGMA foreign_key_list(edges)").fetchall():
        return

    conn.execute("ALTER TABLE edges RENAME TO edges_legacy")
    _create_edges_table(conn)
    conn.execute("""
        INSERT INTO edges (source, target, kind)
        SELECT legacy.source, legacy.target, legacy.kind
        FROM edges_legacy AS legacy
        JOIN nodes AS source_node ON source_node.id = legacy.source
        JOIN nodes AS target_node ON target_node.id = legacy.target
    """)
    conn.execute("DROP TABLE edges_legacy")


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create (or open) the graph database with its schema."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            name TEXT,
            data TEXT NOT NULL,
            fts_content TEXT NOT NULL DEFAULT ''
        )
    """)
    _ensure_edges_schema(conn)
    _create_edge_evidence_table(conn)
    _ensure_fts_schema(conn)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_metadata (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            schema_version TEXT NOT NULL,
            built_at TEXT NOT NULL,
            project_path TEXT NOT NULL,
            git_sha TEXT NOT NULL,
            codegraph_provenance TEXT NOT NULL DEFAULT '{}'
        )
    """)
    metadata_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(graph_metadata)")
    }
    if "codegraph_provenance" not in metadata_columns:
        conn.execute(
            "ALTER TABLE graph_metadata ADD COLUMN "
            "codegraph_provenance TEXT NOT NULL DEFAULT '{}'"
        )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_evidence_source ON edge_evidence(source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_evidence_target ON edge_evidence(target)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS node_embeddings (
            node_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            vector_json TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            PRIMARY KEY (node_id, model_id),
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_model ON node_embeddings(model_id)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS source_chunks (
            file_node_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            text TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            PRIMARY KEY (file_node_id, ordinal),
            FOREIGN KEY (file_node_id) REFERENCES nodes(id) ON DELETE CASCADE
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS source_chunks_fts USING fts5(
            file_node_id UNINDEXED, ordinal UNINDEXED, text
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embedding_metadata (
            model_id TEXT PRIMARY KEY,
            dimension INTEGER NOT NULL,
            corpus_digest TEXT NOT NULL,
            node_count INTEGER NOT NULL,
            indexed_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _save_node_records(
    conn: sqlite3.Connection,
    records: list[tuple[str, str, str, str, str]],
) -> int:
    _validate_node_batch(records)
    for record in records:
        conn.execute(
            """
            INSERT INTO nodes (id, kind, name, data, fts_content)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                kind = excluded.kind,
                name = excluded.name,
                data = excluded.data,
                fts_content = excluded.fts_content
            """,
            record,
        )
    return len(records)


def save_structural_nodes(conn: sqlite3.Connection, nodes: list[dict[str, Any]]) -> int:
    """Validate and upsert structural CodeGraph nodes."""
    records = []
    for raw in nodes:
        node = _require_mapping(raw, "structural node")
        name = node.get("name") or node.get("filePath", "")
        records.append(_validate_node_record(
            node.get("id"),
            node.get("kind"),
            name,
            node,
            allowed_kinds=schema.supported_structural_node_kinds(),
        ))
    return _save_node_records(conn, records)


def save_decision_nodes(conn: sqlite3.Connection, decisions: list[dict[str, Any]]) -> int:
    """Validate and upsert semantic decision_ref nodes."""
    records = []
    for raw in decisions:
        decision = _require_mapping(raw, "decision_ref")
        decision_id = _require_string(decision.get("id"), "decision id")
        title = _require_string(decision.get("title"), "decision title")
        records.append(_validate_node_record(
            f"decision_ref:{decision_id}",
            "decision_ref",
            title,
            decision,
            allowed_kinds=schema.supported_semantic_node_kinds(),
            semantic_id=decision_id,
        ))
    return _save_node_records(conn, records)


def save_topic_nodes(conn: sqlite3.Connection, topics: list[dict[str, Any]]) -> int:
    """Validate and upsert generic documentation topic nodes."""
    from emp_kg.merger import _topic_id

    records = []
    for raw in topics:
        topic = _require_mapping(raw, "topic")
        title = _require_string(topic.get("title"), "topic title")
        _require_string(topic.get("content"), "topic content")
        _require_string(topic.get("source_path"), "topic source_path")
        records.append(_validate_node_record(
            _topic_id(topic),
            "topic",
            title,
            topic,
            allowed_kinds=schema.supported_semantic_node_kinds(),
            verify_data_identity=False,
        ))
    return _save_node_records(conn, records)


def save_story_nodes(conn: sqlite3.Connection, stories: list[dict[str, Any]]) -> int:
    """Validate and upsert story packet nodes."""
    records = []
    for raw in stories:
        story = _require_mapping(raw, "story")
        story_id = _require_string(story.get("id"), "story id")
        title = _require_string(story.get("title"), "story title")
        _require_string(story.get("status"), "story status")
        _require_string(story.get("source_path"), "story source_path")
        records.append(_validate_node_record(
            f"story:{story_id}",
            "story",
            title,
            story,
            allowed_kinds=schema.supported_semantic_node_kinds(),
            semantic_id=story_id,
        ))
    return _save_node_records(conn, records)


def replace_node_embeddings(
    conn: sqlite3.Connection,
    model_id: str,
    embeddings: list[dict[str, Any]],
    *,
    corpus_digest: str,
) -> int:
    """Atomically replace one model's vector index after full validation."""
    model_id = _require_string(model_id, "embedding model_id")
    corpus_digest = _require_string(corpus_digest, "embedding corpus_digest")
    records = []
    dimensions: set[int] = set()
    seen_ids: set[str] = set()
    for item in embeddings:
        node_id = _require_string(item.get("node_id"), "embedding node_id")
        if node_id in seen_ids:
            raise GraphValidationError(f"duplicate embedding node_id {node_id!r}")
        seen_ids.add(node_id)
        if item.get("model_id") != model_id:
            raise GraphValidationError("embedding model_id disagrees with replacement model")
        vector = item.get("vector")
        if not isinstance(vector, list) or not vector:
            raise GraphValidationError("embedding vector must be a nonempty list")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in vector
        ):
            raise GraphValidationError("embedding vector values must be finite numeric values")
        if conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone() is None:
            raise GraphValidationError(f"embedding node does not exist: {node_id!r}")
        content_hash = _require_string(item.get("content_hash"), "embedding content_hash")
        dimensions.add(len(vector))
        records.append((node_id, model_id, len(vector), json.dumps(vector), content_hash))
    if len(dimensions) > 1:
        raise GraphValidationError("embedding vectors must share one dimension per model")
    dimension = next(iter(dimensions), 0)
    conn.execute("SAVEPOINT replace_node_embeddings")
    try:
        conn.execute("DELETE FROM node_embeddings WHERE model_id = ?", (model_id,))
        if records:
            conn.executemany(
                """
                INSERT INTO node_embeddings (node_id, model_id, dimension, vector_json, content_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                records,
            )
        conn.execute(
            """
            INSERT INTO embedding_metadata (model_id, dimension, corpus_digest, node_count, indexed_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                dimension = excluded.dimension,
                corpus_digest = excluded.corpus_digest,
                node_count = excluded.node_count,
                indexed_at = excluded.indexed_at
            """,
            (model_id, dimension, corpus_digest, len(records), datetime.now(timezone.utc).isoformat()),
        )
    except sqlite3.DatabaseError:
        conn.execute("ROLLBACK TO replace_node_embeddings")
        raise
    finally:
        conn.execute("RELEASE replace_node_embeddings")
    return len(records)


def replace_source_chunks(conn: sqlite3.Connection, chunks: list[dict[str, Any]]) -> int:
    """Atomically replace all lexical source chunks after validation."""
    records = []
    seen: set[tuple[str, int]] = set()
    for chunk in chunks:
        file_node_id = _require_string(chunk.get("file_node_id"), "chunk file_node_id")
        ordinal = chunk.get("ordinal")
        start_line = chunk.get("start_line")
        end_line = chunk.get("end_line")
        text = _require_string(chunk.get("text"), "chunk text")
        content_hash = _require_string(chunk.get("content_hash"), "chunk content_hash")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise GraphValidationError("chunk ordinal must be nonnegative")
        if isinstance(start_line, bool) or not isinstance(start_line, int) or start_line <= 0:
            raise GraphValidationError("chunk start_line must be positive")
        if isinstance(end_line, bool) or not isinstance(end_line, int) or end_line < start_line:
            raise GraphValidationError("chunk end_line must be >= start_line")
        key = (file_node_id, ordinal)
        if key in seen:
            raise GraphValidationError(f"duplicate source chunk {key!r}")
        seen.add(key)
        kind = conn.execute("SELECT kind FROM nodes WHERE id = ?", (file_node_id,)).fetchone()
        if kind is None or kind[0] != "file":
            raise GraphValidationError(f"chunk file node does not exist: {file_node_id!r}")
        records.append((file_node_id, ordinal, start_line, end_line, text, content_hash))
    records.sort(key=lambda item: (item[0], item[1]))
    conn.execute("SAVEPOINT replace_source_chunks")
    try:
        conn.execute("DELETE FROM source_chunks")
        conn.execute("DELETE FROM source_chunks_fts")
        if records:
            conn.executemany(
                """
                INSERT INTO source_chunks (
                    file_node_id, ordinal, start_line, end_line, text, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            conn.executemany(
                "INSERT INTO source_chunks_fts (file_node_id, ordinal, text) VALUES (?, ?, ?)",
                [(file_node_id, ordinal, text) for file_node_id, ordinal, _, _, text, _ in records],
            )
    except sqlite3.DatabaseError:
        conn.execute("ROLLBACK TO replace_source_chunks")
        raise
    finally:
        conn.execute("RELEASE replace_source_chunks")
    return len(records)


def query_source_chunks_ranked(
    conn: sqlite3.Connection,
    search: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic BM25-ranked file chunks for an FTS query."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    sql = """
        SELECT chunk.file_node_id, chunk.ordinal, chunk.start_line, chunk.end_line,
               chunk.text, chunk.content_hash, bm25(source_chunks_fts) AS bm25_rank
        FROM source_chunks_fts
        JOIN source_chunks AS chunk
          ON chunk.file_node_id = source_chunks_fts.file_node_id
         AND chunk.ordinal = source_chunks_fts.ordinal
        WHERE source_chunks_fts MATCH ?
        ORDER BY bm25_rank, chunk.file_node_id, chunk.ordinal
    """
    params: list[Any] = [search]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return [
        {
            "file_node_id": file_node_id,
            "ordinal": ordinal,
            "start_line": start_line,
            "end_line": end_line,
            "text": text,
            "content_hash": content_hash,
            "bm25_rank": float(bm25_rank),
        }
        for file_node_id, ordinal, start_line, end_line, text, content_hash, bm25_rank
        in conn.execute(sql, params)
    ]


def save_node_embeddings(
    conn: sqlite3.Connection,
    embeddings: list[dict[str, Any]],
) -> int:
    """Upsert optional local embeddings for existing nodes."""
    records = []
    for item in embeddings:
        node_id = _require_string(item.get("node_id"), "embedding node_id")
        model_id = _require_string(item.get("model_id"), "embedding model_id")
        vector = item.get("vector")
        if not isinstance(vector, list) or not vector:
            raise GraphValidationError("embedding vector must be a nonempty list")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in vector
        ):
            raise GraphValidationError("embedding vector values must be finite numeric values")
        content_hash = _require_string(item.get("content_hash"), "embedding content_hash")
        if conn.execute("SELECT 1 FROM nodes WHERE id = ?", (node_id,)).fetchone() is None:
            raise GraphValidationError(f"embedding node does not exist: {node_id!r}")
        records.append((node_id, model_id, len(vector), json.dumps(vector), content_hash))
    conn.executemany(
        """
        INSERT INTO node_embeddings (node_id, model_id, dimension, vector_json, content_hash)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(node_id, model_id) DO UPDATE SET
            dimension = excluded.dimension,
            vector_json = excluded.vector_json,
            content_hash = excluded.content_hash
        """,
        records,
    )
    return len(records)


def load_node_embeddings(
    conn: sqlite3.Connection,
    model_id: str,
) -> list[dict[str, Any]]:
    """Return deterministic embeddings for one model."""
    rows = conn.execute(
        """
        SELECT node_id, dimension, vector_json, content_hash
        FROM node_embeddings WHERE model_id = ? ORDER BY node_id
        """,
        (model_id,),
    )
    return [
        {
            "node_id": node_id,
            "model_id": model_id,
            "dimension": dimension,
            "vector": json.loads(vector_json),
            "content_hash": content_hash,
        }
        for node_id, dimension, vector_json, content_hash in rows
    ]


def embedding_documents(conn: sqlite3.Connection) -> list[dict[str, str]]:
    """Return nonempty FTS documents in deterministic node-ID order."""
    return [
        {
            "node_id": node_id,
            "content": content,
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
        for node_id, content in conn.execute(
            "SELECT id, fts_content FROM nodes WHERE fts_content != '' ORDER BY id"
        )
    ]


def get_embedding_metadata(conn: sqlite3.Connection, model_id: str) -> dict[str, Any]:
    """Return one model's embedding index metadata, or an empty mapping."""
    row = conn.execute(
        """
        SELECT dimension, corpus_digest, node_count, indexed_at
        FROM embedding_metadata WHERE model_id = ?
        """,
        (model_id,),
    ).fetchone()
    if row is None:
        return {}
    dimension, corpus_digest, node_count, indexed_at = row
    return {
        "model_id": model_id,
        "dimension": dimension,
        "corpus_digest": corpus_digest,
        "node_count": node_count,
        "indexed_at": indexed_at,
    }


def load_fresh_node_embeddings(
    conn: sqlite3.Connection,
    model_id: str,
) -> list[dict[str, Any]]:
    """Return complete current-model vectors only when corpus is fresh."""
    documents = embedding_documents(conn)
    digest = hashlib.sha256(
        "\n".join(
            f"{item['node_id']}:{item['content_hash']}" for item in documents
        ).encode("utf-8")
    ).hexdigest()
    metadata = get_embedding_metadata(conn, model_id)
    if not metadata or metadata["corpus_digest"] != digest:
        return []
    embeddings = load_node_embeddings(conn, model_id)
    if len(embeddings) != len(documents) or any(
        item["dimension"] != metadata["dimension"] for item in embeddings
    ):
        return []
    return embeddings


def delete_stale_embeddings(
    conn: sqlite3.Connection,
    model_id: str,
    current_hashes: dict[str, str],
) -> int:
    """Remove embeddings whose source content hash changed or disappeared."""
    rows = conn.execute(
        "SELECT node_id, content_hash FROM node_embeddings WHERE model_id = ?",
        (model_id,),
    ).fetchall()
    stale = [
        (node_id,)
        for node_id, content_hash in rows
        if current_hashes.get(node_id) != content_hash
    ]
    conn.executemany(
        "DELETE FROM node_embeddings WHERE node_id = ? AND model_id = ?",
        [(node_id, model_id) for node_id, in stale],
    )
    return len(stale)


def save_edges(conn: sqlite3.Connection, edges: list[dict[str, Any]]) -> int:
    """Validate and upsert graph edges."""
    records = [_validate_edge_record(conn, edge) for edge in edges]
    for record in records:
        conn.execute(
            "INSERT OR REPLACE INTO edges (source, target, kind) VALUES (?, ?, ?)",
            record,
        )
    return len(records)


def save_affects_evidence(conn: sqlite3.Connection, evidence: list[dict[str, Any]]) -> int:
    """Validate and replace current-build AFFECTS evidence."""
    records = [_validate_evidence_record(conn, item) for item in evidence]
    conn.execute("DELETE FROM edge_evidence WHERE kind = 'AFFECTS'")
    for record in records:
        conn.execute(
            """
            INSERT INTO edge_evidence (
                source, target, kind, reference, source_path,
                source_line_start, source_line_end, resolution, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            record,
        )
    return len(records)


def validate_graph(conn: sqlite3.Connection) -> dict[str, int]:
    """Assert persisted graph invariants. Returns audited record counts."""
    errors: list[str] = []
    nodes = conn.execute("SELECT id, kind, name, data FROM nodes ORDER BY id").fetchall()
    node_kinds: dict[str, str] = {}
    for node_id, kind, name, data in nodes:
        if kind not in schema.node_kinds():
            errors.append(f"unknown persisted node kind {kind!r} ({node_id})")
            continue
        supported_node_kinds = (
            schema.supported_structural_node_kinds()
            | schema.supported_semantic_node_kinds()
        )
        if kind not in supported_node_kinds:
            errors.append(f"unsupported persisted node kind {kind!r} ({node_id})")
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            errors.append(f"node data is not JSON ({node_id})")
            continue
        if not isinstance(payload, dict):
            errors.append(f"node data is not an object ({node_id})")
        elif kind not in {"decision_ref", "topic"} and (
            payload.get("id") != node_id or payload.get("kind") != kind
        ):
            errors.append(f"node data identity disagrees ({node_id})")
        if not isinstance(name, (str, type(None))):
            errors.append(f"node name is malformed ({node_id})")
        node_kinds[node_id] = kind

    edge_rows = conn.execute(
        "SELECT source, target, kind FROM edges ORDER BY source, target, kind"
    ).fetchall()
    for source, target, kind in edge_rows:
        source_kind = node_kinds.get(source)
        target_kind = node_kinds.get(target)
        if kind not in schema.edge_types():
            errors.append(f"unknown persisted edge kind {kind!r} ({source}->{target})")
        elif source_kind is None or target_kind is None:
            errors.append(f"edge endpoint missing ({source}->{target})")
        elif not schema.is_valid_edge(kind, source_kind, target_kind):
            errors.append(
                f"invalid persisted {kind} endpoints: {source_kind} -> {target_kind}"
            )
        elif kind not in schema.supported_edge_types():
            errors.append(f"unsupported persisted edge kind {kind!r}")
        elif (
            schema.supported_edge_sources(kind)
            and source_kind not in schema.supported_edge_sources(kind)
        ):
            errors.append(f"unsupported persisted {kind} source kind {source_kind}")

    evidence_rows = conn.execute(
        """
        SELECT source, target, kind, reference, source_path, source_line_start,
               source_line_end, resolution, confidence
        FROM edge_evidence ORDER BY id
        """
    ).fetchall()
    for row in evidence_rows:
        source, target, kind, reference, source_path, start, end, resolution, confidence = row
        try:
            _validate_evidence_record(conn, {
                "source": source,
                "target": target,
                "kind": kind,
                "reference": reference,
                "source_path": source_path,
                "source_lines": {"start": start, "end": end},
                "resolution": resolution,
                "confidence": confidence,
            })
        except GraphValidationError as exc:
            errors.append(f"invalid persisted evidence: {exc}")

    node_count = len(nodes)
    fts_count = conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0]
    if fts_count != node_count:
        errors.append(f"FTS row count {fts_count} differs from node count {node_count}")
    try:
        conn.execute(
            "INSERT INTO nodes_fts(nodes_fts, rank) VALUES ('integrity-check', 1)"
        )
    except sqlite3.DatabaseError as exc:
        errors.append(f"FTS index differs from node content: {exc}")

    if errors:
        raise GraphValidationError("; ".join(errors))
    return {
        "nodes": node_count,
        "edges": len(edge_rows),
        "edge_evidence": len(evidence_rows),
        "fts_rows": fts_count,
    }


def get_edge_evidence(
    conn: sqlite3.Connection,
    source: str,
    target: str,
    kind: str,
) -> list[dict[str, Any]]:
    """Return deterministic provenance records for one graph edge."""
    rows = conn.execute(
        """
        SELECT reference, source_path, source_line_start, source_line_end,
               resolution, confidence
        FROM edge_evidence
        WHERE source = ? AND target = ? AND kind = ?
        ORDER BY source_path, source_line_start, source_line_end, reference, resolution
        """,
        (source, target, kind),
    )
    return [
        {
            "kind": kind,
            "confidence": confidence,
            "reference": reference,
            "source_path": source_path,
            "source_lines": {"start": start, "end": end},
            "resolution": resolution,
        }
        for reference, source_path, start, end, resolution, confidence in rows
    ]


def reconcile(
    conn: sqlite3.Connection,
    current_structural_ids: set[str],
    current_decision_ids: set[str],
    current_topic_ids: set[str],
    current_edges: set[tuple[str, str, str]],
    current_story_ids: set[str] | None = None,
) -> dict[str, int]:
    """Delete nodes/edges left over from a previous build that no longer
    exist in the current extraction -- a deleted source file, a renamed
    function, a removed decision, a removed/renamed doc topic or story.
    Without this, `save_*` (INSERT OR REPLACE) only ever adds/updates, so
    stale records accumulate forever.

    Independent categories, each reconciled against its own current set:
    structural nodes (kind not in {'decision_ref', 'topic', 'story'}),
    decision_ref nodes, topic nodes, and story nodes -- since each comes
    from a separate extraction pass. Edges are reconciled against
    current_edges so obsolete relationships between surviving nodes are
    also removed.

    Returns counts of what was removed, for disclosure/logging.
    """
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _current_structural (id TEXT PRIMARY KEY)")
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _current_decisions (id TEXT PRIMARY KEY)")
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _current_topics (id TEXT PRIMARY KEY)")
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _current_stories (id TEXT PRIMARY KEY)")
    conn.execute("""
        CREATE TEMP TABLE IF NOT EXISTS _current_edges (
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            kind TEXT NOT NULL,
            PRIMARY KEY (source, target, kind)
        )
    """)
    conn.execute("DELETE FROM _current_structural")
    conn.execute("DELETE FROM _current_decisions")
    conn.execute("DELETE FROM _current_topics")
    conn.execute("DELETE FROM _current_stories")
    conn.execute("DELETE FROM _current_edges")
    conn.executemany("INSERT INTO _current_structural (id) VALUES (?)", [(i,) for i in current_structural_ids])
    conn.executemany("INSERT INTO _current_decisions (id) VALUES (?)", [(i,) for i in current_decision_ids])
    conn.executemany("INSERT INTO _current_topics (id) VALUES (?)", [(i,) for i in current_topic_ids])
    if current_story_ids:
        conn.executemany("INSERT INTO _current_stories (id) VALUES (?)", [(i,) for i in current_story_ids])
    conn.executemany("INSERT INTO _current_edges (source, target, kind) VALUES (?, ?, ?)", current_edges)

    stale_edges = conn.execute("""
        SELECT COUNT(*) FROM edges AS edge
        WHERE NOT EXISTS (
            SELECT 1 FROM _current_edges AS current
            WHERE current.source = edge.source AND current.target = edge.target AND current.kind = edge.kind
        )
    """).fetchone()[0]
    conn.execute("""
        DELETE FROM edges AS edge
        WHERE NOT EXISTS (
            SELECT 1 FROM _current_edges AS current
            WHERE current.source = edge.source AND current.target = edge.target AND current.kind = edge.kind
        )
    """)

    stale_structural = conn.execute("""
        SELECT COUNT(*) FROM nodes
        WHERE kind NOT IN ('decision_ref', 'topic', 'story')
          AND id NOT IN (SELECT id FROM _current_structural)
    """).fetchone()[0]
    conn.execute("""
        DELETE FROM nodes
        WHERE kind NOT IN ('decision_ref', 'topic', 'story')
          AND id NOT IN (SELECT id FROM _current_structural)
    """)

    stale_decisions = conn.execute("""
        SELECT COUNT(*) FROM nodes
        WHERE kind = 'decision_ref' AND id NOT IN (SELECT id FROM _current_decisions)
    """).fetchone()[0]
    conn.execute("""
        DELETE FROM nodes
        WHERE kind = 'decision_ref' AND id NOT IN (SELECT id FROM _current_decisions)
    """)

    stale_topics = conn.execute("""
        SELECT COUNT(*) FROM nodes
        WHERE kind = 'topic' AND id NOT IN (SELECT id FROM _current_topics)
    """).fetchone()[0]
    conn.execute("""
        DELETE FROM nodes
        WHERE kind = 'topic' AND id NOT IN (SELECT id FROM _current_topics)
    """)

    stale_stories = conn.execute("""
        SELECT COUNT(*) FROM nodes
        WHERE kind = 'story' AND id NOT IN (SELECT id FROM _current_stories)
    """).fetchone()[0]
    conn.execute("""
        DELETE FROM nodes
        WHERE kind = 'story' AND id NOT IN (SELECT id FROM _current_stories)
    """)

    return {
        "stale_structural_nodes_removed": stale_structural,
        "stale_decision_nodes_removed": stale_decisions,
        "stale_topic_nodes_removed": stale_topics,
        "stale_story_nodes_removed": stale_stories,
        "stale_edges_removed": stale_edges,
    }


def rebuild_fts(conn: sqlite3.Connection) -> None:
    """Rebuild the full-text index from the current nodes table."""
    conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES ('rebuild')")


def save_metadata(
    conn: sqlite3.Connection,
    schema_version: str,
    project_path: Path,
    git_sha: str,
    codegraph_provenance: dict[str, Any] | None = None,
) -> None:
    """Record graph, source-revision, and CodeGraph-export provenance."""
    provenance = _json_data(codegraph_provenance or {}, "CodeGraph provenance")
    conn.execute(
        """
        INSERT INTO graph_metadata (
            id, schema_version, built_at, project_path, git_sha, codegraph_provenance
        ) VALUES (1, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            schema_version = excluded.schema_version,
            built_at = excluded.built_at,
            project_path = excluded.project_path,
            git_sha = excluded.git_sha,
            codegraph_provenance = excluded.codegraph_provenance
        """,
        (
            schema_version,
            datetime.now(timezone.utc).isoformat(),
            str(project_path.resolve()),
            git_sha,
            provenance,
        ),
    )


def get_metadata(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return persisted graph and CodeGraph provenance."""
    row = conn.execute(
        """
        SELECT schema_version, built_at, project_path, git_sha, codegraph_provenance
        FROM graph_metadata WHERE id = 1
        """
    ).fetchone()
    if row is None:
        return {}
    schema_version, built_at, project_path, git_sha, provenance = row
    return {
        "schema_version": schema_version,
        "built_at": built_at,
        "project_path": project_path,
        "git_sha": git_sha,
        "codegraph_provenance": json.loads(provenance),
    }


def open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open an existing graph database without allowing mutations."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Graph database not found: {db_path}")
    return sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)


def query_by_name(conn: sqlite3.Connection, search: str) -> list[dict[str, Any]]:
    """Full-text search nodes by name. Returns canonical full node records."""
    node_ids = [
        row[0]
        for row in conn.execute(
            """
            SELECT n.id FROM nodes n
            JOIN nodes_fts f ON n.rowid = f.rowid
            WHERE nodes_fts MATCH ?
            ORDER BY n.id
            """,
            (search,),
        )
    ]
    return query_by_ids(conn, node_ids)


def query_by_name_ranked(
    conn: sqlite3.Connection,
    search: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Full-text search nodes ordered by FTS5 BM25 relevance."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    sql = """
        SELECT n.id, bm25(nodes_fts) AS bm25_rank
        FROM nodes_fts
        JOIN nodes AS n ON n.rowid = nodes_fts.rowid
        WHERE nodes_fts MATCH ?
        ORDER BY bm25_rank, n.id
    """
    params: list[Any] = [search]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    nodes = query_by_ids(conn, [node_id for node_id, _ in rows])
    for node, (_, bm25_rank) in zip(nodes, rows):
        node["_bm25_rank"] = float(bm25_rank)
    return nodes


def query_by_ids(
    conn: sqlite3.Connection,
    node_ids: list[str],
) -> list[dict[str, Any]]:
    """Return full node records for IDs in deterministic input order."""
    if not node_ids:
        return []
    placeholders = ", ".join("?" for _ in node_ids)
    rows = {}
    for node_id, kind, name, data in conn.execute(
        f"SELECT id, kind, name, data FROM nodes WHERE id IN ({placeholders})",
        node_ids,
    ):
        node = json.loads(data)
        original_id = node.get("id")
        if original_id != node_id:
            node.setdefault("semantic_id", original_id)
        node["id"] = node_id
        node["kind"] = kind
        node["name"] = name
        rows[node_id] = node
    return [rows[node_id] for node_id in node_ids if node_id in rows]


def graph_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return graph totals and deterministic counts grouped by kind."""
    return {
        "nodes": conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
        "edges": conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        "nodes_by_kind": dict(conn.execute(
            "SELECT kind, COUNT(*) FROM nodes GROUP BY kind ORDER BY kind"
        )),
        "edges_by_kind": dict(conn.execute(
            "SELECT kind, COUNT(*) FROM edges GROUP BY kind ORDER BY kind"
        )),
    }


def neighbors(
    conn: sqlite3.Connection,
    node_id: str,
    direction: str = "both",
) -> list[dict[str, Any]]:
    """Return adjacent nodes and their connecting edges in requested direction."""
    if direction not in {"in", "out", "both"}:
        raise ValueError(f"invalid direction: {direction}")

    records: list[dict[str, Any]] = []
    if direction in {"out", "both"}:
        rows = conn.execute(
            """
            SELECT edge.source, edge.target, edge.kind, node.data
            FROM edges AS edge
            JOIN nodes AS node ON node.id = edge.target
            WHERE edge.source = ?
            ORDER BY edge.kind, edge.target
            """,
            (node_id,),
        )
        records.extend(
            {
                "direction": "out",
                "edge": {"source": source, "target": target, "kind": kind},
                "node": json.loads(data),
            }
            for source, target, kind, data in rows
        )
    if direction in {"in", "both"}:
        rows = conn.execute(
            """
            SELECT edge.source, edge.target, edge.kind, node.data
            FROM edges AS edge
            JOIN nodes AS node ON node.id = edge.source
            WHERE edge.target = ?
            ORDER BY edge.kind, edge.source
            """,
            (node_id,),
        )
        records.extend(
            {
                "direction": "in",
                "edge": {"source": source, "target": target, "kind": kind},
                "node": json.loads(data),
            }
            for source, target, kind, data in rows
        )
    return records


def shortest_path(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    max_depth: int = 10,
    direction: str = "out",
) -> dict[str, list[dict[str, Any]]]:
    """Find a bounded shortest path with cycle-safe breadth-first search.

    direction="out" follows outgoing edges (the original behavior),
    "in" follows incoming edges, and "both" traverses edges in
    either direction.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if direction not in {"out", "in", "both"}:
        raise ValueError(f"invalid direction: {direction}")

    endpoints = {
        row[0]: json.loads(row[1])
        for row in conn.execute(
            "SELECT id, data FROM nodes WHERE id IN (?, ?)",
            (source_id, target_id),
        )
    }
    if source_id not in endpoints or target_id not in endpoints:
        return {"nodes": [], "edges": []}
    if source_id == target_id:
        return {"nodes": [endpoints[source_id]], "edges": []}

    visited = {source_id}
    predecessors: dict[str, tuple[str, dict[str, str]]] = {}
    frontier = [source_id]

    out_sql = """
        SELECT source, target, kind FROM edges
        WHERE source IN ({placeholders})
        ORDER BY source, kind, target
    """
    in_sql = """
        SELECT source, target, kind FROM edges
        WHERE target IN ({placeholders})
        ORDER BY source, kind, target
    """

    for _ in range(max_depth):
        placeholders = ", ".join("?" for _ in frontier)
        neighbors: list[tuple[str, str, dict[str, str]]] = []
        if direction in {"out", "both"}:
            neighbors.extend(
                (target, source, {"source": source, "target": target, "kind": kind})
                for source, target, kind in conn.execute(
                    out_sql.format(placeholders=placeholders), frontier
                )
            )
        if direction in {"in", "both"}:
            neighbors.extend(
                (source, target, {"source": source, "target": target, "kind": kind})
                for source, target, kind in conn.execute(
                    in_sql.format(placeholders=placeholders), frontier
                )
            )
        next_frontier: list[str] = []
        for neighbor, previous, edge in neighbors:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            predecessors[neighbor] = (previous, edge)
            if neighbor == target_id:
                path_ids = [target_id]
                path_edges: list[dict[str, str]] = []
                current = target_id
                while current != source_id:
                    previous, edge = predecessors[current]
                    path_ids.append(previous)
                    path_edges.append(edge)
                    current = previous
                path_ids.reverse()
                path_edges.reverse()
                path_nodes = {
                    row[0]: json.loads(row[1])
                    for row in conn.execute(
                        f"SELECT id, data FROM nodes WHERE id IN ({', '.join('?' for _ in path_ids)})",
                        path_ids,
                    )
                }
                return {
                    "nodes": [path_nodes[node_id] for node_id in path_ids],
                    "edges": path_edges,
                }
            next_frontier.append(neighbor)
        if not next_frontier:
            break
        frontier = next_frontier

    return {"nodes": [], "edges": []}


def export_html(conn: sqlite3.Connection, output_path: Path) -> None:
    """Export a self-contained, interactive offline graph visualization.

    The view embeds vis-network locally, supports search and kind filtering,
    and turns a node click into a focused, directional neighborhood view.
    """
    nodes_rows = conn.execute("SELECT id, kind, name, data FROM nodes").fetchall()
    edges_rows = conn.execute("SELECT source, target, kind FROM edges").fetchall()

    kind_colors = {
        "decision_ref": "#e11d48",
        "story": "#ea580c",
        "topic": "#059669",
        "file": "#475569",
        "function": "#2563eb",
        "method": "#1d4ed8",
        "class": "#7c3aed",
        "struct": "#6d28d9",
        "interface": "#0d9488",
        "route": "#db2777",
        "table": "#d97706",
        "variable": "#0284c7",
        "import": "#94a3b8",
        "exception_type": "#dc2626",
    }
    kind_shapes = {
        "decision_ref": "diamond",
        "story": "diamond",
        "topic": "diamond",
        "file": "box",
        "class": "ellipse",
        "struct": "ellipse",
        "interface": "ellipse",
        "route": "hexagon",
        "table": "database",
    }
    kind_sizes = {
        "decision_ref": 20,
        "story": 19,
        "topic": 19,
        "file": 17,
        "route": 18,
        "table": 18,
        "import": 8,
        "variable": 11,
    }

    vis_nodes = []
    kind_counts: dict[str, int] = {}
    for node_id, kind, name, data_text in nodes_rows:
        try:
            data = json.loads(data_text)
        except (TypeError, json.JSONDecodeError):
            data = {}
        full_label = name or node_id
        display_label = full_label if len(full_label) <= 34 else full_label[:31] + "..."
        source_path = data.get("source_path") or data.get("filePath") or ""
        color = kind_colors.get(kind, "#64748b")
        size = kind_sizes.get(kind, 14)
        vis_nodes.append({
            "id": node_id,
            "label": display_label,
            "fullLabel": full_label,
            "kind": kind,
            "sourcePath": source_path,
            "title": f"{kind}: {full_label}",
            "shape": kind_shapes.get(kind, "dot"),
            "size": size,
            "baseSize": size,
            "baseColor": color,
            "color": {"background": color, "border": color},
            "borderWidth": 1,
        })
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    vis_edges = [
        {
            "id": f"edge:{index}",
            "from": source,
            "to": target,
            "label": kind,
            "kind": kind,
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.55}},
            "width": 1,
            "color": {"color": "#94a3b8", "opacity": 0.28},
            "font": {
                "size": 8,
                "color": "rgba(71,85,105,0.38)",
                "strokeWidth": 0,
                "align": "middle",
            },
        }
        for index, (source, target, kind) in enumerate(edges_rows)
    ]

    #vis_js_content = _VIS_JS_PATH.read_text(encoding="utf-8")
    # Salar change
    if _VIS_JS_PATH.is_file():
        vis_js_content = _VIS_JS_PATH.read_text(encoding="utf-8")
    else:
        vis_js_content = "/* offline vis fallback */"
    kinds = sorted(kind_counts)
    kind_options = "".join(f'<option value="{kind}">{kind}</option>' for kind in kinds)
    legend_items = "".join(
        '<div class="legend-item">'
        f'<span class="swatch" style="background:{kind_colors.get(kind, "#64748b")}"></span>'
        f'<span>{kind}</span><strong>{kind_counts[kind]}</strong></div>'
        for kind in kinds
    )
    
    # Salar change

def extract_subgraph(
    conn: sqlite3.Connection, 
    seed_ids: list[str], 
    max_depth: int = 1,
    direction: str = "both"
) -> dict[str, Any]:
    """Extract neighboring nodes and relations around seed nodes up to max_depth."""
    if not seed_ids:
        return {"nodes": [], "edges": []}
        
    visited_nodes = set(seed_ids)
    frontier = set(seed_ids)
    subgraph_edges = []
    
    for _ in range(max_depth):
        if not frontier:
            break
        
        placeholders = ", ".join("?" for _ in frontier)
        query_parts = []
        params = []
        
        if direction in {"out", "both"}:
            query_parts.append(f"source IN ({placeholders})")
            params.extend(frontier)
        if direction in {"in", "both"}:
            query_parts.append(f"target IN ({placeholders})")
            params.extend(frontier)
            
        where_clause = " OR ".join(query_parts)
        sql = f"SELECT source, target, kind FROM edges WHERE {where_clause} ORDER BY source, target, kind"
        
        rows = conn.execute(sql, params).fetchall()
        next_frontier = set()
        
        for source, target, kind in rows:
            subgraph_edges.append({"source": source, "target": target, "kind": kind})
            for node_id in (source, target):
                if node_id not in visited_nodes:
                    visited_nodes.add(node_id)
                    next_frontier.add(node_id)
                    
        frontier = next_frontier
        
    extracted_nodes = query_by_ids(conn, list(visited_nodes))
    return {
        "nodes": extracted_nodes,
        "edges": subgraph_edges,
    }
    


def export_html(conn: sqlite3.Connection, output_path: Path) -> None:
    """Export a self-contained, interactive offline graph visualization.

    The view embeds vis-network locally, supports search and kind filtering,
    and turns a node click into a focused, directional neighborhood view.
    """
    nodes_rows = conn.execute("SELECT id, kind, name, data FROM nodes").fetchall()
    edges_rows = conn.execute("SELECT source, target, kind FROM edges").fetchall()

    kind_colors = {
        "decision_ref": "#e11d48",
        "story": "#ea580c",
        "topic": "#059669",
        "file": "#475569",
        "function": "#2563eb",
        "method": "#1d4ed8",
        "class": "#7c3aed",
        "struct": "#6d28d9",
        "interface": "#0d9488",
        "route": "#db2777",
        "table": "#d97706",
        "variable": "#0284c7",
        "import": "#94a3b8",
        "exception_type": "#dc2626",
    }
    kind_shapes = {
        "decision_ref": "diamond",
        "story": "diamond",
        "topic": "diamond",
        "file": "box",
        "class": "ellipse",
        "struct": "ellipse",
        "interface": "ellipse",
        "route": "hexagon",
        "table": "database",
    }
    kind_sizes = {
        "decision_ref": 20,
        "story": 19,
        "topic": 19,
        "file": 17,
        "route": 18,
        "table": 18,
        "import": 8,
        "variable": 11,
    }

    vis_nodes = []
    kind_counts: dict[str, int] = {}
    for node_id, kind, name, data_text in nodes_rows:
        try:
            data = json.loads(data_text)
        except (TypeError, json.JSONDecodeError):
            data = {}
        full_label = name or node_id
        display_label = full_label if len(full_label) <= 34 else full_label[:31] + "..."
        source_path = data.get("source_path") or data.get("filePath") or ""
        color = kind_colors.get(kind, "#64748b")
        size = kind_sizes.get(kind, 14)
        vis_nodes.append({
            "id": node_id,
            "label": display_label,
            "fullLabel": full_label,
            "kind": kind,
            "sourcePath": source_path,
            "title": f"{kind}: {full_label}",
            "shape": kind_shapes.get(kind, "dot"),
            "size": size,
            "baseSize": size,
            "baseColor": color,
            "color": {"background": color, "border": color},
            "borderWidth": 1,
        })
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    vis_edges = [
        {
            "id": f"edge:{index}",
            "from": source,
            "to": target,
            "label": kind,
            "kind": kind,
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.55}},
            "width": 1,
            "color": {"color": "#94a3b8", "opacity": 0.28},
            "font": {
                "size": 8,
                "color": "rgba(71,85,105,0.38)",
                "strokeWidth": 0,
                "align": "middle",
            },
        }
        for index, (source, target, kind) in enumerate(edges_rows)
    ]

    if _VIS_JS_PATH.is_file():
        vis_js_content = _VIS_JS_PATH.read_text(encoding="utf-8")
    else:
        vis_js_content = "/* offline vis fallback */"

    kinds = sorted(kind_counts)
    kind_options = "".join(f'<option value="{kind}">{kind}</option>' for kind in kinds)
    legend_items = "".join(
        '<div class="legend-item">'
        f'<span class="swatch" style="background:{kind_colors.get(kind, "#64748b")}"></span>'
        f'<span>{kind}</span><strong>{kind_counts[kind]}</strong></div>'
        for kind in kinds
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>EMP-KG Knowledge Graph</title>
  <script>{vis_js_content}</script>
  <style>
    :root {{ --ink:#0f172a; --muted:#64748b; --line:#dbe4ee; --panel:#ffffff; --blue:#2563eb; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; overflow:hidden; font-family:Inter,Segoe UI,Arial,sans-serif; color:var(--ink); background:#f8fafc; }}
    #graph {{ width:100vw; height:100vh; background:radial-gradient(circle at center,#fff 0,#f8fafc 72%); }}
    .panel {{ position:absolute; z-index:10; background:rgba(255,255,255,.97); border:1px solid var(--line);
              border-radius:12px; box-shadow:0 8px 28px rgba(15,23,42,.12); }}
    #controls {{ top:16px; left:16px; width:255px; padding:16px; }}
    #controls h1 {{ margin:0 0 4px; font-size:17px; }}
    #controls .hint {{ color:var(--muted); font-size:11px; line-height:1.35; margin-bottom:12px; }}
    #controls label {{ display:block; margin:9px 0 4px; font-weight:700; font-size:12px; }}
    #controls input,#controls select {{ width:100%; padding:8px 9px; border:1px solid #cbd5e1; border-radius:7px; background:white; }}
    .button-row {{ display:flex; gap:7px; margin-top:11px; }}
    button {{ border:0; border-radius:7px; padding:8px 10px; cursor:pointer; font-weight:700; font-size:12px; }}
    #reset-view {{ flex:1; color:white; background:var(--blue); }}
    #fit-view {{ color:#334155; background:#e2e8f0; }}
    #counts {{ margin-top:10px; padding:8px; border-radius:7px; background:#eff6ff; color:#1e40af; font-size:12px; font-weight:700; }}
    #legend {{ max-height:240px; overflow:auto; margin-top:10px; padding-top:8px; border-top:1px solid #e2e8f0; }}
    .legend-item {{ display:grid; grid-template-columns:12px 1fr auto; align-items:center; gap:7px; padding:3px 1px; color:#475569; font-size:11px; }}
    .legend-item strong {{ color:#0f172a; }}
    .swatch {{ width:10px; height:10px; border-radius:50%; }}
    #inspector {{ top:16px; right:16px; width:330px; max-height:calc(100vh - 32px); overflow:auto; padding:17px; }}
    #inspector.empty {{ color:var(--muted); font-size:13px; line-height:1.5; }}
    .kind-pill {{ display:inline-block; padding:3px 7px; border-radius:999px; color:white; font-size:10px; font-weight:800; text-transform:uppercase; }}
    .node-name {{ margin:8px 0 4px; font-size:19px; overflow-wrap:anywhere; }}
    .node-id,.source {{ color:var(--muted); font:11px/1.45 Consolas,monospace; overflow-wrap:anywhere; }}
    .section-title {{ margin:14px 0 6px; font-size:11px; color:#475569; text-transform:uppercase; letter-spacing:.08em; }}
    .connection {{ display:grid; grid-template-columns:auto 1fr; gap:7px; width:100%; margin:4px 0; padding:7px 8px;
                   text-align:left; background:#f8fafc; border:1px solid #e2e8f0; color:#1e293b; }}
    .connection:hover {{ background:#eff6ff; border-color:#93c5fd; }}
    .edge-kind {{ color:#2563eb; font:700 10px Consolas,monospace; }}
    .neighbor-name {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .focus-note {{ position:absolute; left:50%; bottom:17px; transform:translateX(-50%); z-index:9; padding:8px 12px;
                  border:1px solid #bfdbfe; border-radius:999px; background:rgba(239,246,255,.96); color:#1e40af; font-size:12px; }}
    @media (max-width:850px) {{ #inspector {{ width:280px; }} #controls {{ width:220px; }} }}
  </style>
</head>
<body>
  <div id="controls" class="panel">
    <h1>EMP-KG Graph Explorer</h1>
    <div class="hint">Click a node for direct links. Double-click for two-hop context.</div>
    <label for="search">Search node</label>
    <input id="search" type="search" placeholder="function, file, topic...">
    <label for="kind-filter">Node kind</label>
    <select id="kind-filter"><option value="">All kinds</option>{kind_options}</select>
    <div class="button-row"><button id="reset-view">Show whole graph</button><button id="fit-view">Fit</button></div>
    <div id="counts">{len(vis_nodes)} nodes · {len(vis_edges)} edges</div>
    <div id="legend">{legend_items}</div>
  </div>
  <aside id="inspector" class="panel empty">
    <strong>Select a node</strong><br>Related nodes and directional arrows will be isolated and listed here.
  </aside>
  <div id="focus-note" class="focus-note">Whole graph · select a node to isolate its neighborhood</div>
  <div id="graph"></div>
  <script>
    const nodes = new vis.DataSet({json.dumps(vis_nodes)});
    const edges = new vis.DataSet({json.dumps(vis_edges)});
    const allNodeIds = nodes.getIds();
    const allEdgeIds = edges.getIds();
    let selectedNodeId = null;

    const network = new vis.Network(document.getElementById('graph'), {{nodes,edges}}, {{
      autoResize:true,
      interaction:{{hover:true,navigationButtons:true,keyboard:true,tooltipDelay:120}},
      nodes:{{font:{{size:12,color:'#0f172a',strokeWidth:3,strokeColor:'#ffffff'}},shadow:{{enabled:true,size:4,x:1,y:2,color:'rgba(15,23,42,.15)'}}}},
      edges:{{smooth:{{enabled:true,type:'dynamic',roundness:.35}},selectionWidth:4,hoverWidth:2}},
      physics:{{enabled:true,solver:'barnesHut',barnesHut:{{gravitationalConstant:-5200,centralGravity:.18,springLength:135,springConstant:.035,damping:.22,avoidOverlap:.65}},stabilization:{{enabled:true,iterations:650,updateInterval:50}}}},
      layout:{{improvedLayout:true,randomSeed:7}}
    }});

    network.once('stabilizationIterationsDone', () => {{ network.setOptions({{physics:false}}); network.fit({{animation:false}}); }});

    function escapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}})[c]);
    }}

    function resetStyles(fit=true) {{
      selectedNodeId = null;
      nodes.update(nodes.get().map(n => ({{id:n.id,opacity:1,size:n.baseSize,borderWidth:1,
        color:{{background:n.baseColor,border:n.baseColor}},font:{{size:12,color:'#0f172a',strokeWidth:3,strokeColor:'#ffffff'}}}})));
      edges.update(edges.get().map(e => ({{id:e.id,width:1,color:{{color:'#94a3b8',opacity:.28}},
        font:{{size:8,color:'rgba(71,85,105,.38)',strokeWidth:0,background:'rgba(255,255,255,0)'}},
        arrows:{{to:{{enabled:true,scaleFactor:.55}}}}}})));
      document.getElementById('inspector').className='panel empty';
      document.getElementById('inspector').innerHTML='<strong>Select a node</strong><br>Related nodes and directional arrows will be isolated and listed here.';
      document.getElementById('focus-note').textContent='Whole graph · select a node to isolate its neighborhood';
      if (fit) network.fit({{animation:{{duration:450,easingFunction:'easeInOutQuad'}}}});
    }}

    function neighborhood(nodeId, depth) {{
      const visibleNodes = new Set([nodeId]);
      const visibleEdges = new Set();
      let frontier = new Set([nodeId]);
      for (let level=0; level<depth; level++) {{
        const next = new Set();
        edges.forEach(edge => {{
          if (frontier.has(edge.from) || frontier.has(edge.to)) {{
            visibleEdges.add(edge.id);
            const other = frontier.has(edge.from) ? edge.to : edge.from;
            if (!visibleNodes.has(other)) next.add(other);
            visibleNodes.add(edge.from); visibleNodes.add(edge.to);
          }}
        }});
        frontier=next;
      }}
      return {{visibleNodes,visibleEdges}};
    }}

    function renderInspector(nodeId) {{
      const node=nodes.get(nodeId);
      const outgoing=edges.get({{filter:e=>e.from===nodeId}});
      const incoming=edges.get({{filter:e=>e.to===nodeId}});
      const row=(edge,direction) => {{
        const neighborId=direction==='out' ? edge.to : edge.from;
        const neighbor=nodes.get(neighborId);
        const arrow=direction==='out' ? '→' : '←';
        return `<button class="connection" data-node="${{escapeHtml(neighborId)}}"><span class="edge-kind">${{arrow}} ${{escapeHtml(edge.kind)}}</span><span class="neighbor-name">${{escapeHtml(neighbor?.fullLabel||neighborId)}}</span></button>`;
      }};
      const inspector=document.getElementById('inspector');
      inspector.className='panel';
      inspector.innerHTML=`<span class="kind-pill" style="background:${{escapeHtml(node.baseColor)}}">${{escapeHtml(node.kind)}}</span>
        <h2 class="node-name">${{escapeHtml(node.fullLabel)}}</h2>
        <div class="node-id">${{escapeHtml(node.id)}}</div>
        ${{node.sourcePath ? `<div class="source">Source: ${{escapeHtml(node.sourcePath)}}</div>` : ''}}
        <div class="section-title">Outgoing (${{outgoing.length}})</div>${{outgoing.map(e=>row(e,'out')).join('')||'<div class="source">None</div>'}}
        <div class="section-title">Incoming (${{incoming.length}})</div>${{incoming.map(e=>row(e,'in')).join('')||'<div class="source">None</div>'}}`;
      inspector.querySelectorAll('.connection').forEach(button => button.addEventListener('click',()=>focusNeighbors(button.dataset.node,1)));
    }}

    function focusNeighbors(nodeId, depth=1) {{
      selectedNodeId=nodeId;
      const {{visibleNodes,visibleEdges}}=neighborhood(nodeId,depth);
      nodes.update(nodes.get().map(node => {{
        if (node.id===nodeId) return {{id:node.id,opacity:1,size:node.baseSize+10,borderWidth:5,
          color:{{background:node.baseColor,border:'#0f172a'}},font:{{size:15,color:'#0f172a',strokeWidth:4,strokeColor:'#ffffff'}}}};
        if (visibleNodes.has(node.id)) return {{id:node.id,opacity:1,size:node.baseSize+3,borderWidth:2,
          color:{{background:node.baseColor,border:'#ffffff'}},font:{{size:12,color:'#0f172a',strokeWidth:4,strokeColor:'#ffffff'}}}};
        return {{id:node.id,opacity:.055,size:node.baseSize,borderWidth:1,font:{{size:10,color:'rgba(100,116,139,.12)',strokeWidth:0}}}};
      }}));
      edges.update(edges.get().map(edge => {{
        if (!visibleEdges.has(edge.id)) return {{id:edge.id,width:.35,color:{{color:'#cbd5e1',opacity:.06}},
          font:{{size:1,color:'rgba(0,0,0,0)',strokeWidth:0,background:'rgba(255,255,255,0)'}},arrows:{{to:{{enabled:true,scaleFactor:.25}}}}}};
        const direct=edge.from===nodeId||edge.to===nodeId;
        const color=edge.from===nodeId ? '#2563eb' : (edge.to===nodeId ? '#e11d48' : '#059669');
        return {{id:edge.id,width:direct?3.6:2.3,color:{{color,opacity:1}},
          font:{{size:direct?13:11,color:'#0f172a',strokeWidth:3,strokeColor:'#ffffff',background:'rgba(255,255,255,.9)'}},
          arrows:{{to:{{enabled:true,scaleFactor:direct?1.05:.8}}}}}};
      }}));
      renderInspector(nodeId);
      document.getElementById('focus-note').textContent=`${{depth}}-hop focus · ${{visibleNodes.size}} nodes · ${{visibleEdges.size}} arrows`;
      network.fit({{nodes:Array.from(visibleNodes),animation:{{duration:550,easingFunction:'easeInOutQuad'}},maxZoomLevel:1.35}});
    }}

    network.on('click',params=>{{ if(params.nodes.length) focusNeighbors(params.nodes[0],1); else resetStyles(false); }});
    network.on('doubleClick',params=>{{ if(params.nodes.length) focusNeighbors(params.nodes[0],2); }});
    document.getElementById('reset-view').addEventListener('click',()=>{{document.getElementById('search').value='';document.getElementById('kind-filter').value='';resetStyles(true);}});
    document.getElementById('fit-view').addEventListener('click',()=>network.fit({{animation:{{duration:400}}}}));

    document.getElementById('search').addEventListener('input',event=>{{
      const query=event.target.value.trim().toLowerCase();
      if(!query){{resetStyles(false);return;}}
      const hits=[];
      nodes.update(nodes.get().map(node=>{{const hit=(node.fullLabel+' '+node.id+' '+node.sourcePath).toLowerCase().includes(query);if(hit)hits.push(node.id);return{{id:node.id,opacity:hit?1:.04,size:hit?node.baseSize+5:node.baseSize}};}}));
      edges.update(edges.get().map(edge=>({{id:edge.id,color:{{color:'#cbd5e1',opacity:.05}},font:{{size:1,color:'rgba(0,0,0,0)'}}}})));
      document.getElementById('focus-note').textContent=`Search · ${{hits.length}} matching nodes`;
      if(hits.length)network.fit({{nodes:hits,animation:{{duration:350}},maxZoomLevel:1.2}});
    }});

    document.getElementById('kind-filter').addEventListener('change',event=>{{
      const kind=event.target.value;
      if(!kind){{resetStyles(false);return;}}
      const hits=[];
      nodes.update(nodes.get().map(node=>{{const hit=node.kind===kind;if(hit)hits.push(node.id);return{{id:node.id,opacity:hit?1:.04,size:hit?node.baseSize+3:node.baseSize}};}}));
      edges.update(edges.get().map(edge=>({{id:edge.id,color:{{color:'#cbd5e1',opacity:.05}},font:{{size:1,color:'rgba(0,0,0,0)'}}}})));
      document.getElementById('focus-note').textContent=`Kind filter · ${{kind}} · ${{hits.length}} nodes`;
      if(hits.length)network.fit({{nodes:hits,animation:{{duration:350}},maxZoomLevel:1.1}});
    }});
  </script>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


def extract_subgraph(
    conn: sqlite3.Connection, 
    seed_ids: list[str], 
    max_depth: int = 1,
    direction: str = "both"
) -> dict[str, Any]:
    """Extract neighboring nodes and relations around seed nodes up to max_depth."""
    if not seed_ids:
        return {"nodes": [], "edges": []}
        
    visited_nodes = set(seed_ids)
    frontier = set(seed_ids)
    subgraph_edges = []
    
    for _ in range(max_depth):
        if not frontier:
            break
        
        placeholders = ", ".join("?" for _ in frontier)
        query_parts = []
        params = []
        
        if direction in {"out", "both"}:
            query_parts.append(f"source IN ({placeholders})")
            params.extend(frontier)
        if direction in {"in", "both"}:
            query_parts.append(f"target IN ({placeholders})")
            params.extend(frontier)
            
        where_clause = " OR ".join(query_parts)
        sql = f"SELECT source, target, kind FROM edges WHERE {where_clause} ORDER BY source, target, kind"
        
        rows = conn.execute(sql, params).fetchall()
        next_frontier = set()
        
        for source, target, kind in rows:
            subgraph_edges.append({"source": source, "target": target, "kind": kind})
            for node_id in (source, target):
                if node_id not in visited_nodes:
                    visited_nodes.add(node_id)
                    next_frontier.add(node_id)
                    
        frontier = next_frontier
        
    extracted_nodes = query_by_ids(conn, list(visited_nodes))
    return {
        "nodes": extracted_nodes,
        "edges": subgraph_edges,
    }