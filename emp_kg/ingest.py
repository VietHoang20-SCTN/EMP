"""Incremental knowledge ingestion and memory management for emp-kg."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from emp_kg import embeddings, store
from emp_kg.merger import _topic_id


class _HTMLTextExtractor(HTMLParser):
    """Lightweight stdlib HTML-to-text extractor without external dependencies."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip = False
        self.title: str = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = True
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip = False
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title and not self.title:
            self.title = text
        self._chunks.append(text)

    def get_text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _existing_hashes_by_source(conn: sqlite3.Connection) -> dict[str, tuple[str, str]]:
    """Map source_path -> (node_id, content_hash) for persisted topic nodes."""
    rows = conn.execute("SELECT id, data FROM nodes WHERE kind = 'topic'").fetchall()
    mapping: dict[str, tuple[str, str]] = {}
    for node_id, data_text in rows:
        try:
            payload = json.loads(data_text)
        except json.JSONDecodeError:
            continue
        source_path = payload.get("source_path")
        content_hash = payload.get("content_hash", "")
        if isinstance(source_path, str):
            mapping[source_path] = (node_id, content_hash)
    return mapping


def _update_incremental_embeddings(
    conn: sqlite3.Connection,
    node_ids: list[str],
    model_id: str,
) -> int:
    """Re-embed ONLY the specified modified/new node IDs if local embeddings are active."""
    if not node_ids:
        return 0
    docs = [
        doc for doc in embeddings.embedding_content_hashes(store.embedding_documents(conn))
        if doc["node_id"] in set(node_ids)
    ]
    if not docs:
        return 0
    embedder = embeddings.SentenceTransformerEmbedder(model_id)
    vectors = embedder.encode([d["content"] for d in docs])
    embeddings.validate_vectors(vectors, len(docs))
    return store.save_node_embeddings(
        conn,
        [
            {
                "node_id": doc["node_id"],
                "model_id": model_id,
                "vector": vec,
                "content_hash": doc["content_hash"],
            }
            for doc, vec in zip(docs, vectors)
        ],
    )


def ingest_directory_incremental(
    conn: sqlite3.Connection,
    target_dir: Path,
    *,
    extensions: set[str] = {".py", ".md", ".txt", ".rst"},
    use_embeddings: bool = False,
    embedding_model: str = embeddings.DEFAULT_MODEL,
) -> dict[str, Any]:
    """Incrementally ingest only new or modified files without full graph rebuild."""
    root = target_dir.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Target path does not exist: {root}")

    existing = _existing_hashes_by_source(conn)
    files = [root] if root.is_file() else sorted(root.rglob("*"))

    topics_to_upsert: list[dict[str, Any]] = []
    skipped = 0
    seen_sources: set[str] = set()

    for path in files:
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if any(part.startswith(".") for part in path.relative_to(root.parent).parts):
            continue

        rel_source = str(path.relative_to(root.parent)) if root.is_dir() else path.name
        seen_sources.add(rel_source)
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if not content.strip():
            continue

        c_hash = _sha256_text(content)
        prev = existing.get(rel_source)
        if prev is not None and prev[1] == c_hash:
            skipped += 1
            continue

        topics_to_upsert.append({
            "title": path.stem,
            "content": content,
            "source_path": rel_source,
            "content_hash": c_hash,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "ingestion_type": "incremental_file",
        })

    # Detect deleted files under the scanned directory
    deleted_ids: list[str] = []
    if root.is_dir():
        prefix = f"{root.name}/"
        for src, (node_id, _) in existing.items():
            if src.startswith(prefix) and src not in seen_sources:
                deleted_ids.append(node_id)

    updated_node_ids: list[str] = []
    with conn:
        if deleted_ids:
            placeholders = ", ".join("?" for _ in deleted_ids)
            conn.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", deleted_ids)

        if topics_to_upsert:
            store.save_topic_nodes(conn, topics_to_upsert)
            updated_node_ids = [_topic_id(t) for t in topics_to_upsert]

        if topics_to_upsert or deleted_ids:
            store.rebuild_fts(conn)
            store.validate_graph(conn)

        embedded_count = 0
        if use_embeddings and updated_node_ids:
            embedded_count = _update_incremental_embeddings(
                conn, updated_node_ids, embedding_model
            )

    return {
        "updated": len(topics_to_upsert),
        "skipped_unchanged": skipped,
        "deleted": len(deleted_ids),
        "embedded": embedded_count,
    }


def ingest_url(
    conn: sqlite3.Connection,
    url: str,
    *,
    title: str | None = None,
    use_embeddings: bool = False,
    embedding_model: str = embeddings.DEFAULT_MODEL,
) -> dict[str, Any]:
    """Fetch a web URL, extract clean text, and incrementally upsert into the graph."""
    req = urllib.request.Request(url, headers={"User-Agent": "emp-kg/1.0"})
    with urllib.request.urlopen(req, timeout=20.0) as resp:
        raw_html = resp.read().decode("utf-8", errors="replace")

    parser = _HTMLTextExtractor()
    parser.feed(raw_html)
    clean_text = parser.get_text()
    if not clean_text:
        raise ValueError(f"No extractable text found at URL: {url}")

    c_hash = _sha256_text(clean_text)
    existing = _existing_hashes_by_source(conn)
    prev = existing.get(url)
    if prev is not None and prev[1] == c_hash:
        return {"status": "unchanged", "node_id": prev[0], "url": url}

    topic_record = {
        "title": title or parser.title or url,
        "content": clean_text,
        "source_path": url,
        "content_hash": c_hash,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "ingestion_type": "url",
    }
    node_id = _topic_id(topic_record)

    with conn:
        store.save_topic_nodes(conn, [topic_record])
        store.rebuild_fts(conn)
        store.validate_graph(conn)
        if use_embeddings:
            _update_incremental_embeddings(conn, [node_id], embedding_model)

    return {"status": "ingested", "node_id": node_id, "title": topic_record["title"]}


def ingest_note(
    conn: sqlite3.Connection,
    text: str,
    *,
    title: str | None = None,
    tags: list[str] | None = None,
    use_embeddings: bool = False,
    embedding_model: str = embeddings.DEFAULT_MODEL,
) -> dict[str, Any]:
    """Ingest natural-language personal memory or domain knowledge into the graph."""
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Note text cannot be empty")

    c_hash = _sha256_text(clean_text)
    short_id = c_hash[:12]
    source_uri = f"memory://note/{short_id}"
    cleaned_title = title.strip() if title else ""
    resolved_title = cleaned_title or (
        clean_text[:50] + ("..." if len(clean_text) > 50 else "")
    )

    topic_record = {
        "title": resolved_title,
        "content": clean_text,
        "source_path": source_uri,
        "content_hash": c_hash,
        "tags": tags or [],
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "ingestion_type": "natural_language_note",
    }
    node_id = _topic_id(topic_record)

    with conn:
        store.save_topic_nodes(conn, [topic_record])
        store.rebuild_fts(conn)
        store.validate_graph(conn)
        if use_embeddings:
            _update_incremental_embeddings(conn, [node_id], embedding_model)

    return {"status": "saved", "node_id": node_id, "title": resolved_title}


def list_records(
    conn: sqlite3.Connection,
    *,
    kind: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List persisted knowledge records with source metadata."""
    if kind:
        rows = conn.execute(
            "SELECT id, kind, name, data FROM nodes WHERE kind = ? ORDER BY id LIMIT ?",
            (kind, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, kind, name, data FROM nodes ORDER BY kind, id LIMIT ?",
            (limit,),
        ).fetchall()

    results = []
    for node_id, node_kind, name, data_text in rows:
        try:
            payload = json.loads(data_text)
        except json.JSONDecodeError:
            payload = {}
        results.append({
            "id": node_id,
            "kind": node_kind,
            "name": name,
            "source_path": payload.get("source_path") or payload.get("filePath") or "",
            "ingestion_type": payload.get("ingestion_type", "build"),
        })
    return results


def delete_record(conn: sqlite3.Connection, node_id: str) -> bool:
    """Delete a single node record and cascade-clean its edges/embeddings/FTS."""
    with conn:
        cursor = conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        if cursor.rowcount == 0:
            return False
        store.rebuild_fts(conn)
        store.validate_graph(conn)
    return True