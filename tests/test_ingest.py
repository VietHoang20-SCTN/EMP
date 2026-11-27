"""Comprehensive unit and CLI integration tests for incremental ingestion."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from emp_kg import cli, context, embeddings, ingest, store


class _FakeEmbedder:
    """Deterministic lightweight embedder for offline unit tests."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 10 + 1), 0.5, 0.25, 0.1] for t in texts]


@pytest.fixture
def db_conn(tmp_path: Path):
    db_file = tmp_path / "test-ingest.db"
    conn = store.init_db(db_file)
    yield conn, db_file
    conn.close()


# ---------------------------------------------------------------------------
# 1. Incremental Directory & File Ingestion Tests
# ---------------------------------------------------------------------------


def test_ingest_directory_incremental_lifecycle(db_conn, tmp_path: Path):
    conn, _ = db_conn
    docs_dir = tmp_path / "knowledge"
    docs_dir.mkdir()

    f1 = docs_dir / "arch.md"
    f2 = docs_dir / "notes.txt"
    empty_file = docs_dir / "empty.md"
    ignored_file = docs_dir / "binary.bin"
    hidden_file = docs_dir / ".secret.md"

    f1.write_text("# Architecture\nInitial system design.", encoding="utf-8")
    f2.write_text("Operational runbook details.", encoding="utf-8")
    empty_file.write_text("   \n\t  ", encoding="utf-8")
    ignored_file.write_bytes(b"\x00\x01\x02")
    hidden_file.write_text("Should be ignored.", encoding="utf-8")

    # Initial ingest: only f1 and f2 should be ingested
    res1 = ingest.ingest_directory_incremental(conn, docs_dir)
    assert res1 == {
        "updated": 2,
        "skipped_unchanged": 0,
        "deleted": 0,
        "embedded": 0,
    }

    # Re-run without changes: both must be skipped
    res2 = ingest.ingest_directory_incremental(conn, docs_dir)
    assert res2 == {
        "updated": 0,
        "skipped_unchanged": 2,
        "deleted": 0,
        "embedded": 0,
    }

    # Modify f1 and delete f2
    f1.write_text("# Architecture\nUpdated v2 design.", encoding="utf-8")
    f2.unlink()

    res3 = ingest.ingest_directory_incremental(conn, docs_dir)
    assert res3 == {
        "updated": 1,
        "skipped_unchanged": 0,
        "deleted": 1,
        "embedded": 0,
    }

    # Verify FTS search reflects updated content and removed deleted content
    hits_v2 = store.query_by_name(conn, "arch")
    assert len(hits_v2) == 1
    assert store.query_by_name(conn, "notes") == []


def test_ingest_single_file_and_missing_path(db_conn, tmp_path: Path):
    conn, _ = db_conn
    single = tmp_path / "single_doc.rst"
    single.write_text("RST specification content.", encoding="utf-8")

    res = ingest.ingest_directory_incremental(conn, single)
    assert res["updated"] == 1
    assert res["skipped_unchanged"] == 0

    # Re-ingesting unchanged single file skips it
    res_again = ingest.ingest_directory_incremental(conn, single)
    assert res_again["updated"] == 0
    assert res_again["skipped_unchanged"] == 1

    with pytest.raises(FileNotFoundError):
        ingest.ingest_directory_incremental(conn, tmp_path / "nonexistent")


def test_ingest_with_selective_embeddings(db_conn, tmp_path: Path, monkeypatch):
    conn, _ = db_conn
    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", _FakeEmbedder)
    monkeypatch.setattr(embeddings, "validate_vectors", lambda vectors, count: None)

    docs_dir = tmp_path / "emb_docs"
    docs_dir.mkdir()
    a = docs_dir / "a.md"
    b = docs_dir / "b.md"
    a.write_text("First document content", encoding="utf-8")
    b.write_text("Second document content", encoding="utf-8")

    res1 = ingest.ingest_directory_incremental(conn, docs_dir, use_embeddings=True)
    assert res1["updated"] == 2
    assert res1["embedded"] == 2

    # Modify only 'a'; only 1 node should be re-embedded
    a.write_text("First document content modified!", encoding="utf-8")
    res2 = ingest.ingest_directory_incremental(conn, docs_dir, use_embeddings=True)
    assert res2["updated"] == 1
    assert res2["skipped_unchanged"] == 1
    assert res2["embedded"] == 1


# ---------------------------------------------------------------------------
# 2. URL Ingestion Tests
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    def __init__(self, html: str) -> None:
        self._buf = io.BytesIO(html.encode("utf-8"))

    def read(self) -> bytes:
        return self._buf.read()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_ingest_url_extracts_html_and_skips_unchanged(db_conn, monkeypatch):
    conn, _ = db_conn
    html_page = """
    <html>
      <head>
        <title>RFC 9110 HTTP Semantics</title>
        <style>body { color: red; }</style>
        <script>console.log("ignore me");</script>
      </head>
      <body>
        <h1>HTTP Semantics</h1>
        <p>Defines core HTTP methods and status codes.</p>
      </body>
    </html>
    """

    monkeypatch.setattr(
        ingest.urllib.request,
        "urlopen",
        lambda req, timeout=20.0: _FakeHTTPResponse(html_page),
    )

    url = "https://example.org/rfc9110"
    res1 = ingest.ingest_url(conn, url)
    assert res1["status"] == "ingested"
    assert res1["title"] == "RFC 9110 HTTP Semantics"

    # Re-ingesting identical content returns unchanged
    res2 = ingest.ingest_url(conn, url)
    assert res2["status"] == "unchanged"
    assert res2["node_id"] == res1["node_id"]

    # Ensure script/style text was stripped
    row = conn.execute("SELECT data FROM nodes WHERE id = ?", (res1["node_id"],)).fetchone()
    payload = json.loads(row[0])
    assert "console.log" not in payload["content"]
    assert "color: red" not in payload["content"]
    assert "Defines core HTTP methods" in payload["content"]


def test_ingest_url_empty_text_raises(db_conn, monkeypatch):
    conn, _ = db_conn
    empty_html = "<html><head><script>var x=1;</script></head><body>   </body></html>"
    monkeypatch.setattr(
        ingest.urllib.request,
        "urlopen",
        lambda req, timeout=20.0: _FakeHTTPResponse(empty_html),
    )

    with pytest.raises(ValueError, match="No extractable text"):
        ingest.ingest_url(conn, "https://example.org/empty")


# ---------------------------------------------------------------------------
# 3. Natural Language Note (`remember`) & Records Management Tests
# ---------------------------------------------------------------------------


def test_ingest_note_and_records_list_delete(db_conn):
    conn, _ = db_conn

    with pytest.raises(ValueError, match="cannot be empty"):
        ingest.ingest_note(conn, "   \n  ")

    saved = ingest.ingest_note(
        conn,
        "Production PostgreSQL cluster uses PgBouncer on port 6432.",
        title="DB Pooling Config",
        tags=["infra", "postgres"],
    )
    assert saved["status"] == "saved"
    assert saved["title"] == "DB Pooling Config"

    records = ingest.list_records(conn, kind="topic")
    assert len(records) == 1
    assert records[0]["id"] == saved["node_id"]
    assert records[0]["ingestion_type"] == "natural_language_note"

    # Delete existing record
    assert ingest.delete_record(conn, saved["node_id"]) is True
    assert ingest.list_records(conn) == []

    # Deleting non-existent record returns False
    assert ingest.delete_record(conn, "topic:does-not-exist") is False


# ---------------------------------------------------------------------------
# 4. CLI Subcommands End-to-End Tests (`ingest`, `ingest-url`, `remember`, `records`)
# ---------------------------------------------------------------------------


def test_cli_ingestion_commands(tmp_path: Path, monkeypatch, capsys):
    db_file = tmp_path / "cli-graph.db"
    src_dir = tmp_path / "docs"
    src_dir.mkdir()
    (src_dir / "guide.md").write_text("# Setup Guide\nRun pytest.", encoding="utf-8")

    # 1. emp-kg ingest
    monkeypatch.setattr(
        sys, "argv", ["emp-kg", "ingest", str(src_dir), "--db", str(db_file), "--json"]
    )
    cli.main()
    ingest_out = json.loads(capsys.readouterr().out)
    assert ingest_out["updated"] == 1

    # 2. emp-kg remember
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "emp-kg",
            "remember",
            "Deployments freeze every Friday at 16:00 UTC.",
            "--title",
            "Deploy Freeze",
            "--tag",
            "ops",
            "--db",
            str(db_file),
            "--json",
        ],
    )
    cli.main()
    remember_out = json.loads(capsys.readouterr().out)
    assert remember_out["status"] == "saved"
    note_id = remember_out["node_id"]

    # 3. emp-kg ingest-url
    monkeypatch.setattr(
        ingest.urllib.request,
        "urlopen",
        lambda req, timeout=20.0: _FakeHTTPResponse(
            "<html><title>API Docs</title><body>REST v2</body></html>"
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["emp-kg", "ingest-url", "https://example.org/api", "--db", str(db_file), "--json"],
    )
    cli.main()
    url_out = json.loads(capsys.readouterr().out)
    assert url_out["status"] == "ingested"

    # 4. emp-kg records list
    monkeypatch.setattr(
        sys, "argv", ["emp-kg", "records", "list", "--db", str(db_file), "--json"]
    )
    cli.main()
    listed = json.loads(capsys.readouterr().out)
    assert len(listed) == 3

    # 5. emp-kg records delete
    monkeypatch.setattr(
        sys,
        "argv",
        ["emp-kg", "records", "delete", "--id", note_id, "--db", str(db_file), "--json"],
    )
    cli.main()
    del_out = json.loads(capsys.readouterr().out)
    assert del_out == {"deleted": True, "node_id": note_id}


# ---------------------------------------------------------------------------
# 5. Retrieval & Cascade Cleanup on Ingested Notes, URLs, and Files
# ---------------------------------------------------------------------------


def test_retrieval_surfaces_ingested_notes_urls_and_files(
    db_conn, tmp_path: Path, monkeypatch
):
    conn, db_file = db_conn

    # 1. Ingest a markdown file
    docs_dir = tmp_path / "runbooks"
    docs_dir.mkdir()
    (docs_dir / "bgp_policy.md").write_text(
        "# BGP Failover Policy\nAutomatic route failover triggers within 3 seconds.",
        encoding="utf-8",
    )
    ingest.ingest_directory_incremental(conn, docs_dir, use_embeddings=False)

    # 2. Ingest a natural-language memory note
    note_res = ingest.ingest_note(
        conn,
        "OAuth2 refresh tokens expire after 14 days and require rotation.",
        title="Auth Token Lifecycle",
        tags=["security", "auth"],
        use_embeddings=False,
    )

    # 3. Ingest a web URL
    monkeypatch.setattr(
        ingest.urllib.request,
        "urlopen",
        lambda req, timeout=20.0: _FakeHTTPResponse(
            "<html><title>Postgres HA</title><body>PgBouncer connection pooling and replica lag.</body></html>"
        ),
    )
    ingest.ingest_url(
        conn,
        "https://example.org/postgres-ha",
        use_embeddings=False,
    )

    # Query via context builder (BM25 + synonym expansion)
    packet = context.build_context(
        context.ContextRequest(
            task_description="Investigate BGP failover routing behavior",
            semantic_mode="off",
            synonym_mode="auto",
        ),
        db_file,
    )
    assert len(packet.facts) >= 1
    assert any(f.name == "bgp_policy" for f in packet.facts)

    # Query targeting the remembered note
    auth_packet = context.build_context(
        context.ContextRequest(
            task_description="OAuth2 refresh tokens rotation",
            semantic_mode="off",
        ),
        db_file,
    )
    assert any(f.node_id == note_res["node_id"] for f in auth_packet.facts)

    # Delete the note and verify retrieval no longer returns it
    assert ingest.delete_record(conn, note_res["node_id"]) is True
    after_delete_packet = context.build_context(
        context.ContextRequest(
            task_description="OAuth2 refresh tokens rotation",
            semantic_mode="off",
        ),
        db_file,
    )
    assert all(f.node_id != note_res["node_id"] for f in after_delete_packet.facts)


# ---------------------------------------------------------------------------
# 6. Additional Edge Cases (Invalid URL, Duplicate Note, Cascade Embedding Delete)
# ---------------------------------------------------------------------------


def test_edge_cases_invalid_url_duplicate_note_and_cascade_delete(
    db_conn, monkeypatch
):
    conn, _ = db_conn

    # Invalid/unreachable URL raises URLError
    def _failing_urlopen(req, timeout=20.0):
        raise ingest.urllib.request.URLError("connection refused")

    monkeypatch.setattr(ingest.urllib.request, "urlopen", _failing_urlopen)
    with pytest.raises(ingest.urllib.request.URLError):
        ingest.ingest_url(conn, "https://unreachable.invalid/docs")

    # Duplicate note content shares deterministic memory URI and upserts cleanly
    first = ingest.ingest_note(conn, "Zero-downtime schema migrations use expand-contract.")
    second = ingest.ingest_note(conn, "Zero-downtime schema migrations use expand-contract.")
    assert first["node_id"] == second["node_id"]
    assert len(ingest.list_records(conn, kind="topic")) == 1

    # Cascade delete removes related embedding rows cleanly
    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", _FakeEmbedder)
    monkeypatch.setattr(embeddings, "validate_vectors", lambda vectors, count: None)
    note_with_emb = ingest.ingest_note(
        conn,
        "Redis sentinel failover quorum requires 3 nodes.",
        title="Redis Sentinel",
        use_embeddings=True,
    )
    emb_rows_before = conn.execute(
        "SELECT COUNT(*) FROM node_embeddings WHERE node_id = ?",
        (note_with_emb["node_id"],),
    ).fetchone()[0]
    assert emb_rows_before == 1

    assert ingest.delete_record(conn, note_with_emb["node_id"]) is True
    emb_rows_after = conn.execute(
        "SELECT COUNT(*) FROM node_embeddings WHERE node_id = ?",
        (note_with_emb["node_id"],),
    ).fetchone()[0]
    assert emb_rows_after == 0