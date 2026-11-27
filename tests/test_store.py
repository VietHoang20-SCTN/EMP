"""Tests for emp_kg.store."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from emp_kg import store


def graph_fixture(conn: sqlite3.Connection) -> None:
    store.save_structural_nodes(conn, [
        {"id": "function:a", "kind": "function", "name": "alpha"},
        {"id": "function:b", "kind": "function", "name": "beta"},
        {"id": "function:c", "kind": "function", "name": "gamma"},
        {"id": "function:d", "kind": "function", "name": "delta"},
    ])
    store.save_edges(conn, [
        {"source": "function:a", "target": "function:b", "kind": "CALLS"},
        {"source": "function:b", "target": "function:c", "kind": "CALLS"},
        {"source": "function:c", "target": "function:a", "kind": "CALLS"},
        {"source": "function:a", "target": "function:d", "kind": "REFERENCES"},
    ])
    store.rebuild_fts(conn)
    conn.commit()


class TestStoreIntegrity(unittest.TestCase):
    def test_init_db_enables_integrity_and_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = store.init_db(Path(tmpdir) / "graph.db")
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            indexes = {
                row[1] for row in conn.execute("PRAGMA index_list(edges)")
            }
            self.assertIn("idx_edges_source", indexes)
            self.assertIn("idx_edges_target", indexes)
            self.assertIn(
                "idx_nodes_kind",
                {row[1] for row in conn.execute("PRAGMA index_list(nodes)")},
            )
            conn.close()

    def test_init_db_migrates_legacy_fts_to_search_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "legacy.db"
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE nodes (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    name TEXT,
                    data TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE nodes_fts USING fts5(
                    id UNINDEXED, name, content='nodes', content_rowid='rowid'
                )
            """)
            payload = {
                "id": "0001",
                "title": "Generic decision",
                "context": "Authorization boundaries require tenant isolation.",
            }
            conn.execute(
                "INSERT INTO nodes (id, kind, name, data) VALUES (?, ?, ?, ?)",
                ("decision_ref:0001", "decision_ref", "Generic decision", json.dumps(payload)),
            )
            conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES ('rebuild')")
            conn.commit()
            conn.close()

            conn = store.init_db(db_path)
            self.assertEqual(
                [row[1] for row in conn.execute("PRAGMA table_info(nodes_fts)")],
                ["id", "name", "fts_content"],
            )
            self.assertIn("Authorization", conn.execute(
                "SELECT fts_content FROM nodes WHERE id = 'decision_ref:0001'"
            ).fetchone()[0])
            self.assertEqual(
                [node["id"] for node in store.query_by_name(conn, '"Authorization"')],
                ["decision_ref:0001"],
            )
            conn.close()

    def test_content_search_indexes_semantic_and_structural_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = store.init_db(Path(tmpdir) / "graph.db")
            store.save_structural_nodes(conn, [{
                "id": "function:opaque",
                "kind": "function",
                "name": "opaque",
                "signature": "(tenant_id: str)",
                "qualifiedName": "auth.authorize_tenant",
                "filePath": "src/security/authorize.py",
            }])
            store.save_topic_nodes(conn, [{
                "title": "Generic topic",
                "content": "Enforces workspace isolation and least privilege.",
                "source_path": "docs/security.md",
            }])
            store.rebuild_fts(conn)
            self.assertEqual(
                [node["id"] for node in store.query_by_name(conn, '"least"')][0].split(":")[0],
                "topic",
            )
            self.assertEqual(
                [node["id"] for node in store.query_by_name(conn, '"tenant_id"')],
                ["function:opaque"],
            )
            conn.close()

    def test_content_search_indexes_story_criteria_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = store.init_db(Path(tmpdir) / "graph.db")
            store.save_story_nodes(conn, [{
                "id": "US-042",
                "title": "Rotate credentials",
                "status": "planned",
                "lane": "normal",
                "acceptance_criteria": [
                    "Refresh credentials without interrupting active sessions.",
                ],
                "design_notes": ["Use token rotation endpoint."],
                "source_path": "docs/stories/US-042.md",
            }])
            store.rebuild_fts(conn)
            self.assertEqual(
                [node["id"] for node in store.query_by_name(conn, '"interrupting"')],
                ["story:US-042"],
            )
            self.assertEqual(
                [node["id"] for node in store.query_by_name(conn, '"rotation"')],
                ["story:US-042"],
            )
            conn.close()

    def test_embedding_persistence_and_stale_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = store.init_db(Path(tmpdir) / "graph.db")
            store.save_structural_nodes(conn, [{
                "id": "function:semantic", "kind": "function", "name": "semantic",
            }])
            self.assertEqual(store.save_node_embeddings(conn, [{
                "node_id": "function:semantic",
                "model_id": "fake-model",
                "vector": [1.0, 0.0],
                "content_hash": "hash-a",
            }]), 1)
            loaded = store.load_node_embeddings(conn, "fake-model")
            self.assertEqual(loaded[0]["dimension"], 2)
            self.assertEqual(loaded[0]["vector"], [1.0, 0.0])
            self.assertEqual(
                store.delete_stale_embeddings(
                    conn, "fake-model", {"function:semantic": "hash-b"}
                ),
                1,
            )
            self.assertEqual(store.load_node_embeddings(conn, "fake-model"), [])
            conn.close()

    def test_embedding_replacement_preserves_prior_index_on_invalid_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = store.init_db(Path(tmpdir) / "graph.db")
            store.save_structural_nodes(conn, [
                {"id": "function:a", "kind": "function", "name": "a"},
                {"id": "function:b", "kind": "function", "name": "b"},
            ])
            store.replace_node_embeddings(
                conn,
                "fake-model",
                [{
                    "node_id": "function:a", "model_id": "fake-model",
                    "vector": [1.0, 0.0], "content_hash": "hash-a",
                }],
                corpus_digest="digest-a",
            )
            with self.assertRaisesRegex(store.GraphValidationError, "share one dimension"):
                store.replace_node_embeddings(
                    conn,
                    "fake-model",
                    [
                        {"node_id": "function:a", "model_id": "fake-model", "vector": [1.0], "content_hash": "hash-a"},
                        {"node_id": "function:b", "model_id": "fake-model", "vector": [1.0, 0.0], "content_hash": "hash-b"},
                    ],
                    corpus_digest="digest-b",
                )
            self.assertEqual(
                [item["node_id"] for item in store.load_node_embeddings(conn, "fake-model")],
                ["function:a"],
            )
            self.assertEqual(store.get_embedding_metadata(conn, "fake-model")["corpus_digest"], "digest-a")
            conn.close()

    def test_stale_embedding_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = store.init_db(Path(tmpdir) / "graph.db")
            store.save_structural_nodes(conn, [{
                "id": "function:semantic", "kind": "function", "name": "semantic",
            }])
            store.rebuild_fts(conn)
            docs = store.embedding_documents(conn)
            digest = hashlib.sha256(
                "\n".join(f"{item['node_id']}:{item['content_hash']}" for item in docs).encode("utf-8")
            ).hexdigest()
            store.replace_node_embeddings(
                conn,
                "fake-model",
                [{
                    "node_id": "function:semantic", "model_id": "fake-model",
                    "vector": [1.0], "content_hash": docs[0]["content_hash"],
                }],
                corpus_digest=digest,
            )
            self.assertEqual(len(store.load_fresh_node_embeddings(conn, "fake-model")), 1)
            conn.execute("UPDATE nodes SET fts_content = 'changed' WHERE id = 'function:semantic'")
            store.rebuild_fts(conn)
            self.assertEqual(store.load_fresh_node_embeddings(conn, "fake-model"), [])
            conn.close()

    def test_embedding_rejects_unknown_node(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = store.init_db(Path(tmpdir) / "graph.db")
            with self.assertRaisesRegex(store.GraphValidationError, "does not exist"):
                store.save_node_embeddings(conn, [{
                    "node_id": "function:missing",
                    "model_id": "fake-model",
                    "vector": [1.0],
                    "content_hash": "hash",
                }])
            conn.close()

    def test_init_db_migrates_legacy_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE nodes (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    name TEXT,
                    data TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE edges (
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    PRIMARY KEY (source, target, kind)
                )
            """)
            conn.executemany(
                "INSERT INTO nodes (id, kind, name, data) VALUES (?, ?, ?, ?)",
                [
                    ("function:source", "function", "source", "{}"),
                    ("function:target", "function", "target", "{}"),
                ],
            )
            conn.executemany(
                "INSERT INTO edges (source, target, kind) VALUES (?, ?, ?)",
                [
                    ("function:source", "function:target", "CALLS"),
                    ("function:missing", "function:target", "CALLS"),
                ],
            )
            conn.commit()
            conn.close()

            conn = store.init_db(db_path)
            self.assertTrue(conn.execute("PRAGMA foreign_key_list(edges)").fetchall())
            self.assertEqual(
                conn.execute("SELECT source, target, kind FROM edges").fetchall(),
                [("function:source", "function:target", "CALLS")],
            )
            conn.close()

    def test_edge_evidence_supports_multiple_records_and_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = store.init_db(Path(tmpdir) / "graph.db")
            store.save_structural_nodes(conn, [
                {"id": "function:target", "kind": "function", "name": "target"},
            ])
            store.save_decision_nodes(conn, [{"id": "0001", "title": "Decision"}])
            edge = {"source": "decision_ref:0001", "target": "function:target", "kind": "AFFECTS"}
            store.save_edges(conn, [edge])
            evidence = [
                {**edge, "confidence": 1.0, "reference": "target", "source_path": "README.md", "source_lines": {"start": 1, "end": 2}, "resolution": "unique_global_name"},
                {**edge, "confidence": 1.0, "reference": "function:target", "source_path": "docs/A.md", "source_lines": {"start": 3, "end": 4}, "resolution": "exact_node_id"},
            ]
            self.assertEqual(store.save_affects_evidence(conn, evidence), 2)
            self.assertEqual(store.get_edge_evidence(conn, **{key: edge[key] for key in ("source", "target", "kind")}), [
                {"kind": "AFFECTS", "confidence": 1.0, "reference": "target", "source_path": "README.md", "source_lines": {"start": 1, "end": 2}, "resolution": "unique_global_name"},
                {"kind": "AFFECTS", "confidence": 1.0, "reference": "function:target", "source_path": "docs/A.md", "source_lines": {"start": 3, "end": 4}, "resolution": "exact_node_id"},
            ])
            conn.execute("DELETE FROM edges WHERE source = ?", (edge["source"],))
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM edge_evidence").fetchone()[0],
                0,
            )
            conn.close()

    def test_edges_require_existing_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = store.init_db(Path(tmpdir) / "graph.db")
            try:
                with self.assertRaisesRegex(store.GraphValidationError, "endpoint does not exist"):
                    store.save_edges(conn, [{
                        "source": "function:missing",
                        "target": "function:also-missing",
                        "kind": "CALLS",
                    }])
            finally:
                conn.close()

    def test_store_rejects_invalid_records_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = store.init_db(Path(tmpdir) / "graph.db")
            try:
                with self.assertRaisesRegex(store.GraphValidationError, "unknown node kind"):
                    store.save_structural_nodes(conn, [{
                        "id": "mystery:1", "kind": "mystery", "name": "bad",
                    }])
                with self.assertRaisesRegex(store.GraphValidationError, "conflicting duplicate node id"):
                    store.save_structural_nodes(conn, [
                        {"id": "function:a", "kind": "function", "name": "a"},
                        {"id": "function:a", "kind": "function", "name": "different"},
                    ])
                store.save_structural_nodes(conn, [
                    {"id": "function:a", "kind": "function", "name": "a"},
                    {"id": "function:b", "kind": "function", "name": "b"},
                ])
                with self.assertRaisesRegex(store.GraphValidationError, "invalid IMPORTS endpoints"):
                    store.save_edges(conn, [{
                        "source": "function:a", "target": "function:b", "kind": "IMPORTS",
                    }])
                store.save_edges(conn, [{
                    "source": "function:a", "target": "function:a", "kind": "CALLS",
                }])
                self.assertEqual(
                    conn.execute(
                        "SELECT source, target, kind FROM edges "
                        "WHERE source = 'function:a' AND target = 'function:a'"
                    ).fetchall(),
                    [("function:a", "function:a", "CALLS")],
                )
            finally:
                conn.close()

    def test_mixed_valid_and_invalid_batch_leaves_no_partial_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = store.init_db(Path(tmpdir) / "graph.db")
            try:
                store.save_structural_nodes(conn, [
                    {"id": "function:a", "kind": "function", "name": "a"},
                    {"id": "function:b", "kind": "function", "name": "b"},
                ])
                valid = {"source": "function:a", "target": "function:b", "kind": "CALLS"}
                invalid = {"source": "function:a", "target": "function:b", "kind": "BOGUS"}
                with self.assertRaisesRegex(store.GraphValidationError, "unknown edge kind"):
                    store.save_edges(conn, [valid, invalid])
                self.assertEqual(
                    conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
                    0,
                    "invalid batch must not leave partial edge writes",
                )
            finally:
                conn.close()

    def test_evidence_rejects_malformed_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = store.init_db(Path(tmpdir) / "graph.db")
            try:
                store.save_structural_nodes(conn, [
                    {"id": "function:target", "kind": "function", "name": "target"},
                ])
                store.save_decision_nodes(conn, [{"id": "0001", "title": "Decision"}])
                edge = {"source": "decision_ref:0001", "target": "function:target", "kind": "AFFECTS"}
                store.save_edges(conn, [edge])
                evidence = {
                    **edge,
                    "reference": "target",
                    "source_path": "README.md",
                    "source_lines": {"start": 1, "end": 3},
                    "resolution": "unique_global_name",
                    "confidence": 1.5,
                }
                with self.assertRaisesRegex(store.GraphValidationError, "confidence"):
                    store.save_affects_evidence(conn, [evidence])
                with self.assertRaisesRegex(store.GraphValidationError, "end must be"):
                    store.save_affects_evidence(conn, [{
                        **evidence,
                        "confidence": 1.0,
                        "source_lines": {"start": 3, "end": 1},
                    }])
            finally:
                conn.close()

    def test_validate_graph_detects_corruption_and_fts_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = store.init_db(Path(tmpdir) / "graph.db")
            try:
                graph_fixture(conn)
                self.assertEqual(store.validate_graph(conn)["nodes"], 4)
                conn.execute(
                    "INSERT INTO nodes (id, kind, name, data) VALUES (?, ?, ?, ?)",
                    ("function:rogue", "function", "rogue", '{"id":"function:rogue","kind":"function","name":"rogue"}'),
                )
                with self.assertRaisesRegex(store.GraphValidationError, "FTS (row count|index differs)"):
                    store.validate_graph(conn)
                store.rebuild_fts(conn)
                conn.execute(
                    "UPDATE nodes SET data = ? WHERE id = ?",
                    ('{"id":"wrong","kind":"function"}', "function:a"),
                )
                with self.assertRaisesRegex(store.GraphValidationError, "identity disagrees"):
                    store.validate_graph(conn)
            finally:
                conn.close()

    def test_save_topic_nodes_persists_and_replaces_deterministic_nodes(self) -> None:
        from emp_kg.merger import _topic_id

        topic = {
            "title": "Build flow",
            "content": "References `process_payment`.",
            "source_path": "docs/ARCHITECTURE.md",
            "source_lines": {"start": 1, "end": 3},
        }
        replacement = {**topic, "content": "Updated content."}
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = store.init_db(Path(tmpdir) / "graph.db")
            self.assertEqual(store.save_topic_nodes(conn, [topic]), 1)
            self.assertEqual(store.save_topic_nodes(conn, [replacement]), 1)
            row = conn.execute(
                "SELECT id, kind, name, data FROM nodes"
            ).fetchone()
            self.assertEqual(row[:3], (_topic_id(topic), "topic", "Build flow"))
            self.assertEqual(json.loads(row[3]), replacement)
            conn.close()

    def test_reconcile_removes_stale_topics_independently(self) -> None:
        from emp_kg.merger import _topic_id

        current = {
            "title": "Current",
            "content": "Current.",
            "source_path": "README.md",
        }
        stale = {
            "title": "Stale",
            "content": "Stale.",
            "source_path": "README.md",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            conn = store.init_db(Path(tmpdir) / "graph.db")
            store.save_topic_nodes(conn, [current, stale])
            conn.commit()
            removed = store.reconcile(
                conn,
                current_structural_ids=set(),
                current_decision_ids=set(),
                current_topic_ids={_topic_id(current)},
                current_edges=set(),
            )
            self.assertEqual(removed["stale_topic_nodes_removed"], 1)
            self.assertEqual(
                conn.execute("SELECT name FROM nodes WHERE kind = 'topic'").fetchall(),
                [("Current",)],
            )
            conn.close()

    def test_metadata_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            conn = store.init_db(db_path)
            project = Path(tmpdir) / "project"
            project.mkdir()
            store.save_metadata(conn, "1.6.0", project, "abc123")
            conn.commit()
            row = conn.execute(
                "SELECT schema_version, project_path, git_sha, built_at FROM graph_metadata"
            ).fetchone()
            self.assertEqual(row[:3], ("1.6.0", str(project.resolve()), "abc123"))
            self.assertTrue(row[3])
            conn.close()


class TestGraphQueries(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "graph.db"
        self.conn = store.init_db(self.db_path)
        graph_fixture(self.conn)

    def tearDown(self) -> None:
        self.conn.close()
        self.tmpdir.cleanup()

    def test_graph_stats_groups_node_and_edge_kinds(self) -> None:
        self.assertEqual(
            store.graph_stats(self.conn),
            {
                "nodes": 4,
                "edges": 4,
                "nodes_by_kind": {"function": 4},
                "edges_by_kind": {"CALLS": 3, "REFERENCES": 1},
            },
        )

    def test_neighbors_respects_direction(self) -> None:
        outgoing = store.neighbors(self.conn, "function:a", "out")
        incoming = store.neighbors(self.conn, "function:a", "in")
        both = store.neighbors(self.conn, "function:a", "both")
        self.assertEqual({record["node"]["id"] for record in outgoing}, {"function:b", "function:d"})
        self.assertEqual({record["node"]["id"] for record in incoming}, {"function:c"})
        self.assertEqual(len(both), 3)

    def test_shortest_path_is_directed_cycle_safe_and_bounded(self) -> None:
        path = store.shortest_path(self.conn, "function:a", "function:c")
        self.assertEqual(
            [node["id"] for node in path["nodes"]],
            ["function:a", "function:b", "function:c"],
        )
        self.assertEqual([edge["kind"] for edge in path["edges"]], ["CALLS", "CALLS"])
        self.assertEqual(store.shortest_path(self.conn, "function:a", "function:c", 1), {"nodes": [], "edges": []})
        self.assertEqual(store.shortest_path(self.conn, "function:d", "function:a"), {"nodes": [], "edges": []})
        self.assertEqual(store.shortest_path(self.conn, "function:missing", "function:a"), {"nodes": [], "edges": []})

    def test_shortest_path_direction_in_follows_incoming_edges(self) -> None:
        # a->b is an incoming edge to b; "in" traverses it backwards.
        path = store.shortest_path(
            self.conn, "function:b", "function:a", direction="in"
        )
        self.assertEqual(
            [node["id"] for node in path["nodes"]],
            ["function:b", "function:a"],
        )
        self.assertEqual([edge["kind"] for edge in path["edges"]], ["CALLS"])

    def test_shortest_path_direction_in_is_directed(self) -> None:
        # d has no incoming edges; a is unreachable backwards from d.
        self.assertEqual(
            store.shortest_path(self.conn, "function:a", "function:d", direction="in"),
            {"nodes": [], "edges": []},
        )

    def test_shortest_path_direction_both_uses_incoming_edge(self) -> None:
        # a->b is incoming to b; "both" must traverse it backwards.
        # (The old bug treated it as outgoing: b's only outgoing neighbor
        # is c, yielding the wrong detour b->c->a.)
        path = store.shortest_path(
            self.conn, "function:b", "function:a", direction="both"
        )
        self.assertEqual(
            [node["id"] for node in path["nodes"]],
            ["function:b", "function:a"],
        )
        self.assertEqual([edge["kind"] for edge in path["edges"]], ["CALLS"])

    def test_shortest_path_direction_both_reaches_leaf_via_incoming_edge(self) -> None:
        # d has no outgoing edges; only a->d (incoming to d) reaches it.
        # (The old bug dropped the edge entirely and reported no path.)
        path = store.shortest_path(
            self.conn, "function:d", "function:a", direction="both"
        )
        self.assertEqual(
            [node["id"] for node in path["nodes"]],
            ["function:d", "function:a"],
        )
        self.assertEqual([edge["kind"] for edge in path["edges"]], ["REFERENCES"])

    def test_shortest_path_direction_both_is_cycle_safe(self) -> None:
        # "both" follows c->a backwards at layer 0, reaching c in one hop.
        path = store.shortest_path(
            self.conn, "function:a", "function:c", direction="both"
        )
        self.assertEqual(
            [node["id"] for node in path["nodes"]],
            ["function:a", "function:c"],
        )

    def test_open_readonly_disallows_mutation(self) -> None:
        self.conn.close()
        self.conn = store.open_readonly(self.db_path)
        with self.assertRaises(sqlite3.OperationalError):
            self.conn.execute("DELETE FROM nodes")

    def test_export_html_is_offline_with_search_and_kind_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "graph.html"
            store.export_html(self.conn, out)
            html = out.read_text(encoding="utf-8")

            # No external loads -- the vis library is bundled inline.
            self.assertNotIn('src="http', html)
            self.assertNotIn('href="http', html)
            self.assertIn("vis-network", html)
            # Search box and kind filter are present.
            self.assertIn('id="search"', html)
            self.assertIn('id="kind-filter"', html)
            self.assertIn("focusNeighbors", html)
            self.assertIn("renderInspector", html)
            self.assertIn("doubleClick", html)
            self.assertIn("Show whole graph", html)
            self.assertIn("Outgoing", html)
            self.assertIn("Incoming", html)
            self.assertIn("gravitationalConstant", html)
            # Graph data is embedded.
            self.assertIn("function:a", html)
            self.assertIn("CALLS", html)


if __name__ == "__main__":
    unittest.main()
