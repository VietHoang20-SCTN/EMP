"""Tests for emp_kg.cli."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from emp_kg import cli, schema, store, structural
from tests.test_store import graph_fixture


class TestBuildReconciliation(unittest.TestCase):
    def setUp(self) -> None:
        self.prepare_codegraph = patch.object(
            cli.structural,
            "prepare_codegraph",
            return_value={"index_last_indexed": "2100-01-01T00:00:00Z"},
        )
        self.prepare_codegraph.start()
        self.addCleanup(self.prepare_codegraph.stop)
        self.export_raw = patch.object(
            cli.structural,
            "export_raw",
            return_value={"nodes": [], "edges": []},
        )
        self.export_raw.start()
        self.addCleanup(self.export_raw.stop)

    def test_build_enforces_codegraph_sync_before_extraction(self) -> None:
        current_node = {
            "id": "function:current",
            "kind": "function",
            "name": "current",
        }
        calls: list[str] = []

        def prepare(_project: Path) -> dict[str, str]:
            calls.append("prepare")
            return {"index_last_indexed": "2100-01-01T00:00:00Z"}

# Salar Change

        def extract(_project: Path, *, index_last_indexed: str, raw: dict | None = None, include_provenance: bool = False, **_kwargs: Any):
            calls.append(f"extract:{index_last_indexed}")
            return [current_node], [], []

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            with patch.object(
                cli.structural,
                "prepare_codegraph",
                side_effect=prepare,
            ), patch.object(
                cli.structural,
                "extract_nodes_and_edges",
                side_effect=extract,
            ):
                cli.build(tmpdir, str(db_path))

        self.assertEqual(
            calls,
            ["prepare", "extract:2100-01-01T00:00:00Z"],
        )

    def test_build_aborts_before_extraction_when_codegraph_sync_fails(self) -> None:
        current_node = {
            "id": "function:current",
            "kind": "function",
            "name": "current",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            conn = store.init_db(db_path)
            store.save_structural_nodes(conn, [current_node])
            conn.commit()
            conn.close()

            with patch.object(
                cli.structural,
                "prepare_codegraph",
                side_effect=structural.CodeGraphFreshnessError("index stale"),
            ), patch.object(
                cli.structural,
                "extract_nodes_and_edges",
            ) as extract:
                with self.assertRaisesRegex(
                    structural.CodeGraphFreshnessError,
                    "index stale",
                ):
                    cli.build(tmpdir, str(db_path))

            extract.assert_not_called()
            conn = store.init_db(db_path)
            self.assertEqual(
                conn.execute("SELECT id FROM nodes").fetchall(),
                [(current_node["id"],)],
            )
            conn.close()

    def test_build_local_embeddings_uses_optional_embedder(self) -> None:
        current_node = {
            "id": "function:semantic",
            "kind": "function",
            "name": "semantic",
            "description": "meaning without shared lexical terms",
        }

        class FakeEmbedder:
            def __init__(self, model_id: str) -> None:
                self.model_id = model_id

            def encode(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, float(index)] for index, _ in enumerate(texts)]

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            with patch.object(
                cli.structural,
                "extract_nodes_and_edges",
                return_value=([current_node], [], []),
            ), patch.object(cli, "SentenceTransformerEmbedder", FakeEmbedder):
                cli.build(
                    tmpdir,
                    str(db_path),
                    embeddings="local",
                    embedding_model="fake-model",
                )

            conn = store.init_db(db_path)
            embeddings = store.load_node_embeddings(conn, "fake-model")
            conn.close()
            self.assertEqual(len(embeddings), 1)
            self.assertEqual(embeddings[0]["node_id"], "function:semantic")
            self.assertEqual(embeddings[0]["dimension"], 2)

    def test_build_persists_codegraph_provenance_metadata(self) -> None:
        current_node = {
            "id": "function:current",
            "kind": "function",
            "name": "current",
        }
        provenance = {
            "codegraph_version": "1.2.3",
            "index_last_indexed": "2026-08-18T12:00:00Z",
            "index_state": "complete",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            with patch.object(
                cli.structural,
                "prepare_codegraph",
                return_value=provenance,
            ), patch.object(
                cli.structural,
                "extract_nodes_and_edges",
                return_value=([current_node], [], []),
            ):
                cli.build(tmpdir, str(db_path))

            conn = store.init_db(db_path)
            provenance = json.loads(
                conn.execute(
                    "SELECT codegraph_provenance FROM graph_metadata"
                ).fetchone()[0]
            )
            conn.close()
            self.assertEqual(provenance["codegraph_version"], "1.2.3")
            self.assertEqual(provenance["index_last_indexed"], "2026-08-18T12:00:00Z")
            self.assertEqual(provenance["index_state"], "complete")

    def test_build_persists_raw_export_hash_provenance(self) -> None:
        current_node = {
            "id": "function:current",
            "kind": "function",
            "name": "current",
        }
        raw_export = {
            "nodes": [current_node],
            "edges": [],
        }
        canonical = json.dumps(
            raw_export,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected_hash = hashlib.sha256(canonical).hexdigest()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            with patch.object(
                cli.structural,
                "prepare_codegraph",
                return_value={"index_last_indexed": "2100-01-01T00:00:00Z"},
            ), patch.object(
                cli.structural,
                "export_raw",
                return_value=raw_export,
            ):
                cli.build(tmpdir, str(db_path))

            conn = store.init_db(db_path)
            provenance = json.loads(
                conn.execute(
                    "SELECT codegraph_provenance FROM graph_metadata"
                ).fetchone()[0]
            )
            conn.close()
            self.assertEqual(provenance["raw_export_sha256"], expected_hash)
            self.assertEqual(provenance["raw_node_count"], 1)
            self.assertEqual(provenance["raw_edge_count"], 0)

    def test_build_removes_stale_node_and_orphan_edge(self) -> None:
        current_node = {
            "id": "function:current",
            "kind": "function",
            "name": "current",
        }
        stale_node = {
            "id": "function:stale",
            "kind": "function",
            "name": "stale",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            conn = store.init_db(db_path)
            store.save_structural_nodes(conn, [current_node, stale_node])
            store.save_edges(conn, [{
                "source": stale_node["id"],
                "target": current_node["id"],
                "kind": "CALLS",
            }])
            conn.commit()
            conn.close()

            output = io.StringIO()
            with patch.object(
                cli.structural,
                "extract_nodes_and_edges",
                return_value=([current_node], [], []),
            ), redirect_stdout(output):
                cli.build(tmpdir, str(db_path))

            conn = store.init_db(db_path)
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE id = ?",
                    (stale_node["id"],),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE source = ?",
                    (stale_node["id"],),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE id = ?",
                    (current_node["id"],),
                ).fetchone()[0],
                1,
            )
            conn.close()

        self.assertIn(
            "Reconciled stale records: 1 structural nodes, 0 decisions, 0 topics, 0 stories, 1 edges removed",
            output.getvalue(),
        )

    def test_build_removes_stale_edge_between_current_nodes(self) -> None:
        source_node = {
            "id": "function:source",
            "kind": "function",
            "name": "source",
        }
        target_node = {
            "id": "function:target",
            "kind": "function",
            "name": "target",
        }
        stale_edge = {
            "source": source_node["id"],
            "target": target_node["id"],
            "kind": "CALLS",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            conn = store.init_db(db_path)
            store.save_structural_nodes(conn, [source_node, target_node])
            store.save_edges(conn, [stale_edge])
            conn.commit()
            conn.close()

            output = io.StringIO()
            with patch.object(
                cli.structural,
                "extract_nodes_and_edges",
                return_value=([source_node, target_node], [], []),
            ), redirect_stdout(output):
                cli.build(tmpdir, str(db_path))

            conn = store.init_db(db_path)
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE id IN (?, ?)",
                    (source_node["id"], target_node["id"]),
                ).fetchone()[0],
                2,
            )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE source = ? AND target = ? AND kind = ?",
                    (stale_edge["source"], stale_edge["target"], stale_edge["kind"]),
                ).fetchone()[0],
                0,
            )
            conn.close()

        self.assertIn(
            "Reconciled stale records: 0 structural nodes, 0 decisions, 0 topics, 0 stories, 1 edges removed",
            output.getvalue(),
        )

    def test_build_rolls_back_when_late_step_fails(self) -> None:
        previous_node = {
            "id": "function:previous",
            "kind": "function",
            "name": "previous",
        }
        current_node = {
            "id": "function:current",
            "kind": "function",
            "name": "current",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            conn = store.init_db(db_path)
            store.save_structural_nodes(conn, [previous_node])
            conn.commit()
            conn.close()
            (Path(tmpdir) / "README.md").write_text(
                "## Rolled back topic\n\nCall `current`.\n",
                encoding="utf-8",
            )

            with patch.object(
                cli.structural,
                "extract_nodes_and_edges",
                return_value=([current_node], [], []),
            ), patch.object(
                cli.store,
                "rebuild_fts",
                side_effect=RuntimeError("forced late failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "forced late failure"):
                    cli.build(tmpdir, str(db_path))

            conn = store.init_db(db_path)
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE id = ?",
                    (previous_node["id"],),
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE id = ?",
                    (current_node["id"],),
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM nodes WHERE kind = 'topic'").fetchone()[0],
                0,
            )
            conn.close()

    def test_build_rolls_back_when_graph_validation_fails(self) -> None:
        previous_node = {
            "id": "function:previous",
            "kind": "function",
            "name": "previous",
        }
        current_node = {
            "id": "function:current",
            "kind": "function",
            "name": "current",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            conn = store.init_db(db_path)
            store.save_structural_nodes(conn, [previous_node])
            conn.commit()
            conn.close()

            with patch.object(
                cli.structural,
                "extract_nodes_and_edges",
                return_value=([current_node], [], []),
            ), patch.object(
                cli.store,
                "validate_graph",
                side_effect=store.GraphValidationError("forced invariant failure"),
            ):
                with self.assertRaisesRegex(store.GraphValidationError, "forced invariant"):
                    cli.build(tmpdir, str(db_path))

            conn = store.init_db(db_path)
            self.assertEqual(
                conn.execute("SELECT id FROM nodes").fetchall(),
                [(previous_node["id"],)],
            )
            conn.close()

    def test_build_writes_metadata(self) -> None:
        current_node = {
            "id": "function:current",
            "kind": "function",
            "name": "current",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            with patch.object(
                cli.structural,
                "extract_nodes_and_edges",
                return_value=([current_node], [], []),
            ), patch.object(cli, "_git_sha", return_value="test-sha"):
                cli.build(tmpdir, str(db_path))

            conn = store.init_db(db_path)
            try:
                row = conn.execute(
                    "SELECT schema_version, project_path, git_sha FROM graph_metadata"
                ).fetchone()
            finally:
                conn.close()
            self.assertEqual(row, (schema.load_schema()["version"], str(Path(tmpdir).resolve()), "test-sha"))

    def test_discover_docs_includes_general_markdown_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            root_doc = project / "README.md"
            architecture = project / "docs" / "ARCHITECTURE.md"
            decision = project / "docs" / "decisions" / "0001-test.md"
            story = project / "docs" / "stories" / "test.md"
            for path in [root_doc, architecture, decision, story]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("## Topic\n\nContent.\n", encoding="utf-8")

            discovered = set(cli._discover_docs(project))

        self.assertEqual(discovered, {root_doc, architecture})

    def test_build_persists_topics_and_topic_affects_edges(self) -> None:
        current_node = {
            "id": "function:current",
            "kind": "function",
            "name": "current",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            db_path = project / "graph.db"
            (project / "README.md").write_text(
                "# Project\n\n## Runtime\n\nCall `current`.\n",
                encoding="utf-8",
            )
            with patch.object(
                cli.structural,
                "extract_nodes_and_edges",
                return_value=([current_node], [], []),
            ), patch.object(cli, "_git_sha", return_value="test-sha"):
                cli.build(str(project), str(db_path))

            conn = store.init_db(db_path)
            self.assertEqual(
                conn.execute("SELECT kind, name FROM nodes WHERE kind = 'topic'").fetchall(),
                [("topic", "Runtime")],
            )
            self.assertEqual(
                conn.execute(
                    "SELECT target, kind FROM edges WHERE kind = 'AFFECTS'"
                ).fetchall(),
                [("function:current", "AFFECTS")],
            )
            evidence = conn.execute(
                """
                SELECT reference, source_path, source_line_start, source_line_end,
                       resolution, confidence
                FROM edge_evidence
                """
            ).fetchone()
            self.assertEqual(evidence[0], "current")
            self.assertTrue(evidence[1].endswith("README.md"))
            self.assertEqual(evidence[2:], (3, 5, "unique_global_name", 1.0))
            conn.close()

    def test_build_reports_unresolved_semantic_reference(self) -> None:
        current_node = {
            "id": "function:current",
            "kind": "function",
            "name": "current",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            db_path = project / "graph.db"
            (project / "README.md").write_text(
                "# Project\n\n## Runtime\n\nCall `missing`.\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch.object(
                cli.structural,
                "extract_nodes_and_edges",
                return_value=([current_node], [], []),
            ), redirect_stdout(output):
                cli.build(str(project), str(db_path))

            conn = store.init_db(db_path)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM edge_evidence").fetchone()[0], 0)
            conn.close()

        self.assertIn("1 unresolved semantic reference(s)", output.getvalue())
        self.assertIn("'missing' (unresolved_reference)", output.getvalue())

    def test_build_removes_stale_topic(self) -> None:
        stale_topic = {
            "title": "Stale",
            "content": "Old topic.",
            "source_path": "README.md",
        }
        current_node = {
            "id": "function:current",
            "kind": "function",
            "name": "current",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            conn = store.init_db(db_path)
            store.save_topic_nodes(conn, [stale_topic])
            conn.commit()
            conn.close()

            output = io.StringIO()
            with patch.object(
                cli.structural,
                "extract_nodes_and_edges",
                return_value=([current_node], [], []),
            ), redirect_stdout(output):
                cli.build(tmpdir, str(db_path))

            conn = store.init_db(db_path)
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM nodes WHERE kind = 'topic'").fetchone()[0],
                0,
            )
            conn.close()

        self.assertIn("0 decisions, 1 topics, 0 stories, 0 edges removed", output.getvalue())


class TestIncrementalBuild(unittest.TestCase):
    """build reuses the cached graph when the source revision is unchanged."""

    def _build_once(self, db_path: Path) -> None:
        with tempfile.TemporaryDirectory() as project_dir:
            project = Path(project_dir)
            cli.build(str(project), str(db_path))

    def test_second_build_skips_extraction_when_sha_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            calls: list[str] = []

            def prepare(_project: Path) -> dict[str, str]:
                calls.append("prepare")
                return {"index_last_indexed": "2100-01-01T00:00:00Z"}

            def extract(_project: Path, **kwargs):
                calls.append("extract")
                return [], [], []

            with patch.object(cli.structural, "prepare_codegraph", side_effect=prepare), \
                 patch.object(cli.structural, "export_raw", return_value={"nodes": [], "edges": []}), \
                 patch.object(cli.structural, "extract_nodes_and_edges", side_effect=extract), \
                 patch.object(cli, "_git_sha", return_value="abc123"):
                cli.build(str(Path(tmpdir)), str(db_path))

            # First build ran extraction.
            self.assertIn("prepare", calls)
            self.assertIn("extract", calls)

            calls.clear()
            with patch.object(cli.structural, "prepare_codegraph", side_effect=prepare), \
                 patch.object(cli.structural, "export_raw", side_effect=AssertionError("export must not run")), \
                 patch.object(cli.structural, "extract_nodes_and_edges", side_effect=extract), \
                 patch.object(cli, "_git_sha", return_value="abc123"):
                cli.build(str(Path(tmpdir)), str(db_path))

            # Second build skipped extraction entirely.
            self.assertEqual(calls, [])

    def test_cached_build_can_add_local_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            node = {"id": "function:cached", "kind": "function", "name": "cached"}
            conn = store.init_db(db_path)
            store.save_structural_nodes(conn, [node])
            store.rebuild_fts(conn)
            store.save_metadata(conn, schema.load_schema()["version"], Path(tmpdir), "abc123")
            conn.commit()
            conn.close()

            class FakeEmbedder:
                def __init__(self, model_id: str) -> None:
                    self.model_id = model_id

                def encode(self, texts: list[str]) -> list[list[float]]:
                    return [[1.0, 0.0] for _ in texts]

            with patch.object(cli, "_git_sha", return_value="abc123"), \
                 patch.object(cli, "SentenceTransformerEmbedder", FakeEmbedder):
                cli.build(
                    tmpdir,
                    str(db_path),
                    embeddings="local",
                    embedding_model="fake-model",
                )

            conn = store.init_db(db_path)
            self.assertEqual(
                [item["node_id"] for item in store.load_node_embeddings(conn, "fake-model")],
                ["function:cached"],
            )
            conn.close()

    def test_force_rebuilds_even_when_sha_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.db"
            calls: list[str] = []

            def prepare(_project: Path) -> dict[str, str]:
                calls.append("prepare")
                return {"index_last_indexed": "2100-01-01T00:00:00Z"}

            def extract(_project: Path, **kwargs):
                calls.append("extract")
                return [], [], []

            with patch.object(cli.structural, "prepare_codegraph", side_effect=prepare), \
                 patch.object(cli.structural, "export_raw", return_value={"nodes": [], "edges": []}), \
                 patch.object(cli.structural, "extract_nodes_and_edges", side_effect=extract), \
                 patch.object(cli, "_git_sha", return_value="abc123"):
                cli.build(str(Path(tmpdir)), str(db_path))
                calls.clear()
                cli.build(str(Path(tmpdir)), str(db_path), force=True)

            self.assertIn("prepare", calls)
            self.assertIn("extract", calls)


class TestQueryCommands(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "graph.db"
        conn = store.init_db(self.db_path)
        graph_fixture(conn)
        conn.close()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def run_cli(self, *args: str) -> str:
        output = io.StringIO()
        with patch.object(sys, "argv", ["emp-kg", *args]), redirect_stdout(output):
            cli.main()
        return output.getvalue()

    def test_benchmark_context_json_and_text(self) -> None:
        cases_path = Path(self.tmpdir.name) / "benchmark.json"
        cases_path.write_text(json.dumps({
            "version": 1,
            "cases": [{
                "id": "alpha", "task": "alpha",
                "expected_node_ids": ["function:a"], "lane": "tiny", "k": 1,
            }],
        }), encoding="utf-8")
        payload = json.loads(self.run_cli(
            "benchmark-context", "--db", str(self.db_path), "--cases", str(cases_path), "--json"
        ))
        self.assertEqual(payload["aggregate"]["recall_at_k"], 1.0)
        output = self.run_cli(
            "benchmark-context", "--db", str(self.db_path), "--cases", str(cases_path)
        )
        self.assertIn("Recall@K 1.000", output)

    def test_stats_and_search_json(self) -> None:
        stats = json.loads(self.run_cli("stats", "--db", str(self.db_path), "--json"))
        self.assertEqual(stats["nodes"], 4)
        search = json.loads(
            self.run_cli("search", "alpha", "--db", str(self.db_path), "--json")
        )
        self.assertEqual([node["id"] for node in search], ["function:a"])

    def test_neighbors_and_path_text_output(self) -> None:
        neighbors = self.run_cli(
            "neighbors", "function:a", "--direction", "out", "--db", str(self.db_path)
        )
        self.assertIn("out\tCALLS\tfunction:b", neighbors)
        path = self.run_cli(
            "path", "function:a", "function:c", "--db", str(self.db_path)
        )
        self.assertIn("function:a", path)
        self.assertIn("-[CALLS]-> function:b", path)


if __name__ == "__main__":
    unittest.main()
