"""Golden three-build regression for the complete emp-kg pipeline."""
from __future__ import annotations

import io
import json
import shutil
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from emp_kg import cli, schema, store, structural
from tests.fixtures import golden_exports


_FIXTURE_PROJECT = Path(__file__).resolve().parent / "fixtures" / "golden_project"


class TestGoldenBuild(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.project = Path(self.tmpdir.name) / "golden-project"
        shutil.copytree(_FIXTURE_PROJECT, self.project)
        self.db_path = self.project / "graph.db"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def build(self, raw_export: dict, sha: str = "golden-sha") -> str:
        output = io.StringIO()
        with patch.object(
            structural,
            "prepare_codegraph",
            return_value={
                "codegraph_version": "unavailable",
                "index_last_indexed": "2100-01-01T00:00:00Z",
                "index_state": "complete",
            },
        ), patch.object(structural, "export_raw", return_value=raw_export), patch.object(
            cli, "_git_sha", return_value=sha
        ), redirect_stdout(output):
            cli.build(str(self.project), str(self.db_path))
        return output.getvalue()

    def graph_rows(self) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
        conn = store.init_db(self.db_path)
        try:
            nodes = conn.execute("SELECT id, kind FROM nodes ORDER BY id").fetchall()
            edges = conn.execute(
                "SELECT source, target, kind FROM edges ORDER BY source, target, kind"
            ).fetchall()
            return nodes, edges
        finally:
            conn.close()

    def test_three_builds_produce_exact_graph_and_reconciliation(self) -> None:
        output_one = self.build(golden_exports.build_one())
        conn = store.init_db(self.db_path)
        try:
            self.assertEqual(store.validate_graph(conn), {
                "nodes": 10,
                "edges": 14,
                "edge_evidence": 5,
                "fts_rows": 10,
            })
            self.assertEqual(conn.execute(
                "SELECT schema_version, git_sha FROM graph_metadata"
            ).fetchone(), (schema.load_schema()["version"], "golden-sha"))
            provenance = json.loads(
                conn.execute(
                    "SELECT codegraph_provenance FROM graph_metadata"
                ).fetchone()[0]
            )
            self.assertEqual(provenance["codegraph_version"], "unavailable")
            self.assertEqual(provenance["index_last_indexed"], "2100-01-01T00:00:00Z")
            self.assertEqual(provenance["index_state"], "complete")
            self.assertEqual(provenance["raw_node_count"], 8)
            self.assertRegex(provenance["raw_export_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(store.graph_stats(conn), {
                "nodes": 10,
                "edges": 14,
                "nodes_by_kind": {
                    "class": 1,
                    "constant": 1,
                    "decision_ref": 1,
                    "file": 2,
                    "function": 3,
                    "import": 1,
                    "topic": 1,
                },
                "edges_by_kind": {
                    "AFFECTS": 4,
                    "CALLS": 1,
                    "CONTAINS": 4,
                    "DEFINES": 2,
                    "IMPORTS": 1,
                    "INSTANTIATES": 1,
                    "REFERENCES": 1,
                },
            })
            self.assertEqual(
                [node["title"] for node in store.query_by_name(conn, "Runtime")],
                ["0001 Service Runtime", "Runtime"],
            )
            self.assertEqual(
                conn.execute(
                    """
                    SELECT source, target, kind FROM edges
                    WHERE kind = 'AFFECTS' ORDER BY source, target
                    """
                ).fetchall(),
                [
                    ("decision_ref:0001", "class:service", "AFFECTS"),
                    ("decision_ref:0001", "function:service.run", "AFFECTS"),
                    (self.topic_id(), "function:legacy", "AFFECTS"),
                    (self.topic_id(), "function:service.run", "AFFECTS"),
                ],
            )
            evidence = conn.execute(
                """
                SELECT source, target, reference, source_path, source_line_start,
                       source_line_end, resolution, confidence
                FROM edge_evidence
                ORDER BY source, reference
                """
            ).fetchall()
            self.assertEqual(
                [(source, target, reference, start, end, resolution, confidence)
                 for source, target, reference, _, start, end, resolution, confidence in evidence],
                [
                    ("decision_ref:0001", "class:service", "Service", 17, 25, "unique_global_name", 1.0),
                    ("decision_ref:0001", "function:service.run", "service.run", 9, 11, "qualified_name", 1.0),
                    ("decision_ref:0001", "function:service.run", "service.run", 13, 15, "qualified_name", 1.0),
                    (self.topic_id(), "function:legacy", "legacy", 3, 5, "unique_global_name", 1.0),
                    (self.topic_id(), "function:service.run", "service.run", 3, 5, "qualified_name", 1.0),
                ],
            )
        finally:
            conn.close()
        self.assertIn("0 unresolved semantic references", output_one)

        output_two = self.build(golden_exports.build_two(), sha="golden-sha-2")
        nodes_two, edges_two = self.graph_rows()
        self.assertNotIn(("function:legacy", "function"), nodes_two)
        self.assertIn(("function:service.run", "function"), nodes_two)
        self.assertNotIn((self.topic_id(), "function:legacy", "AFFECTS"), edges_two)
        self.assertIn(("decision_ref:0001", "function:service.run", "AFFECTS"), edges_two)
        self.assertIn("1 structural nodes, 0 decisions, 0 topics, 0 stories, 1 edges removed", output_two)

        self._rename_document_references()
        output_three = self.build(golden_exports.build_three(), sha="golden-sha-3")
        nodes_three, edges_three = self.graph_rows()
        self.assertNotIn(("function:service.run", "function"), nodes_three)
        self.assertIn(("function:service.execute", "function"), nodes_three)
        self.assertEqual(
            [edge for edge in edges_three if edge[2] == "AFFECTS"],
            [
                ("decision_ref:0001", "class:service", "AFFECTS"),
                ("decision_ref:0001", "function:service.execute", "AFFECTS"),
                (self.topic_id(), "function:service.execute", "AFFECTS"),
            ],
        )
        conn = store.init_db(self.db_path)
        try:
            self.assertEqual(
                [row[0] for row in conn.execute(
                    "SELECT reference FROM edge_evidence ORDER BY source, reference"
                )],
                ["Service", "service.execute", "service.execute", "service.execute"],
            )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM edge_evidence WHERE target = 'function:service.run'"
                ).fetchone()[0],
                0,
            )
        finally:
            conn.close()
        self.assertIn("1 structural nodes, 0 decisions, 0 topics, 0 stories, 4 edges removed", output_three)
        self.assertIn("'legacy' (unresolved_reference)", output_three)

    def test_raw_fixture_covers_every_supported_structural_mapping(self) -> None:
        with patch.object(structural, "export_raw", return_value=golden_exports.build_one()):
            nodes, edges, warnings = structural.extract_nodes_and_edges(self.project)
        self.assertEqual(warnings, [])
        self.assertEqual(
            {node["kind"] for node in nodes},
            {"file", "import", "function", "constant", "class"},
        )
        self.assertEqual(
            {edge["kind"] for edge in edges},
            {"CALLS", "DEFINES", "IMPORTS", "REFERENCES", "CONTAINS", "INSTANTIATES"},
        )

    def topic_id(self) -> str:
        from emp_kg.merger import _topic_id

        return _topic_id({
            "title": "Runtime",
            "source_path": str((self.project / "README.md").as_posix()),
        })

    def _rename_document_references(self) -> None:
        for path in [
            self.project / "README.md",
            self.project / "docs" / "decisions" / "0001-service-runtime.md",
        ]:
            path.write_text(
                path.read_text(encoding="utf-8").replace("service.run", "service.execute"),
                encoding="utf-8",
            )
