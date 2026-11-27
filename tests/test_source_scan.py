"""Tests for source-based structural edge derivation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from emp_kg.source_scan import (
    derive_source_graph,
    enrich_file_nodes_with_source_text,
    enrich_nodes_with_source_text,
    extract_docstrings_and_body,
)


class TestDeriveSourceGraph(unittest.TestCase):
    def test_derives_extends_throws_and_queries(self) -> None:
        source = """\
class Animal:
    pass

class Dog(Animal):
    pass

class Bird(ExternalBird):
    pass


def validate():
    raise ValueError("bad")


def persist(cursor):
    cursor.execute("SELECT * FROM users")
    cursor.execute("INSERT INTO events VALUES (1)")
    cursor.execute("UPDATE records SET value = 1")
    cursor.execute("DELETE FROM logs")
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "src" / "domain.py"
            path.parent.mkdir()
            path.write_text(source, encoding="utf-8")
            nodes = [
                {"id": "class:animal", "kind": "class", "name": "Animal", "filePath": "src/domain.py", "startLine": 1, "endLine": 2},
                {"id": "class:dog", "kind": "class", "name": "Dog", "filePath": "src/domain.py", "startLine": 4, "endLine": 5},
                {"id": "class:bird", "kind": "class", "name": "Bird", "filePath": "src/domain.py", "startLine": 7, "endLine": 8},
                {"id": "function:validate", "kind": "function", "name": "validate", "filePath": "src/domain.py", "startLine": 11, "endLine": 12},
                {"id": "function:persist", "kind": "function", "name": "persist", "filePath": "src/domain.py", "startLine": 15, "endLine": 19},
            ]

            derived_nodes, edges = derive_source_graph(nodes, [], Path(tmpdir))

        self.assertEqual(
            {(edge["source"], edge["target"], edge["kind"]) for edge in edges},
            {
                ("class:dog", "class:animal", "EXTENDS"),
                ("class:bird", next(node["id"] for node in derived_nodes if node["name"] == "ExternalBird"), "EXTENDS"),
                ("function:validate", next(node["id"] for node in derived_nodes if node["name"] == "ValueError"), "THROWS"),
                ("function:persist", next(node["id"] for node in derived_nodes if node["name"] == "users"), "QUERIES"),
                ("function:persist", next(node["id"] for node in derived_nodes if node["name"] == "events"), "QUERIES"),
                ("function:persist", next(node["id"] for node in derived_nodes if node["name"] == "records"), "QUERIES"),
                ("function:persist", next(node["id"] for node in derived_nodes if node["name"] == "logs"), "QUERIES"),
            },
        )
        self.assertEqual(
            {(node["kind"], node["name"]) for node in derived_nodes},
            {
                ("class", "ExternalBird"),
                ("exception_type", "ValueError"),
                ("table", "users"),
                ("table", "events"),
                ("table", "records"),
                ("table", "logs"),
            },
        )

    def test_derives_implements_and_python_route(self) -> None:
        source = """\
class Service implements Runnable, Closeable {
}

@app.get("/health")
def health():
    pass
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "src" / "api.ts"
            path.parent.mkdir()
            path.write_text(source, encoding="utf-8")
            nodes = [
                {"id": "class:service", "kind": "class", "name": "Service", "filePath": "src/api.ts", "startLine": 1, "endLine": 2},
                {"id": "function:health", "kind": "function", "name": "health", "filePath": "src/api.ts", "startLine": 5, "endLine": 6},
            ]

            derived_nodes, edges = derive_source_graph(nodes, [], Path(tmpdir))

        interface_names = {node["name"] for node in derived_nodes if node["kind"] == "interface"}
        self.assertEqual(interface_names, {"Runnable", "Closeable"})
        self.assertIn(
            ("class:service", next(node["id"] for node in derived_nodes if node["name"] == "Runnable"), "IMPLEMENTS"),
            {(edge["source"], edge["target"], edge["kind"]) for edge in edges},
        )
        self.assertIn(
            ("class:service", next(node["id"] for node in derived_nodes if node["name"] == "Closeable"), "IMPLEMENTS"),
            {(edge["source"], edge["target"], edge["kind"]) for edge in edges},
        )

    def test_derives_python_route_handler(self) -> None:
        source = '@app.get("/health")\ndef health():\n    pass\n'
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "src" / "api.py"
            path.parent.mkdir()
            path.write_text(source, encoding="utf-8")
            nodes = [
                {"id": "function:health", "kind": "function", "name": "health", "filePath": "src/api.py", "startLine": 2, "endLine": 3},
            ]

            derived_nodes, edges = derive_source_graph(nodes, [], Path(tmpdir))

        route = next(node for node in derived_nodes if node["kind"] == "route")
        self.assertEqual(route["name"], "GET /health")
        self.assertEqual(edges, [{"source": route["id"], "target": "function:health", "kind": "ROUTE_HANDLER"}])

    def test_derives_javascript_route_handler(self) -> None:
        source = 'function login() {}\napp.post("/login", login);\n'
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "src" / "api.js"
            path.parent.mkdir()
            path.write_text(source, encoding="utf-8")
            nodes = [
                {"id": "function:login", "kind": "function", "name": "login", "filePath": "src/api.js", "startLine": 1, "endLine": 1},
            ]

            derived_nodes, edges = derive_source_graph(nodes, [], Path(tmpdir))

        route = next(node for node in derived_nodes if node["kind"] == "route")
        self.assertEqual(route["name"], "POST /login")
        self.assertEqual(edges, [{"source": route["id"], "target": "function:login", "kind": "ROUTE_HANDLER"}])

    def test_extracts_source_body_and_docstrings(self) -> None:
        source = '''def refresh():
    """Refresh credentials for active sessions."""
    return True
'''
        docstring, body = extract_docstrings_and_body(source, 1, 3)
        self.assertIn("Refresh credentials", docstring)
        self.assertIn("return True", body)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "auth.py"
            path.write_text(source, encoding="utf-8")
            nodes = [{
                "id": "function:refresh", "kind": "function", "name": "refresh",
                "filePath": "auth.py", "startLine": 1, "endLine": 3,
            }]
            enrich_nodes_with_source_text(nodes, Path(tmpdir))
        self.assertIn("Refresh credentials", nodes[0]["docstring"])
        self.assertIn("return True", nodes[0]["source_body"])

    def test_file_nodes_get_bounded_source_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "app.py"
            path.write_text("needle = 'file-level search'\n" * 2000, encoding="utf-8")
            nodes = [{
                "id": "file:app.py", "kind": "file", "name": "app.py",
                "filePath": "app.py",
            }]
            enrich_file_nodes_with_source_text(nodes, Path(tmpdir))
        self.assertEqual(len(nodes[0]["source_text"]), 32000)
        self.assertIn("file-level search", nodes[0]["source_text"])

    def test_file_index_skips_missing_and_binary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            binary = root / "binary.dat"
            binary.write_bytes(b"\x00binary")
            nodes = [
                {"id": "file:missing.py", "kind": "file", "name": "missing.py", "filePath": "missing.py"},
                {"id": "file:binary.dat", "kind": "file", "name": "binary.dat", "filePath": "binary.dat"},
            ]
            enrich_file_nodes_with_source_text(nodes, root)
        self.assertNotIn("source_text", nodes[0])
        self.assertNotIn("source_text", nodes[1])

    def test_reuses_existing_targets_and_skips_missing_files(self) -> None:
        nodes = [
            {"id": "function:missing", "kind": "function", "name": "missing", "filePath": "gone.py", "startLine": 1, "endLine": 2},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertEqual(derive_source_graph(nodes, [], Path(tmpdir)), ([], []))


if __name__ == "__main__":
    unittest.main()
