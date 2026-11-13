"""Tests for supported-scope structural extraction."""
from __future__ import annotations

import os
import tempfile
import hashlib
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from emp_kg import structural


def _git(args: list[str], cwd: Path) -> CompletedProcess[str]:
    import subprocess
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def _init_repo(root: Path) -> None:
    """Create a bare git repo with .gitignore at root."""
    _git(["init"], root)
    _git(["config", "user.email", "test@test"], root)
    _git(["config", "user.name", "Test"], root)
    (root / ".gitignore").write_text(".claude/\nnode_modules/\n", encoding="utf-8")
    _git(["add", ".gitignore"], root)
    _git(["commit", "-m", "init", "--allow-empty"], root)


class TestStructuralImplementationScope(unittest.TestCase):
    def test_all_source_files_are_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            source = project / "src" / "main.py"
            source.parent.mkdir()
            source.write_text("print('indexed')\n", encoding="utf-8")
            result = CompletedProcess(
                args=[],
                returncode=0,
                stdout='[{"path": "src/main.py"}]',
                stderr="",
            )

            with patch.object(
                structural, "_source_files_on_disk", return_value={"src/main.py"}
            ), patch.object(structural.subprocess, "run", return_value=result):
                structural._assert_all_source_files_indexed("codegraph", project)

    def test_unindexed_source_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            source = project / "src" / "missing.py"
            source.parent.mkdir()
            source.write_text("print('missing')\n", encoding="utf-8")
            result = CompletedProcess(
                args=[],
                returncode=0,
                stdout='[{"path": "src/indexed.py"}]',
                stderr="",
            )

            with patch.object(
                structural, "_source_files_on_disk", return_value={"src/missing.py"}
            ), patch.object(structural.subprocess, "run", return_value=result):
                with self.assertRaisesRegex(
                    structural.CodeGraphFreshnessError,
                    "src/missing.py",
                ):
                    structural._assert_all_source_files_indexed("codegraph", project)

    def test_excluded_and_unsupported_files_are_not_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            ignored = project / ".claude" / "scratch.py"
            unsupported = project / "notes.txt"
            ignored.parent.mkdir()
            ignored.write_text("print('ignored')\n", encoding="utf-8")
            unsupported.write_text("notes\n", encoding="utf-8")
            result = CompletedProcess(
                args=[],
                returncode=0,
                stdout='[{"path": "src/indexed.py"}]',
                stderr="",
            )

            with patch.object(
                structural, "_source_files_on_disk", return_value=set()
            ), patch.object(structural.subprocess, "run", return_value=result):
                structural._assert_all_source_files_indexed("codegraph", project)

    def test_invalid_indexed_file_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = CompletedProcess(
                args=[],
                returncode=0,
                stdout="not-json",
                stderr="",
            )
            with patch.object(structural.subprocess, "run", return_value=result):
                with self.assertRaisesRegex(
                    structural.CodeGraphFreshnessError,
                    "invalid JSON",
                ):
                    structural._indexed_source_files("codegraph", Path(tmpdir))

    def test_failed_indexed_file_query_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = CompletedProcess(
                args=[],
                returncode=1,
                stdout="",
                stderr="index unavailable",
            )
            with patch.object(structural.subprocess, "run", return_value=result):
                with self.assertRaisesRegex(
                    structural.CodeGraphFreshnessError,
                    "index unavailable",
                ):
                    structural._indexed_source_files("codegraph", Path(tmpdir))


class TestSourceFilesOnDisk(unittest.TestCase):
    """Git-aware _source_files_on_disk respects .gitignore and nested repos."""

    def test_gitignored_files_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            _init_repo(project)
            tracked = project / "src" / "main.py"
            tracked.parent.mkdir()
            tracked.write_text("print('hello')\n", encoding="utf-8")
            ignored = project / ".claude" / "scratch.py"
            ignored.parent.mkdir()
            ignored.write_text("print('scratch')\n", encoding="utf-8")
            _git(["add", "src/main.py"], project)
            _git(["commit", "-m", "add main"], project)

            result = structural._source_files_on_disk(project)
            self.assertIn("src/main.py", result)
            self.assertEqual(len([p for p in result if ".claude" in p]), 0)

    def test_nested_gitignored_repo_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            _init_repo(project)
            (project / "src").mkdir()
            (project / "src" / "main.py").write_text("print('main')\n")
            _git(["add", "."], project)
            _git(["commit", "-m", "init"], project)

            # Nested repo is gitignored by the parent BEFORE it ever exists,
            # so the parent never tracks it (no gitlink entry).
            (project / ".gitignore").write_text(
                ".claude/\nworkspace/other-lib/\n", encoding="utf-8"
            )
            _git(["add", ".gitignore"], project)
            _git(["commit", "-m", "ignore nested"], project)

            nested = project / "workspace" / "other-lib"
            nested.mkdir(parents=True)
            _git(["init"], nested)
            _git(["config", "user.email", "test@test"], nested)
            _git(["config", "user.name", "Test"], nested)
            (nested / "lib.py").write_text("x = 1\n")
            _git(["add", "."], nested)
            _git(["commit", "-m", "init"], nested)

            result = structural._source_files_on_disk(project)
            self.assertIn("src/main.py", result)
            self.assertFalse(any("other-lib" in p for p in result))

    def test_non_gitignored_nested_repo_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            _init_repo(project)
            (project / "src").mkdir()
            (project / "src" / "main.py").write_text("print('main')\n")
            _git(["add", "."], project)
            _git(["commit", "-m", "init"], project)

            nested = project / "lib" / "sub"
            nested.mkdir(parents=True)
            _git(["init"], nested)
            _git(["config", "user.email", "test@test"], nested)
            _git(["config", "user.name", "Test"], nested)
            (nested / "helper.py").write_text("y = 2\n")
            _git(["add", "."], nested)
            _git(["commit", "-m", "init"], nested)
            _git(["add", "lib/sub"], project)
            _git(["commit", "-m", "add nested"], project)

            result = structural._source_files_on_disk(project)
            self.assertIn("src/main.py", result)
            self.assertIn("lib/sub/helper.py", result)

    def test_non_git_directory_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            with self.assertRaisesRegex(
                structural.CodeGraphFreshnessError, "not a git repository"
            ):
                structural._source_files_on_disk(project)

    def test_raw_export_sha256_is_canonical_and_deterministic(self) -> None:
        raw = {
            "edges": [{"source": "a", "target": "b", "kind": "calls"}],
            "nodes": [{"id": "function:main", "kind": "function", "name": "main"}],
        }
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
        expected = hashlib.sha256(canonical).hexdigest()
        self.assertEqual(structural.raw_export_sha256(raw), expected)
        self.assertEqual(structural.raw_export_sha256(dict(reversed(list(raw.items())))), expected)

    def test_source_newer_than_index_rejects_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            source = project / "src" / "main.py"
            source.parent.mkdir()
            source.write_text("print('new')\n", encoding="utf-8")
            indexed_at = datetime.now(timezone.utc).timestamp() - 60
            os.utime(source, (indexed_at + 1, indexed_at + 1))
            raw = {
                "nodes": [
                    {"id": "file:src/main.py", "kind": "file", "filePath": "src/main.py"},
                ],
                "edges": [],
            }

            with patch.object(structural, "export_raw", return_value=raw):
                with self.assertRaisesRegex(
                    structural.CodeGraphFreshnessError,
                    "src/main.py",
                ):
                    structural.extract_nodes_and_edges(
                        project,
                        index_last_indexed=datetime.fromtimestamp(
                            indexed_at,
                            timezone.utc,
                        ).isoformat(),
                    )

    def test_source_not_newer_than_index_allows_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            source = project / "src" / "main.py"
            source.parent.mkdir()
            source.write_text("print('old')\n", encoding="utf-8")
            indexed_at = datetime.now(timezone.utc).timestamp()
            os.utime(source, (indexed_at - 1, indexed_at - 1))
            raw = {
                "nodes": [
                    {"id": "file:src/main.py", "kind": "file", "filePath": "src/main.py"},
                ],
                "edges": [],
            }

            with patch.object(structural, "export_raw", return_value=raw):
                nodes, edges, warnings = structural.extract_nodes_and_edges(
                    project,
                    index_last_indexed=datetime.fromtimestamp(
                        indexed_at,
                        timezone.utc,
                    ).isoformat(),
                )

            self.assertEqual(nodes, raw["nodes"])
            self.assertEqual(edges, [])
            self.assertEqual(warnings, [])

    def test_non_file_nodes_do_not_trigger_mtime_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = {
                "nodes": [
                    {"id": "function:main", "kind": "function", "name": "main"},
                ],
                "edges": [],
            }
            with patch.object(structural, "export_raw", return_value=raw):
                nodes, edges, warnings = structural.extract_nodes_and_edges(
                    Path(tmpdir),
                    index_last_indexed="1970-01-01T00:00:00Z",
                )

            self.assertEqual(nodes, raw["nodes"])
            self.assertEqual(edges, [])
            self.assertEqual(warnings, [])

    def test_derives_defines_from_file_containment_only(self) -> None:
        raw = {
            "nodes": [
                {"id": "file:src/main.py", "kind": "file", "name": "main.py"},
                {"id": "function:run", "kind": "function", "name": "run"},
                {"id": "class:service", "kind": "class", "name": "Service"},
                {"id": "variable:state", "kind": "variable", "name": "state"},
                {"id": "method:start", "kind": "method", "name": "start"},
            ],
            "edges": [
                {"source": "file:src/main.py", "target": "function:run", "kind": "contains"},
                {"source": "file:src/main.py", "target": "class:service", "kind": "contains"},
                {"source": "file:src/main.py", "target": "variable:state", "kind": "contains"},
                {"source": "function:run", "target": "method:start", "kind": "contains"},
                {"source": "function:run", "target": "method:start", "kind": "calls"},
            ],
        }
        with patch.object(structural, "export_raw", return_value=raw):
            _, edges, warnings = structural.extract_nodes_and_edges(Path("."))

        self.assertEqual(
            edges,
            [
                {"source": "file:src/main.py", "target": "function:run", "kind": "CONTAINS"},
                {"source": "file:src/main.py", "target": "class:service", "kind": "CONTAINS"},
                {"source": "file:src/main.py", "target": "variable:state", "kind": "CONTAINS"},
                {"source": "function:run", "target": "method:start", "kind": "CONTAINS"},
                {"source": "function:run", "target": "method:start", "kind": "CALLS"},
                {"source": "file:src/main.py", "target": "function:run", "kind": "DEFINES"},
                {"source": "file:src/main.py", "target": "class:service", "kind": "DEFINES"},
            ],
        )
        self.assertEqual(warnings, [])

    def test_derives_tests_from_test_function_calls_only(self) -> None:
        raw = {
            "nodes": [
                {"id": "function:test_build", "kind": "function", "name": "test_build"},
                {"id": "function:build", "kind": "function", "name": "build"},
                {"id": "function:main", "kind": "function", "name": "main"},
                {"id": "method:setup", "kind": "method", "name": "setup"},
                {"id": "method:test_run", "kind": "method", "name": "test_run"},
            ],
            "edges": [
                {"source": "function:test_build", "target": "function:build", "kind": "calls"},
                {"source": "function:main", "target": "function:build", "kind": "calls"},
                {"source": "function:test_build", "target": "function:test_build", "kind": "calls"},
                {"source": "method:setup", "target": "function:build", "kind": "calls"},
                {"source": "method:test_run", "target": "function:build", "kind": "calls"},
            ],
        }
        with patch.object(structural, "export_raw", return_value=raw):
            _, edges, warnings = structural.extract_nodes_and_edges(Path("."))

        self.assertEqual(
            edges,
            [
                {"source": "function:test_build", "target": "function:build", "kind": "CALLS"},
                {"source": "function:main", "target": "function:build", "kind": "CALLS"},
                {"source": "function:test_build", "target": "function:test_build", "kind": "CALLS"},
                {"source": "method:setup", "target": "function:build", "kind": "CALLS"},
                {"source": "method:test_run", "target": "function:build", "kind": "CALLS"},
                {"source": "function:test_build", "target": "function:build", "kind": "TESTS"},
                {"source": "method:test_run", "target": "function:build", "kind": "TESTS"},
            ],
        )
        self.assertEqual(warnings, [])


    def test_drops_declared_but_unsupported_node_kind_with_warning(self) -> None:
        raw = {
            "nodes": [
                {"id": "function:build", "kind": "function", "name": "build"},
                {"id": "story:001", "kind": "story", "name": "Story"},
            ],
            "edges": [],
        }
        with patch.object(structural, "export_raw", return_value=raw):
            nodes, edges, warnings = structural.extract_nodes_and_edges(Path("."))

        self.assertEqual(nodes, [raw["nodes"][0]])
        self.assertEqual(edges, [])
        self.assertEqual(
            warnings,
            ["declared but unsupported node kind 'story' (id=story:001)"],
        )
        

    


if __name__ == "__main__":
    unittest.main()

    def test_parse_go_ast_nodes():
        go_code = """
        package main

        type User struct {
            Name string
        }

        func (u *User) GetName() string {
            return u.Name
        }

        func CalculateTotal(a int, b int) int {
            return a + b
        }
        """
        nodes = parse_go_ast_nodes(go_code)
        node_names = {n["name"] for n in nodes}
        
        assert "User" in node_names
        assert "GetName" in node_names
        assert "CalculateTotal" in node_names


    def test_parse_cpp_ast_nodes():
        cpp_code = """
        #include <iostream>

        class Calculator {
        public:
            int add(int a, int b) {
                return a + b;
            }
        };

        void processData() {
            // do something
        }
        """
        nodes = parse_cpp_ast_nodes(cpp_code)
        node_names = {n["name"] for n in nodes}

        assert "Calculator" in node_names
        assert "add" in node_names
        assert "processData" in node_names


    def test_parse_java_ast_nodes():
        java_code = """
        public class OrderProcessor {
            private String status;

            public void processOrder() {
                // processing
            }
        }

        public interface Notifiable {
            void notifyUser();
        }
        """
        nodes = parse_java_ast_nodes(java_code)
        node_names = {n["name"] for n in nodes}

        assert "OrderProcessor" in node_names
        assert "processOrder" in node_names
        assert "Notifiable" in node_names
        assert "notifyUser" in node_names