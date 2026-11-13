"""Structural extraction and CodeGraph freshness checks for emp-kg."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import tree_sitter_go as tsgo
import tree_sitter_cpp as tscpp
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Query, QueryCursor

from emp_kg import schema
from emp_kg.source_scan import derive_source_graph


_EDGE_KIND_MAP = {
    "calls": "CALLS",
    "imports": "IMPORTS",
    "references": "REFERENCES",
    "contains": "CONTAINS",
    "instantiates": "INSTANTIATES",
}

_SUPPORTED_SOURCE_EXTENSIONS = {
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".ets",
    ".py", ".go", ".rs", ".java", ".cs", ".php", ".rb",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".m", ".mm",
    ".metal", ".cu", ".cuh",
    ".swift", ".kt", ".kts", ".scala", ".sc", ".dart",
    ".svelte", ".vue", ".astro", ".liquid",
    ".pas", ".dpr", ".dpk", ".lpr",
    ".lua", ".R", ".r", ".luau",
    ".cfc", ".cfm", ".cfs",
    ".cbl", ".cob", ".cpy",
    ".vb", ".erl", ".hrl", ".escript",
    ".sol", ".tf", ".tfvars", ".tofu", ".nix",
}
_EXCLUDED_SOURCE_DIRS = {
    "node_modules", "vendor", "dist", "build", "target",
    ".venv", "Pods", ".next", ".git", ".codegraph", ".claude",
}
_MAX_SOURCE_FILE_BYTES = 1_000_000


class CodeGraphFreshnessError(RuntimeError):
    """CodeGraph cannot provide a complete index for a safe graph build."""


def _find_codegraph() -> str:
    exe = shutil.which("codegraph") or shutil.which("codegraph.cmd")
    if not exe:
        raise FileNotFoundError(
            "Could not find the codegraph CLI on PATH. "
            "Install CodeGraph and ensure it is discoverable."
        )
    return exe


def _run_codegraph(
    codegraph: str,
    args: list[str],
    project_path: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [codegraph, *args, str(project_path)],
        cwd=str(project_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _status(codegraph: str, project_path: Path) -> dict[str, Any]:
    result = _run_codegraph(codegraph, ["status", "--json"], project_path)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CodeGraphFreshnessError(f"codegraph status failed: {detail}")
    try:
        status = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CodeGraphFreshnessError("codegraph status returned invalid JSON") from exc
    if not isinstance(status, dict):
        raise CodeGraphFreshnessError("codegraph status returned an invalid payload")
    return status


def _pending_count(status: dict[str, Any]) -> int:
    pending = status.get("pendingChanges")
    if not isinstance(pending, dict):
        return -1
    values = [pending.get(key) for key in ("added", "modified", "removed")]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return -1
    return sum(values)


def _assert_fresh_status(status: dict[str, Any]) -> None:
    if status.get("initialized") is not True:
        raise CodeGraphFreshnessError(
            "CodeGraph project is not initialized. Run `codegraph init` first."
        )
    pending = _pending_count(status)
    if pending < 0:
        raise CodeGraphFreshnessError("codegraph status omitted valid pendingChanges")
    if pending:
        raise CodeGraphFreshnessError(
            f"CodeGraph index remains stale after sync ({pending} pending change(s))."
        )
    index = status.get("index")
    if not isinstance(index, dict) or index.get("state") != "complete":
        raise CodeGraphFreshnessError("CodeGraph index is not complete")
    if index.get("reindexRecommended") is True:
        raise CodeGraphFreshnessError("CodeGraph requires a full reindex")
    if index.get("pendingRefs") not in (0, None):
        raise CodeGraphFreshnessError("CodeGraph has unresolved pending references")
    if status.get("worktreeMismatch") not in (None, ""):
        raise CodeGraphFreshnessError(
            f"CodeGraph worktree mismatch: {status['worktreeMismatch']}"
        )
    if not isinstance(status.get("lastIndexed"), str) or not status["lastIndexed"]:
        raise CodeGraphFreshnessError("CodeGraph status omitted lastIndexed")


def _path_values(value: Any) -> set[str]:
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"path", "file", "filePath", "relativePath"} and isinstance(item, str):
                paths.add(item.replace("\\", "/").lstrip("/"))
            paths.update(_path_values(item))
    elif isinstance(value, list):
        for item in value:
            paths.update(_path_values(item))
    return paths


def _indexed_source_files(codegraph: str, project_path: Path) -> set[str]:
    result = subprocess.run(
        [codegraph, "files", "--json", "--format", "flat", "--path", str(project_path)],
        cwd=str(project_path),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CodeGraphFreshnessError(f"codegraph files failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CodeGraphFreshnessError("codegraph files returned invalid JSON") from exc
    paths = _path_values(payload)
    if not paths:
        raise CodeGraphFreshnessError("codegraph files returned no indexed paths")
    return paths


def _git_ls_files(repo: Path, extra_args: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", *extra_args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise CodeGraphFreshnessError(
            f"git ls-files failed in {repo}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.splitlines()


def _find_nested_git_repos(repo_root: Path) -> list[Path]:
    nested: list[Path] = []
    for git_dir in repo_root.rglob(".git"):
        if git_dir == repo_root / ".git":
            continue
        parent = git_dir.parent
        try:
            relative = parent.relative_to(repo_root)
        except ValueError:
            continue
        if any(part in _EXCLUDED_SOURCE_DIRS for part in relative.parts):
            continue
        nested.append(parent)
    return nested


def _is_gitignored(root: Path, path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "-q", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode not in (0, 1):
        raise CodeGraphFreshnessError(
            f"git check-ignore failed in {root}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.returncode == 0


def _git_source_files(repo: Path) -> set[str]:
    tracked = _git_ls_files(repo, [])
    untracked = _git_ls_files(repo, ["--others", "--exclude-standard"])
    source_files: set[str] = set()
    for relative in set(tracked) | set(untracked):
        rel_path = Path(relative)
        if rel_path.suffix not in _SUPPORTED_SOURCE_EXTENSIONS:
            continue
        if any(part in _EXCLUDED_SOURCE_DIRS for part in rel_path.parts):
            continue
        try:
            if (repo / rel_path).stat().st_size > _MAX_SOURCE_FILE_BYTES:
                continue
        except OSError:
            continue
        source_files.add(rel_path.as_posix())
    return source_files


def _source_files_on_disk(project_path: Path) -> set[str]:
    if not (project_path / ".git").exists():
        raise CodeGraphFreshnessError(f"not a git repository: {project_path}")
    source_files = _git_source_files(project_path)
    for nested_repo in _find_nested_git_repos(project_path):
        rel_nested = nested_repo.relative_to(project_path)
        if _is_gitignored(project_path, nested_repo):
            continue
        source_files |= {
            (rel_nested / Path(f)).as_posix()
            for f in _git_source_files(nested_repo)
        }
    return source_files


def _assert_all_source_files_indexed(codegraph: str, project_path: Path) -> None:
    missing = sorted(_source_files_on_disk(project_path) - _indexed_source_files(codegraph, project_path))
    if missing:
        raise CodeGraphFreshnessError(
            "Source files missing from the CodeGraph index: " + ", ".join(missing)
        )


def _codegraph_version(codegraph: str, project_path: Path) -> str:
    result = subprocess.run(
        [codegraph, "version"],
        cwd=str(project_path),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CodeGraphFreshnessError(f"codegraph version failed: {detail}")
    version = result.stdout.strip()
    if not version:
        raise CodeGraphFreshnessError("codegraph version returned no version")
    return version


def prepare_codegraph(project_path: Path) -> dict[str, Any]:
    codegraph = _find_codegraph()
    before = _status(codegraph, project_path)
    if before.get("initialized") is not True:
        raise CodeGraphFreshnessError(
            "CodeGraph project is not initialized. Run `codegraph init` first."
        )

    sync = _run_codegraph(codegraph, ["sync"], project_path)
    if sync.returncode != 0:
        detail = sync.stderr.strip() or sync.stdout.strip()
        raise CodeGraphFreshnessError(f"codegraph sync failed: {detail}")

    after_sync = _status(codegraph, project_path)
    _assert_fresh_status(after_sync)
    _assert_all_source_files_indexed(codegraph, project_path)
    return {
        "codegraph_version": _codegraph_version(codegraph, project_path),
        "index_last_indexed": after_sync["lastIndexed"],
        "index_state": after_sync["index"]["state"],
        "index_built_with_version": after_sync["index"].get("builtWithVersion"),
        "index_extraction_version": after_sync["index"].get("builtWithExtractionVersion"),
        "index_current_extraction_version": after_sync["index"].get("currentExtractionVersion"),
        "pending_changes_before_sync": _pending_count(before),
        "pending_changes_after_sync": _pending_count(after_sync),
    }


def _parse_indexed_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CodeGraphFreshnessError(f"invalid CodeGraph lastIndexed value {value!r}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _source_files_newer_than_index(
    project_path: Path,
    raw: dict[str, Any],
    last_indexed: str,
) -> list[str]:
    indexed_at = _parse_indexed_at(last_indexed).timestamp()
    newer: list[str] = []
    for node in raw.get("nodes", []):
        if node.get("kind") != "file" or not isinstance(node.get("filePath"), str):
            continue
        file_path = project_path / node["filePath"]
        if file_path.is_file() and file_path.stat().st_mtime > indexed_at:
            newer.append(node["filePath"])
    return sorted(set(newer))


def _assert_source_files_not_newer(
    project_path: Path,
    raw: dict[str, Any],
    last_indexed: str,
) -> None:
    newer = _source_files_newer_than_index(project_path, raw, last_indexed)
    if newer:
        raise CodeGraphFreshnessError(
            "Source files are newer than the CodeGraph index: " + ", ".join(newer)
        )


def raw_export_sha256(raw: dict[str, Any]) -> str:
    canonical_raw = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_raw).hexdigest()


def finalize_codegraph_provenance(
    project_path: Path,
    raw: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    codegraph = _find_codegraph()
    after_export = _status(codegraph, project_path)
    _assert_fresh_status(after_export)
    _assert_source_files_not_newer(
        project_path,
        raw,
        str(after_export["lastIndexed"]),
    )
    return {
        **provenance,
        "index_last_indexed": after_export["lastIndexed"],
        "pending_changes_after_export": _pending_count(after_export),
        "source_files_newer_than_index": 0,
        "raw_export_sha256": raw_export_sha256(raw),
        "raw_node_count": len(raw.get("nodes", [])),
        "raw_edge_count": len(raw.get("edges", [])),
    }


def unavailable_provenance() -> dict[str, Any]:
    return {
        "codegraph_version": "unavailable",
        "index_last_indexed": None,
        "index_state": "unavailable",
        "raw_export_sha256": None,
        "source_files_newer_than_index": None,
    }


def export_raw(project_path: Path) -> dict[str, Any]:
    codegraph = _find_codegraph()
    with tempfile.NamedTemporaryFile(
        mode="r", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            [codegraph, "export", "--format", "json", "--output", str(tmp_path)],
            cwd=str(project_path),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"codegraph export failed (exit {result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        with open(tmp_path, encoding="utf-8") as file:
            return json.load(file)
    finally:
        tmp_path.unlink(missing_ok=True)


def extract_nodes_and_edges(
    project_path: Path,
    *,
    include_provenance: bool = False,
    index_last_indexed: str | None = None,
    raw: dict[str, Any] | None = None,
) -> tuple[list[dict], list[dict], list[str]] | tuple[list[dict], list[dict], list[str], dict[str, Any]]:
    provenance = prepare_codegraph(project_path) if include_provenance else None
    raw = export_raw(project_path) if raw is None else raw
    if provenance is not None:
        provenance = finalize_codegraph_provenance(project_path, raw, provenance)
    elif index_last_indexed is not None:
        _assert_source_files_not_newer(project_path, raw, index_last_indexed)

    valid_kinds = schema.supported_structural_node_kinds()
    declared_kinds = schema.node_kinds()

    valid_nodes: list[dict] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    for node in raw.get("nodes", []):
        kind = node.get("kind")
        if kind not in valid_kinds:
            status = "declared but unsupported" if kind in declared_kinds else "unknown"
            warnings.append(f"{status} node kind {kind!r} (id={node.get('id')})")
            continue
        file_path = node.get("filePath", "")
        if file_path.startswith(".claude/worktrees/"):
            warnings.append(f"skipped worktree scratch copy: {file_path}")
            continue
        valid_nodes.append(node)
        seen_ids.add(node["id"])

    nodes_by_id = {node["id"]: node for node in valid_nodes}
    valid_edges: list[dict] = []
    for edge in raw.get("edges", []):
        source_id, target_id = edge.get("source"), edge.get("target")
        if source_id not in seen_ids or target_id not in seen_ids:
            warnings.append(
                f"edge kind={edge.get('kind')} references a node dropped/missing "
                f"(source={source_id!r}, target={target_id!r})"
            )
            continue

        raw_kind = edge.get("kind")
        edge_kind = _EDGE_KIND_MAP.get(raw_kind)
        if edge_kind is None:
            warnings.append(
                f"unknown raw edge kind {raw_kind!r} "
                f"(source={source_id!r}, target={target_id!r})"
            )
            continue

        source_kind = nodes_by_id[source_id]["kind"]
        target_kind = nodes_by_id[target_id]["kind"]
        if not schema.is_valid_edge(edge_kind, source_kind, target_kind):
            warnings.append(
                f"invalid edge kind={edge_kind!r} "
                f"(source={source_kind}:{source_id!r}, "
                f"target={target_kind}:{target_id!r})"
            )
            continue

        valid_edges.append({**edge, "kind": edge_kind})

    existing_edge_keys = {
        (edge["source"], edge["target"], edge["kind"])
        for edge in valid_edges
    }
    defining_target_kinds = {"function", "method", "struct", "class"}
    for edge in list(valid_edges):
        if edge["kind"] != "CONTAINS":
            continue
        source_kind = nodes_by_id[edge["source"]]["kind"]
        target_kind = nodes_by_id[edge["target"]]["kind"]
        if source_kind != "file" or target_kind not in defining_target_kinds:
            continue
        derived = {
            "source": edge["source"],
            "target": edge["target"],
            "kind": "DEFINES",
        }
        key = (derived["source"], derived["target"], derived["kind"])
        if key not in existing_edge_keys and schema.is_valid_edge(
            derived["kind"], source_kind, target_kind
        ):
            valid_edges.append(derived)
            existing_edge_keys.add(key)

    callable_kinds = {"function", "method"}
    for edge in list(valid_edges):
        if edge["kind"] != "CALLS" or edge["source"] == edge["target"]:
            continue
        source = nodes_by_id[edge["source"]]
        target = nodes_by_id[edge["target"]]
        source_kind = source["kind"]
        target_kind = target["kind"]
        if (
            source_kind not in callable_kinds
            or target_kind not in callable_kinds
            or not str(source.get("name", "")).startswith("test_")
        ):
            continue
        derived = {
            "source": edge["source"],
            "target": edge["target"],
            "kind": "TESTS",
        }
        key = (derived["source"], derived["target"], derived["kind"])
        if key not in existing_edge_keys and schema.is_valid_edge(
            derived["kind"], source_kind, target_kind
        ):
            valid_edges.append(derived)
            existing_edge_keys.add(key)

    source_nodes, source_edges = derive_source_graph(
        valid_nodes,
        valid_edges,
        project_path,
    )
    for node in source_nodes:
        if node["id"] not in seen_ids:
            valid_nodes.append(node)
            seen_ids.add(node["id"])
            nodes_by_id[node["id"]] = node
    for edge in source_edges:
        source_kind = nodes_by_id[edge["source"]]["kind"]
        target_kind = nodes_by_id[edge["target"]]["kind"]
        if schema.is_valid_edge(edge["kind"], source_kind, target_kind):
            valid_edges.append(edge)

    from emp_kg.source_scan import (
        enrich_file_nodes_with_source_text,
        enrich_nodes_with_source_text,
    )

    enrich_nodes_with_source_text(valid_nodes, project_path)
    enrich_file_nodes_with_source_text(valid_nodes, project_path)

    if provenance is not None:
        return valid_nodes, valid_edges, warnings, provenance
    return valid_nodes, valid_edges, warnings


# ==========================================
# توابع پارسر چندزبانه (Go, C/C++, Java)
# ==========================================

def parse_go_ast_nodes(file_path: str, source_code: str | None = None) -> list[dict[str, Any]]:
    """Parse Go source code using Tree-sitter and extract AST nodes."""
    if source_code is None:
        source_code = Path(file_path).read_text(encoding="utf-8")
    
    go_language = Language(tsgo.language())
    parser = Parser(go_language)
    tree = parser.parse(bytes(source_code, "utf-8"))

    extracted_nodes: list[dict[str, Any]] = []
    query_scm = """
    (function_declaration name: (identifier) @func.name) @func.def
    (method_declaration name: (field_identifier) @method.name) @method.def
    (type_declaration (type_spec name: (type_identifier) @struct.name)) @struct.def
    """
    query = Query(go_language, query_scm)
    cursor = QueryCursor(query)
    captures = cursor.captures(tree.root_node)

    for capture_name, nodes in captures.items():
        if capture_name in ("func.name", "method.name", "struct.name"):
            node_type = "function" if "func" in capture_name or "method" in capture_name else "struct"
            for node in nodes:
                node_text = source_code[node.start_byte:node.end_byte]
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                node_id = f"{file_path}:{node_type}:{node_text}:{start_line}"
                if not any(n["id"] == node_id for n in extracted_nodes):
                    extracted_nodes.append({
                        "id": node_id,
                        "kind": node_type,
                        "name": node_text,
                        "filePath": file_path,
                        "startLine": start_line,
                        "endLine": end_line,
                    })

    return extracted_nodes


def parse_cpp_ast_nodes(file_path: str, source_code: str | None = None) -> list[dict[str, Any]]:
    """Parse C/C++ source code using Tree-sitter and extract AST nodes."""
    if source_code is None:
        source_code = Path(file_path).read_text(encoding="utf-8")
        
    cpp_language = Language(tscpp.language())
    parser = Parser(cpp_language)
    tree = parser.parse(bytes(source_code, "utf-8"))

    extracted_nodes: list[dict[str, Any]] = []
    query_scm = """
    (function_definition declarator: (function_declarator declarator: (identifier) @func.name))
    (class_specifier name: (type_identifier) @class.name)
    (struct_specifier name: (type_identifier) @struct.name)
    """
    query = Query(cpp_language, query_scm)
    cursor = QueryCursor(query)
    captures = cursor.captures(tree.root_node)

    for capture_name, nodes in captures.items():
        node_type = "function" if "func" in capture_name else ("class" if "class" in capture_name else "struct")
        for node in nodes:
            node_text = source_code[node.start_byte:node.end_byte]
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            node_id = f"{file_path}:{node_type}:{node_text}:{start_line}"
            if not any(n["id"] == node_id for n in extracted_nodes):
                extracted_nodes.append({
                    "id": node_id,
                    "kind": node_type,
                    "name": node_text,
                    "filePath": file_path,
                    "startLine": start_line,
                    "endLine": end_line,
                })

    return extracted_nodes


def parse_java_ast_nodes(file_path: str, source_code: str | None = None) -> list[dict[str, Any]]:
    """Parse Java source code using Tree-sitter and extract AST nodes."""
    if source_code is None:
        source_code = Path(file_path).read_text(encoding="utf-8")
        
    java_language = Language(tsjava.language())
    parser = Parser(java_language)
    tree = parser.parse(bytes(source_code, "utf-8"))

    extracted_nodes: list[dict[str, Any]] = []
    query_scm = """
    (method_declaration name: (identifier) @method.name)
    (class_declaration name: (identifier) @class.name)
    (interface_declaration name: (identifier) @interface.name)
    """
    query = Query(java_language, query_scm)
    cursor = QueryCursor(query)
    captures = cursor.captures(tree.root_node)

    for capture_name, nodes in captures.items():
        node_type = "method" if "method" in capture_name else ("class" if "class" in capture_name else "interface")
        for node in nodes:
            node_text = source_code[node.start_byte:node.end_byte]
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            node_id = f"{file_path}:{node_type}:{node_text}:{start_line}"
            if not any(n["id"] == node_id for n in extracted_nodes):
                extracted_nodes.append({
                    "id": node_id,
                    "kind": node_type,
                    "name": node_text,
                    "filePath": file_path,
                    "startLine": start_line,
                    "endLine": end_line,
                })

    return extracted_nodes