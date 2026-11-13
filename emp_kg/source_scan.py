"""Derive graph records from source patterns using Tree-sitter with Regex fallback."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, List, Tuple, Dict, Optional

from emp_kg import schema
from emp_kg.parser_comments import extract_comments
from emp_kg.parser_python import extract_python_docstrings


# C++ and Java Inheritance Patterns
_CPP_CLASS_RE = re.compile(r"class\s+(?P<child>[A-Za-z_]\w*)\s*:\s*(?:public|protected|private)?\s*(?P<parent>[A-Za-z_][\w.]*)")
_JAVA_EXTENDS_RE = re.compile(r"class\s+(?P<child>[A-Za-z_]\w*)\s+extends\s+(?P<parent>[A-Za-z_][\w.]*)")



# Check Tree-sitter availability
HAS_TREE_SITTER = False
try:
    import tree_sitter
    from tree_sitter import Language, Parser
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False

# Regex Fallback Patterns
_CLASS_RE = re.compile(r"^\s*class\s+(?P<child>[A-Za-z_]\w*)\s*\(\s*(?P<parent>[A-Za-z_][\w.]*)")
_RAISE_RE = re.compile(r"\braise\s+(?P<exception>[A-Za-z_][\w.]*)")
_EXECUTE_RE = re.compile(r"\.execute\(\s*(?:f)?(?P<quote>['\"])(?P<sql>.*?)(?P=quote)", re.IGNORECASE)
_TABLE_PATTERNS = (
    re.compile(r"\bFROM\s+([A-Za-z_]\w*)", re.IGNORECASE),
    re.compile(r"\bINSERT\s+INTO\s+([A-Za-z_]\w*)", re.IGNORECASE),
    re.compile(r"\bUPDATE\s+([A-Za-z_]\w*)", re.IGNORECASE),
    re.compile(r"\bDELETE\s+FROM\s+([A-Za-z_]\w*)", re.IGNORECASE),
)
_IMPLEMENTS_RE = re.compile(r"class\s+(\w+)\s+implements\s+([\w\s,]+)")
_PYTHON_ROUTE_RE = re.compile(r'@(\w+)\.(get|post|put|delete|patch)\s*\(\s*"([^"]+)"')
_JS_ROUTE_RE = re.compile(r'(\w+)\.(get|post|put|delete|patch)\s*\(\s*"([^"]+)"\s*,\s*(\w+)')

_SOURCE_BODY_LIMIT = 2000
_FILE_SOURCE_LIMIT = 32000
_CHUNK_TARGET = 8000
_CHUNK_OVERLAP = 1000
_CHUNK_HARD_LIMIT = 12000


def _language_for_file(file_path: str) -> str:
    """Return simplified language family for parser dispatch."""
    if file_path.endswith(".py"):
        return "python"
    if any(file_path.endswith(ext) for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs")):
        return "javascript"
    if file_path.endswith(".rs"):
        return "rust"
    if any(file_path.endswith(ext) for ext in (".java", ".cs")):
        return "java"
    return "generic"


def extract_docstrings_and_body(source_text, start_line, end_line, file_path="", limit=_SOURCE_BODY_LIMIT):
    """Extract bounded docstring/comment text and full source span using language-appropriate parsers."""
    lines = source_text.splitlines()
    start = max(0, start_line - 1)
    end = min(len(lines), end_line)
    span = lines[start:end]
    body = "\n".join(span).strip()[:limit]

    language = _language_for_file(file_path)

    if language in {"python", "generic"}:
        docstrings = extract_python_docstrings(source_text)
        relevant = [
            span
            for span in docstrings
            if span.start_line >= start_line and span.end_line <= end_line
        ]
        if relevant:
            text = " ".join(span.text for span in relevant)
            return text[:limit], body

    all_comments = extract_comments(source_text.splitlines())
    relevant = [(s, e, t) for s, e, t in all_comments if s >= start_line and e <= end_line]
    if relevant:
        text = " ".join(t for _, _, t in relevant)
        return text[:limit], body

    return "", body


def enrich_nodes_with_source_text(nodes, project_path):
    """Add bounded source bodies and comments/docstrings to structural nodes."""
    cache = {}
    for node in nodes:
        if node.get("kind") not in {"file", "function", "method", "class", "struct", "interface"}:
            continue
        file_path = node.get("filePath")
        if not isinstance(file_path, str) or not file_path:
            continue
        if file_path not in cache:
            try:
                cache[file_path] = (project_path / file_path).read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                cache[file_path] = None
        source = cache[file_path]
        if source is None:
            continue
        lines = source.splitlines()
        start = node.get("startLine", 1)
        end = node.get("endLine", len(lines))
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        docstring, body = extract_docstrings_and_body(
            source, start, end, file_path=file_path
        )
        if docstring:
            node["docstring"] = docstring
        if body:
            node["source_body"] = body


def enrich_file_nodes_with_source_text(nodes, project_path, limit=_FILE_SOURCE_LIMIT):
    """Add bounded whole-file text to every indexed nonbinary file node."""
    for node in nodes:
        if node.get("kind") != "file":
            continue
        file_path = node.get("filePath")
        if not isinstance(file_path, str) or not file_path:
            continue
        try:
            raw = (project_path / file_path).read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:8192]:
            continue
        text = raw.decode("utf-8", errors="replace")
        if text:
            node["source_text"] = text[:limit]


def collect_source_chunks(nodes, project_path, target=_CHUNK_TARGET, overlap=_CHUNK_OVERLAP, hard_limit=_CHUNK_HARD_LIMIT):
    """Return deterministic overlapping source chunks for indexed file nodes."""
    if target <= 0 or overlap < 0 or overlap >= target or hard_limit < target:
        raise ValueError("invalid source chunk limits")
    chunks = []
    for node in nodes:
        if node.get("kind") != "file":
            continue
        node_id = node.get("id")
        file_path = node.get("filePath")
        if not isinstance(node_id, str) or not isinstance(file_path, str) or not file_path:
            continue
        try:
            raw = (project_path / file_path).read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:8192]:
            continue
        lines = raw.decode("utf-8", errors="replace").splitlines(keepends=True)
        if not lines:
            continue
        ordinal = 0
        start = 0
        while start < len(lines):
            if len(lines[start]) > hard_limit:
                line = lines[start]
                char_start = 0
                while char_start < len(line):
                    text = line[char_start:char_start + target]
                    chunks.append({
                        "file_node_id": node_id,
                        "ordinal": ordinal,
                        "start_line": start + 1,
                        "end_line": start + 1,
                        "text": text,
                        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    })
                    ordinal += 1
                    if char_start + target >= len(line):
                        break
                    char_start += target - overlap
                start += 1
                continue
            end = start
            length = 0
            while end < len(lines):
                line_length = len(lines[end])
                if end > start and length + line_length > target:
                    break
                length += line_length
                end += 1
                if length >= target:
                    break
            text = "".join(lines[start:end])
            chunks.append({
                "file_node_id": node_id,
                "ordinal": ordinal,
                "start_line": start + 1,
                "end_line": end,
                "text": text,
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            })
            ordinal += 1
            if end >= len(lines):
                break
            retained = 0
            next_start = end
            while next_start > start and retained < overlap:
                next_start -= 1
                retained += len(lines[next_start])
            start = next_start if next_start > start else end
    return chunks


def _synthetic_id(kind, name):
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    return f"{kind}:derived:{digest}"


def _synthetic_node(kind, name):
    node_id = _synthetic_id(kind, name)
    return {"id": node_id, "kind": kind, "name": name, "qualifiedName": name, "filePath": "", "derived_from": "source_scan"}


def _node_index(nodes):
    index = {}
    for node in nodes:
        kind, name = node.get("kind"), node.get("name")
        if isinstance(kind, str) and isinstance(name, str):
            index.setdefault((kind, name), []).append(node)
    return index


def _callable_at_line(nodes, file_path, line):
    candidates = []
    for node in nodes:
        if node.get("kind") not in {"function", "method"} or node.get("filePath") != file_path:
            continue
        start, end = node.get("startLine"), node.get("endLine")
        if isinstance(start, int) and isinstance(end, int) and start <= line <= end:
            candidates.append(node)
    return min(candidates, key=lambda n: n["endLine"] - n["startLine"]) if candidates else None


def _class_at_line(nodes, file_path, line, name):
    for node in nodes:
        if node.get("kind") != "class" or node.get("filePath") != file_path or node.get("name") != name:
            continue
        start, end = node.get("startLine"), node.get("endLine")
        if not isinstance(start, int) or not isinstance(end, int) or start <= line <= end:
            return node
    return None


def _table_name(sql):
    for pattern in _TABLE_PATTERNS:
        match = pattern.search(sql)
        if match:
            return match.group(1)
    return None


def derive_source_graph(valid_nodes, valid_edges, project_path):
    """Return synthetic nodes and source-derived schema-valid edges using AST or fallback Regex."""
    nodes_by_file = {}
    for node in valid_nodes:
        path = node.get("filePath")
        if isinstance(path, str) and path.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".rs",".go", ".cpp", ".cc", ".c", ".h", ".java")):
            nodes_by_file.setdefault(path, []).append(node)

    by_kind_name = _node_index(valid_nodes)
    synthetic_nodes = []
    synthetic_by_key = {}
    existing_edges = {(e.get("source"), e.get("target"), e.get("kind")) for e in valid_edges}
    derived_edges = []

    def target_node(kind, name):
        key = (kind, name)
        if by_kind_name.get(key):
            return by_kind_name[key][0]
        if key not in synthetic_by_key:
            node = _synthetic_node(kind, name)
            synthetic_by_key[key] = node
            synthetic_nodes.append(node)
            by_kind_name.setdefault(key, []).append(node)
        return synthetic_by_key[key]

    def add_edge(source, target, kind):
        sid, tid = source.get("id"), target.get("id")
        sk, tk = source.get("kind"), target.get("kind")
        if not all(isinstance(v, str) for v in (sid, tid, sk, tk)):
            return
        key = (sid, tid, kind)
        if key in existing_edges or not schema.is_valid_edge(kind, sk, tk):
            return
        existing_edges.add(key)
        derived_edges.append({"source": sid, "target": tid, "kind": kind})

    for file_path, nodes in nodes_by_file.items():
        try:
            content = (project_path / file_path).read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
        except OSError:
            continue

        for line_number, line in enumerate(lines, start=1):
            class_match = _CLASS_RE.match(line)
            if class_match:
                child = _class_at_line(nodes, file_path, line_number, class_match.group("child"))
                if child:
                    parent_name = class_match.group("parent")
                    parent = by_kind_name.get(("class", parent_name)) or by_kind_name.get(("struct", parent_name))
                    add_edge(child, parent[0] if parent else target_node("class", parent_name), "EXTENDS")
            # بررسی ارث‌بری C++
            cpp_class_match = _CPP_CLASS_RE.match(line)
            if cpp_class_match:
                child = _class_at_line(nodes, file_path, line_number, cpp_class_match.group("child"))
                if child:
                    parent_name = cpp_class_match.group("parent")
                    parent = by_kind_name.get(("class", parent_name)) or by_kind_name.get(("struct", parent_name))
                    add_edge(child, parent[0] if parent else target_node("class", parent_name), "EXTENDS")

            # بررسی ارث‌بری Java (extends)
            java_extends_match = _JAVA_EXTENDS_RE.match(line)
            if java_extends_match:
                child = _class_at_line(nodes, file_path, line_number, java_extends_match.group("child"))
                if child:
                    parent_name = java_extends_match.group("parent")
                    parent = by_kind_name.get(("class", parent_name))
                    add_edge(child, parent[0] if parent else target_node("class", parent_name), "EXTENDS")
 
                    
                    
                    

            implements_match = _IMPLEMENTS_RE.search(line)
            if implements_match:
                child = _class_at_line(nodes, file_path, line_number, implements_match.group(1))
                if child:
                    for iface in implements_match.group(2).split(","):
                        iface_name = iface.strip()
                        if iface_name:
                            interface = by_kind_name.get(("interface", iface_name))
                            add_edge(child, interface[0] if interface else target_node("interface", iface_name), "IMPLEMENTS")

            callable_node = _callable_at_line(nodes, file_path, line_number)
            if callable_node:
                raise_match = _RAISE_RE.search(line)
                if raise_match:
                    exc_name = raise_match.group("exception")
                    exc = by_kind_name.get(("exception_type", exc_name))
                    add_edge(callable_node, exc[0] if exc else target_node("exception_type", exc_name), "THROWS")

                execute_match = _EXECUTE_RE.search(line)
                if execute_match:
                    tname = _table_name(execute_match.group("sql"))
                    if tname:
                        table = by_kind_name.get(("table", tname))
                        add_edge(callable_node, table[0] if table else target_node("table", tname), "QUERIES")

            python_route = _PYTHON_ROUTE_RE.search(line)
            if python_route:
                method, path = python_route.group(2).upper(), python_route.group(3)
                route_name = f"{method} {path}"
                next_line = lines[line_number] if line_number < len(lines) else ""
                handler_match = re.match(r"\s*def\s+(\w+)", next_line)
                if handler_match:
                    handler = None
                    for n in nodes:
                        if n.get("name") == handler_match.group(1) and n.get("filePath") == file_path:
                            handler = n
                            break
                    if not handler:
                        handler = next((n for n in valid_nodes if n.get("name") == handler_match.group(1)), None)
                    route = target_node("route", route_name)
                    if handler:
                        add_edge(route, handler, "ROUTE_HANDLER")

            js_route = _JS_ROUTE_RE.search(line)
            if js_route:
                method, path, handler_name = js_route.group(2).upper(), js_route.group(3), js_route.group(4)
                route_name = f"{method} {path}"
                handler = next((n for n in nodes if n.get("name") == handler_name and n.get("filePath") == file_path), None)
                if not handler:
                    handler = next((n for n in valid_nodes if n.get("name") == handler_name), None)
                route = target_node("route", route_name)
                if handler:
                    add_edge(route, handler, "ROUTE_HANDLER")

    return synthetic_nodes, derived_edges