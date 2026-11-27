"""Python docstring extraction using the standard-library AST."""
from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class DocstringSpan:
    start_line: int
    end_line: int
    text: str


def extract_python_docstrings(source_text: str) -> list[DocstringSpan]:
    """Return module, class, function, and method docstrings with line spans."""
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return []

    spans: list[DocstringSpan] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        text = first.value.value.strip()
        if not text:
            continue
        spans.append(DocstringSpan(
            start_line=getattr(first, "lineno", 1),
            end_line=getattr(first, "end_lineno", getattr(first, "lineno", 1)),
            text=text,
        ))
    return sorted(spans, key=lambda span: (span.start_line, span.end_line, span.text))
