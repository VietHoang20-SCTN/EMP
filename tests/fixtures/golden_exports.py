"""Deterministic CodeGraph raw exports for golden integration builds."""
from __future__ import annotations


def node(node_id: str, kind: str, name: str, file_path: str, qualified: str | None = None) -> dict:
    return {
        "id": node_id,
        "kind": kind,
        "name": name,
        "filePath": file_path,
        "qualifiedName": qualified or name,
    }


FILE_APP = node("file:src/app.py", "file", "app.py", "src/app.py", "src/app.py")
FILE_SERVICE = node("file:src/service.py", "file", "service.py", "src/service.py", "src/service.py")
IMPORT_SERVICE = node("import:service", "import", "service", "src/app.py")
FUNCTION_MAIN = node("function:main", "function", "main", "src/app.py", "app.main")
FUNCTION_LEGACY = node("function:legacy", "function", "legacy", "src/app.py", "app.legacy")
CONSTANT_CONFIG = node("constant:config", "constant", "CONFIG", "src/app.py", "app.CONFIG")
CLASS_SERVICE = node("class:service", "class", "Service", "src/service.py", "service.Service")
FUNCTION_RUN = node("function:service.run", "function", "run", "src/service.py", "service.run")
FUNCTION_EXECUTE = node("function:service.execute", "function", "execute", "src/service.py", "service.execute")


def edge(source: str, target: str, kind: str) -> dict:
    return {"source": source, "target": target, "kind": kind}


BASE_NODES = [
    FILE_APP,
    FILE_SERVICE,
    IMPORT_SERVICE,
    FUNCTION_MAIN,
    CONSTANT_CONFIG,
    CLASS_SERVICE,
]


def build_one() -> dict:
    return {
        "nodes": [*BASE_NODES, FUNCTION_LEGACY, FUNCTION_RUN],
        "edges": [
            edge("file:src/app.py", "import:service", "imports"),
            edge("file:src/app.py", "function:main", "contains"),
            edge("file:src/app.py", "constant:config", "contains"),
            edge("file:src/service.py", "class:service", "contains"),
            edge("class:service", "function:service.run", "contains"),
            edge("function:main", "function:service.run", "calls"),
            edge("function:main", "constant:config", "references"),
            edge("function:main", "class:service", "instantiates"),
        ],
    }


def build_two() -> dict:
    """Remove legacy while leaving the runtime flow unchanged."""
    return {
        "nodes": [*BASE_NODES, FUNCTION_RUN],
        "edges": [
            edge("file:src/app.py", "import:service", "imports"),
            edge("file:src/app.py", "function:main", "contains"),
            edge("file:src/app.py", "constant:config", "contains"),
            edge("file:src/service.py", "class:service", "contains"),
            edge("class:service", "function:service.run", "contains"),
            edge("function:main", "function:service.run", "calls"),
            edge("function:main", "constant:config", "references"),
            edge("function:main", "class:service", "instantiates"),
        ],
    }


def build_three() -> dict:
    """Rename run to execute and restore call/containment relationships."""
    return {
        "nodes": [*BASE_NODES, FUNCTION_EXECUTE],
        "edges": [
            edge("file:src/app.py", "import:service", "imports"),
            edge("file:src/app.py", "function:main", "contains"),
            edge("file:src/app.py", "constant:config", "contains"),
            edge("file:src/service.py", "class:service", "contains"),
            edge("class:service", "function:service.execute", "contains"),
            edge("function:main", "function:service.execute", "calls"),
            edge("function:main", "constant:config", "references"),
            edge("function:main", "class:service", "instantiates"),
        ],
    }
