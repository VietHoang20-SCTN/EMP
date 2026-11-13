from pathlib import Path
import pytest

from emp_kg.structural import (
    parse_go_ast_nodes,
    parse_cpp_ast_nodes,
    parse_java_ast_nodes,
)
from emp_kg.source_scan import derive_source_graph, _language_for_file


# 1. Test precise file extension dispatcher for new languages
def test_language_dispatcher_extensions():
    assert _language_for_file("main.go") == "generic"
    assert _language_for_file("core.cpp") == "generic"
    assert _language_for_file("header.h") == "generic"
    assert _language_for_file("App.java") == "java"
    assert _language_for_file("Service.cs") == "java"


# 2. Rigorous test for Go parser (extracting functions and structs)
def test_go_parser_functions_and_structs(tmp_path: Path):
    go_file = tmp_path / "main.go"
    go_file.write_text("""
    package main

    type User struct {
        Name string
    }

    func ProcessData(x int) string {
        return "ok"
    }
    """, encoding="utf-8")
    
    nodes = parse_go_ast_nodes(str(go_file))
    names = {n["name"] for n in nodes}
    kinds = {n["kind"] for n in nodes}

    assert "User" in names
    assert "ProcessData" in names
    assert "struct" in kinds
    assert "function" in kinds


# 3. Test C++ parser (extracting classes and functions)
def test_cpp_parser_classes_and_functions(tmp_path: Path):
    cpp_file = tmp_path / "main.cpp"
    cpp_file.write_text("""
    class Engine {
    public:
        void start();
    };

    int main() {
        return 0;
    }
    """, encoding="utf-8")
    
    nodes = parse_cpp_ast_nodes(str(cpp_file))
    names = {n["name"] for n in nodes}
    
    assert "Engine" in names
    assert "main" in names
    assert any(n["kind"] == "class" for n in nodes)
    assert any(n["kind"] == "function" for n in nodes)


# 4. Test C++ inheritance edge derivation (EXTENDS)
def test_cpp_edge_derivation_inheritance(tmp_path: Path):
    cpp_file = tmp_path / "derived.cpp"
    cpp_file.write_text("class Base {};\nclass Derived : public Base {};", encoding="utf-8")

    nodes = [
        {"id": "file:derived.cpp", "kind": "file", "name": "derived.cpp", "filePath": "derived.cpp"},
        {"id": "c1", "kind": "class", "name": "Base", "filePath": "derived.cpp", "startLine": 1, "endLine": 1},
        {"id": "c2", "kind": "class", "name": "Derived", "filePath": "derived.cpp", "startLine": 2, "endLine": 2},
    ]
    
    synthetic, edges = derive_source_graph(nodes, [], tmp_path)
    
    extends_edges = [e for e in edges if e["kind"] == "EXTENDS"]
    assert len(extends_edges) >= 1
    assert any(e["source"] == "c2" and e["target"] == "c1" for e in extends_edges)


# 5. Rigorous test for Java parser (classes and methods)
def test_java_parser_classes_and_methods(tmp_path: Path):
    java_file = tmp_path / "Account.java"
    java_file.write_text("""
    public class Account {
        private double balance;
        
        public void deposit(double amount) {
            balance += amount;
        }
    }
    """, encoding="utf-8")
    
    nodes = parse_java_ast_nodes(str(java_file))
    names = {n["name"] for n in nodes}
    kinds = {n["kind"] for n in nodes}

    assert "Account" in names
    assert "deposit" in names
    assert "class" in kinds
    assert "method" in kinds


# 6. Test Java parser (interfaces)
def test_java_parser_interfaces(tmp_path: Path):
    java_file = tmp_path / "Repository.java"
    java_file.write_text("""
    public interface Repository {
        void save();
    }
    """, encoding="utf-8")
    
    nodes = parse_java_ast_nodes(str(java_file))
    interface_nodes = [n for n in nodes if n["kind"] == "interface"]
    
    assert len(interface_nodes) == 1
    assert interface_nodes[0]["name"] == "Repository"


# 7. Test Java inheritance edge derivation (EXTENDS)
def test_java_edge_derivation_extends(tmp_path: Path):
    java_file = tmp_path / "Child.java"
    java_file.write_text("class Parent {}\nclass Child extends Parent {}", encoding="utf-8")

    nodes = [
        {"id": "f1", "kind": "file", "name": "Child.java", "filePath": "Child.java"},
        {"id": "p_node", "kind": "class", "name": "Parent", "filePath": "Child.java", "startLine": 1, "endLine": 1},
        {"id": "c_node", "kind": "class", "name": "Child", "filePath": "Child.java", "startLine": 2, "endLine": 2},
    ]

    _, edges = derive_source_graph(nodes, [], tmp_path)
    extends_edges = [e for e in edges if e["kind"] == "EXTENDS" and e["source"] == "c_node"]
    
    assert len(extends_edges) == 1
    assert extends_edges[0]["target"] == "p_node"


# 8. Test Java implementation edge derivation (IMPLEMENTS)
def test_java_edge_derivation_implements(tmp_path: Path):
    java_file = tmp_path / "Impl.java"
    java_file.write_text("interface Serializable {}\nclass MyClass implements Serializable {}", encoding="utf-8")

    nodes = [
        {"id": "f2", "kind": "file", "name": "Impl.java", "filePath": "Impl.java"},
        {"id": "iface_node", "kind": "interface", "name": "Serializable", "filePath": "Impl.java", "startLine": 1, "endLine": 1},
        {"id": "cls_node", "kind": "class", "name": "MyClass", "filePath": "Impl.java", "startLine": 2, "endLine": 2},
    ]

    _, edges = derive_source_graph(nodes, [], tmp_path)
    implements_edges = [e for e in edges if e["kind"] == "IMPLEMENTS"]
    
    assert len(implements_edges) == 1
    assert implements_edges[0]["source"] == "cls_node"
    assert implements_edges[0]["target"] == "iface_node"


# 9. Test robustness and fallback behavior against malformed syntax
def test_parsers_with_syntax_errors(tmp_path: Path):
    go_file = tmp_path / "bad.go"
    go_file.write_text("func broken( { syntax error", encoding="utf-8")
    cpp_file = tmp_path / "bad.cpp"
    cpp_file.write_text("class { incomplete", encoding="utf-8")
    java_file = tmp_path / "bad.java"
    java_file.write_text("public class {", encoding="utf-8")

    assert isinstance(parse_go_ast_nodes(str(go_file)), list)
    assert isinstance(parse_cpp_ast_nodes(str(cpp_file)), list)
    assert isinstance(parse_java_ast_nodes(str(java_file)), list)


# 10. Test that unsupported files/extensions are ignored cleanly without false edges
def test_derive_graph_ignores_unsupported_extensions(tmp_path: Path):
    txt_file = tmp_path / "readme.txt"
    txt_file.write_text("just a text file with class MyClass", encoding="utf-8")

    nodes = [
        {"id": "txt1", "kind": "file", "name": "readme.txt", "filePath": "readme.txt"},
    ]
    
    synthetic, edges = derive_source_graph(nodes, [], tmp_path)
    assert len(edges) == 0

