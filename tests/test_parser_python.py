"""Tests for AST-backed Python docstring extraction."""
from __future__ import annotations

import unittest

from emp_kg.parser_python import extract_python_docstrings


class TestPythonDocstrings(unittest.TestCase):
    def test_extracts_module_class_and_function_docstrings(self) -> None:
        source = '''"""Module purpose."""

class Session:
    """Session lifecycle."""

    def refresh(self):
        """Refresh active credentials."""
        return True
'''
        spans = extract_python_docstrings(source)

        self.assertEqual(
            [(span.start_line, span.text) for span in spans],
            [(1, "Module purpose."), (4, "Session lifecycle."), (7, "Refresh active credentials.")],
        )

    def test_returns_empty_for_invalid_python(self) -> None:
        self.assertEqual(extract_python_docstrings("def broken(:\n"), [])


if __name__ == "__main__":
    unittest.main()
