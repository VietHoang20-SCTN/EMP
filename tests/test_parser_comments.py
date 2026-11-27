"""Tests for shared multiline comment extraction."""
from __future__ import annotations

import unittest

from emp_kg.parser_comments import extract_comments


class TestComments(unittest.TestCase):
    def test_extracts_line_and_block_comments(self) -> None:
        comments = extract_comments([
            "// route documentation",
            "const text = '/* not a comment */';",
            "/**",
            " * Refresh credentials.",
            " * @param user account owner",
            " */",
            "/* storage note */",
        ])

        self.assertEqual(comments[0], (1, 1, "route documentation"))
        self.assertEqual(comments[1], (3, 6, "Refresh credentials. @param user account owner"))
        self.assertEqual(comments[2], (7, 7, "storage note"))

    def test_extracts_html_comments(self) -> None:
        self.assertEqual(
            extract_comments(["<!-- page purpose", "     for login -->"]),
            [(1, 2, "page purpose for login")],
        )


if __name__ == "__main__":
    unittest.main()
