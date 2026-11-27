"""Tests for schema vocabulary and implementation scope."""
from __future__ import annotations

import unittest

from emp_kg import schema


class TestImplementationScope(unittest.TestCase):
    def test_scope_separates_supported_and_planned_capabilities(self) -> None:
        self.assertIn("CALLS", schema.supported_edge_types())
        self.assertIn("decision_ref", schema.supported_semantic_node_kinds())
        self.assertIn("topic", schema.supported_semantic_node_kinds())
        self.assertIn("story", schema.supported_semantic_node_kinds())
        self.assertIn("DEFINES", schema.supported_edge_types())
        self.assertIn("TESTS", schema.supported_edge_types())
        self.assertIn("EXTENDS", schema.supported_edge_types())
        self.assertIn("THROWS", schema.supported_edge_types())
        self.assertIn("QUERIES", schema.supported_edge_types())
        self.assertIn("exception_type", schema.supported_structural_node_kinds())
        self.assertIn("table", schema.supported_structural_node_kinds())
        self.assertIn("IMPLEMENTS", schema.supported_edge_types())
        self.assertIn("ROUTE_HANDLER", schema.supported_edge_types())
        self.assertEqual(schema.implementation_scope()["planned_edge_types"], [])

    def test_affects_supported_sources_are_explicit(self) -> None:
        self.assertEqual(
            schema.supported_edge_sources("AFFECTS"),
            {"decision_ref", "story", "topic"},
        )
        self.assertEqual(schema.supported_edge_sources("CALLS"), set())

    def test_full_vocabulary_remains_available(self) -> None:
        self.assertIn("story", schema.node_kinds())
        self.assertIn("DEFINES", schema.edge_types())
        self.assertTrue(schema.is_valid_edge("AFFECTS", "story", "file"))


if __name__ == "__main__":
    unittest.main()
