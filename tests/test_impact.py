"""Tests for Impact Analysis Engine."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from emp_kg import impact, store


class TestImpact(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db = Path(self.tmpdir.name) / "graph.db"
        conn = store.init_db(self.db)
        store.save_structural_nodes(conn, [
            {"id": "file:auth.py", "kind": "file", "name": "auth.py"},
            {"id": "function:login", "kind": "function", "name": "login"},
            {"id": "function:validate_token", "kind": "function", "name": "validate_token"},
            {"id": "function:test_login", "kind": "function", "name": "test_login"},
        ])
        store.save_decision_nodes(conn, [{"id": "0001", "title": "Auth Decision"}])
        store.save_edges(conn, [
            {"source": "file:auth.py", "target": "function:login", "kind": "DEFINES"},
            {"source": "function:login", "target": "function:validate_token", "kind": "CALLS"},
            {"source": "function:test_login", "target": "function:login", "kind": "TESTS"},
            {"source": "decision_ref:0001", "target": "function:login", "kind": "AFFECTS"},
        ])
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_login_blast_radius_has_semantic_and_test_impact(self) -> None:
        report = impact.analyze_impact(self.db, impact.ImpactRequest("function:login"))
        direct_ids = {item["id"] for item in report.directly_affected}
        self.assertIn("decision_ref:0001", direct_ids)
        self.assertIn("function:test_login", direct_ids)
        self.assertIn("function:validate_token", direct_ids)
        self.assertEqual(report.risk_level, "high")
        self.assertIn("decision_ref:0001", report.affected_decisions)
        self.assertIn("function:test_login", report.recommended_tests)

    def test_invalid_change_type_and_depth_rejected(self) -> None:
        with self.assertRaises(ValueError):
            impact.analyze_impact(self.db, impact.ImpactRequest("function:login", change_type="bad"))
        with self.assertRaises(ValueError):
            impact.analyze_impact(self.db, impact.ImpactRequest("function:login", depth=-1))

    def test_unknown_target_has_low_impact(self) -> None:
        report = impact.analyze_impact(self.db, impact.ImpactRequest("function:missing"))
        self.assertEqual(report.risk_level, "low")
        self.assertFalse(report.directly_affected)

    def test_direction_out_excludes_incoming_decision(self) -> None:
        report = impact.analyze_impact(self.db, impact.ImpactRequest("function:login", direction="out"))
        ids = {item["id"] for item in report.directly_affected}
        self.assertIn("function:validate_token", ids)
        self.assertNotIn("decision_ref:0001", ids)


if __name__ == "__main__":
    unittest.main()
