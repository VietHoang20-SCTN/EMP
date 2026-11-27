"""Tests for task-specific context assembly."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from emp_kg import context, store

#Salar Changes
from emp_kg.context import reciprocal_rank_fusion

class TestContextBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "graph.db"
        conn = store.init_db(self.db_path)
        store.save_structural_nodes(conn, [
            {"id": "file:auth.py", "kind": "file", "name": "auth.py", "filePath": "auth.py"},
            {"id": "function:login", "kind": "function", "name": "login", "filePath": "auth.py", "signature": "(user)"},
            {"id": "function:validate_token", "kind": "function", "name": "validate_token", "filePath": "auth.py"},
            {"id": "function:unrelated", "kind": "function", "name": "unrelated", "filePath": "other.py"},
        ])
        store.save_decision_nodes(conn, [
            {
                "id": "0001",
                "title": "Login architecture",
                "source_path": "docs/decisions/0001-login.md",
            },
        ])
        store.save_edges(conn, [
            {"source": "file:auth.py", "target": "function:login", "kind": "DEFINES"},
            {"source": "function:login", "target": "function:validate_token", "kind": "CALLS"},
            {"source": "decision_ref:0001", "target": "function:login", "kind": "AFFECTS"},
        ])
        store.save_affects_evidence(conn, [
            {
                "source": "decision_ref:0001",
                "target": "function:login",
                "kind": "AFFECTS",
                "reference": "login",
                "source_path": "docs/decisions/0001-login.md",
                "source_lines": {"start": 12, "end": 14},
                "resolution": "unique_global_name",
                "confidence": 1.0,
            },
            {
                "source": "decision_ref:0001",
                "target": "function:login",
                "kind": "AFFECTS",
                "reference": "function:login",
                "source_path": "docs/decisions/0001-login.md",
                "source_lines": {"start": 20, "end": 20},
                "resolution": "exact_node_id",
                "confidence": 1.0,
            },
        ])
        store.rebuild_fts(conn)
        conn.commit()
        conn.close()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_seed_and_neighbor_expansion(self) -> None:
        result = context.build_context(
            context.ContextRequest("login", lane="normal"), self.db_path
        )

        self.assertEqual(result.lane, "normal")
        self.assertIn("function:login", {fact.node_id for fact in result.facts})
        self.assertTrue(all(fact.kind for fact in result.facts))
        self.assertIn("function:validate_token", {fact.node_id for fact in result.facts})

    def test_content_search_retrieves_decision_context(self) -> None:
        conn = store.init_db(self.db_path)
        store.save_decision_nodes(conn, [{
            "id": "0002",
            "title": "Generic decision",
            "context": "Authorization needs tenant isolation.",
        }])
        store.rebuild_fts(conn)
        conn.commit()
        conn.close()

        result = context.build_context(
            context.ContextRequest("authorization", lane="tiny"), self.db_path
        )
        self.assertIn("decision_ref:0002", {fact.node_id for fact in result.facts})

    def test_affected_file_falls_back_to_basename(self) -> None:
        conn = store.init_db(self.db_path)
        store.save_structural_nodes(conn, [{
            "id": "file:src/auth.py",
            "kind": "file",
            "name": "auth.py",
            "filePath": "auth.py",
        }])
        store.rebuild_fts(conn)
        conn.commit()
        conn.close()

        result = context.build_context(
            context.ContextRequest(
                "unrelated", lane="tiny", affected_files=["src/auth.py"]
            ),
            self.db_path,
        )
        facts = {fact.node_id: fact for fact in result.facts}
        self.assertIn("file:src/auth.py", facts)
        self.assertEqual(facts["file:src/auth.py"].provenance.matched_terms, ["auth.py"])

    def test_two_character_identifiers_are_searchable(self) -> None:
        conn = store.init_db(self.db_path)
        store.save_structural_nodes(conn, [
            {"id": "function:db", "kind": "function", "name": "db"},
            {"id": "function:io", "kind": "function", "name": "io"},
            {"id": "function:os", "kind": "function", "name": "os"},
            {"id": "function:id", "kind": "function", "name": "id"},
        ])
        store.rebuild_fts(conn)
        conn.commit()
        conn.close()

        result = context.build_context(
            context.ContextRequest("db io os id x", lane="tiny"), self.db_path
        )
        self.assertTrue({"function:db", "function:io", "function:os", "function:id"}.issubset(
            {fact.node_id for fact in result.facts}
        ))
        self.assertNotIn("x", context._keywords("db io os id x"))

    def test_context_direction_filters_neighbor_expansion(self) -> None:
        expected_presence = {
            "out": {"function:validate_token": True, "file:auth.py": False},
            "in": {"function:validate_token": False, "file:auth.py": True},
            "both": {"function:validate_token": True, "file:auth.py": True},
        }
        for direction, expectations in expected_presence.items():
            with self.subTest(direction=direction):
                result = context.build_context(
                    context.ContextRequest("login", lane="tiny", direction=direction),
                    self.db_path,
                )
                facts = {fact.node_id: fact for fact in result.facts}
                for node_id, present in expectations.items():
                    self.assertEqual(node_id in facts, present)
                for fact in facts.values():
                    for step in fact.provenance.path:
                        self.assertEqual(step.direction, direction if direction != "both" else step.direction)

        with self.assertRaises(ValueError):
            context.ContextRequest("login", direction="sideways")

    def test_ranked_search_returns_bm25_order_and_canonical_ids(self) -> None:
        conn = store.open_readonly(self.db_path)
        try:
            matches = store.query_by_name_ranked(conn, '"login"')
            legacy = store.query_by_name(conn, '"login"')
        finally:
            conn.close()

        self.assertTrue(matches)
        self.assertEqual(
            {node["id"] for node in matches},
            {node["id"] for node in legacy},
        )
        self.assertTrue(all("_bm25_rank" in node for node in matches))
        self.assertEqual(
            [node["_bm25_rank"] for node in matches],
            sorted(node["_bm25_rank"] for node in matches),
        )
        self.assertNotIn("_bm25_rank", legacy[0])

    def test_bm25_scores_are_deterministic_and_composed_with_graph_score(self) -> None:
        first = context.build_context(
            context.ContextRequest("login architecture", lane="normal"), self.db_path
        )
        second = context.build_context(
            context.ContextRequest("login architecture", lane="normal"), self.db_path
        )
        first_scores = [(fact.node_id, fact.relevance_score) for fact in first.facts]
        second_scores = [(fact.node_id, fact.relevance_score) for fact in second.facts]
        self.assertEqual(first_scores, second_scores)
        self.assertEqual(first_scores[0][0], "decision_ref:0001")
        self.assertGreaterEqual(first_scores[0][1], first_scores[-1][1])

    def test_semantic_term_expansion_is_bounded_and_deterministic(self) -> None:
        first = context._expand_semantic_terms("authentication")
        second = context._expand_semantic_terms("authentication")

        self.assertEqual(first, second)
        self.assertEqual(first[0], ("authentication", False))
        self.assertLessEqual(len(first), context._MAX_SYNONYMS_PER_TERM + 1)
        self.assertIn(("login", True), first)
        self.assertEqual(context._expand_semantic_terms("unmapped"), [("unmapped", False)])

    def test_synonym_search_retrieves_paraphrased_content_with_lower_score(self) -> None:
        conn = store.init_db(self.db_path)
        store.save_decision_nodes(conn, [{
            "id": "0003",
            "title": "Credential policy",
            "context": "Authentication requires strong credentials.",
        }])
        store.rebuild_fts(conn)
        conn.commit()
        conn.close()

        result = context.build_context(
            context.ContextRequest("signin", lane="tiny"), self.db_path
        )
        facts = {fact.node_id: fact for fact in result.facts}
        self.assertIn("decision_ref:0003", facts)
        self.assertLessEqual(facts["decision_ref:0003"].relevance_score, 0.85)
        self.assertEqual(facts["decision_ref:0003"].provenance.matched_terms, ["signin"])

    def test_content_search_finds_story_by_criteria_keyword(self) -> None:
        conn = store.init_db(self.db_path)
        store.save_story_nodes(conn, [{
            "id": "US-042",
            "title": "Rotate credentials",
            "status": "planned",
            "lane": "normal",
            "acceptance_criteria": [
                "Refresh credentials without interrupting active sessions.",
            ],
            "design_notes": ["Use token rotation endpoint."],
            "source_path": "docs/stories/US-042.md",
        }])
        store.rebuild_fts(conn)
        conn.commit()
        conn.close()

        result = context.build_context(
            context.ContextRequest("interrupting", lane="tiny"), self.db_path
        )
        facts = {fact.node_id for fact in result.facts}
        self.assertIn("story:US-042", facts)

    def test_local_semantic_retrieval_adds_semantic_only_seed(self) -> None:
        conn = store.init_db(self.db_path)
        store.save_structural_nodes(conn, [{
            "id": "function:unlock", "kind": "function", "name": "unlock",
        }])
        store.save_node_embeddings(conn, [{
            "node_id": "function:unlock",
            "model_id": "fake-model",
            "vector": [1.0, 0.0],
            "content_hash": "unlock-hash",
        }])
        conn.commit()
        conn.close()

        with patch.object(context, "semantic_seeds", return_value=[{
            "node_id": "function:unlock",
            "cosine": 0.9,
            "model_id": "fake-model",
            "content_hash": "unlock-hash",
        }]):
            result = context.build_context(
                context.ContextRequest(
                    "open secured resource", lane="tiny", semantic_mode="local",
                    embedding_model="fake-model", semantic_weight=0.8,
                ),
                self.db_path,
            )
        facts = {fact.node_id: fact for fact in result.facts}
        self.assertIn("function:unlock", facts)
        self.assertEqual(facts["function:unlock"].provenance.matched_terms, ["semantic task match"])
        self.assertAlmostEqual(facts["function:unlock"].relevance_score, 0.72)
        self.assertEqual(facts["function:unlock"].score_components["cosine"], 0.9)
        self.assertEqual(facts["function:unlock"].score_components["semantic_weight"], 0.8)
        self.assertEqual(facts["function:unlock"].score_components["final"], 0.72)
        self.assertEqual(facts["function:unlock"].retrieval_metadata, {
            "mode": "local", "model_id": "fake-model", "content_hash": "unlock-hash",
        })

    def test_local_semantic_mode_reports_missing_runtime(self) -> None:
        with patch.object(
            context, "semantic_seeds", side_effect=context.EmbeddingUnavailableError("install semantic extra")
        ):
            with self.assertRaisesRegex(context.ContextRetrievalError, "install semantic extra"):
                context.build_context(
                    context.ContextRequest("meaning", semantic_mode="local"), self.db_path
                )

    def test_direct_match_outscores_synonym_match(self) -> None:
        direct = context.build_context(
            context.ContextRequest("login", lane="tiny"), self.db_path
        )
        synonym = context.build_context(
            context.ContextRequest("signin", lane="tiny"), self.db_path
        )
        direct_facts = {fact.node_id: fact for fact in direct.facts}
        synonym_facts = {fact.node_id: fact for fact in synonym.facts}

        self.assertEqual(direct_facts["function:login"].relevance_score, 1.0)
        self.assertLessEqual(synonym_facts["function:login"].relevance_score, 0.85)
        self.assertEqual(
            synonym_facts["function:login"].provenance.matched_terms,
            ["signin"],
        )
        self.assertGreater(
            direct_facts["function:login"].relevance_score,
            synonym_facts["function:login"].relevance_score,
        )

    def test_structured_lane_classification(self) -> None:
        tiny = context.build_context(
            context.ContextRequest("rename auth label", request_class="maintenance_request"),
            self.db_path,
        )
        self.assertEqual(tiny.lane, "high-risk")

        normal = context.build_context(
            context.ContextRequest(
                "rename label", risk_flags={"public_contracts", "existing_behavior"}
            ),
            self.db_path,
        )
        self.assertEqual(normal.lane, "normal")
        self.assertIn("2 flags", normal.classification_reason)

        high = context.build_context(
            context.ContextRequest(
                "rename label",
                risk_flags={"public_contracts", "existing_behavior", "weak_proof", "multi_domain"},
            ),
            self.db_path,
        )
        self.assertEqual(high.lane, "high-risk")

        with self.assertRaises(ValueError):
            context.ContextRequest("inspect", risk_flags={"unknown"})

    def test_cosmetic_scope_narrows_keyword_only_hard_gate(self) -> None:
        cosmetic = context.build_context(
            context.ContextRequest("fix auth label", scope="cosmetic"),
            self.db_path,
        )
        self.assertEqual(cosmetic.lane, "tiny")
        self.assertIn("human-narrowed cosmetic scope", cosmetic.classification_reason)

        multi_file = context.build_context(
            context.ContextRequest(
                "fix auth label",
                scope="cosmetic",
                affected_files=["a.py", "b.py"],
            ),
            self.db_path,
        )
        self.assertEqual(multi_file.lane, "normal")

        explicit_hard_gate = context.build_context(
            context.ContextRequest(
                "fix auth label",
                scope="cosmetic",
                risk_flags={"auth"},
            ),
            self.db_path,
        )
        self.assertEqual(explicit_hard_gate.lane, "high-risk")
        self.assertIn("explicit hard-gate", explicit_hard_gate.classification_reason)

        behavioral = context.build_context(
            context.ContextRequest("fix auth label"), self.db_path
        )
        self.assertEqual(behavioral.lane, "high-risk")

        with self.assertRaises(ValueError):
            context.ContextRequest("inspect", scope="unknown")

    def test_lane_auto_classification_and_override(self) -> None:
        self.assertEqual(
            context.build_context(context.ContextRequest("fix auth"), self.db_path).lane,
            "high-risk",
        )
        self.assertEqual(
            context.build_context(
                context.ContextRequest("fix auth", lane="tiny"), self.db_path
            ).lane,
            "tiny",
        )
        self.assertEqual(
            context.build_context(
                context.ContextRequest("fix auth", lane="high-risk"), self.db_path
            ).lane,
            "high-risk",
        )

    def test_lane_matches_feature_intake_hard_gates(self) -> None:
        for task in (
            "change authentication flow",
            "update authorization roles",
            "audit privacy logs",
            "integrate external provider webhook",
            "delete retained data",
            "change validation requirements",
            "change payment session token",
            "configure cloud queue",
            "update external SDK",
            "send email notification",
        ):
            with self.subTest(task=task):
                self.assertEqual(
                    context.build_context(context.ContextRequest(task), self.db_path).lane,
                    "high-risk",
                )

    def test_lane_matching_avoids_substring_false_positive(self) -> None:
        for task in ("rename author field", "change categorical labels"):
            with self.subTest(task=task):
                result = context.build_context(context.ContextRequest(task), self.db_path)
                self.assertEqual(result.lane, "tiny")

    def test_json_output_excludes_bm25_rank(self) -> None:
        result = context.build_context(
            context.ContextRequest("login", lane="tiny"), self.db_path
        )
        payload = result.to_dict()
        for fact in payload["facts"]:
            self.assertNotIn("_bm25_rank", fact)

    def test_lane_budgets_match_context_rules(self) -> None:
        expected = {"tiny": 2_000, "normal": 5_000, "high-risk": 10_000}
        for lane, budget in expected.items():
            result = context.build_context(
                context.ContextRequest("inspect login", lane=lane), self.db_path
            )
            self.assertEqual(result.budget_tokens, budget)

        with self.assertRaises(ValueError):
            context.build_context(context.ContextRequest("inspect", lane="unknown"), self.db_path)


    def test_explicit_encoding_is_reported_and_counts_tokens(self) -> None:
        class FakeTokenizer:
            def encode(self, text: str) -> list[int]:
                return [1] * len(text)

        with patch.object(context, "get_tokenizer", return_value=FakeTokenizer()):
            result = context.build_context(
                context.ContextRequest("login", lane="tiny", encoding="fake"), self.db_path
            )
        self.assertEqual(result.encoding, "fake")
        self.assertGreater(result.total_tokens, 0)

    def test_unknown_encoding_rejected(self) -> None:
        with patch.object(context, "get_tokenizer", side_effect=ValueError("unknown")):
            with self.assertRaisesRegex(context.ContextRetrievalError, "unknown"):
                context.build_context(
                    context.ContextRequest("login", encoding="unknown"), self.db_path
                )

    def test_budget_pruning(self) -> None:
        result = context.build_context(
            context.ContextRequest("login", lane="high-risk", max_tokens=1), self.db_path
        )

        self.assertLessEqual(result.total_tokens, 1)
        self.assertEqual(result.facts, [])

    def test_empty_graph_returns_empty_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_db = Path(tmpdir) / "empty.db"
            conn = store.init_db(empty_db)
            conn.commit()
            conn.close()
            result = context.build_context(context.ContextRequest("nothing"), empty_db)

        self.assertEqual(result.facts, [])
        self.assertEqual(result.graph_summary["seed_nodes"], 0)

    def test_invalid_request_rejected(self) -> None:
        with self.assertRaises(ValueError):
            context.build_context(context.ContextRequest("", lane="tiny"), self.db_path)
        with self.assertRaises(ValueError):
            context.build_context(context.ContextRequest("login", lane="bad"), self.db_path)

    def test_provenance_tracks_affects_evidence(self) -> None:
        result = context.build_context(
            context.ContextRequest("architecture", lane="normal"), self.db_path
        )
        facts = {fact.node_id: fact for fact in result.facts}

        decision = facts["decision_ref:0001"]
        self.assertEqual(decision.provenance.seed_id, "decision_ref:0001")
        self.assertEqual(decision.provenance.matched_terms, ["architecture"])
        self.assertEqual(decision.provenance.path, [])

        login = facts["function:login"]
        self.assertEqual(login.provenance.seed_id, "decision_ref:0001")
        self.assertEqual(len(login.provenance.path), 1)
        step = login.provenance.path[0]
        self.assertEqual(
            (step.source_id, step.target_id, step.kind, step.direction),
            ("decision_ref:0001", "function:login", "AFFECTS", "out"),
        )
        self.assertEqual(len(step.evidence), 2)
        self.assertEqual(step.evidence[0]["source_lines"], {"start": 12, "end": 14})
        self.assertIn("docs/decisions/0001-login.md", result.sources)

    def test_provenance_is_deterministic_for_multiple_seed_paths(self) -> None:
        result = context.build_context(
            context.ContextRequest("login architecture", lane="normal"), self.db_path
        )
        facts = {fact.node_id: fact for fact in result.facts}

        fact = facts["function:validate_token"]
        self.assertEqual(fact.provenance.seed_id, "decision_ref:0001")
        self.assertEqual(
            [step.kind for step in fact.provenance.path],
            ["AFFECTS", "CALLS"],
        )
        self.assertTrue(any(
            path.seed_id == "function:login"
            and [step.kind for step in path.path] == ["CALLS"]
            for path in fact.alternate_paths
        ))

    def test_render_markdown_structure(self) -> None:
        result = context.build_context(
            context.ContextRequest("architecture", lane="normal"), self.db_path
        )
        rendered = context.render_markdown(result)

        for heading in (
            "# Context: architecture",
            "## Related Decisions",
            "## Related Stories",
            "## Topics",
            "## Code Symbols",
            "## Provenance",
            "## Sources",
        ):
            self.assertIn(heading, rendered)
        self.assertIn("| Login architecture |", rendered)
        self.assertIn("Step 1: `decision_ref:0001` -[`AFFECTS`]-> `function:login` (out)", rendered)
        self.assertIn("docs/decisions/0001-login.md:12-14", rendered)

    def test_cli_text_includes_provenance(self) -> None:
        result = subprocess.run(
            [
                sys.executable, "-m", "emp_kg.cli", "context", "architecture",
                "--lane", "normal", "--db", str(self.db_path),
            ],
            capture_output=True, text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Seed: decision_ref:0001 (architecture)", result.stdout)
        self.assertIn("Path: decision_ref:0001 -[AFFECTS]-> function:login (out)", result.stdout)
        self.assertIn("Evidence: docs/decisions/0001-login.md:12-14 login", result.stdout)

    def test_cli_markdown_format(self) -> None:
        result = subprocess.run(
            [
                sys.executable, "-m", "emp_kg.cli", "context", "login",
                "--lane", "tiny", "--db", str(self.db_path),
                "--format", "markdown",
            ],
            capture_output=True, text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith("# Context: login"))
        self.assertIn("## Code Symbols", result.stdout)

    def test_cli_json_output(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "emp_kg.cli",
                "context",
                "login",
                "--lane",
                "tiny",
                "--db",
                str(self.db_path),
                "--json",
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["lane"], "tiny")
        self.assertIn("facts", payload)

#SalarChange
def test_reciprocal_rank_fusion():
    bm25_results = [
        {'id': 'doc1', 'text': 'Python AST parsing'},
        {'id': 'doc2', 'text': 'SQLite storage'}
    ]
    vector_results = [
        {'id': 'doc2', 'text': 'SQLite storage'},
        {'id': 'doc3', 'text': 'Vector embeddings'}
    ]
    
    results = reciprocal_rank_fusion(bm25_results, vector_results, k=60)
    
  
    assert results[0]['id'] == 'doc2'
    assert len(results) == 3



if __name__ == "__main__":
    unittest.main()
