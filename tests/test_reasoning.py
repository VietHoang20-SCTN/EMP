"""Tests for provider-neutral reasoning boundary."""
from __future__ import annotations

import unittest

from emp_kg.context import ContextFact, ContextProvenance, ContextResult
from emp_kg.reasoning import NoOpReasoner, ReasoningRequest


class TestReasoning(unittest.TestCase):
    def _context(self) -> ContextResult:
        return ContextResult(
            task_description="inspect login",
            lane="tiny",
            classification_reason="test",
            affected_files=[],
            budget_tokens=2000,
            graph_summary={"seed_nodes": 1, "candidate_nodes": 1, "selected_facts": 1},
            facts=[ContextFact(
                fact_type="code",
                node_id="function:login",
                name="login",
                kind="function",
                snippet="Authenticate user",
                relevance_score=1.0,
                provenance=ContextProvenance(
                    seed_id="function:login", seed_name="login", seed_kind="function",
                    matched_terms=["login"], path=[],
                ),
            )],
            sources=[],
            total_tokens=10,
        )

    def test_noop_reasoner_is_deterministic_except_timing_and_id(self) -> None:
        request = ReasoningRequest(task="inspect login", context=self._context())
        first = NoOpReasoner().reason(request)
        second = NoOpReasoner().reason(request)
        self.assertTrue(first.ok)
        self.assertEqual(first.text, second.text)
        self.assertEqual(first.provider, "none")
        self.assertEqual(first.citation_facts[0].fact_id, "function:login")


if __name__ == "__main__":
    unittest.main()
