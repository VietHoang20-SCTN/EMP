"""Tests for Anthropic Claude reasoning adapter."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from emp_kg.context import ContextFact, ContextProvenance, ContextResult
from emp_kg.reasoning import AnthropicReasoner, AnthropicUnavailableError, ReasoningRequest


class TestAnthropicReasoner(unittest.TestCase):
    def _context(self) -> ContextResult:
        return ContextResult(
            task_description="inspect login",
            lane="tiny",
            classification_reason="test",
            affected_files=[],
            budget_tokens=2000,
            graph_summary={"seed_nodes": 1, "candidate_nodes": 1, "selected_facts": 1},
            facts=[ContextFact(
                fact_type="code", node_id="function:login", name="login",
                kind="function", snippet="Authenticate user", relevance_score=1.0,
                provenance=ContextProvenance(
                    seed_id="function:login", seed_name="login", seed_kind="function",
                    matched_terms=["login"], path=[],
                ),
            )],
            sources=[], total_tokens=10,
        )

    def test_missing_sdk_raises_clear_error(self) -> None:
        with patch.dict("sys.modules", {"anthropic": None}):
            with self.assertRaises(AnthropicUnavailableError):
                AnthropicReasoner()

    def test_success_path_returns_citations_and_metadata(self) -> None:
        client = MagicMock()
        content_block = MagicMock()
        content_block.type = "text"
        content_block.text = "Login uses validate_token. Citation: fact_id:function:login."
        response = MagicMock()
        response.content = [content_block]
        response.stop_reason = "end_turn"
        response._request_id = "req_test123"
        client.messages.create.return_value = response

        reasoner = AnthropicReasoner(client=client)
        result = reasoner.reason(
            ReasoningRequest(task="inspect login", context=self._context())
        )
        self.assertTrue(result.ok)
        self.assertIn("function:login", result.text)
        self.assertEqual(result.model, "claude-opus-5")
        self.assertEqual(result.provider, "anthropic")
        self.assertTrue(result.citation_valid)
        self.assertEqual(result.error_code, "ok")
        self.assertFalse(result.unknown_citation_ids)
        self.assertTrue(any(cf.fact_id == "function:login" for cf in result.citation_facts))

    def test_unknown_citation_is_rejected(self) -> None:
        client = MagicMock()
        block = MagicMock(type="text", text="Claim [fact_id:function:missing]")
        response = MagicMock(content=[block], stop_reason="end_turn")
        client.messages.create.return_value = response
        result = AnthropicReasoner(client=client).reason(
            ReasoningRequest(task="inspect login", context=self._context())
        )
        self.assertFalse(result.ok)
        self.assertFalse(result.citation_valid)
        self.assertEqual(result.error_code, "invalid_citation")
        self.assertEqual(result.unknown_citation_ids, ["function:missing"])

    def test_refusal_detected(self) -> None:
        client = MagicMock()
        response = MagicMock()
        response.content = []
        response.stop_reason = "refusal"
        client.messages.create.return_value = response

        reasoner = AnthropicReasoner(client=client)
        result = reasoner.reason(
            ReasoningRequest(task="inspect login", context=self._context())
        )
        self.assertFalse(result.ok)
        self.assertTrue(result.refusal)

    def test_opus_refusal_fallback_uses_beta_messages(self) -> None:
        client = MagicMock()
        block = MagicMock(type="fallback")
        text_block = MagicMock(type="text", text="Login uses fact_id:function:login.")
        response = MagicMock(
            content=[block, text_block], stop_reason="end_turn", model="claude-opus-4-8"
        )
        client.beta.messages.create.return_value = response

        reasoner = AnthropicReasoner(
            client=client, enable_refusal_fallback=True,
        )
        result = reasoner.reason(
            ReasoningRequest(task="inspect login", context=self._context())
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.metadata["fallback_occurred"])
        self.assertEqual(result.metadata["served_model"], "claude-opus-4-8")
        client.beta.messages.create.assert_called_once()
        kwargs = client.beta.messages.create.call_args.kwargs
        self.assertEqual(kwargs["betas"], ["server-side-fallback-2026-07-01"])
        self.assertEqual(kwargs["fallbacks"], "default")
        client.messages.create.assert_not_called()

    def test_opus_refusal_fallback_preserves_final_refusal(self) -> None:
        client = MagicMock()
        response = MagicMock(content=[], stop_reason="refusal")
        client.beta.messages.create.return_value = response

        reasoner = AnthropicReasoner(
            client=client, enable_refusal_fallback=True,
        )
        result = reasoner.reason(
            ReasoningRequest(task="inspect login", context=self._context())
        )

        self.assertFalse(result.ok)
        self.assertTrue(result.refusal)
        self.assertFalse(result.metadata["fallback_occurred"])

    def test_fallback_rejected_for_other_models(self) -> None:
        client = MagicMock()
        with self.assertRaises(ValueError):
            AnthropicReasoner(
                model="claude-sonnet-5", client=client,
                enable_refusal_fallback=True,
            )

    def test_sdk_error_reported(self) -> None:
        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("network failure")

        reasoner = AnthropicReasoner(client=client)
        result = reasoner.reason(
            ReasoningRequest(task="inspect login", context=self._context())
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "provider_error")
        self.assertFalse(result.retryable)

    def test_invalid_limits_rejected(self) -> None:
        client = MagicMock()
        with self.assertRaises(ValueError):
            AnthropicReasoner(client=client, max_tokens=0)
        with self.assertRaises(ValueError):
            AnthropicReasoner(client=client, timeout=-1)
        with self.assertRaises(ValueError):
            AnthropicReasoner(client=client, max_retries=-1)


if __name__ == "__main__":
    unittest.main()
