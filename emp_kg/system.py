"""End-to-end task context and reasoning orchestration."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from emp_kg.context import ContextRequest, ContextResult, build_context
from emp_kg.reasoning import (
    AnthropicReasoner, AnthropicUnavailableError, NoOpReasoner,
    Reasoner, ReasoningRequest, ReasoningResult,
)


@dataclass(frozen=True)
class SystemResult:
    context: ContextResult
    reasoning: ReasoningResult
    elapsed_ms: float


def reasoner_from_provider(
    provider: str,
    *,
    model: str = "claude-opus-5",
    max_tokens: int = 4096,
    timeout: float = 30.0,
    max_retries: int = 2,
) -> Reasoner | None:
    """Create a reasoner for the given provider, or None for noop."""
    if provider == "noop":
        return None
    if provider == "anthropic":
        try:
            return AnthropicReasoner(
                model=model, max_tokens=max_tokens,
                timeout=timeout, max_retries=max_retries,
            )
        except AnthropicUnavailableError as exc:
            raise RuntimeError(str(exc)) from exc
    raise ValueError(f"unknown reasoning provider: {provider!r}")


def reason_over_context(
    request: ContextRequest,
    db_path: Path,
    *,
    reasoner: Reasoner | None = None,
    provider: str = "noop",
    model: str = "claude-opus-5",
    max_tokens: int = 4096,
    timeout: float = 30.0,
    max_retries: int = 2,
) -> SystemResult:
    """Build bounded context then pass it to an explicitly selected reasoner."""
    start = time.perf_counter()
    context_result = build_context(request, db_path)
    active_reasoner = reasoner or reasoner_from_provider(
        provider, model=model, max_tokens=max_tokens,
        timeout=timeout, max_retries=max_retries,
    )
    if active_reasoner is None:
        active_reasoner = NoOpReasoner()
    reasoning_result = active_reasoner.reason(
        ReasoningRequest(
            task=request.task_description,
            context=context_result,
            output_size=max_tokens,
            timeout_seconds=timeout,
        )
    )
    return SystemResult(
        context=context_result,
        reasoning=reasoning_result,
        elapsed_ms=round((time.perf_counter() - start) * 1000, 3),
    )
