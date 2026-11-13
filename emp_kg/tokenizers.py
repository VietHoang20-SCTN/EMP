"""Pluggable tokenizer for accurate token counting."""
from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class _FallbackTokenizer:
    """Character-based fallback returning approximate token count."""

    def encode(self, text: str) -> list[int]:
        return list(range(max(1, (len(text) + 3) // 4)))


def get_tokenizer(name: str | None) -> Tokenizer | None:
    if name is None:
        return None
    normalized = name.casefold()
    if normalized == "estimate" or not normalized:
        return None
    if normalized == "tiktoken" or normalized.startswith("cl100k") or normalized.startswith("o200k"):
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                f"tiktoken not installed; run pip install tiktoken to use encoding {name!r}"
            ) from exc
        encoding_name = normalized if normalized.startswith("cl100k") or normalized.startswith("o200k") else "cl100k_base"
        return tiktoken.get_encoding(encoding_name)
    raise ValueError(f"unsupported tokenizer: {name!r}")
