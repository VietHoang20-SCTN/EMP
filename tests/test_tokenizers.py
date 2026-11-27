"""Tests for pluggable tokenizer accounting."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from emp_kg.tokenizers import _FallbackTokenizer, get_tokenizer


class TestTokenizers(unittest.TestCase):
    def test_fallback_returns_len_divided_by_4(self) -> None:
        t = _FallbackTokenizer()
        self.assertEqual(len(t.encode("hello")), 2)
        self.assertEqual(len(t.encode("x")), 1)

    def test_get_tokenizer_returns_none_for_estimate(self) -> None:
        self.assertIsNone(get_tokenizer(None))
        self.assertIsNone(get_tokenizer("estimate"))
        self.assertIsNone(get_tokenizer(""))

    def test_get_tokenizer_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            get_tokenizer("unknown_model")

    def test_tiktoken_loader_uses_default_encoding(self) -> None:
        class FakeTikToken:
            @staticmethod
            def get_encoding(name: str):
                return name

        with patch.dict("sys.modules", {"tiktoken": FakeTikToken}):
            self.assertEqual(get_tokenizer("tiktoken"), "cl100k_base")


if __name__ == "__main__":
    unittest.main()
