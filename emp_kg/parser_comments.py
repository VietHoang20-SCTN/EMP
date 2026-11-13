"""Deterministic extraction of line and block comments."""
from __future__ import annotations


_BLOCK_MARKERS = (("/**", "*/"), ("/*", "*/"), ("<!--", "-->"))


def extract_comments(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return normalized comment spans, ignoring marker text inside strings."""
    comments: list[tuple[int, int, str]] = []
    block_end: str | None = None
    block_start = 0
    fragments: list[str] = []

    for line_number, line in enumerate(lines, start=1):
        index = 0
        quote: str | None = None
        while index < len(line):
            if block_end is not None:
                end_index = line.find(block_end, index)
                if end_index < 0:
                    fragments.append(line[index:].strip("* "))
                    break
                fragments.append(line[index:end_index].strip("* "))
                text = " ".join(fragment for fragment in fragments if fragment).strip()
                if text:
                    comments.append((block_start, line_number, text))
                block_end = None
                fragments = []
                index = end_index + len("*/")
                continue

            char = line[index]
            if quote is not None:
                if char == "\\":
                    index += 2
                    continue
                if char == quote:
                    quote = None
                index += 1
                continue
            if char in {"'", '"', "`"}:
                quote = char
                index += 1
                continue
            if line.startswith("//", index) or line.startswith("#", index):
                text = line[index + (2 if line.startswith("//", index) else 1):].strip()
                if text:
                    comments.append((line_number, line_number, text))
                break
            marker = next((item for item in _BLOCK_MARKERS if line.startswith(item[0], index)), None)
            if marker is not None:
                start_marker, block_end = marker
                block_start = line_number
                fragments = []
                index += len(start_marker)
                continue
            index += 1

    return comments
