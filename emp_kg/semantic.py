"""
Semantic extraction layer for emp-kg.

Reads project documentation (currently: ADRs under a decisions directory)
and produces structured decision_ref dicts -- concepts, decisions, and
rationale that CodeGraph's structural extraction cannot see. Markdown
remains the source of truth; this module only reads and parses, never
modifies source documents.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

def decision_ref(
    *,
    id: str,
    title: str,
    date: str,
    status: str,
    context: str,
    decision: str,
    alternatives: list[str],
    consequences: dict[str, list[str]],
    follow_up: list[str],
    source_path: str,
    source_lines: dict[str, tuple[int, int] | None],
) -> dict:
    """Return a typed decision_ref dict."""
    return {
        "id": id,
        "title": title,
        "date": date,
        "status": status,
        "context": context,
        "decision": decision,
        "alternatives": alternatives,
        "consequences": consequences,
        "follow_up": follow_up,
        "source_path": source_path,
        "source_lines": source_lines,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"^Date:\s*(\d{4}-\d{2}-\d{2})\s*$")
_H2_RE = re.compile(r"^## (.+)$")
_NUMBERED_RE = re.compile(r"^\d+\.\s+(.+)")
_BULLET_RE = re.compile(r"^[-*]\s+(.+)")

# Maps ADR section heading text to the canonical decision_ref field key.
_SECTION_FIELD = {
    "Status": "status",
    "Context": "context",
    "Decision": "decision",
    "Consequences": "consequences",
    "Alternatives Considered": "alternatives",
    "Follow-Up": "follow_up",
}


def _extract_id(file_path: Path, lines: list[str]) -> str:
    """Extract four-digit numeric ID from the filename stem or H1 heading."""
    stem = file_path.stem
    m = re.match(r"^(\d{4})", stem)
    if m:
        return m.group(1)
    # Fallback: ID embedded in the H1 heading, e.g. "# 0099 No Status".
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            m = re.match(r"^#\s+(\d{4})", stripped)
            if m:
                return m.group(1)
    raise ValueError(f"{file_path}: cannot extract ID from filename")


def _find_date(lines: list[str]) -> tuple[str, int] | None:
    """Return (YYYY-MM-DD, 1-based line number) if found, else None."""
    for i, line in enumerate(lines, start=1):
        m = _DATE_RE.match(line.strip())
        if m:
            return m.group(1), i
    return None


def _split_sections(lines: list[str]) -> list[tuple[str, int, list[str]]]:
    """Split lines into (section_name, heading_line, body_lines) tuples.

    Section 0 is the preamble (before first ## heading).  Heading line
    numbers are 1-based.
    """
    sections: list[tuple[str, int, list[str]]] = []
    current_name = "__preamble__"
    current_start = 1
    current_body: list[str] = []

    for i, line in enumerate(lines, start=1):
        m = _H2_RE.match(line.strip())
        if m:
            sections.append((current_name, current_start, current_body))
            current_name = m.group(1).strip()
            current_start = i
            current_body = []
        else:
            current_body.append(line)

    sections.append((current_name, current_start, current_body))
    return sections


def _trimmed_body(body: list[str]) -> tuple[str, int, int]:
    """Strip leading/trailing blank lines from section body.

    Returns (content, start_offset, end_offset) where offsets are 0-based
    indices into body.  Content is the trimmed multiline string.
    """
    first = 0
    while first < len(body) and body[first].strip() == "":
        first += 1
    last = len(body) - 1
    while last >= first and body[last].strip() == "":
        last -= 1
    if first > last:
        return "", first, first - 1  # empty
    content = "\n".join(body[first:last + 1])
    return content, first, last


def _collect_bullets(body: list[str]) -> list[str]:
    """Extract unordered bullet items from body lines."""
    items: list[str] = []
    for line in body:
        m = _BULLET_RE.match(line.strip())
        if m:
            items.append(m.group(1))
    return items


def _collect_numbered(body: list[str]) -> list[str]:
    """Extract numbered list items from body lines (non-continuation)."""
    items: list[str] = []
    for line in body:
        m = _NUMBERED_RE.match(line.strip())
        if m:
            items.append(m.group(1))
    return items


def _split_consequences(body: list[str]) -> dict[str, list[str]]:
    """Parse consequences body into positive/tradeoffs lists."""
    result: dict[str, list[str]] = {"positive": [], "tradeoffs": []}
    current_key: str | None = None
    for line in body:
        stripped = line.strip()
        if stripped.lower() == "positive:":
            current_key = "positive"
            continue
        if stripped.lower() == "tradeoffs:":
            current_key = "tradeoffs"
            continue
        m = _BULLET_RE.match(stripped)
        if m and current_key is not None:
            result[current_key].append(m.group(1))
    return result


def _line_range(
    heading_line: int,
    body: list[str],
    body_offset_start: int,
    body_offset_end: int,
) -> tuple[int, int]:
    """Compute (start, end) 1-based inclusive line range for a section.

    heading_line is 1-based. body_offset_start/end are 0-based offsets.
    If body is empty, range is just the heading line.
    """
    start = heading_line
    end = heading_line + body_offset_end + 1 if body_offset_end >= body_offset_start else heading_line
    return start, end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_adr(file_path: Path) -> dict:
    """Parse one ADR Markdown file into a structured decision_ref dict.

    Returns a decision_ref with keys: id, title, date, status, context,
    decision, alternatives, consequences, follow_up, source_path, source_lines.

    Raises ValueError for malformed required fields (missing H1, missing
    date, missing required sections).
    """
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    source_path = str(file_path.as_posix())

    # --- Preamble: H1 title + Date ---
    sections = _split_sections(lines)
    preamble = sections[0]
    preamble_name, preamble_start, preamble_body = preamble

    # Find H1 in preamble
    title_line_num: int | None = None
    title_text: str | None = None
    for i, line in enumerate(preamble_body):
        stripped = line.strip()
        if stripped.startswith("# "):
            title_line_num = preamble_start + i
            title_text = stripped[2:].strip()
            break

    if title_text is None or title_line_num is None:
        raise ValueError(f"{file_path}: missing H1 title")

    adr_id = _extract_id(file_path, lines)

    # Date
    date_result = _find_date(preamble_body)
    if date_result is None:
        raise ValueError(f"{file_path}: missing Date field")
    date_value, date_line_offset = date_result
    try:
        date.fromisoformat(date_value)
    except ValueError as exc:
        raise ValueError(f"{file_path}: invalid Date field {date_value!r}") from exc
    date_line_num = preamble_start + date_line_offset - 1

    # --- Required sections ---
    section_map: dict[str, tuple[int, list[str]]] = {}
    for name, heading_line, body in sections:
        if name != "__preamble__":
            section_map[name] = (heading_line, body)

    required = ["Status", "Context", "Decision", "Consequences"]
    for req in required:
        if req not in section_map:
            raise ValueError(f"{file_path}: missing required section '{req}'")

    # --- Build source_lines ---
    source_lines: dict[str, tuple[int, int] | None] = {}

    # Title
    source_lines["title"] = (title_line_num, title_line_num)
    source_lines["date"] = (date_line_num, date_line_num)

    # Scalar sections
    for name in ["Status", "Context", "Decision"]:
        heading_line, body = section_map[name]
        trimmed, t_start, t_end = _trimmed_body(body)
        sr = _line_range(heading_line, body, t_start, t_end)
        source_lines[name.lower()] = sr

    # Consequences: special parsing
    heading_line, body = section_map["Consequences"]
    trimmed_c, t_start_c, t_end_c = _trimmed_body(body)
    sr_cons = _line_range(heading_line, body, t_start_c, t_end_c)
    source_lines["consequences"] = sr_cons
    consequences = _split_consequences(body)

    # Optional sections
    for name in ["Alternatives Considered", "Follow-Up"]:
        field = _SECTION_FIELD[name]
        if name in section_map:
            heading_line, body = section_map[name]
            _, t_start_o, t_end_o = _trimmed_body(body)
            source_lines[field] = _line_range(heading_line, body, t_start_o, t_end_o)
        else:
            source_lines[field] = None

    # --- Scalar field values ---
    status_heading, status_body = section_map["Status"]
    _, ts, te = _trimmed_body(status_body)
    status = "\n".join(status_body[ts:te + 1]) if ts <= te else ""

    context_heading, context_body = section_map["Context"]
    _, tc, tec = _trimmed_body(context_body)
    context = "\n".join(context_body[tc:tec + 1]) if tc <= tec else ""

    decision_heading, decision_body = section_map["Decision"]
    _, td, ted = _trimmed_body(decision_body)
    decision = "\n".join(decision_body[td:ted + 1]) if td <= ted else ""

    # --- Alternatives ---
    if "Alternatives Considered" in section_map:
        alt_heading, alt_body = section_map["Alternatives Considered"]
        alternatives = _collect_numbered(alt_body)
    else:
        alternatives = []

    # --- Follow-Up ---
    if "Follow-Up" in section_map:
        fu_heading, fu_body = section_map["Follow-Up"]
        follow_up = _collect_bullets(fu_body)
    else:
        follow_up = []

    return decision_ref(
        id=adr_id,
        title=title_text,
        date=date_value,
        status=status,
        context=context,
        decision=decision,
        alternatives=alternatives,
        consequences=consequences,
        follow_up=follow_up,
        source_path=source_path,
        source_lines=source_lines,
    )


def parse_all_adrs(
    decisions_dir: Path = Path("docs/decisions"),
) -> list[dict]:
    """Parse every NNNN-*.md ADR file in the directory (excluding README.md).

    Returns a list of decision_ref dicts sorted by ADR ID.
    """
    md_files = sorted(decisions_dir.glob("[0-9][0-9][0-9][0-9]-*.md"))
    return [parse_adr(f) for f in md_files]


def parse_doc(file_path: Path) -> list[dict]:
    """Parse a generic Markdown document (README, architecture notes, design
    docs -- anything that isn't an ADR) into topic dicts, one per H2 section.

    Unlike parse_adr, no fields are required -- a generic doc has no Status/
    Date/Consequences structure to enforce. Each H2-delimited section becomes
    one topic: its heading text as the title, the section body as content,
    and source location for traceability.
    """
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    source_path = str(file_path.as_posix())

    sections = _split_sections(lines)
    topics = []
    for name, heading_line, body in sections:
        if name == "__preamble__":
            continue
        trimmed, t_start, t_end = _trimmed_body(body)
        if not trimmed:
            continue
        start, end = _line_range(heading_line, body, t_start, t_end)
        topics.append({
            "title": name,
            "content": trimmed,
            "source_path": source_path,
            "source_lines": {"start": start, "end": end},
        })
    return topics


def parse_all_docs(
    docs_paths: list[Path],
) -> list[dict]:
    """Parse every given Markdown file into topic dicts. Files that fail to
    parse (e.g. binary content, encoding issues) are skipped with a printed
    warning rather than crashing the whole build.
    """
    all_topics = []
    for path in docs_paths:
        try:
            all_topics.extend(parse_doc(path))
        except (UnicodeDecodeError, OSError) as exc:
            print(f"  [SKIP] {path}: {exc}")
    return all_topics


# ---------------------------------------------------------------------------
# Story packets
# ---------------------------------------------------------------------------

_STORY_STATUSES = {"planned", "in_progress", "implemented", "changed", "retired"}
_HIGH_RISK_SUPPORTING_DOCS = (
    "overview.md",
    "design.md",
    "execplan.md",
    "validation.md",
)


def _collect_supporting_docs(story_dir: Path) -> list[dict]:
    """Read high-risk packet docs with per-file provenance."""
    docs = []
    for name in _HIGH_RISK_SUPPORTING_DOCS:
        path = story_dir / name
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        docs.append({
            "name": name,
            "content": content,
            "source_path": str(path.as_posix()),
            "source_lines": (1, len(content.splitlines())),
        })
    return docs


def parse_story(file_path: Path) -> dict:
    """Parse one story packet Markdown file into a structured story dict.

    Returns a story with keys: id, title, status, lane, acceptance_criteria,
    design_notes, source_path, source_lines.

    Raises ValueError for malformed required fields (missing H1 with a
    US-XXX-style ID, missing Status, missing Lane).
    """
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    source_path = str(file_path.as_posix())

    sections = _split_sections(lines)
    preamble_name, preamble_start, preamble_body = sections[0]

    # H1 title must carry the story ID (e.g. "US-001 ...").
    title_text: str | None = None
    story_id: str | None = None
    for i, line in enumerate(preamble_body):
        stripped = line.strip()
        if stripped.startswith("# "):
            title_text = stripped[2:].strip()
            match = re.match(r"^(US-\d+)\b", title_text)
            if match:
                story_id = match.group(1)
            break
    if title_text is None :
        raise ValueError(f"{file_path}: missing H1 title")
    if story_id is None:
        raise ValueError(f"{file_path}: story H1 must start with a US-XXX id")

    section_map: dict[str, tuple[int, list[str]]] = {}
    for name, heading_line, body in sections:
        if name != "__preamble__":
            section_map[name] = (heading_line, body)

    required = ["Status", "Lane"]
    for req in required:
        if req not in section_map:
            raise ValueError(f"{file_path}: missing required section {req!r}")

    def _first_content_line(section: str) -> str:
        return next(
            (line.strip() for line in section_map[section][1] if line.strip()),
            "",
        )

    status = _first_content_line("Status")
    if status.casefold() not in _STORY_STATUSES:
        raise ValueError(f"{file_path}: invalid story Status {status!r}")

    def _bullets(section: str) -> list[str]:
        if section not in section_map:
            return []
        return [
            line.strip().lstrip("- ").strip()
            for line in section_map[section][1]
            if line.strip().startswith("- ")
        ]

    source_lines = {
        "title": (1, 1),
        "status": (
            section_map["Status"][0],
            section_map["Status"][0] + len(section_map["Status"][1]),
        ),
        "lane": (
            section_map["Lane"][0],
            section_map["Lane"][0] + len(section_map["Lane"][1]),
        ),
    }
    for section in ("Acceptance Criteria", "Design Notes"):
        if section in section_map:
            heading, body = section_map[section]
            source_lines[section] = (heading, heading + len(body))

    return {
        "id": story_id,
        "title": title_text,
        "status": status,
        "lane": _first_content_line("Lane"),
        "acceptance_criteria": _bullets("Acceptance Criteria"),
        "design_notes": _bullets("Design Notes"),
        "source_path": source_path,
        "source_lines": source_lines,
    }


def parse_all_stories(stories_dir: Path = Path("docs/stories")) -> list[dict]:
    """Parse every story packet Markdown file recursively under stories_dir.

    Discovery is by filename: only `US-*.md` files count, matching the
    documented normal-story layout `epics/E*-domain/US-XXX-title.md`.
    README.md, backlog.md, and high-risk packet components (overview.md,
    design.md, execplan.md, validation.md) are excluded by the name filter.

    Returns a list of story dicts sorted by story ID. A discovered US-*.md
    with malformed required fields raises ValueError -- a file presented as
    a normal story packet must not disappear silently from the graph.
    """
    normal_files = set(stories_dir.rglob("US-*.md"))
    high_risk_manifests = {
        path
        for path in stories_dir.rglob("story.md")
        if path.parent.name.startswith("US-")
    }
    md_files = sorted(normal_files | high_risk_manifests)
    stories = []
    for path in md_files:
        story = parse_story(path)
        story["supporting_docs"] = (
            _collect_supporting_docs(path.parent)
            if path in high_risk_manifests
            else []
        )
        stories.append(story)
    return stories
