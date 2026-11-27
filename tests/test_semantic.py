"""Tests for scripts/adr_parser.py"""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from emp_kg.semantic import (
    parse_adr,
    parse_all_adrs,
    parse_all_docs,
    parse_all_stories,
    parse_doc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DECISIONS_DIR = Path(__file__).resolve().parent / "fixtures" / "decisions"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseAdrFull(unittest.TestCase):
    """0003 has all sections present."""

    def setUp(self) -> None:
        self.result = parse_adr(DECISIONS_DIR / "0003-generic-spec-intake-harness.md")

    def test_id(self) -> None:
        self.assertEqual(self.result["id"], "0003")

    def test_title(self) -> None:
        self.assertEqual(self.result["title"], "0003 Generic Spec Intake Harness")

    def test_date(self) -> None:
        self.assertEqual(self.result["date"], "2026-05-05")

    def test_status(self) -> None:
        self.assertEqual(self.result["status"], "Accepted")

    def test_context_not_empty(self) -> None:
        self.assertIn("Harness v0", self.result["context"])

    def test_decision_not_empty(self) -> None:
        self.assertIn("Remove the tracked project-specific spec", self.result["decision"])

    def test_alternatives_count(self) -> None:
        self.assertEqual(len(self.result["alternatives"]), 2)

    def test_alternatives_items(self) -> None:
        self.assertIn("Keep the original `SPEC.md`", self.result["alternatives"][0])

    def test_consequences_positive(self) -> None:
        self.assertGreater(len(self.result["consequences"]["positive"]), 0)

    def test_consequences_tradeoffs(self) -> None:
        self.assertGreater(len(self.result["consequences"]["tradeoffs"]), 0)

    def test_follow_up(self) -> None:
        self.assertEqual(len(self.result["follow_up"]), 1)
        self.assertIn("spec-intake template", self.result["follow_up"][0])

    def test_source_path(self) -> None:
        self.assertIn("0003-generic-spec-intake-harness.md", self.result["source_path"])

    def test_source_lines_title(self) -> None:
        sl = self.result["source_lines"]
        self.assertEqual(sl["title"], (1, 1))

    def test_source_lines_has_all_keys(self) -> None:
        expected_keys = {
            "title", "date", "status", "context", "decision",
            "alternatives", "consequences", "follow_up",
        }
        self.assertEqual(set(self.result["source_lines"].keys()), expected_keys)


class TestParseAdrLegacy(unittest.TestCase):
    """0001 has no Alternatives or Follow-Up sections."""

    def setUp(self) -> None:
        self.result = parse_adr(DECISIONS_DIR / "0001-harness-first-development.md")

    def test_id(self) -> None:
        self.assertEqual(self.result["id"], "0001")

    def test_title(self) -> None:
        self.assertEqual(self.result["title"], "0001 Harness-First Development")

    def test_date(self) -> None:
        self.assertEqual(self.result["date"], "2026-05-05")

    def test_status(self) -> None:
        self.assertEqual(self.result["status"], "Accepted")

    def test_alternatives_empty(self) -> None:
        self.assertEqual(self.result["alternatives"], [])

    def test_follow_up_empty(self) -> None:
        self.assertEqual(self.result["follow_up"], [])

    def test_source_lines_alternatives_none(self) -> None:
        self.assertIsNone(self.result["source_lines"]["alternatives"])

    def test_source_lines_follow_up_none(self) -> None:
        self.assertIsNone(self.result["source_lines"]["follow_up"])

    def test_consequences_positive(self) -> None:
        self.assertEqual(len(self.result["consequences"]["positive"]), 4)


class TestParseAdrStatusVariants(unittest.TestCase):
    """Status field variations: superseded (0002), amended (0005)."""

    def test_superseded(self) -> None:
        r = parse_adr(DECISIONS_DIR / "0002-post-spec-product-lifecycle.md")
        self.assertIn("Superseded", r["status"])
        self.assertIn("0003", r["status"])

    def test_amended(self) -> None:
        r = parse_adr(DECISIONS_DIR / "0005-prebuilt-rust-harness-cli.md")
        self.assertIn("Accepted", r["status"])
        self.assertIn("amended", r["status"])
        self.assertIn("2026-05-31", r["status"])


class TestParseAllAdrs(unittest.TestCase):
    """parse_all_adrs covers all 4 ADRs, excludes README."""

    def setUp(self) -> None:
        self.results = parse_all_adrs(DECISIONS_DIR)

    def test_count(self) -> None:
        self.assertEqual(len(self.results), 4)

    def test_ids(self) -> None:
        ids = [r["id"] for r in self.results]
        self.assertEqual(ids, ["0001", "0002", "0003", "0005"])

    def test_no_readme(self) -> None:
        paths = [r["source_path"] for r in self.results]
        for p in paths:
            self.assertNotIn("README", p)

    def test_all_dates_valid(self) -> None:
        import re
        date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for r in self.results:
            self.assertRegex(r["date"], date_re, msg=f"{r['id']} has invalid date")


class TestParseDoc(unittest.TestCase):
    def test_extracts_nonempty_h2_topics_with_source_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ARCHITECTURE.md"
            path.write_text(textwrap.dedent("""\
                # Architecture

                Preamble ignored.

                ## First topic

                References `build_graph`.

                ## Empty topic

                ## Second topic
                Second body.
            """), encoding="utf-8")

            topics = parse_doc(path)

        self.assertEqual([topic["title"] for topic in topics], ["First topic", "Second topic"])
        self.assertEqual(topics[0]["content"], "References `build_graph`.")
        self.assertEqual(topics[0]["source_path"], path.as_posix())
        self.assertEqual(topics[0]["source_lines"], {"start": 5, "end": 7})
        self.assertEqual(topics[1]["content"], "Second body.")
        self.assertEqual(topics[1]["source_lines"], {"start": 11, "end": 12})

    def test_parse_all_docs_aggregates_and_skips_unreadable_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "README.md"
            second = root / "docs.md"
            missing = root / "missing.md"
            first.write_text("## First\n\nOne.\n", encoding="utf-8")
            second.write_text("## Second\n\nTwo.\n", encoding="utf-8")

            topics = parse_all_docs([first, missing, second])

        self.assertEqual([topic["title"] for topic in topics], ["First", "Second"])


class TestParseAllStories(unittest.TestCase):
    STORY = """\
# US-001 Nested story

## Status

planned

## Lane

normal

## Acceptance Criteria

- Users can call `process_payment`.
"""

    def test_discovers_nested_epic_story(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stories = Path(tmpdir) / "stories"
            path = stories / "epics" / "E01-domain" / "US-001-nested.md"
            path.parent.mkdir(parents=True)
            path.write_text(self.STORY, encoding="utf-8")

            result = parse_all_stories(stories)

        self.assertEqual([story["id"] for story in result], ["US-001"])
        self.assertEqual(result[0]["title"], "US-001 Nested story")
        self.assertEqual(result[0]["source_path"], path.as_posix())

    def test_skips_high_risk_component_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stories = Path(tmpdir) / "stories" / "epics" / "E02-risk" / "US-012-risky"
            stories.mkdir(parents=True)
            (stories / "overview.md").write_text(
                "# Overview\n\n## Current Behavior\n\nDetails.\n",
                encoding="utf-8",
            )

            result = parse_all_stories(stories.parents[2])

        self.assertEqual(result, [])

    def test_skips_readme_and_backlog_at_any_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stories = Path(tmpdir) / "stories" / "epics" / "E01"
            stories.mkdir(parents=True)
            (stories / "README.md").write_text(self.STORY, encoding="utf-8")
            (stories / "backlog.md").write_text(self.STORY, encoding="utf-8")

            result = parse_all_stories(stories.parents[1])

        self.assertEqual(result, [])

    def test_flat_story_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stories = Path(tmpdir) / "stories"
            stories.mkdir()
            (stories / "US-002-flat.md").write_text(self.STORY.replace("US-001", "US-002"), encoding="utf-8")

            result = parse_all_stories(stories)

        self.assertEqual([story["id"] for story in result], ["US-002"])

    def test_malformed_discovered_story_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stories = Path(tmpdir) / "stories" / "epics" / "E01"
            stories.mkdir(parents=True)
            (stories / "US-003-broken.md").write_text(
                "# US-003 Broken\n\n## Status\n\nplanned\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Lane"):
                parse_all_stories(stories.parents[1])

    def test_discovers_high_risk_story_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stories = Path(tmpdir) / "stories"
            manifest = (
                stories / "epics" / "E02-risk" / "US-012-risky" / "story.md"
            )
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                self.STORY.replace("US-001", "US-012").replace("normal", "high-risk"),
                encoding="utf-8",
            )

            result = parse_all_stories(stories)

        self.assertEqual([story["id"] for story in result], ["US-012"])
        self.assertEqual(result[0]["lane"], "high-risk")
        self.assertEqual(result[0]["source_path"], manifest.as_posix())

    def test_high_risk_without_manifest_not_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stories = Path(tmpdir) / "stories"
            packet = stories / "epics" / "E02-risk" / "US-013-no-manifest"
            packet.mkdir(parents=True)
            (packet / "overview.md").write_text("# Overview\n", encoding="utf-8")

            result = parse_all_stories(stories)

        self.assertEqual(result, [])

    def test_root_template_story_md_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stories = Path(tmpdir) / "stories"
            stories.mkdir()
            (stories / "story.md").write_text(self.STORY, encoding="utf-8")

            result = parse_all_stories(stories)

        self.assertEqual(result, [])

    def test_high_risk_story_collects_supporting_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stories = Path(tmpdir) / "stories"
            packet = stories / "epics" / "E02-risk" / "US-012-risky"
            packet.mkdir(parents=True)
            (packet / "story.md").write_text(
                self.STORY.replace("US-001", "US-012").replace("normal", "high-risk"),
                encoding="utf-8",
            )
            (packet / "overview.md").write_text(
                "# Overview\n\n## Current Behavior\n\nCall `process_payment` here.\n",
                encoding="utf-8",
            )
            (packet / "design.md").write_text(
                "# Design\n\nUse `harness_cli` for traces.\n",
                encoding="utf-8",
            )

            result = parse_all_stories(stories)

        docs = result[0]["supporting_docs"]
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0]["name"], "overview.md")
        self.assertIn("process_payment", docs[0]["content"])
        self.assertEqual(docs[0]["source_lines"], (1, 5))
        self.assertTrue(docs[0]["source_path"].endswith("overview.md"))

    def test_normal_story_has_empty_supporting_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stories = Path(tmpdir) / "stories"
            stories.mkdir()
            (stories / "US-001-flat.md").write_text(self.STORY, encoding="utf-8")

            result = parse_all_stories(stories)

        self.assertEqual(result[0]["supporting_docs"], [])


class TestParseAdrMalformed(unittest.TestCase):
    """Missing required fields raise ValueError."""

    def test_missing_h1(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("Date: 2026-01-01\n\n## Status\n\nAccepted\n")
            f.flush()
            with self.assertRaises(ValueError) as ctx:
                parse_adr(Path(f.name))
            self.assertIn("H1", str(ctx.exception))

    def test_missing_date(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# 0099 Bad Date\n\n## Status\n\nAccepted\n")
            f.flush()
            with self.assertRaises(ValueError) as ctx:
                parse_adr(Path(f.name))
            self.assertIn("Date", str(ctx.exception))

    def test_missing_status(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(textwrap.dedent("""\
                # 0099 No Status

                Date: 2026-01-01

                ## Context

                Some context.

                ## Decision

                Some decision.

                ## Consequences

                Positive:

                - Item.

                Tradeoffs:

                - Item.
            """))
            f.flush()
            with self.assertRaises(ValueError) as ctx:
                parse_adr(Path(f.name))
            self.assertIn("Status", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
