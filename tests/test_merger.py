"""Tests for emp_kg.merger."""
from __future__ import annotations

import unittest

from emp_kg import schema
from emp_kg.merger import (
    _is_junk_reference,
    _link,
    classify_unresolved,
    _topic_id,
    link_decisions_to_code,
    link_topics_to_code,
    resolve_stories_to_code,
    resolve_topics_to_code,
)


class TestJunkReferenceFilter(unittest.TestCase):
    def test_rejects_cli_flags(self) -> None:
        self.assertTrue(_is_junk_reference("--json"))
        self.assertTrue(_is_junk_reference("-n"))

    def test_rejects_punctuation_only(self) -> None:
        self.assertTrue(_is_junk_reference(","))
        self.assertTrue(_is_junk_reference("|"))
        self.assertTrue(_is_junk_reference("..."))

    def test_rejects_prose_fragments(self) -> None:
        self.assertTrue(_is_junk_reference("query tools"))
        self.assertTrue(_is_junk_reference("git status --short"))
        self.assertTrue(_is_junk_reference(", and"))

    def test_rejects_language_labels(self) -> None:
        self.assertTrue(_is_junk_reference("bash"))
        self.assertTrue(_is_junk_reference("json"))
        self.assertTrue(_is_junk_reference("powershell"))

    def test_rejects_file_extensions_and_paths(self) -> None:
        self.assertTrue(_is_junk_reference("docs/product/"))
        self.assertTrue(_is_junk_reference("SPEC.md"))
        self.assertTrue(_is_junk_reference("harness.db"))
        self.assertTrue(_is_junk_reference("Cargo.lock"))
        self.assertTrue(_is_junk_reference("scripts/bin/harness-cli"))

    def test_rejects_bare_numbers_and_single_chars(self) -> None:
        self.assertTrue(_is_junk_reference("0"))
        self.assertTrue(_is_junk_reference("1"))
        self.assertTrue(_is_junk_reference("42"))
        self.assertTrue(_is_junk_reference("1800"))
        self.assertTrue(_is_junk_reference("a"))

    def test_rejects_prose_stopwords(self) -> None:
        for word in ("yes", "no", "or", "and", "to", "is", "the", "id", "result"):
            with self.subTest(word=word):
                self.assertTrue(_is_junk_reference(word))

    def test_rejects_label_prefixed_code_fences(self) -> None:
        dumps = (
            "json\n{\n  \"protocol_version\": 1, \"ok\": true}",
            "text\nscore = orphaned_stories * 10\n      + unverified_stories * 5",
            "bash\ninstall-harness.sh --merge --fast",
            "powershell\ninstall-harness.ps1 -Merge -Fast",
        )
        for dump in dumps:
            with self.subTest(prefix=dump.splitlines()[0]):
                self.assertTrue(_is_junk_reference(dump))

    def test_accepts_short_and_single_word_code_identifiers(self) -> None:
        self.assertFalse(_is_junk_reference("os"))
        self.assertFalse(_is_junk_reference("run"))
        self.assertFalse(_is_junk_reference("sqlite3"))
        self.assertFalse(_is_junk_reference("Service"))
        self.assertFalse(_is_junk_reference("json.dump"))

    def test_accepts_code_identifiers(self) -> None:
        self.assertFalse(_is_junk_reference("process_payment"))
        self.assertFalse(_is_junk_reference("find_nested_git_repos"))
        self.assertFalse(_is_junk_reference("harness_friction"))
        self.assertFalse(_is_junk_reference("trace.harness_friction"))
        self.assertFalse(_is_junk_reference("verify_command"))


class TestClassifyUnresolved(unittest.TestCase):
    @staticmethod
    def unresolved(reference: str) -> dict:
        return {
            "source": "topic:test",
            "reference": reference,
            "source_path": "README.md",
            "source_lines": {"start": 1, "end": 2},
            "reason": "unresolved_reference",
        }

    def test_classifies_external_paths_and_urls(self) -> None:
        items = [
            self.unresolved("crates/harness-cli/src/main.rs"),
            self.unresolved("https://github.com/acme/repo"),
        ]

        groups = classify_unresolved(items)

        self.assertEqual(groups, {"external_path": items})

    def test_classifies_system_signals_and_external_symbols(self) -> None:
        sigterm = self.unresolved("SIGTERM")
        sqlite = self.unresolved("sqlite3")

        groups = classify_unresolved([sigterm, sqlite])

        self.assertEqual(groups["system_signal"], [sigterm])
        self.assertEqual(groups["external_ref"], [sqlite])

    def test_classifies_workflow_and_dotted_references(self) -> None:
        workflow = self.unresolved("score-context")
        dotted = self.unresolved("story.verify_command")

        groups = classify_unresolved([workflow, dotted])

        self.assertEqual(groups["workflow_ref"], [workflow])
        self.assertEqual(groups["dotted_domain"], [dotted])

    def test_classifies_numeric_values(self) -> None:
        item = self.unresolved("1800")

        self.assertEqual(classify_unresolved([item]), {"numeric_value": [item]})

    def test_classifies_documentation_and_literal_categories(self) -> None:
        items = [
            self.unresolved("task_summary"),
            self.unresolved("completed"),
            self.unresolved("harness"),
            self.unresolved("datetime('now')"),
            self.unresolved("name:type:required"),
            self.unresolved(".git"),
        ]

        groups = classify_unresolved(items)

        self.assertEqual(groups["doc_schema_field"], [items[0]])
        self.assertEqual(groups["status_value"], [items[1]])
        self.assertEqual(groups["tool_or_component"], [items[2]])
        self.assertEqual(groups["sql_literal"], [items[3]])
        self.assertEqual(groups["json_literal"], [items[4]])
        self.assertEqual(groups["repo_path"], [items[5]])

    def test_classification_conserves_all_entries(self) -> None:
        items = [self.unresolved(value) for value in (
            "unknown_symbol", "trace.errors", "SIGTERM", "score-context", "1800"
        )]
        groups = classify_unresolved(items)
        self.assertEqual(sum(len(group) for group in groups.values()), len(items))

    def test_keeps_unknown_lowercase_symbols_visible(self) -> None:
        items = [self.unresolved("run"), self.unresolved("missing")]

        self.assertEqual(classify_unresolved(items), {"other": items})

    def test_empty_input_returns_empty_groups(self) -> None:
        self.assertEqual(classify_unresolved([]), {})


class TestLinkDecisionsToCode(unittest.TestCase):
    def test_matches_backtick_wrapped_function_name(self) -> None:
        decisions = [{
            "id": "0001",
            "context": "We changed how `process_payment` works.",
            "decision": "",
            "consequences": {"positive": [], "tradeoffs": []},
            "alternatives": [],
            "follow_up": [],
        }]
        nodes = [{"id": "function:abc123", "kind": "function", "name": "process_payment"}]
        edges = link_decisions_to_code(decisions, nodes)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["target"], "function:abc123")

    def test_ignores_prose_mention_without_backticks(self) -> None:
        decisions = [{
            "id": "0002",
            "context": "We merged to main after review.",
            "decision": "",
            "consequences": {"positive": [], "tradeoffs": []},
            "alternatives": [],
            "follow_up": [],
        }]
        nodes = [{"id": "function:xyz789", "kind": "function", "name": "main"}]
        edges = link_decisions_to_code(decisions, nodes)
        self.assertEqual(len(edges), 0)

    def test_excludes_variable_kind_even_with_backticks(self) -> None:
        decisions = [{
            "id": "0003",
            "context": "We renamed `count` for clarity.",
            "decision": "",
            "consequences": {"positive": [], "tradeoffs": []},
            "alternatives": [],
            "follow_up": [],
        }]
        nodes = [{"id": "variable:def456", "kind": "variable", "name": "count"}]
        edges = link_decisions_to_code(decisions, nodes)
        self.assertEqual(len(edges), 0)

    def test_matches_file_by_basename(self) -> None:
        decisions = [{
            "id": "0004",
            "context": "See `src/utils.py` for the implementation.",
            "decision": "",
            "consequences": {"positive": [], "tradeoffs": []},
            "alternatives": [],
            "follow_up": [],
        }]
        nodes = [{"id": "file:src/utils.py", "kind": "file", "filePath": "src/utils.py"}]
        edges = link_decisions_to_code(decisions, nodes)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["target"], "file:src/utils.py")


class TestLinkTopicsToCode(unittest.TestCase):
    def test_junk_references_do_not_enter_unresolved_report(self) -> None:
        topic = {
            "title": "CLI",
            "content": (
                "Run `--json` with `bash` and see `docs/product/`. "
                "The flag `,` is not code."
            ),
            "source_path": "docs/HARNESS.md",
        }
        nodes = [
            {"id": "function:abc123", "kind": "function", "name": "process_payment"},
        ]

        result = resolve_topics_to_code([topic], nodes)

        self.assertEqual(result.edges, [])
        self.assertEqual(result.unresolved, [])

    def test_matches_backtick_reference_with_stable_topic_source(self) -> None:
        topic = {
            "title": "Build flow",
            "content": "Call `process_payment` after validation.",
            "source_path": "docs/ARCHITECTURE.md",
        }
        nodes = [
            {"id": "function:abc123", "kind": "function", "name": "process_payment"},
        ]

        edges = link_topics_to_code([topic], nodes)

        self.assertEqual(edges, [{
            "source": _topic_id(topic),
            "target": "function:abc123",
            "kind": "AFFECTS",
        }])
        self.assertEqual(_topic_id(topic), _topic_id(dict(topic)))

    def test_links_bare_name_but_excludes_ineligible_kinds(self) -> None:
        topics = [{
            "title": "State",
            "content": "process_payment uses `count`.",
            "source_path": "README.md",
        }]
        nodes = [
            {"id": "function:abc123", "kind": "function", "name": "process_payment"},
            {"id": "variable:def456", "kind": "variable", "name": "count"},
        ]

        self.assertEqual(link_topics_to_code(topics, nodes), [{
            "source": _topic_id(topics[0]),
            "target": "function:abc123",
            "kind": "AFFECTS",
        }])

    def test_story_source_can_emit_affects(self) -> None:
        story = {
            "id": "US-001",
            "title": "US-001 Login flow",
            "status": "planned",
            "acceptance_criteria": ["Users can log in via `process_payment`."],
            "design_notes": [],
            "source_path": "docs/stories/US-001-login.md",
        }
        nodes = [
            {"id": "function:abc123", "kind": "function", "name": "process_payment"},
        ]
        result = resolve_stories_to_code([story], nodes)
        self.assertEqual(len(result.edges), 1)
        self.assertEqual(result.edges[0]["target"], "function:abc123")
        self.assertEqual(result.edges[0]["source"], "story:US-001")
        self.assertIn("story", schema.supported_edge_sources("AFFECTS"))

    def test_story_links_references_from_supporting_docs(self) -> None:
        story = {
            "id": "US-012",
            "title": "US-012 Risk flow",
            "status": "planned",
            "acceptance_criteria": [],
            "design_notes": [],
            "source_path": "docs/stories/US-012/story.md",
            "supporting_docs": [{
                "name": "overview.md",
                "content": "Call `process_payment` from this workflow.",
                "source_path": "docs/stories/US-012/overview.md",
                "source_lines": (1, 3),
            }],
        }
        nodes = [
            {"id": "function:abc123", "kind": "function", "name": "process_payment"},
        ]

        result = resolve_stories_to_code([story], nodes)

        self.assertEqual(result.edges, [{
            "source": "story:US-012",
            "target": "function:abc123",
            "kind": "AFFECTS",
        }])
        self.assertEqual(result.evidence[0]["source_path"], "docs/stories/US-012/overview.md")
        self.assertEqual(result.evidence[0]["source_lines"], {"start": 1, "end": 3})


class TestBareNameReferences(unittest.TestCase):
    def test_plain_code_shaped_name_links_without_backticks(self) -> None:
        topic = {
            "title": "Runtime",
            "content": "The entry point calls process_payment after validation.",
            "source_path": "docs/ARCHITECTURE.md",
        }
        nodes = [
            {"id": "function:pp", "kind": "function", "name": "process_payment"},
        ]

        edges = link_topics_to_code([topic], nodes)

        self.assertEqual(edges, [{
            "source": _topic_id(topic),
            "target": "function:pp",
            "kind": "AFFECTS",
        }])

    def test_ordinary_prose_does_not_create_false_positive_link(self) -> None:
        topic = {
            "title": "Onboarding",
            "content": "The service API is public and remains available.",
            "source_path": "docs/ARCHITECTURE.md",
        }
        nodes = [
            {"id": "class:service", "kind": "class", "name": "Service"},
            {"id": "function:main", "kind": "function", "name": "main"},
        ]

        edges = link_topics_to_code([topic], nodes)

        self.assertEqual(edges, [])


class TestDeterministicResolution(unittest.TestCase):
    def test_resolves_exact_id_before_other_forms_with_evidence(self) -> None:
        topic = {
            "title": "References",
            "content": "Use `function:exact`.",
            "source_path": "docs/ARCHITECTURE.md",
            "source_lines": {"start": 10, "end": 12},
        }
        nodes = [
            {"id": "function:exact", "kind": "function", "name": "other"},
            {"id": "function:other", "kind": "function", "name": "exact"},
        ]

        result = resolve_topics_to_code([topic], nodes)

        self.assertEqual(result.edges, [{
            "source": _topic_id(topic), "target": "function:exact", "kind": "AFFECTS",
        }])
        self.assertEqual(result.evidence, [{
            "source": _topic_id(topic),
            "target": "function:exact",
            "kind": "AFFECTS",
            "confidence": 1.0,
            "reference": "function:exact",
            "source_path": "docs/ARCHITECTURE.md",
            "source_lines": {"start": 10, "end": 12},
            "resolution": "exact_node_id",
        }])
        self.assertEqual(result.unresolved, [])

    def test_resolves_file_path_and_qualified_name(self) -> None:
        topic = {
            "title": "References",
            "content": "See `src\\worker.py` and `worker.run`.",
            "source_path": "docs/ARCHITECTURE.md",
            "source_lines": {"start": 1, "end": 2},
        }
        nodes = [
            {"id": "file:src/worker.py", "kind": "file", "filePath": "src/worker.py"},
            {
                "id": "function:run", "kind": "function", "name": "run",
                "qualifiedName": "worker.run", "filePath": "src/worker.py",
            },
        ]

        result = resolve_topics_to_code([topic], nodes)

        self.assertEqual(
            [(item["target"], item["resolution"]) for item in result.evidence],
            [("file:src/worker.py", "file_path"), ("function:run", "qualified_name")],
        )

    def test_resolves_file_local_then_unique_global_name(self) -> None:
        topic = {
            "title": "References",
            "content": "Use `src/a.py::run` then `unique`.",
            "source_path": "docs/ARCHITECTURE.md",
            "source_lines": {"start": 1, "end": 2},
        }
        nodes = [
            {"id": "function:a-run", "kind": "function", "name": "run", "filePath": "src/a.py"},
            {"id": "function:b-run", "kind": "function", "name": "run", "filePath": "src/b.py"},
            {"id": "function:unique", "kind": "function", "name": "unique", "filePath": "src/a.py"},
        ]

        result = resolve_topics_to_code([topic], nodes)

        self.assertEqual(
            [(item["target"], item["resolution"]) for item in result.evidence],
            [("function:a-run", "file_local_name"), ("function:unique", "unique_global_name")],
        )

    def test_reports_ambiguous_and_unresolved_references_without_edges(self) -> None:
        topic = {
            "title": "References",
            "content": "Try `run` and `missing`.",
            "source_path": "docs/ARCHITECTURE.md",
            "source_lines": {"start": 3, "end": 4},
        }
        nodes = [
            {"id": "function:a-run", "kind": "function", "name": "run", "filePath": "src/a.py"},
            {"id": "function:b-run", "kind": "function", "name": "run", "filePath": "src/b.py"},
        ]

        result = resolve_topics_to_code([topic], nodes)

        self.assertEqual(result.edges, [])
        self.assertEqual(result.evidence, [])
        self.assertEqual(
            [(item["reference"], item["reason"]) for item in result.unresolved],
            [("run", "ambiguous_global_name"), ("missing", "unresolved_reference")],
        )

    def test_keeps_multiple_references_as_separate_evidence_for_one_edge(self) -> None:
        topic = {
            "title": "References",
            "content": "Call `unique`; then call `unique` again.",
            "source_path": "docs/ARCHITECTURE.md",
            "source_lines": {"start": 1, "end": 3},
        }
        nodes = [{"id": "function:unique", "kind": "function", "name": "unique"}]

        result = resolve_topics_to_code([topic], nodes)

        self.assertEqual(len(result.edges), 1)
        self.assertEqual(len(result.evidence), 2)


if __name__ == "__main__":
    unittest.main()
