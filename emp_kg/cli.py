"""Command-line interface for emp-kg."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast
from emp_kg.mcp_server import run_stdio
#from emp_kg import benchmark, compression, context, impact, merger, schema, semantic, source_scan, store, structural, system
from emp_kg import benchmark, context, impact, merger, schema, semantic, source_scan, store, structural, system
import emp_kg.context_compression as compression
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time



from emp_kg.embeddings import (
    DEFAULT_MODEL,
    SentenceTransformerEmbedder,
    corpus_digest,
    embedding_content_hashes,
    validate_vectors,
)


def _git_sha(project: Path) -> str:
    """Return the build source revision, or an explicit unavailable marker."""
    try:
        result = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _discover_docs(project: Path) -> list[Path]:
    """Find general documentation Markdown files worth semantic extraction --
    root-level and docs/, excluding decisions/stories (their own dedicated
    parser) and reference-only material (workspace/, .claude/)."""
    candidates: list[Path] = []
    for md in project.glob("*.md"):
        candidates.append(md)
    docs_dir = project / "docs"
    if docs_dir.exists():
        for md in docs_dir.rglob("*.md"):
            rel = md.relative_to(project)
            parts = rel.parts
            if "decisions" in parts or "stories" in parts:
                continue
            candidates.append(md)
    return candidates


def _build_embeddings(conn, model_id: str) -> int:
    """Atomically replace one model's vectors from current FTS documents."""
    documents = embedding_content_hashes(store.embedding_documents(conn))
    current_digest = corpus_digest(documents)
    metadata = store.get_embedding_metadata(conn, model_id)
    if metadata and metadata.get("corpus_digest") == current_digest:
        return 0

    embedder = SentenceTransformerEmbedder(model_id)
    vectors = embedder.encode([item["content"] for item in documents])
    validate_vectors(vectors, len(documents))
    return store.replace_node_embeddings(
        conn,
        model_id,
        [
            {
                "node_id": item["node_id"],
                "model_id": model_id,
                "vector": vector,
                "content_hash": item["content_hash"],
            }
            for item, vector in zip(documents, vectors)
        ],
        corpus_digest=current_digest,
    )


def build(
    project_path: str,
    db_path: str | None = None,
    force: bool = False,
    embeddings: str = "off",
    embedding_model: str = DEFAULT_MODEL,
    profile: bool = False,
) -> None:
    if embeddings not in {"off", "local"}:
        raise ValueError(f"invalid embeddings mode: {embeddings}")
    started_at = time.perf_counter()
    project = Path(project_path).resolve()
    decisions_dir = project / "docs" / "decisions"
    output_db = Path(db_path) if db_path else project / "emp-kg.db"

    print(f"Building knowledge graph for: {project}")

    current_sha = _git_sha(project)
    if not force and output_db.exists():
        try:
            conn_ro = store.open_readonly(output_db)
            existing = store.get_metadata(conn_ro)
            conn_ro.close()
        except FileNotFoundError:
            existing = None
        if existing and existing.get("git_sha") == current_sha:
            print(f"No source changes since {current_sha[:12]} -- reusing cached graph")
            conn = store.init_db(output_db)
            try:
                with conn:
                    embedded = _build_embeddings(conn, embedding_model) if embeddings == "local" else 0
                    store.save_metadata(
                        conn,
                        schema_version=schema.load_schema()["version"],
                        project_path=project,
                        git_sha=current_sha,
                        codegraph_provenance=existing.get("codegraph_provenance"),
                    )
            finally:
                conn.close()
            if embedded:
                print(f"  {embedded} local embedding(s) indexed")
            print(f"Done. Graph already current at {output_db}")
            if profile:
                elapsed = time.perf_counter() - started_at
                size = output_db.stat().st_size if output_db.exists() else 0
                print(f"  profile: build_seconds={elapsed:.3f} db_bytes={size} cached=true")
            return

    print("Synchronizing CodeGraph index...")
    codegraph_state = structural.prepare_codegraph(project)

    print("Extracting structural nodes/edges from CodeGraph...")
    raw_export = structural.export_raw(project)
    extracted = structural.extract_nodes_and_edges(
        project,
        index_last_indexed=codegraph_state["index_last_indexed"],
        raw=raw_export,
        include_provenance=False,
    )
    nodes, edges, warnings = cast(
        tuple[list[dict], list[dict], list[str]],
        extracted,
    )
    codegraph_state["raw_export_sha256"] = structural.raw_export_sha256(raw_export)
    codegraph_state["raw_node_count"] = len(raw_export.get("nodes", []))
    codegraph_state["raw_edge_count"] = len(raw_export.get("edges", []))
    print(f"  {len(nodes)} nodes, {len(edges)} edges, {len(warnings)} warnings")

    decisions = []
    if decisions_dir.exists():
        print(f"Parsing decisions from {decisions_dir}...")
        decisions = semantic.parse_all_adrs(decisions_dir)
        print(f"  {len(decisions)} decisions parsed")
    else:
        print(f"No decisions directory found at {decisions_dir}, skipping semantic layer")

    doc_paths = _discover_docs(project)
    topics = []
    if doc_paths:
        print(f"Parsing {len(doc_paths)} general documentation file(s)...")
        topics = semantic.parse_all_docs(doc_paths)
        print(f"  {len(topics)} topics parsed")

    stories = []
    stories_dir = project / "docs" / "stories"
    if stories_dir.exists():
        print(f"Parsing stories from {stories_dir}...")
        stories = semantic.parse_all_stories(stories_dir)
        print(f"  {len(stories)} stories parsed")
    else:
        print("No stories directory found, skipping story ingestion")

    affects_edges = []
    affects_evidence = []
    unresolved_references = []
    if decisions:
        print("Linking decisions to code...")
        decision_links = merger.resolve_decisions_to_code(decisions, nodes)
        affects_edges = decision_links.edges
        affects_evidence.extend(decision_links.evidence)
        unresolved_references.extend(decision_links.unresolved)
        print(f"  {len(affects_edges)} AFFECTS edges (decisions) found")

    topic_affects_edges = []
    if topics:
        print("Linking doc topics to code...")
        topic_links = merger.resolve_topics_to_code(topics, nodes)
        topic_affects_edges = topic_links.edges
        affects_evidence.extend(topic_links.evidence)
        unresolved_references.extend(topic_links.unresolved)
        print(f"  {len(topic_affects_edges)} AFFECTS edges (topics) found")

    story_affects_edges = []
    if stories:
        print("Linking stories to code...")
        story_links = merger.resolve_stories_to_code(stories, nodes)
        story_affects_edges = story_links.edges
        affects_evidence.extend(story_links.evidence)
        unresolved_references.extend(story_links.unresolved)
        print(f"  {len(story_affects_edges)} AFFECTS edges (stories) found")

    if unresolved_references:
        print(f"  {len(unresolved_references)} unresolved semantic reference(s)")
        grouped = merger.classify_unresolved(unresolved_references)
        for category in sorted(grouped):
            refs = grouped[category]
            print(f"    [{category}] {len(refs)} refs")
            if category == "other" or len(refs) <= 5:
                for unresolved in refs:
                    print(
                        f"      {unresolved['source']}: "
                        f"{unresolved['reference']!r} ({unresolved['reason']})"
                    )
    else:
        print("  0 unresolved semantic references")


    print(f"Writing graph to {output_db}...")
    conn = store.init_db(output_db)
    try:
        with conn:
            store.save_structural_nodes(conn, nodes)
            source_chunks = source_scan.collect_source_chunks(nodes, project)
            store.replace_source_chunks(conn, source_chunks)
            store.save_decision_nodes(conn, decisions)
            store.save_topic_nodes(conn, topics)
            store.save_story_nodes(conn, stories)
            store.save_edges(conn, edges)
            store.save_edges(conn, affects_edges)
            store.save_edges(conn, topic_affects_edges)
            store.save_edges(conn, story_affects_edges)
            store.save_affects_evidence(conn, affects_evidence)

            from emp_kg.merger import _topic_id

            current_structural_ids = {node["id"] for node in nodes}
            current_decision_ids = {
                f"decision_ref:{decision['id']}" for decision in decisions
            }
            current_topic_ids = {_topic_id(t) for t in topics}
            current_story_ids = {f"story:{s['id']}" for s in stories}
            current_edges = {
                (edge["source"], edge["target"], edge["kind"])
                for edge in [
                    *edges, *affects_edges,
                    *topic_affects_edges, *story_affects_edges,
                ]
            }
            removed = store.reconcile(
                conn,
                current_structural_ids=current_structural_ids,
                current_decision_ids=current_decision_ids,
                current_topic_ids=current_topic_ids,
                current_story_ids=current_story_ids,
                current_edges=current_edges,
            )
            store.rebuild_fts(conn)
            embedded = _build_embeddings(conn, embedding_model) if embeddings == "local" else 0
            store.validate_graph(conn)
            store.save_metadata(
                conn,
                schema_version=schema.load_schema()["version"],
                project_path=project,
                git_sha=current_sha,
                codegraph_provenance=codegraph_state,
            )
    finally:
        conn.close()

    print(f"  {len(source_chunks)} source chunk(s) indexed")
    print(
        "  Reconciled stale records: "
        f"{removed['stale_structural_nodes_removed']} structural nodes, "
        f"{removed['stale_decision_nodes_removed']} decisions, "
        f"{removed['stale_topic_nodes_removed']} topics, "
        f"{removed['stale_story_nodes_removed']} stories, "
        f"{removed['stale_edges_removed']} edges removed"
    )
    if embedded:
        print(f"  {embedded} local embedding(s) indexed")
    print(f"Done. Graph saved to {output_db}")
    if profile:
        elapsed = time.perf_counter() - started_at
        size = output_db.stat().st_size if output_db.exists() else 0
        print(f"  profile: build_seconds={elapsed:.3f} db_bytes={size}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _read_db(db_path: str):
    return store.open_readonly(Path(db_path))


def _run_benchmark(
    db_path: str,
    cases_path: str,
    mode: str,
    k: int | None,
    embedding_model: str,
    as_json: bool,
) -> None:
    cases = benchmark.load_cases(Path(cases_path))
    result = benchmark.run_benchmark(
        Path(db_path), cases, mode=mode, default_k=k, embedding_model=embedding_model
    )
    if as_json:
        _print_json(result)
        return
    aggregate = result["aggregate"]
    print(
        f"{result['mode']}: {aggregate['case_count']} cases | "
        f"Recall@K {aggregate['recall_at_k']:.3f} | "
        f"Precision@K {aggregate['precision_at_k']:.3f} | MRR {aggregate['mrr']:.3f}"
    )
    for case in result["cases"]:
        print(
            f"  {case['id']}: {case['hit_count']}/{len(case['expected_node_ids'])} hits | "
            f"MRR {case['mrr']:.3f}"
        )


def _run_stats(db_path: str, as_json: bool) -> None:
    with closing(_read_db(db_path)) as conn:
        result = store.graph_stats(conn)
    if as_json:
        _print_json(result)
        return
    print(f"{result['nodes']} nodes, {result['edges']} edges")
    print("Nodes by kind:")
    for kind, count in result["nodes_by_kind"].items():
        print(f"  {kind}: {count}")
    print("Edges by kind:")
    for kind, count in result["edges_by_kind"].items():
        print(f"  {kind}: {count}")


def _run_search(db_path: str, term: str, as_json: bool) -> None:
    with closing(_read_db(db_path)) as conn:
        result = store.query_by_name(conn, term)
    if as_json:
        _print_json(result)
        return
    for node in result:
        print(f"{node['id']}\t{node.get('kind', '')}\t{node.get('name', '')}")


def _run_neighbors(db_path: str, node_id: str, direction: str, as_json: bool) -> None:
    with closing(_read_db(db_path)) as conn:
        result = store.neighbors(conn, node_id, direction)
    if as_json:
        _print_json(result)
        return
    for record in result:
        edge = record["edge"]
        node = record["node"]
        print(
            f"{record['direction']}\t{edge['kind']}\t"
            f"{node['id']}\t{node.get('name', '')}"
        )


def _run_impact(
    db_path: str,
    target: str,
    change_type: str,
    depth: int,
    direction: str,
    as_json: bool,
) -> None:
    report = impact.analyze_impact(
        Path(db_path),
        impact.ImpactRequest(target, change_type, depth, direction),
    )
    payload = asdict(report)
    if as_json:
        _print_json(payload)
        return
    print(f"Impact: {target} [{report.risk_level}] {len(report.directly_affected)} direct, {len(report.transitive_affected)} transitive")
    for item in report.directly_affected:
        print(f"  direct {item['edge_kind']} {item['id']} ({item['kind']})")
    for item in report.transitive_affected:
        print(f"  depth {item['distance']} {item['edge_kind']} {item['id']} ({item['kind']})")
    if report.recommended_tests:
        print("Recommended tests: " + ", ".join(report.recommended_tests))


def _run_compress(
    db_path: str,
    task: str,
    target_tokens: int,
    max_facts: int,
    as_json: bool,
) -> None:
    result = context.build_context(context.ContextRequest(task), Path(db_path))
    compressed = compression.compress_context(
        result,
        compression.CompressionConfig(
            target_tokens=target_tokens,
            max_facts=max_facts,
        ),
    )
    payload = asdict(compressed)
    if as_json:
        _print_json(payload)
        return
    print(
        f"Compressed {compressed.original_tokens} to {compressed.compressed_tokens} tokens "
        f"(ratio {compressed.compression_ratio:.3f}); "
        f"{len(compressed.retained_facts)} facts retained"
    )
    for fact in compressed.retained_facts:
        print(f"  {fact.node_id}\t{fact.relevance_score:.3f}\t{fact.name}")


def _run_reason(
    db_path: str,
    task: str,
    provider: str,
    model: str,
    max_tokens: int,
    timeout: float,
    as_json: bool,
) -> None:
    try:
        result = system.reason_over_context(
            context.ContextRequest(task),
            Path(db_path),
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    except (RuntimeError, ValueError, context.ContextRetrievalError) as exc:
        raise SystemExit(f"reasoning failed: {exc}") from exc
    payload = {
        "context": result.context.to_dict(),
        "reasoning": asdict(result.reasoning),
        "elapsed_ms": result.elapsed_ms,
    }
    if as_json:
        _print_json(payload)
        return
    print(f"provider={result.reasoning.provider} model={result.reasoning.model} elapsed_ms={result.elapsed_ms}")
    print(result.reasoning.text or result.reasoning.error or "no reasoning output")


def _run_path(
    db_path: str,
    source_id: str,
    target_id: str,
    max_depth: int,
    as_json: bool,
    direction: str = "out",
) -> None:
    with closing(_read_db(db_path)) as conn:
        result = store.shortest_path(conn, source_id, target_id, max_depth, direction)
    if as_json:
        _print_json(result)
        return
    if not result["nodes"]:
        print("No path found")
        return
    print(result["nodes"][0]["id"])
    for edge, node in zip(result["edges"], result["nodes"][1:]):
        print(f"  -[{edge['kind']}]-> {node['id']}")


def _run_context(
    db_path: str,
    task_description: str,
    lane: str | None,
    affected_files: list[str],
    max_tokens: int | None,
    as_json: bool,
    output_format: str = "text",
    direction: Literal["in", "out", "both"] = "both",
    request_class: str | None = None,
    risk_flags: list[str] | None = None,
    scope: Literal["behavioral", "cosmetic"] = "behavioral",
    semantic_mode: Literal["off", "local"] = "off",
    synonym_mode: Literal["off", "auto"] = "auto",
    embedding_model: str = DEFAULT_MODEL,
    semantic_weight: float = 0.85,
    encoding: str | None = None,
) -> None:
    request = context.ContextRequest(
        task_description=task_description,
        lane=lane,
        affected_files=affected_files,
        max_tokens=max_tokens,
        direction=direction,
        request_class=request_class,
        risk_flags=set(risk_flags or []),
        scope=scope,
        semantic_mode=semantic_mode,
        synonym_mode=synonym_mode,
        embedding_model=embedding_model,
        semantic_weight=semantic_weight,
        encoding=encoding,
    )
    result = context.build_context(request, Path(db_path))
    if as_json or output_format == "json":
        _print_json(result.to_dict())
        return
    if output_format == "markdown":
        print(context.render_markdown(result), end="")
        return
    print(
        f"Lane: {result.lane} ({result.classification_reason}) | "
        f"{len(result.facts)} facts | ~{result.total_tokens} tokens"
    )
    for fact in result.facts:
        source = f" [{fact.source_path}]" if fact.source_path else ""
        print(f"\n[{fact.relevance_score:.2f}] {fact.kind} {fact.name}{source}")
        print(f"  {fact.snippet}")
        if fact.score_components:
            components = ", ".join(
                f"{name}={value:.3f}"
                for name, value in sorted(fact.score_components.items())
            )
            print(f"  Score components: {components}")
        if fact.retrieval_metadata:
            metadata = ", ".join(
                f"{name}={value}"
                for name, value in sorted(fact.retrieval_metadata.items())
            )
            print(f"  Retrieval metadata: {metadata}")
        if fact.match_location:
            location = fact.match_location
            print(
                f"  Match location: {location.get('source_path', '')}:"
                f"{location.get('start_line', '')}-{location.get('end_line', '')} "
                f"(chunk {location.get('chunk_ordinal', '')})"
            )
        if fact.provenance:
            provenance = fact.provenance
            terms = ", ".join(provenance.matched_terms) or "none"
            print(f"  Seed: {provenance.seed_id} ({terms})")
            for step in provenance.path:
                print(
                    f"  Path: {step.source_id} -[{step.kind}]-> {step.target_id} "
                    f"({step.direction})"
                )
                for evidence in step.evidence:
                    lines = evidence.get("source_lines", {})
                    print(
                        f"  Evidence: {evidence.get('source_path', '')}:"
                        f"{lines.get('start', '')}-{lines.get('end', '')} "
                        f"{evidence.get('reference', '')}"
                    )
            for alternate_index, alternate in enumerate(fact.alternate_paths, start=1):
                print(f"  Alternative path {alternate_index}: Seed: {alternate.seed_id}")
                for step in alternate.path:
                    print(
                        f"  Path: {step.source_id} -[{step.kind}]-> {step.target_id} "
                        f"({step.direction})"
                    )
                    for evidence in step.evidence:
                        lines = evidence.get("source_lines", {})
                        print(
                            f"  Evidence: {evidence.get('source_path', '')}:"
                            f"{lines.get('start', '')}-{lines.get('end', '')} "
                            f"{evidence.get('reference', '')}"
                        )


def _add_read_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", required=True, help="Path to graph database.")
    parser.add_argument("--json", action="store_true", dest="as_json")



#Salar Changes

class _ProjectChangeHandler(FileSystemEventHandler):
    def __init__(self, project_path: str, db_path: str | None, embeddings: str, embedding_model: str) -> None:
        self.project_path = project_path
        self.db_path = db_path
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        self.last_triggered = 0.0

    def on_any_event(self, event):
        if event.is_directory:
            return
        valid_exts = (".py", ".ts", ".rs", ".go", ".c", ".cpp", ".h", ".java", ".md")
        if not event.src_path.endswith(valid_exts):
            return
        now = time.time()
        if now - self.last_triggered > 1.5:
            self.last_triggered = now
            print(f"\n[Watch] Change detected in {event.src_path}. Rebuilding graph...")
            try:
                build(
                    project_path=self.project_path,
                    db_path=self.db_path,
                    force=False,
                    embeddings=self.embeddings,
                    embedding_model=self.embedding_model,
                )
            except Exception as e:
                print(f"Error during auto-rebuild: {e}")


def _run_watch(project_path: str, db_path: str | None, embeddings: str, embedding_model: str) -> None:
    event_handler = _ProjectChangeHandler(project_path, db_path, embeddings, embedding_model)
    observer = Observer()
    observer.schedule(event_handler, path=project_path, recursive=True)
    observer.start()
    print(f"Watching project directory for changes: {project_path}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    




def main() -> None:
    parser = argparse.ArgumentParser(
        prog="emp-kg",
        description="Knowledge Graph Construction Engine for EMP.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the knowledge graph for a project.")
    build_parser.add_argument("--project", required=True, help="Path to the target project.")
    build_parser.add_argument("--db", required=False, help="Output database path (default: <project>/emp-kg.db).")
    build_parser.add_argument(
        "--force", action="store_true", help="Rebuild even if the source revision is unchanged."
    )
    
    watch_parser = subparsers.add_parser("watch", help="Watch project files for changes and auto-rebuild graph.")
    watch_parser.add_argument("--project", required=True, help="Path to the target project.")
    watch_parser.add_argument("--db", required=False, help="Output database path.")
    watch_parser.add_argument(
        "--embeddings", choices=("off", "local"), default="off",
        help="Optional local embedding index; default off.",
    )
    watch_parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    
    
    build_parser.add_argument(
        "--embeddings", choices=("off", "local"), default="off",
        help="Optional local embedding index; default off.",
    )
    build_parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    build_parser.add_argument("--profile", action="store_true", help="Print build timing and database size.")

    benchmark_parser = subparsers.add_parser(
        "benchmark-context", help="Measure labeled Context Builder retrieval quality."
    )
    benchmark_parser.add_argument("--cases", required=True, help="Versioned benchmark JSON fixture.")
    benchmark_parser.add_argument("--mode", choices=("bm25", "synonyms", "local"), default="bm25")
    benchmark_parser.add_argument("--k", type=_positive_int, default=None)
    benchmark_parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    _add_read_options(benchmark_parser)

    stats_parser = subparsers.add_parser("stats", help="Show graph counts.")
    _add_read_options(stats_parser)

    search_parser = subparsers.add_parser("search", help="Search graph nodes by name.")
    search_parser.add_argument("term", help="FTS query for node names.")
    _add_read_options(search_parser)

    neighbors_parser = subparsers.add_parser("neighbors", help="Show adjacent graph nodes.")
    neighbors_parser.add_argument("node_id", help="Node ID to inspect.")
    neighbors_parser.add_argument(
        "--direction", choices=("in", "out", "both"), default="both"
    )
    _add_read_options(neighbors_parser)

    subparsers.add_parser("mcp-server", help="Run the MCP stdio JSON-RPC server.")

    impact_parser = subparsers.add_parser("impact", help="Analyze change impact blast radius.")
    impact_parser.add_argument("--db", required=True)
    impact_parser.add_argument("--target", required=True)
    impact_parser.add_argument("--change-type", choices=("modified", "deleted", "renamed"), default="modified")
    impact_parser.add_argument("--depth", type=_positive_int, default=3)
    impact_parser.add_argument("--direction", choices=("in", "out", "both"), default="both")
    impact_parser.add_argument("--json", action="store_true", dest="as_json")

    compress_parser = subparsers.add_parser("compress-context", help="Compress a context packet.")
    compress_parser.add_argument("--db", required=True)
    compress_parser.add_argument("--target-tokens", type=_positive_int, default=2000)
    compress_parser.add_argument("--max-facts", type=_positive_int, default=50)
    compress_parser.add_argument("--json", action="store_true", dest="as_json")

    reason_parser = subparsers.add_parser("reason", help="Build context and run AI reasoning.")
    reason_parser.add_argument("task")
    reason_parser.add_argument("--db", required=True)
    reason_parser.add_argument("--provider", choices=("noop", "anthropic"), default="noop")
    reason_parser.add_argument("--model", default="claude-opus-5")
    reason_parser.add_argument("--max-tokens", type=_positive_int, default=4096)
    reason_parser.add_argument("--timeout", type=float, default=30.0)
    reason_parser.add_argument("--json", action="store_true", dest="as_json")

    path_parser = subparsers.add_parser("path", help="Find a shortest directed path.")
    path_parser.add_argument("source_id", help="Starting node ID.")
    path_parser.add_argument("target_id", help="Destination node ID.")
    path_parser.add_argument("--max-depth", type=_positive_int, default=10)
    path_parser.add_argument(
        "--direction", choices=("out", "in", "both"), default="out"
    )
    _add_read_options(path_parser)

    context_parser = subparsers.add_parser(
        "context",
        help="Build task-specific context from the knowledge graph.",
    )
    context_parser.add_argument("task", help="Task description to retrieve context for.")
    context_parser.add_argument(
        "--lane",
        choices=("tiny", "normal", "high-risk"),
        default=None,
        help="Context breadth; defaults to automatic classification.",
    )
    context_parser.add_argument(
        "--affected-files",
        nargs="*",
        default=[],
        help="Optional affected file paths used as extra search seeds.",
    )
    context_parser.add_argument("--max-tokens", type=_positive_int, default=None)
    context_parser.add_argument(
        "--direction",
        choices=("in", "out", "both"),
        default="both",
        help="Graph traversal direction for context expansion.",
    )
    context_parser.add_argument(
        "--request-class",
        choices=(
            "new_spec", "spec_slice", "change_request", "new_initiative",
            "maintenance_request", "harness_improvement",
        ),
        default=None,
        dest="request_class",
    )
    context_parser.add_argument(
        "--risk-flag",
        action="append",
        choices=(
            "auth", "authorization", "data_model", "audit_security",
            "external_systems", "public_contracts", "cross_platform",
            "existing_behavior", "weak_proof", "multi_domain",
        ),
        default=[],
        dest="risk_flags",
    )
    context_parser.add_argument(
        "--scope",
        choices=("behavioral", "cosmetic"),
        default="behavioral",
        help="Declared scope; cosmetic scope narrows heuristic hard-gate hits only.",
    )
    context_parser.add_argument(
        "--synonym-mode",
        choices=("off", "auto"),
        default="auto",
        help="Deterministic synonym expansion; default auto.",
    )
    context_parser.add_argument(
        "--semantic-mode",
        choices=("off", "local"),
        default="off",
        help="Optional local embedding retrieval; default off.",
    )
    context_parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    context_parser.add_argument("--semantic-weight", type=float, default=0.85)
    context_parser.add_argument("--encoding", default=None, help="Tokenizer for budget accounting (e.g. tiktoken, estimate).")
    context_parser.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
        dest="output_format",
        help="Output format; --json remains an alias for JSON.",
    )
    _add_read_options(context_parser)

    args = parser.parse_args()

    if args.command == "mcp-server":
        run_stdio()
    elif args.command == "impact":
        _run_impact(args.db, args.target, args.change_type, args.depth, args.direction, args.as_json)
    elif args.command == "compress-context":
        _run_compress(args.db, args.task, args.target_tokens, args.max_facts, args.as_json)
    elif args.command == "reason":
        _run_reason(args.db, args.task, args.provider, args.model, args.max_tokens, args.timeout, args.as_json)
    elif args.command == "benchmark-context":
        _run_benchmark(
            args.db, args.cases, args.mode, args.k, args.embedding_model, args.as_json
        )
    elif args.command == "build":
        build(
            args.project, args.db, force=args.force,
            embeddings=args.embeddings, embedding_model=args.embedding_model,
            profile=args.profile,
        )
    elif args.command == "stats":
        _run_stats(args.db, args.as_json)
    elif args.command == "search":
        _run_search(args.db, args.term, args.as_json)
    elif args.command == "neighbors":
        _run_neighbors(args.db, args.node_id, args.direction, args.as_json)
    elif args.command == "path":
        _run_path(args.db, args.source_id, args.target_id, args.max_depth, args.as_json, args.direction)
    elif args.command == "context":
        _run_context(
            args.db,
            args.task,
            args.lane,
            args.affected_files,
            args.max_tokens,
            args.as_json,
            args.output_format,
            args.direction,
            args.request_class,
            args.risk_flags,
            args.scope,
            args.semantic_mode,
            args.synonym_mode,
            args.embedding_model,
            args.semantic_weight,
            args.encoding,
        )
    elif args.command == "watch":
        _run_watch(args.project, args.db, args.embeddings, args.embedding_model)


if __name__ == "__main__":
    main()
