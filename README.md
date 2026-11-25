# emp-kg

> **Knowledge Graph Construction Engine & Hybrid Context Retrieval CLI**  
> Built for codebases, architecture decisions, operational runbooks, technical documentation, and personal/domain memory.

```text
+-------------------------------------------------------------------+
|                         emp-kg CLI & MCP                          |
|  build | watch | ingest | ingest-url | remember | records         |
|  context | impact | compress-context | reason | path | benchmark  |
+-------------------------------------------------------------------+
                 |                                   |
                 v                                   v
+---------------------------------+   +-----------------------------+
|   Incremental & Graph Writers   |   |   Retrieval & Analysis      |
|  - Code & doc graph builder     |   |  - BM25 + FTS5 search       |
|  - Incremental file/dir ingest  |   |  - Synonym expansion        |
|  - HTML-to-text URL ingestion   |   |  - Local semantic RRF       |
|  - Natural-language memory note |   |  - Impact & shortest path   |
|  - Record list & cascade delete |   |  - Token budget compression |
+---------------------------------+   +-----------------------------+
                 |                                   |
                 +----------------+------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                   SQLite + FTS5 Graph Store                       |
|  nodes | edges | edge_evidence | source_chunks | node_embeddings  |
+-------------------------------------------------------------------+
```

---

## Key Capabilities

| Area                         | What It Does                                                                                                                                          | Key Commands / Modules                                                            |
|:---------------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------------------------------- |
| **Graph Construction**       | Builds structural code nodes/edges and links ADRs, docs, and stories into a SQLite + FTS5 store with reconciliation and validation.                   | `emp-kg build`, `emp-kg watch`, `store.py`, `merger.py`                           |
| **Incremental Ingestion**    | Ingests new or modified files/directories, skips unchanged content via SHA-256 hashes, and removes nodes for deleted files.                           | `emp-kg ingest`, `ingest.py`                                                      |
| **Web URL Ingestion**        | Fetches HTTP/HTTPS pages, strips `<script>`, `<style>`, and `<noscript>` tags, extracts clean text/title, and deduplicates unchanged pages.           | `emp-kg ingest-url`, `ingest.py`                                                  |
| **Memory & Records**         | Saves natural-language notes with deterministic `memory://note/<hash12>` URIs, lists records, and deletes nodes with FTS rebuild and cascade cleanup. | `emp-kg remember`, `emp-kg records`, `ingest.py`                                  |
| **Hybrid Retrieval & Tools** | Assembles token-budgeted context packets, analyzes blast radius, compresses context, runs benchmarks, and exposes an MCP stdio server.                | `emp-kg context`, `impact`, `compress-context`, `benchmark-context`, `mcp-server` |

---

## CLI Usage

### 1. Build, Watch & Inspect the Graph

```bash
emp-kg build --project /path/to/repo --db ./emp-kg.db --profile
emp-kg watch --project /path/to/repo --db ./emp-kg.db
emp-kg stats --db ./emp-kg.db --json
emp-kg search "build_context" --db ./emp-kg.db
emp-kg neighbors "func:..." --db ./emp-kg.db --direction both
emp-kg path "node:a" "node:b" --db ./emp-kg.db
```

### 2. Incremental Ingestion & Memory Management

```bash
# Incrementally ingest files or a directory
emp-kg ingest ./docs --db ./emp-kg.db --json

# Fetch and ingest a web URL
emp-kg ingest-url [https://example.org/runbook](https://example.org/runbook) --db ./emp-kg.db --json

# Store a natural-language memory note
emp-kg remember "OAuth2 refresh tokens expire after 14 days." \
  --title "Auth Token Policy" \
  --tag security \
  --db ./emp-kg.db \
  --json

# List or delete persisted records
emp-kg records list --db ./emp-kg.db --kind topic --json
emp-kg records delete --db ./emp-kg.db --id "topic:..." --json
```

### 3. Context Retrieval, Impact Analysis & MCP

```bash
emp-kg context "Investigate BGP failover routing behavior" --db ./emp-kg.db --format markdown
emp-kg impact --db ./emp-kg.db --target "emp_kg/context.py" --change-type modified
emp-kg compress-context --db ./emp-kg.db --target-tokens 2000 --max-facts 50
emp-kg benchmark-context --db ./emp-kg.db --cases ./benchmarks/cases.json --mode bm25
emp-kg mcp-server
```

---

## Recent Engineering & Testing Highlights

* Stabilized incremental file ingestion (`ingest_directory_incremental`), HTML URL extraction (`ingest_url`), natural-language note persistence (`ingest_note`), and record listing/deletion (`list_records`, `delete_record`) in `emp_kg/ingest.py` and `emp_kg/cli.py`[cite: 1, 2, 3].
* Cleaned up duplicate/unused imports and obsolete commented blocks across `emp_kg/cli.py`, `emp_kg/ingest.py`, and `tests/test_ingest.py`[cite: 1, 2, 3].
* Built and verified a 9-test suite in `tests/test_ingest.py` covering incremental lifecycles, single/missing paths, selective embeddings, HTML extraction, CLI JSON workflows, retrieval after ingestion/deletion, and edge cases (`URLError`, duplicate notes, and embedding cascade deletion)[cite: 1, 2, 3].
* Verified the full `pytest` suite across the repository with zero regressions[cite: 1, 2, 3].

---

## Development & Testing

```bash
pytest -v
```
