# Document Chunking with Contextualized Retrieval for Graphiti

This is a port of Zep's [chunking-example](https://github.com/getzep/zep/tree/main/examples/python/chunking-example)
to the Graphiti framework. It demonstrates Anthropic's [**Contextual
Retrieval**](https://www.anthropic.com/engineering/contextual-retrieval)
technique: the document is chunked, an LLM generates a short contextual
description for each chunk, and the contextualized chunks are ingested
into Graphiti as episodes.

By default this uses Claude with explicit prompt-cache breakpoints
(`cache_control`), the mechanism the technique was designed around: the
full document is cached once and re-read at a 90% discount for every
chunk instead of being re-billed at full price per call. Pass `--provider
openai` to use an OpenAI model instead -- it relies on OpenAI's automatic
prompt caching (no explicit cache markers, ~50% discount), which works
but wasn't the technique's original design point. Anthropic's own writeup
reports this approach cuts retrieval failure rates by 35% (49% combined
with BM25, 67% with reranking added) versus chunking without context.

## Why Contextualized Retrieval?

Chunking a document and processing each piece in isolation loses context.
A chunk that says "Employees may carry over up to 5 unused PTO days" is
ambiguous on its own -- which company, which policy, as of when?

Contextualized retrieval solves this by prepending a brief description
that situates each chunk within the full document before it's ingested:

**Before:**
```
Employees may carry over up to 5 unused PTO days to the following year.
```

**After:**
```
This chunk describes ACME Corporation's PTO carryover policy from the
Employee Handbook effective January 1, 2024. It appears in the Time Off
and Leave Policies section.

---

Employees may carry over up to 5 unused PTO days to the following year.
```

This gives Graphiti's entity/fact extraction more context to work with,
which particularly helps for chunks referencing "the policy", "the plan",
or other document-local terms that would otherwise be meaningless.

**Note:** Graphiti already applies its own automatic chunking to dense
episodes before extraction (see
[examples/quickstart/dense_vs_normal_ingestion.py](../quickstart/dense_vs_normal_ingestion.py)),
driven by entity density rather than a fixed character count. That's a
different concern from this script -- this script pre-chunks a whole
*document* into episode-sized pieces and contextualizes each one before
handing it to Graphiti at all.

## What's different from the Zep version

- **Ingestion target**: `zep_client.graph.add(user_id=...)` becomes
  `graphiti.add_episode(group_id=...)`. Graphiti uses `group_id` as its
  graph-namespacing concept (equivalent to Zep's per-user graph) -- pass
  whatever id you want this document's episodes scoped under (a user id,
  a document id, a project id, etc).
- **No user provisioning step**: Zep requires a `user.get`/`user.add`
  check before ingesting. Graphiti has no separate user object -- a
  `group_id` is created implicitly on first use.
- **No 10K-character episode cap**: Zep enforces a 10,000-character
  episode limit, so the original script truncates the contextual prefix
  if needed. Graphiti has no equivalent hard cap (practical limits come
  from your LLM's context window instead), so that truncation step is
  dropped.
- **No `--wait` flag**: Zep processes episodes asynchronously after
  `graph.add` returns, so the original script offers a `--wait` option
  to block until processing finishes. Graphiti's `add_episode` already
  runs extraction synchronously and returns only once the episode has
  been fully processed into nodes and edges, so there's nothing to wait
  for.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables (or set them in the repo root `.env`):
   ```bash
   # Required for the default provider (--provider anthropic)
   export ANTHROPIC_API_KEY=your_anthropic_api_key

   # Required only if using --provider openai
   export OPENAI_API_KEY=your_openai_api_key

   # Optional Neo4j connection parameters (defaults shown)
   export NEO4J_URI=bolt://localhost:7687
   export NEO4J_USER=neo4j
   export NEO4J_PASSWORD=password
   ```

## Usage

```bash
python chunk_and_ingest.py sample_document.txt --group-id acme-handbook
```

Use OpenAI instead of Anthropic for contextualization:

```bash
python chunk_and_ingest.py sample_document.txt --group-id acme-handbook --provider openai
```

### Dry run

Chunk and contextualize without ingesting into Graphiti:

```bash
python chunk_and_ingest.py sample_document.txt --group-id acme-handbook --dry-run
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `document` | Path to the document to process | (required) |
| `--group-id` | Graphiti `group_id` to scope this document's episodes into | (required) |
| `--provider` | LLM used to contextualize chunks: `anthropic` or `openai` | `anthropic` |
| `--dry-run` | Chunk and contextualize without ingesting into Graphiti | False |

## How It Works

1. **Document Chunking**: paragraph-first, sentence fallback, with a
   50-character overlap between chunks (configured via the `CHUNK_SIZE`
   and `CHUNK_OVERLAP` constants at the top of the script).
2. **Contextualization**: each chunk is sent to an LLM along with the
   full document, which returns a short description situating the
   chunk.
   - With `--provider anthropic` (default): the document is placed in
     its own `cache_control`-marked content block. The first chunk is
     sent alone to prime the cache (Anthropic's cache is only populated
     once a request completes, so firing every chunk concurrently from
     a cold cache would make them all pay the cache-write price), then
     the remaining chunks are contextualized concurrently against the
     now-warm cache, read at a 90% discount, with a 1-hour TTL.
   - With `--provider openai`: the document and chunk are sent together
     in one prompt, relying on OpenAI's automatic caching (~50%
     discount) instead of an explicit cache breakpoint.
3. **Ingestion**: the contextualized chunk (context + `---` separator +
   original chunk text) is added to Graphiti via `add_episode`, which
   extracts entities (nodes) and facts (edges) from it and merges them
   into the graph under the given `group_id`.

## Sample Document

`sample_document.txt` is a short fictional employee handbook covering
remote work, time off, professional development, performance reviews,
conduct, security, and benefits -- structured content that benefits from
contextualization since many chunks reference "the policy" or "the
review" without repeating which one.
