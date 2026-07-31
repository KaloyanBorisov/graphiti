# Podcast Transcript Example

This example demonstrates how to feed real-world, unstructured conversation data into Graphiti
and build a knowledge graph from it incrementally, with custom entity and edge types.

The transcript (`podcast_transcript.txt`) is an excerpt from a Freakonomics Radio episode:
Stephen Dubner interviewing Tania Tetlow, president of Fordham University.

## Files

- `podcast_transcript.txt` — raw transcript, split into blocks of `speaker_index (timestamp): text`
- `transcript_parser.py` — parses the raw transcript into structured `ParsedMessage` objects,
  mapping each speaker index to a name/role (e.g. index 1 = "Tania Tetlow, Guest") and converting
  relative timestamps (`23s`, `1m 4s`) into real datetimes, counting backward from "now" using the
  last timestamp in the file
- `podcast_runner.py` — the runnable example: ingests parsed messages into Graphiti and queries
  the resulting graph

## What `podcast_runner.py` does

1. Spins up a local embedded FalkorDB (no external database needed) and an OpenAI LLM client.
2. Takes a slice of the parsed messages (11 lines of dialogue) and adds them to Graphiti one at a
   time as "episodes" — each episode is one line of dialogue, tagged with its speaker and
   timestamp.
3. Before adding each episode, it retrieves the graph's memory of recent prior episodes, so
   Graphiti has conversational context (this lets it resolve references like "he" or "the
   university" correctly).
4. It defines custom entity types (`Person`, `City`) and custom edge types
   (`IS_PRESIDENT_OF`, `INTERPERSONAL_RELATIONSHIP`, `LOCATED_IN`) as Pydantic models, and maps
   which edge types are valid between which node type pairs (`edge_type_map`). This makes the LLM
   extract specifically-typed nodes and edges instead of a generic graph — e.g. recognizing "Tania
   Tetlow" as a `Person` who `IS_PRESIDENT_OF` "Fordham University". Some edge types are
   deliberately reused across multiple node-pair signatures (e.g. `INTERPERSONAL_RELATIONSHIP`
   applies to both `Person`-`Person` and `Person`-`Entity`) to exercise that case.
5. After ingestion, it prints a summary of LLM token usage by prompt type, for cost visibility.
6. Finally, it runs a few test search queries against the resulting graph (e.g. "Who is the
   president of Fordham University?") and prints back the facts and nodes Graphiti found —
   demonstrating that the graph now "knows" things it only learned from the transcript.

There's also a `use_bulk` path (`main(True)`) that adds all episodes at once via
`add_episode_bulk` instead of one at a time, for comparison.

## Prerequisites

- Python 3.12+ (required by the embedded `falkordblite` package)
- `OPENAI_API_KEY` environment variable (used for both LLM extraction and embeddings)
- Dependencies: `graphiti-core[falkordblite]`, `python-dotenv`

## Running it

```bash
cd examples/podcast
python podcast_runner.py
```

No external database setup is required — the script uses an embedded FalkorDB instance stored in
a temp file.
