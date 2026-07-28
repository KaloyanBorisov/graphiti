# GLiNER2 Hybrid LLM Client Example (Experimental)

> **Note:** The `GLiNER2Client` is experimental and may change in future releases.

This example demonstrates using [GLiNER2](https://github.com/fastino-ai/GLiNER2) as a hybrid LLM client for Graphiti. GLiNER2 handles entity extraction (NER) locally on CPU, while a general-purpose LLM client handles edge/fact extraction, deduplication, summarization, and other reasoning tasks.

- Paper: [GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface](https://arxiv.org/abs/2507.18546)
- Models on HuggingFace:
  - [fastino/gliner2-base-v1](https://huggingface.co/fastino/gliner2-base-v1) (205M params)
  - [fastino/gliner2-large-v1](https://huggingface.co/fastino/gliner2-large-v1) (340M params)
  - [fastino/gliner2-multi-v1](https://huggingface.co/fastino/gliner2-multi-v1) (multilingual)

## How It Works

Graphiti doesn't know it's talking to two different models. Normally you'd hand Graphiti one "LLM client" and it uses that single client for everything — extracting entities, extracting facts, deduplicating, summarizing. In this example, the object you hand it *looks* like one LLM client, but it's actually a wrapper: a GLiNER2 client that has a full Gemini client tucked inside it. Every time Graphiti asks this wrapper to do something, the wrapper first checks: is this an entity-extraction request, or something else? If it's entity extraction, GLiNER2 handles it directly, locally, on the CPU. For anything else, the wrapper just quietly forwards the request to the Gemini client it's holding onto. Graphiti never has to know the split is happening — as far as it's concerned, it's making one API.

So here's what happens when you feed it one episode of text:

1. Graphiti asks "what entities are in this text?" — GLiNER2 answers that, fast and cheap, since it's just scanning the text for spans matching your predefined categories (Person, Organization, Location, Initiative).
2. Graphiti asks "what facts connect these entities?" — this gets routed to Gemini, because relationship reasoning needs a real language model.
3. Graphiti then asks "have I seen this entity/fact before?" — also routed to Gemini, since deciding whether "Harris" in a Spanish sentence is the same person as "Kamala Harris" in an earlier English one takes judgment, not just pattern matching.
4. If any entity needs a written summary, that's another Gemini call.
5. Everything — the entities, the facts, the summaries — gets written into Neo4j as graph nodes and edges, with timestamps attached where the text implied a date or a validity period.

Search works differently again. When you later ask a question, Graphiti doesn't call GLiNER2 or Gemini's reasoning at all — it uses a third piece, the Gemini *embedder*, to turn your question into a vector and find graph facts whose meaning is closest to it, combined with plain keyword matching and following graph connections. So across the whole example, Gemini does two separate jobs (reasoning-through-the-wrapper, and embedding-for-search), and GLiNER2 does one narrow job (spotting entities) — Graphiti's own code is what stitches all three together into one pipeline, run once per episode.

The two extraction steps in more detail:

**1. GLiNER2 finds the "things" in the text.** You tell it up front what kinds of entities to look for — people, organizations, locations, initiatives — along with a short description of each. GLiNER2 isn't a chatbot-style model that writes an answer; it's a small, specialized model that reads the text once and directly scores every plausible word or phrase in it against your list of categories. Anything that scores above the confidence threshold gets pulled out as a match: "Kamala Harris" scores high against "Person," "California" scores high against "Location," and so on. Because it's just scoring spans of text rather than generating language, it's cheap and fast enough to run on a CPU instead of calling out to an API. The result is a plain list of named things found in the episode, each tagged with its category.

**2. Gemini figures out how those things relate.** Now that the entities are known, the episode text goes to Gemini with a different question: given these specific entities, what facts connect them? Gemini is a full reasoning model, so this step suits it — it reads the sentence structure, figures out which two entities a statement is really about, and phrases it as a fact, like "Kamala Harris held the position of Attorney General of California." It's also told the rules to follow: only connect two different entities (not an entity to itself), keep specific details rather than vaguing them out, don't repeat a fact it already extracted from an earlier episode unless this version is more detailed, and figure out date ranges where the text implies them (e.g., "she was in office from 2011 to 2017" becomes a validity window on that fact). Gemini also sees the previous episode's text and any facts already known, so it can avoid duplicates and resolve references that only make sense with earlier context.

**Why split the work this way:** finding entities is a narrow, repetitive task — a small local model handles it well and for free. Figuring out relationships and reasoning about time, duplicates, and phrasing takes real language understanding — that's worth paying for a capable model to do.

## How Deduplication Works

This example deliberately repeats the same handful of entities (Kamala Harris, Gavin Newsom, California, San Francisco) across languages and formats, to show off how Graphiti avoids creating duplicate nodes and facts every time an episode mentions something it already knows about. It works in escalating, increasingly expensive steps — cheap checks first, an LLM call only when there's real ambiguity to resolve:

**Entities.** When GLiNER2 pulls a new entity out of an episode, Graphiti doesn't just add it — it first searches for existing nodes with a similar *name embedding*, scoped to the same dataset. If a candidate has the exact same name (after normalizing case/punctuation), it's merged immediately, no LLM involved. If names are close-but-not-identical, a fast fuzzy-matching check (comparing text shingles) can also merge them for free. Only when neither check is confident — for example, "Harris" in a Spanish sentence next to an English "Kamala Harris" node, where the strings genuinely don't match — does Graphiti ask Gemini to make the call.

**Facts.** New facts go through the same idea: Graphiti gathers existing facts between the same two entities as duplicate candidates, plus a broader search for facts that might be *related* (not necessarily identical). An exact text match reuses the existing fact instantly. Otherwise, Gemini looks at both candidate sets and decides whether the new fact is a duplicate, a genuinely new fact, or a contradiction of an old one.

**Contradictions and time.** This last case is where the bi-temporal model shows up: if a new fact contradicts an older one (e.g., a later episode says someone left a position they were previously said to hold), Graphiti doesn't delete the old fact — it marks it invalid as of the new fact's timestamp, so the graph still remembers it was true for a while. This is exactly how "Attorney General from 2011 to 2017" ends up as a fact with an actual validity window instead of a permanent, undated claim.

## Prerequisites

- Python 3.11+
- Neo4j 5.26+ ([Neo4j Desktop](https://neo4j.com/download/) or Docker)
- An LLM provider API key (Google, OpenAI, Anthropic, etc.)

## Setup

```bash
# Install graphiti with the gliner2 extra
pip install graphiti-core[gliner2]

# Copy and configure environment variables (from the repo root)
cp ../../.env.example ../../.env
```

The GLiNER2 model weights are downloaded automatically on first run.

## LLM and Embedding Providers

The example uses Google Gemini (`gemini-2.5-flash`) for the LLM and embeddings, but `GLiNER2Client` accepts any Graphiti `LLMClient`. To swap providers, replace `GeminiClient` and `GeminiEmbedder` with the equivalent from another provider:

- `graphiti_core.llm_client.openai_client.OpenAIClient`
- `graphiti_core.llm_client.anthropic_client.AnthropicClient`
- `graphiti_core.llm_client.groq_client.GroqClient`
- `graphiti_core.embedder.openai.OpenAIEmbedder`
- `graphiti_core.embedder.voyage.VoyageAIEmbedder`

## Configuration

| Parameter | Description | Default |
|---|---|---|
| `threshold` | GLiNER2 confidence threshold (0.0-1.0). Higher values reduce spurious extractions. | `0.5` |
| `GLINER2_MODEL` | HuggingFace model ID | `fastino/gliner2-large-v1` |

## Running

```bash
python gliner2_neo4j.py
```
