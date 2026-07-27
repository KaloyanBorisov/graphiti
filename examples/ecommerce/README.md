# Ecommerce Example

This example (`runner.ipynb`, with an equivalent script in `runner.py`) builds a small
shoe-shopping knowledge graph — a product catalog plus two scripted customer conversations —
and then walks through Graphiti's search and reranking options against it.

## Prerequisites

- Neo4j running locally (or reachable via `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`)
- `ANTHROPIC_API_KEY` (the notebook uses `AnthropicClient` for extraction)
- `OPENAI_API_KEY` (used for the default embedder, and for the default cross-encoder reranker
  shown below)

## What the notebook does

1. Ingests the ManyBirds product catalog (`data/manybirds_products.json`) as bulk episodes
2. Ingests two scripted conversations introducing a customer, "John"
3. Runs the same handful of queries through different search/reranking configurations so you
   can compare the resulting fact ordering directly

## Search methods

Graphiti finds candidate facts/entities using up to three independent methods, then merges and
reranks them:

| Method | What it does | Good for | Weak on |
|---|---|---|---|
| **BM25** (full-text) | Keyword/lexical scoring, like a search-engine index | Exact terms, names, SKUs, IDs | Paraphrases, synonyms |
| **Cosine similarity** | Embedding-based semantic match | Paraphrased/reworded queries | Exact identifiers |
| **BFS** (breadth-first search) | Pure graph traversal from a seed node, no text relevance at all | "Everything connected to X" | Needs an anchor node to start from |

Most recipes combine BM25 + cosine similarity; BFS only joins in when a `center_node_uuid` /
`bfs_origin_node_uuids` is supplied.

## Rerankers

The merged candidate pool is then reordered by one of:

| Reranker | How it orders results |
|---|---|
| **RRF** (Reciprocal Rank Fusion) | Default. Fuses the BM25 and cosine-similarity rankings by position, not raw score |
| **Node-distance** | Reorders by graph proximity to a `center_node_uuid` |
| **MMR** (Maximal Marginal Relevance) | Balances relevance against diversity, penalizing near-duplicate results |
| **Cross-encoder** | Scores each candidate directly against the query with a dedicated model — most precise, most expensive |
| **Episode-mentions** | Ranks by how many episodes/conversations mention the fact |

## Which one to use — by scenario

**RRF (default) — general Q&A, "just find relevant facts"**
```python
r = await client.search("What is John's shoe size?")
```
No special structure to exploit — combine keyword + semantic signals and return the best match.

**Node-distance — "relevant to *this specific entity*"**
```python
r = await client.search("What shoes has John purchased?", center_node_uuid=john_uuid)
```
Use inside a conversation/session anchored on one entity, so facts close to *that* entity in the
graph outrank textually-similar facts belonging to someone/something else.

**MMR — "give me a diverse spread, not five near-duplicates"**
```python
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_MMR

mmr_config = EDGE_HYBRID_SEARCH_MMR.model_copy(update={'reranker_min_score': -1})
r = await client._search("wool shoes", mmr_config)
```
Use when the graph has a lot of redundant/overlapping facts (a product catalog with many similar
SKUs, or a fact repeated across episodes) and you're feeding results into something with limited
context, where five near-identical facts waste the budget.

> **Gotcha:** MMR's score (`λ·cos(query, candidate) + (λ-1)·max_similarity_to_other_candidates`)
> can go negative when candidates are near-duplicates of each other — unlike RRF's always-positive
> scores. `SearchConfig.reranker_min_score` defaults to `0`, which is a no-op for RRF but can
> silently filter out *every* MMR result. Override it to a negative value as shown above.

**Cross-encoder — high-stakes precision, worth the latency/cost**
```python
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_CROSS_ENCODER

r = await client._search(
    "Can John wear ManyBirds Wool Runners?",
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    bfs_origin_node_uuids=[john_uuid],
)
```
Use when the top result's correctness really matters and you can afford an extra model call per
candidate — e.g. a direct factual answer shown to the user, not just retrieval context. Defaults
to `OpenAIRerankerClient`, so it needs `OPENAI_API_KEY`.

**Episode-mentions — "what's the most well-established/repeated fact"**
```python
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_EPISODE_MENTIONS
```
Favors facts corroborated across multiple episodes over a single passing mention.

## Who decides which method/reranker runs?

There's no automatic query-based selection — it's always an explicit choice by the caller:

- **`client.search(...)`** (high-level): hardcoded to `EDGE_HYBRID_SEARCH_RRF` unless you pass
  `center_node_uuid`, in which case it switches to `EDGE_HYBRID_SEARCH_NODE_DISTANCE`. That's the
  only lever this method exposes.
- **`client._search(...)` / `client.search_(...)`** (low-level): you pass a `SearchConfig`
  directly — either a premade recipe from `graphiti_core.search.search_config_recipes`, or one you
  build yourself by setting `search_methods` and `reranker` on `EdgeSearchConfig` /
  `NodeSearchConfig`. If no config is given, it falls back to
  `COMBINED_HYBRID_SEARCH_CROSS_ENCODER`.

## Rule of thumb

Start with plain `client.search()` (RRF). Add `center_node_uuid` the moment you have a natural
anchor entity. Reach for MMR when results look redundant. Reach for cross-encoder only for the
specific query whose top answer really matters and can tolerate extra latency.
