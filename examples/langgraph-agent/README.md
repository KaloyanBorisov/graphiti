# LangGraph Agent Example

This example (`agent.ipynb`) builds a shoe-sales chatbot, "SalesBot," using LangGraph for agent
orchestration and Graphiti for long-term memory. It's the ManyBirds shoe catalog again, this time
wired into a conversational agent instead of queried directly.

## Prerequisites

- Neo4j running locally (or reachable via `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`)
- `OPENAI_API_KEY` (used for both the `ChatOpenAI` chat model and Graphiti's default embedder)
- `pip install graphiti-core langchain-openai langgraph ipywidgets`

## What the notebook does

1. Ingests the ManyBirds product catalog (`data/manybirds_products.json`)
2. Creates a user node for a customer, "jess"
3. Defines a `get_shoe_data` tool that lets the agent query Graphiti for product facts
4. Builds a LangGraph agent loop (`agent` ↔ `tools`) with a `chatbot` node that pulls relevant
   facts from Graphiti into the system prompt on every turn, and writes each exchange back to
   Graphiti afterward
5. Runs the agent once programmatically, then optionally interactively via `ipywidgets`

## Why there are two graphs

The notebook involves two separate graphs that are easy to conflate because both are called
"graph" in the code:

| | LangGraph `StateGraph` | Graphiti knowledge graph |
|---|---|---|
| **What it is** | A control-flow graph | A data/memory graph (in Neo4j) |
| **Nodes represent** | Steps in agent execution (`agent`, `tools`) | Entities (`jess`, shoe products) |
| **Edges represent** | "what runs next" | Facts (e.g. `jess` `INTERESTED_IN` `TinyBirds Wool Runner`) |
| **Lifetime** | Rebuilt each process run; per-turn state kept by `MemorySaver` | Persists in Neo4j across sessions |
| **Purpose** | Decides *what the agent does next* | Stores *what the agent knows* |
| **Visualized via** | `graph.get_graph().draw_mermaid_png()` | `tinybirds-jess.png` (Neo4j Desktop screenshot) |

Concretely:

- The **LangGraph graph** (`graph_builder = StateGraph(State)`) wires together two nodes: `agent`
  (the `chatbot` function) and `tools` (`get_shoe_data`). `should_continue` decides whether the
  agent needs to call the tool again or can end the turn. This is pure orchestration logic — it
  has no memory of past conversations beyond the current thread's checkpointed state.

- The **Graphiti graph** is the actual knowledge store. The `chatbot` node queries it
  (`client.search(...)`, centered on `user_node_uuid`) to pull relevant facts about the user and
  prior conversation into the system prompt, then asynchronously writes the new exchange back as
  an episode. This is what makes the agent remember Jess's shoe size or preferences in a
  conversation next week, not just later in the same thread.

In short: one graph is the agent's wiring diagram, the other is its memory. They're
complementary — LangGraph doesn't persist facts long-term, and Graphiti doesn't orchestrate
control flow.
