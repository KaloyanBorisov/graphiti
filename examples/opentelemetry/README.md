# OpenTelemetry Stdout Tracing Example

Configure Graphiti with OpenTelemetry to output trace spans to stdout.

## What this example does

This script feeds Graphiti a few facts (some sentences about Kamala Harris's
career, and a structured record about Gavin Newsom's roles in California
politics) and then asks it questions about them. Behind the scenes, Graphiti
reads each piece of text, figures out who and what is being talked about,
works out how those things relate to each other, and stitches that into a
growing knowledge graph — a network of facts and relationships, not just a
pile of text. When you then ask a question, it searches that graph to pull
back the relevant facts, rather than doing a plain keyword search over the
raw text.

Normally all of that background work happens invisibly. This example turns
on OpenTelemetry tracing so that each internal step is printed to your
terminal as it happens — showing which steps ran, which ones called out to
the LLM, and how long each one took. The point isn't the facts about Harris
or Newsom — it's to give you visibility into Graphiti's internal pipeline so
you can understand (and later debug or monitor) what happens every time you
add information or ask a question.

## Setup

Run from the repository root using the root `pyproject.toml` / `uv.lock`:

```bash
uv sync --extra dev --extra kuzu --extra tracing
export OPENAI_API_KEY=your_api_key_here
uv run examples/opentelemetry/otel_stdout_example.py
```

## Configure OpenTelemetry with Graphiti

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# Set up OpenTelemetry with stdout exporter
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

# Get tracer and pass to Graphiti
tracer = trace.get_tracer(__name__)
graphiti = Graphiti(
    graph_driver=kuzu_driver,
    tracer=tracer,
    trace_span_prefix='graphiti.example'
)
```
