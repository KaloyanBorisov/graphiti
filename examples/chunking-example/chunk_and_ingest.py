#!/usr/bin/env python3
"""
Copyright 2025, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Document Chunking with Contextualized Retrieval for Graphiti
--------------------------------------------------------------
Port of Zep's contextualized-retrieval example
(https://github.com/getzep/zep/tree/main/examples/python/chunking-example)
to the Graphiti framework.

This script demonstrates Anthropic's contextualized retrieval technique:
1. Chunks a document into manageable pieces (paragraph-first, sentence
   fallback, with overlap for continuity).
2. Uses OpenAI to generate a short "situating" context for each chunk,
   based on the full document.
3. Ingests each contextualized chunk into Graphiti as an episode via
   `add_episode`, scoped to a `group_id` (Graphiti's equivalent of a
   per-user/per-document graph namespace).

Note: Graphiti already chunks dense episodes internally before entity
extraction (see examples/quickstart/dense_vs_normal_ingestion.py). That
chunking is entity-density driven and happens automatically. What this
script adds on top is Anthropic-style *contextualization* -- prepending
a short summary that situates each chunk within the whole document --
which helps the LLM extract entities/facts from a chunk without losing
surrounding context, and is not something Graphiti does for you.
"""

import argparse
import asyncio
import hashlib
import os
import re
from datetime import datetime, timezone

from dotenv import dotenv_values
from openai import AsyncOpenAI

# Load .env file from repository root BEFORE importing graphiti_core
_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.env'))
_env_values = dotenv_values(dotenv_path=_env_path)
for _key, _value in _env_values.items():
    if _value:
        os.environ[_key] = _value

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

# Configuration
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks for continuity
OPENAI_MODEL = 'gpt-5.4-mini'
MAX_CONCURRENCY = 5  # Concurrent contextualization calls in flight at once


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using common delimiters."""
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs."""
    paragraphs = re.split(r'\n\n+', text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_document(document: str) -> list[str]:
    """
    Chunk a document into smaller pieces suitable for processing.

    Strategy:
    1. First split by paragraphs
    2. If a paragraph is too large, split by sentences
    3. Combine small paragraphs/sentences until chunk_size is reached
    4. Maintain overlap between chunks for continuity
    """
    chunks = []
    paragraphs = split_into_paragraphs(document)
    current_chunk = ''

    for paragraph in paragraphs:
        if len(paragraph) > CHUNK_SIZE:
            sentences = split_into_sentences(paragraph)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= CHUNK_SIZE:
                    current_chunk = (
                        f'{current_chunk} {sentence}'.strip() if current_chunk else sentence
                    )
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        overlap_text = (
                            current_chunk[-CHUNK_OVERLAP:]
                            if len(current_chunk) > CHUNK_OVERLAP
                            else current_chunk
                        )
                        current_chunk = f'{overlap_text} {sentence}'.strip()
                    else:
                        chunks.append(sentence[:CHUNK_SIZE])
                        current_chunk = ''
        else:
            if len(current_chunk) + len(paragraph) + 2 <= CHUNK_SIZE:
                current_chunk = (
                    f'{current_chunk}\n\n{paragraph}'.strip() if current_chunk else paragraph
                )
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    overlap_text = (
                        current_chunk[-CHUNK_OVERLAP:]
                        if len(current_chunk) > CHUNK_OVERLAP
                        else current_chunk
                    )
                    current_chunk = f'{overlap_text}\n\n{paragraph}'.strip()
                else:
                    current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


async def contextualize_chunk(
    openai_client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    full_document: str,
    chunk: str,
    cache_key: str,
) -> str:
    """
    Use OpenAI to situate a chunk within the document context.

    This implements Anthropic's contextualized retrieval technique,
    which improves search retrieval by adding contextual information
    to each chunk before it is handed to Graphiti for entity/fact
    extraction.

    Every call resends the full document, so chunks are contextualized
    concurrently (bounded by `semaphore`) rather than one at a time --
    otherwise wall-clock time scales linearly with chunk count. The
    document is placed first and the chunk last, per OpenAI's prompt
    caching guidance (static content before variable content), and
    `cache_key` keeps concurrent calls for the same document routed to
    the same cache so they still hit it despite running in parallel.
    """
    prompt = f"""<document>
{full_document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the
overall document for the purposes of improving search retrieval of the
chunk. If the document has a publication date, please include the date
in your context. Answer only with the succinct context and nothing else."""

    max_retries = 3
    retry_delay = 1

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{'role': 'user', 'content': prompt}],
                    max_completion_tokens=256,
                    prompt_cache_key=cache_key,
                )
                usage = response.usage
                cached = usage.prompt_tokens_details.cached_tokens if usage else 0
                total = usage.prompt_tokens if usage else 0
                print(f'  [tokens] prompt={total} cached={cached}')
                context = response.choices[0].message.content.strip()
                return f'{context}\n\n---\n\n{chunk}'

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f'  Error contextualizing: {e}')
                    print(f'  Retrying in {retry_delay}s...')
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise


async def ingest_to_graphiti(
    graphiti: Graphiti, group_id: str, name: str, contextualized_chunk: str
) -> str:
    """Ingest a contextualized chunk into Graphiti as an episode."""
    result = await graphiti.add_episode(
        name=name,
        episode_body=contextualized_chunk,
        source=EpisodeType.text,
        source_description='chunked document (contextualized retrieval)',
        reference_time=datetime.now(timezone.utc),
        group_id=group_id,
    )
    return result.episode.uuid


async def process_document(document_path: str, group_id: str, dry_run: bool = False):
    """
    Process a document through the full pipeline:
    1. Read and chunk the document
    2. Contextualize each chunk using OpenAI
    3. Ingest each contextualized chunk into Graphiti
    """
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError('OPENAI_API_KEY environment variable not set')

    openai_client = AsyncOpenAI(api_key=openai_api_key)

    graphiti = None
    if not dry_run:
        neo4j_uri = os.environ.get('NEO4J_URI')
        neo4j_user = os.environ.get('NEO4J_USER')
        neo4j_password = os.environ.get('NEO4J_PASSWORD')

        if not neo4j_uri or not neo4j_user or not neo4j_password:
            raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set in .env')

        graphiti = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
        await graphiti.build_indices_and_constraints()

    try:
        print(f'\nReading document: {document_path}')
        with open(document_path, encoding='utf-8') as f:
            document_content = f.read()
        print(f'Document size: {len(document_content):,} characters')

        print(f'\nChunking document (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...')
        chunks = chunk_document(document_content)
        print(f'Created {len(chunks)} chunks')

        print(f'\nContextualizing {len(chunks)} chunks (up to {MAX_CONCURRENCY} concurrently)...')
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        # Stable per-document key so concurrent calls hit the same prompt
        # cache instead of scattering across backend replicas.
        cache_key = hashlib.sha256(document_content.encode()).hexdigest()[:32]
        contextualized_results = await asyncio.gather(
            *(
                contextualize_chunk(openai_client, semaphore, document_content, chunk, cache_key)
                for chunk in chunks
            ),
            return_exceptions=True,
        )

        print('\nProcessing chunks:')
        print('-' * 60)

        document_name = os.path.splitext(os.path.basename(document_path))[0]
        succeeded = 0
        failed = 0

        for i, (chunk, contextualized) in enumerate(zip(chunks, contextualized_results)):
            print(f'\nChunk {i + 1}/{len(chunks)} ({len(chunk):,} chars)')

            if isinstance(contextualized, BaseException):
                print(f'  ERROR contextualizing: {contextualized}')
                failed += 1
                continue

            context_end = contextualized.find('\n\n---\n\n')
            if context_end > 0:
                context_text = contextualized[:context_end]
                print(f'  Context: "{context_text}"')

            if dry_run:
                print('  [dry-run] Skipping ingestion')
                succeeded += 1
                continue

            print('  Ingesting to Graphiti...')
            try:
                episode_uuid = await ingest_to_graphiti(
                    graphiti, group_id, f'{document_name}-chunk-{i}', contextualized
                )
                print(f'  Created episode: {episode_uuid}')
                succeeded += 1
            except Exception as e:
                print(f'  ERROR ingesting: {e}')
                failed += 1

        print('\n' + '=' * 60)
        print('PROCESSING SUMMARY')
        print('=' * 60)
        print(f'Total chunks: {len(chunks)}')
        print(f'Successfully processed: {succeeded}')
        print(f'Failed: {failed}')
        print('=' * 60)

    finally:
        if graphiti is not None:
            await graphiti.close()
            print('\nConnection closed')


def main():
    parser = argparse.ArgumentParser(
        description='Chunk a document, contextualize each chunk, and ingest into Graphiti.'
    )
    parser.add_argument('document', help='Path to the document to process')
    parser.add_argument(
        '--group-id',
        required=True,
        help="Graphiti group_id to scope this document's episodes into (e.g. a user or doc id)",
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Chunk and contextualize without ingesting into Graphiti',
    )
    args = parser.parse_args()

    print('=' * 60)
    print('DOCUMENT CHUNKING WITH CONTEXTUALIZED RETRIEVAL (Graphiti)')
    print('=' * 60)
    print(f'Document: {args.document}')
    print(f'Group ID: {args.group_id}')
    print(f'Chunk size: {CHUNK_SIZE}')
    print(f'Chunk overlap: {CHUNK_OVERLAP}')
    print(f'Dry run: {args.dry_run}')

    asyncio.run(process_document(args.document, args.group_id, args.dry_run))


if __name__ == '__main__':
    main()
