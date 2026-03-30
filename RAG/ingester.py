"""
ingester.py

Why this exists:
    This is the pipeline orchestrator — it connects all 3 pieces:
        crawler  → fetches raw docs
        chunker  → splits into semantic pieces
        embedder → converts to vectors
    
    And then writes everything to Postgres (pgvector).
    It also owns the RETRIEVAL side — given a user query,
    find the top-K most relevant chunks via cosine similarity.

How the DB storage works:
    We store chunks in a `rag_documents` table (separate from AgentMemories).
    
    Why separate?
        AgentMemories  = per-agent episodic/semantic memory (runtime data)
        rag_documents  = static knowledge corpus (build-time data)
    
    They serve different purposes. Don't collapse them.

    The similarity search uses pgvector's <=> operator (cosine distance).
    Lower = more similar. We flip it to score = 1 - distance.

Where to use it:
    1. Run once to build the corpus:
           python -m rag.ingester --source all
    
    2. Import retrieve() in your builder layer (Month 2):
           from rag.ingester import Retriever
           chunks = await retriever.retrieve("how to define a LangGraph node")
    
    3. Add to a weekly cron to keep docs fresh.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from uuid import uuid4

import asyncpg
from dotenv import load_dotenv

from RAG.crawler import DocsCrawler, SEED_URLS
from RAG.chunker import MarkdownChunker
from RAG.embedder import Embedder, EMBEDDING_DIMENSION

load_dotenv()
logger = logging.getLogger(__name__)

# ─── DB setup ────────────────────────────────────────────────────────────────
# Using asyncpg directly (not SQLModel) because:
# 1. pgvector types aren't natively supported by SQLModel
# 2. Bulk INSERT with COPY is much faster than ORM individual inserts
# 3. This is a batch job, not a web request — no need for the ORM layer

async def get_db_conn() -> asyncpg.Connection:
    password = os.getenv("POSTGRES_PASSWORD")
    return await asyncpg.connect(
        f"postgresql://postgres:{password}@localhost:5432/ARCA_db"
    )


# ─── Schema ───────────────────────────────────────────────────────────────────
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS rag_documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source      TEXT NOT NULL,              -- e.g. "langchain", "langgraph"
    url         TEXT NOT NULL,
    title       TEXT,
    chunk_index INTEGER,
    content     TEXT NOT NULL,              -- the raw chunk text
    embedding   vector({EMBEDDING_DIMENSION}),  -- pgvector column
    metadata    JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- IVFFlat index for fast approximate nearest-neighbour search
-- lists=100 is a good default for <1M rows. Increase to 200 for >1M.
CREATE INDEX IF NOT EXISTS rag_embedding_idx
    ON rag_documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Index for filtering by source before vector search
CREATE INDEX IF NOT EXISTS rag_source_idx ON rag_documents (source);
"""


# ─── Ingester ─────────────────────────────────────────────────────────────────
class Ingester:
    def __init__(self):
        self.crawler  = DocsCrawler(max_pages_per_source=30, max_depth=2)
        self.chunker  = MarkdownChunker(chunk_size=512, chunk_overlap=100)
        self.embedder = Embedder()

    async def setup(self):
        """Create rag_documents table + pgvector index if they don't exist."""
        conn = await get_db_conn()
        try:
            # Enable pgvector extension (idempotent)
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute(CREATE_TABLE_SQL)
            logger.info("[ingester] Table rag_documents ready")
        finally:
            await conn.close()

    async def ingest(self, source: str = "all"):
        """
        Full pipeline: crawl → chunk → embed → store.
        Pass source="langchain" to ingest one source,
        or source="all" to ingest everything.
        """
        await self.setup()

        # Select which sources to crawl
        seed_urls = SEED_URLS if source == "all" else {source: SEED_URLS[source]}

        # ── Step 1: Crawl ────────────────────────────────────────────────
        logger.info(f"[ingester] Step 1/3 — Crawling {list(seed_urls.keys())}")
        pages = await self.crawler.crawl_all(seed_urls)
        logger.info(f"[ingester] Crawled {len(pages)} pages")

        if not pages:
            logger.warning("[ingester] No pages crawled — check seed URLs and network")
            return

        # ── Step 2: Chunk ────────────────────────────────────────────────
        logger.info("[ingester] Step 2/3 — Chunking pages")
        chunks = self.chunker.chunk_pages(pages)
        logger.info(f"[ingester] Created {len(chunks)} chunks")

        # ── Step 3: Embed ────────────────────────────────────────────────
        logger.info("[ingester] Step 3/3 — Embedding chunks")
        embedded_chunks = self.embedder.embed_chunks(chunks)  # sync — BGE-M3 is CPU/GPU bound
        logger.info(f"[ingester] Embedded {len(embedded_chunks)} chunks")

        # ── Step 4: Store ────────────────────────────────────────────────
        await self._store(embedded_chunks, source)

    async def _store(self, embedded_chunks, source: str):
        """Bulk-insert embedded chunks into Postgres."""
        if not embedded_chunks:
            logger.warning("[ingester] Nothing to store")
            return

        conn = await get_db_conn()
        try:
            # Delete existing docs for this source before re-ingesting
            # This makes re-runs idempotent — no duplicate chunks
            if source == "all":
                deleted = await conn.execute("DELETE FROM rag_documents")
            else:
                deleted = await conn.execute(
                    "DELETE FROM rag_documents WHERE source = $1", source
                )
            logger.info(f"[ingester] Cleared old docs for source='{source}'")

            # Bulk insert using executemany
            # vector type requires string format "[f1, f2, ...]" for asyncpg
            rows = [
                (
                    str(uuid4()),
                    ec.chunk.source,
                    ec.chunk.url,
                    ec.chunk.title,
                    ec.chunk.chunk_index,
                    ec.chunk.text,
                    json.dumps(ec.embedding),   # pgvector accepts JSON array string
                    json.dumps(ec.chunk.metadata),
                )
                for ec in embedded_chunks
            ]

            await conn.executemany(
                """
                INSERT INTO rag_documents
                    (id, source, url, title, chunk_index, content, embedding, metadata)
                VALUES
                    ($1, $2, $3, $4, $5, $6, $7::vector, $8::jsonb)
                """,
                rows,
            )

            logger.info(f"[ingester] Stored {len(rows)} chunks to rag_documents")

        finally:
            await conn.close()


# ─── Retriever ────────────────────────────────────────────────────────────────
class Retriever:
    """
    Used at query time by the builder layer (Month 2).
    Given a natural language query, returns the top-K most relevant chunks.
    """
    def __init__(self):
        self.embedder = Embedder()

    async def retrieve(
        self,
        query     : str,
        top_k     : int = 5,
        source    : str | None = None,    # filter by source if provided
        threshold : float = 0.5,          # minimum similarity score (0-1)
    ) -> list[dict]:
        """
        Semantic search over rag_documents.

        Returns list of dicts:
            [{ content, source, url, title, score }, ...]
        """
        # Embed the query using the same model used during ingestion
        query_vector = self.embedder.embed_query(query)  # sync
        query_vector_str = json.dumps(query_vector)

        conn = await get_db_conn()
        try:
            # pgvector cosine distance: <=> operator
            # 1 - distance converts to similarity score (higher = more similar)
            if source:
                rows = await conn.fetch(
                    """
                    SELECT
                        content,
                        source,
                        url,
                        title,
                        chunk_index,
                        1 - (embedding <=> $1::vector) AS score
                    FROM rag_documents
                    WHERE source = $2
                      AND 1 - (embedding <=> $1::vector) >= $3
                    ORDER BY embedding <=> $1::vector
                    LIMIT $4
                    """,
                    query_vector_str, source, threshold, top_k,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT
                        content,
                        source,
                        url,
                        title,
                        chunk_index,
                        1 - (embedding <=> $1::vector) AS score
                    FROM rag_documents
                    WHERE 1 - (embedding <=> $1::vector) >= $2
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """,
                    query_vector_str, threshold, top_k,
                )

            return [dict(row) for row in rows]

        finally:
            await conn.close()

    async def retrieve_for_builder(self, user_prompt: str) -> str:
        """
        Higher-level method called by the builder layer.
        Returns a formatted context string ready to inject into an LLM prompt.

        Usage in Month 2:
            context = await retriever.retrieve_for_builder("build me a customer support agent")
            prompt  = f"You are an agent builder. Context:\n{context}\n\nUser request: {user_prompt}"
        """
        chunks = await self.retrieve(user_prompt, top_k=5)

        if not chunks:
            return "No relevant documentation found."

        lines = []
        for i, chunk in enumerate(chunks, 1):
            lines.append(
                f"[{i}] Source: {chunk['source']} | {chunk['title']}\n"
                f"URL: {chunk['url']}\n"
                f"Relevance: {chunk['score']:.2f}\n\n"
                f"{chunk['content']}\n"
                f"{'─' * 60}"
            )

        return "\n".join(lines)


# ─── CLI entry point ──────────────────────────────────────────────────────────
# Run with: python -m rag.ingester
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="ARCA RAG Ingester")
    parser.add_argument(
        "--source",
        default = "all",
        choices = list(SEED_URLS.keys()) + ["all"],
        help    = "Which source to ingest (default: all)",
    )
    args = parser.parse_args()

    ingester = Ingester()
    asyncio.run(ingester.ingest(source=args.source))