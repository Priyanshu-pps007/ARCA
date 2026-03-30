"""
chunker.py

Why this exists:
    You can't embed an entire documentation page as one vector — it's too long
    (LLMs have token limits) and too broad (one vector can't represent
    many different concepts accurately). You need to split it into smaller
    pieces where each chunk is about ONE specific thing.

How it works:
    We use RecursiveCharacterTextSplitter from LangChain — it tries to split
    on natural boundaries in this order:
        1. Double newline (paragraph break)    ← preferred
        2. Single newline
        3. Space
        4. Character                           ← last resort

    Overlap: each chunk shares ~100 tokens with the next chunk.
    Why? If a concept spans a chunk boundary, the overlap ensures
    neither chunk loses the full context of that concept.

    Chunk size: 512 tokens (~400 words).
    Why 512? Small enough to be specific (good retrieval precision),
    large enough to have full context (good answer quality).
    Adjust based on your embedding model's sweet spot.

Where to use it:
    Called by ingester.py after crawler returns CrawledPage objects.
"""

import logging
from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from RAG.crawler import CrawledPage

logger = logging.getLogger(__name__)


# ─── Data contract ────────────────────────────────────────────────────────────
@dataclass
class TextChunk:
    text       : str
    source     : str       # e.g. "langchain", "langgraph"
    url        : str       # original page URL
    title      : str       # page title
    chunk_index: int       # position in the page (0, 1, 2, ...)
    metadata   : dict


# ─── Chunker ──────────────────────────────────────────────────────────────────
class MarkdownChunker:
    def __init__(
        self,
        chunk_size    : int = 512,   # tokens per chunk
        chunk_overlap : int = 100,   # overlap between consecutive chunks
    ):
        # RecursiveCharacterTextSplitter measures in characters by default.
        # We pass a tiktoken-based length function so it counts tokens instead.
        # This matters because embedding models have TOKEN limits, not char limits.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size        = chunk_size * 4,    # ~4 chars per token (rough estimate)
            chunk_overlap     = chunk_overlap * 4,
            length_function   = len,
            separators        = [
                "\n## ",    # H2 heading    ← split on new sections first
                "\n### ",   # H3 heading
                "\n\n",     # paragraph break
                "\n",       # line break
                " ",        # word boundary
                "",         # character     ← last resort
            ],
        )

    def chunk_pages(self, pages: list[CrawledPage]) -> list[TextChunk]:
        """
        Chunk a list of crawled pages.
        Returns flat list of TextChunk objects ready for embedding.
        """
        all_chunks = []

        for page in pages:
            chunks = self.chunk_page(page)
            all_chunks.extend(chunks)
            logger.debug(f"[chunker] {page.url}: {len(chunks)} chunks")

        logger.info(f"[chunker] Total chunks: {len(all_chunks)} from {len(pages)} pages")
        return all_chunks

    def chunk_page(self, page: CrawledPage) -> list[TextChunk]:
        """Chunk a single CrawledPage into TextChunk objects."""
        if not page.markdown.strip():
            return []

        # Split the markdown
        raw_chunks = self.splitter.split_text(page.markdown)

        # Wrap each raw chunk in our TextChunk dataclass
        chunks = []
        for idx, text in enumerate(raw_chunks):
            text = text.strip()
            if len(text) < 50:          # skip tiny fragments
                continue

            chunks.append(TextChunk(
                text        = text,
                source      = page.source,
                url         = page.url,
                title       = page.title,
                chunk_index = idx,
                metadata    = {
                    **page.metadata,
                    "chunk_index" : idx,
                    "total_chunks": len(raw_chunks),
                    "char_count"  : len(text),
                }
            ))

        return chunks