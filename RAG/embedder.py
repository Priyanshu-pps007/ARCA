"""
embedder.py

Why BGE-M3 over OpenAI text-embedding-3-small:
    1. Free — runs locally, zero per-token cost. At scale this matters a lot.
    2. 1024 dimensions — smaller vector = faster index, less storage.
    3. Multi-vector retrieval — BGE-M3 supports three retrieval modes
       in a single model:
         - Dense  : standard semantic similarity (cosine distance)
         - Sparse : keyword-aware (like BM25 but neural)
         - ColBERT: token-level late interaction (most precise, most expensive)
       We use Dense for now. Hybrid (Dense + Sparse) is a one-line change
       when you want better precision on technical queries.
    4. No internet dependency — ingestion works fully offline after first download.
    5. Multi-lingual out of the box — ARCA can support non-English docs later.

How it works:
    FlagEmbedding loads BAAI/bge-m3 weights (~2.2GB) locally.
    First run downloads from HuggingFace, subsequent runs use cache.

    GPU: if CUDA is available, FlagEmbedding uses it automatically.
    CPU: still works, slower (~5s per batch vs ~0.2s on GPU).

    Batching: process chunks in batches of 32 (GPU/RAM bound, not rate-limit bound).

Install:
    pip install FlagEmbedding
"""

import logging
from dataclasses import dataclass

from FlagEmbedding import BGEM3FlagModel
from RAG.chunker import TextChunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL     = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024      # BGE-M3 dense vector output dimension
BATCH_SIZE          = 32        # 32 → safe for 8GB VRAM
                                # 64 → safe for 16GB VRAM
                                # 16 → safe for CPU-only


# ─── Data contract ────────────────────────────────────────────────────────────
@dataclass
class EmbeddedChunk:
    chunk     : TextChunk
    embedding : list[float]     # 1024-dim dense vector


# ─── Embedder ─────────────────────────────────────────────────────────────────
class Embedder:
    def __init__(self, use_fp16: bool = True):
        """
        Args:
            use_fp16: Half-precision inference. Cuts VRAM by 50%,
                      negligible quality loss for retrieval tasks.
                      Set False if you hit NaN issues on older GPUs.
        """
        logger.info(f"[embedder] Loading {EMBEDDING_MODEL} (first run downloads ~2.2GB)")

        self.model = BGEM3FlagModel(
            EMBEDDING_MODEL,
            use_fp16=use_fp16,
        )

        logger.info("[embedder] Model loaded")

    def embed_chunks(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
        """
        Embed a list of TextChunks in batches.

        Note: BGE-M3 via FlagEmbedding is synchronous (local GPU/CPU).
        No async needed — this is CPU/GPU-bound, not I/O-bound.
        """
        if not chunks:
            return []

        embedded = []
        batches  = self._batch(chunks, BATCH_SIZE)

        logger.info(f"[embedder] Embedding {len(chunks)} chunks in {len(batches)} batches")

        for i, batch in enumerate(batches):
            try:
                texts = [chunk.text for chunk in batch]

                # encode() returns a dict:
                #   "dense_vecs"      → shape (batch_size, 1024)  ← what we use now
                #   "lexical_weights" → sparse weights (hybrid search, Month 3)
                #   "colbert_vecs"    → token-level vectors (re-ranking, Month 3)
                output = self.model.encode(
                    texts,
                    batch_size          = len(batch),
                    max_length          = 512,
                    return_dense        = True,
                    return_sparse       = False,
                    return_colbert_vecs = False,
                )

                dense_vecs = output["dense_vecs"]   # numpy (batch_size, 1024)

                for j, chunk in enumerate(batch):
                    embedded.append(EmbeddedChunk(
                        chunk     = chunk,
                        embedding = dense_vecs[j].tolist(),
                    ))

                logger.debug(f"[embedder] Batch {i+1}/{len(batches)} done")

            except Exception as e:
                logger.error(f"[embedder] Batch {i+1} failed: {e}")
                continue

        logger.info(f"[embedder] Embedded {len(embedded)}/{len(chunks)} chunks")
        return embedded

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query at retrieval time.
        Must use the same model as ingestion — vector spaces are incompatible
        across different models.
        """
        output = self.model.encode(
            [text.strip()],
            batch_size          = 1,
            max_length          = 512,
            return_dense        = True,
            return_sparse       = False,
            return_colbert_vecs = False,
        )
        return output["dense_vecs"][0].tolist()

    def _batch(self, items: list, size: int) -> list[list]:
        return [items[i:i + size] for i in range(0, len(items), size)]


# ─── Month 3 upgrade: hybrid search ──────────────────────────────────────────
# Enable when you need better precision on exact technical terms
# e.g. "IVFFlat lists parameter", "LangGraph interrupt() usage"
#
# output = self.model.encode(
#     texts,
#     return_dense  = True,
#     return_sparse = True,         # ← flip this on
# )
# dense_vecs      = output["dense_vecs"]
# lexical_weights = output["lexical_weights"]
#
# Final score = 0.7 * dense_score + 0.3 * sparse_score