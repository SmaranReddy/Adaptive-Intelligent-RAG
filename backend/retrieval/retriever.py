import os
import time
import hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional
from utils.model import get_embedding_model, EMBED_MODEL

MAX_CACHE_SIZE = 500
HASH_THRESHOLD = 200

# Module-level flag: once the index is confirmed to exist, skip the check
_INDEX_READY: bool = False


class RetrieverAgent:
    # Class-level OrderedDict — preserves insertion order for FIFO eviction
    _cache: OrderedDict = OrderedDict()

    def _make_key(self, query: str, top_k: int) -> str:
        normalized = query.strip().lower()
        if len(normalized) > HASH_THRESHOLD:
            normalized = hashlib.md5(normalized.encode()).hexdigest()
        return f"{normalized}::{top_k}"

    def _insert(self, key: str, value) -> None:
        RetrieverAgent._cache[key] = value
        if len(RetrieverAgent._cache) > MAX_CACHE_SIZE:
            RetrieverAgent._cache.popitem(last=False)  # evict oldest (FIFO)
        if len(RetrieverAgent._cache) % 50 == 0:
            print(f"[CACHE] retrieval cache size: {len(RetrieverAgent._cache)}/{MAX_CACHE_SIZE}")

    def __init__(self):
        global _INDEX_READY
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "re-search"
        if not _INDEX_READY:
            if index_name not in pc.list_indexes().names():
                print(f"[SETUP] Pinecone index '{index_name}' not found — creating it...")
                pc.create_index(
                    name=index_name,
                    dimension=384,  # all-MiniLM-L6-v2 produces 384-dim vectors
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                print(f"[OK] Pinecone index '{index_name}' created.")
            _INDEX_READY = True
        self.index = pc.Index(index_name)
        self.model = get_embedding_model()
        print("[OK] RetrieverAgent connected to Pinecone index.")

    def embed_query(self, text: str) -> list:
        return self.model.encode(text).tolist()

    def retrieve(self, query: str, top_k: int = 5, min_score: float = None, per_paper_cap: int = None):
        """Retrieve top relevant chunks for the query"""
        cache_key = self._make_key(query, top_k)
        if cache_key in RetrieverAgent._cache:
            print(f"[CACHE HIT] retrieval")
            return RetrieverAgent._cache[cache_key]
        print(f"[CACHE MISS] retrieval")

        query_vector = self.embed_query(query)

        response = self.index.query(
            namespace="default",
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        matches = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {}) or {}
            text = meta.get("text", "") or ""
            matches.append({
                "id": match.get("id"),
                "score": match.get("score"),
                "metadata": meta,
                "text": text
            })

        if min_score is not None:
            matches = [m for m in matches if (m["score"] or 0) >= min_score]

        if per_paper_cap is not None:
            capped = []
            counts = {}
            for m in matches:
                title = (m["metadata"] or {}).get("title", "unknown")
                if counts.get(title, 0) < per_paper_cap:
                    capped.append(m)
                    counts[title] = counts.get(title, 0) + 1
            matches = capped

        print(f"[SEARCH] Retrieved {len(matches)} matches (requested top_k={top_k}).")
        self._insert(cache_key, matches)
        return matches

    def retrieve_many(self, queries: list, top_k: int = 5, timeout_s: float = 5.0) -> list:
        """
        Retrieve docs for multiple queries with guaranteed I/O parallelism.

        Why this exists instead of threading retrieve():
          SentenceTransformer.encode() holds the GIL during CPU tensor ops, so
          parallel threads embedding the same shared model serialize — no speedup.
          This method separates the two stages:
            1. Embed all queries sequentially  (CPU-bound, GIL-safe, ~50-200ms each)
            2. Query Pinecone for all in parallel (pure network I/O, GIL released)
          Result: wall-clock ≈ max(Pinecone latency) instead of sum(all latencies).
        """
        # ── Phase 1: serve cache hits, collect what needs embedding ──────────
        uncached: list = []
        cached_docs: dict = {}
        for q in queries:
            key = self._make_key(q, top_k)
            if key in RetrieverAgent._cache:
                print(f"[CACHE HIT] '{q[:50]}'")
                cached_docs[q] = RetrieverAgent._cache[key]
            else:
                print(f"[CACHE MISS] '{q[:50]}'")
                uncached.append(q)

        # ── Phase 2: embed all uncached queries (sequential, avoids GIL race) ─
        query_vectors: dict = {}
        for q in uncached:
            t0 = time.monotonic()
            query_vectors[q] = self.embed_query(q)
            print(f"[QUERY_TIME] embed  '{q[:50]}' → {int((time.monotonic()-t0)*1000)}ms")

        # ── Phase 3: fire all Pinecone queries in parallel (pure I/O) ─────────
        def _pinecone_query(q: str) -> list:
            t0 = time.monotonic()
            response = self.index.query(
                namespace="default",
                vector=query_vectors[q],
                top_k=top_k,
                include_metadata=True,
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            print(f"[QUERY_TIME] pinecone '{q[:50]}' → {elapsed_ms}ms")
            matches = []
            for match in response.get("matches", []):
                meta = match.get("metadata", {}) or {}
                matches.append({
                    "id":       match.get("id"),
                    "score":    match.get("score"),
                    "metadata": meta,
                    "text":     meta.get("text", ""),
                })
            self._insert(self._make_key(q, top_k), matches)
            return matches

        pinecone_docs: dict = {}
        if uncached:
            with ThreadPoolExecutor(max_workers=len(uncached)) as pool:
                futures = [(pool.submit(_pinecone_query, q), q) for q in uncached]
                for fut, q in futures:
                    try:
                        pinecone_docs[q] = fut.result(timeout=timeout_s)
                    except FutureTimeoutError:
                        print(f"[WARN] Pinecone timeout ({timeout_s}s) for '{q[:50]}'")
                        pinecone_docs[q] = []
                    except Exception as exc:
                        print(f"[WARN] Pinecone error for '{q[:50]}': {exc}")
                        pinecone_docs[q] = []

        # ── Merge in original query order ─────────────────────────────────────
        all_docs: list = []
        for q in queries:
            all_docs.extend(cached_docs.get(q) or pinecone_docs.get(q) or [])
        return all_docs


# ---------------------------------------------------------------------------
# Module-level singleton — avoids recreating Pinecone client on every step
# ---------------------------------------------------------------------------

_retriever_instance: "RetrieverAgent | None" = None


def get_retriever() -> "RetrieverAgent":
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RetrieverAgent()
    return _retriever_instance
