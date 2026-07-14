"""Shared external memory (RAG+).

The memory is the network's "hippocampus": a vector store that all
experts in the pipeline read from and write to. Before an expert
generates, it calls :meth:`Memory.retrieve` to fetch relevant chunks;
after it generates, it may call :meth:`Memory.write` to store structured
outputs for later experts.

Two design choices distinguish this from vanilla RAG:

1. **Shared across experts.** All experts in the pipeline see the same
   memory. The Analyst's writes are visible to the Synthesizer.
2. **Latent-aware retrieval.** Entries carry the role of the expert that
   wrote them, allowing retrieval to filter by "written by the Reasoner"
   or "high-confidence."

The default backend is :class:`InMemoryMemory` (numpy-based, no deps,
deterministic). A ChromaDB backend is on the roadmap.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class MemoryEntry:
    """One entry in the shared memory store.

    Attributes:
        text: The stored text (e.g. an extracted entity, a summary, a fact).
        embedding: Optional precomputed embedding. If absent, retrieval
            computes one on the fly.
        role: Role of the expert that wrote this entry
            (``ExpertRole.SYNTHESIZER`` etc.), or empty if written by the
            user / external source.
        expert_name: Name of the expert that wrote this entry.
        confidence: 0..1 confidence score. Used by retrieval for filtering.
        timestamp: Wall-clock time the entry was written.
        metadata: Free-form metadata (e.g. ``{"source": "pdf page 3"}``).
    """

    text: str
    embedding: np.ndarray | None = None
    role: str = ""
    expert_name: str = ""
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Stable id derived from text + role + expert."""
        h = hashlib.blake2b(
            f"{self.role}|{self.expert_name}|{self.text}".encode(),
            digest_size=8,
        )
        return h.hexdigest()


class Memory(Protocol):
    """Shared external memory protocol."""

    def write(self, entry: MemoryEntry) -> str:
        """Add an entry. Returns the entry's id."""
        ...

    def retrieve(
        self,
        query: str,
        k: int = 4,
        *,
        role: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[MemoryEntry]:
        """Return the top-k entries most similar to ``query``.

        Args:
            query: The query text.
            k: Maximum number of entries to return.
            role: If set, only return entries written by an expert with
                this role.
            min_confidence: Filter out entries below this confidence.
        """
        ...

    def count(self) -> int:
        """Total number of entries."""
        ...

    def clear(self) -> None:
        """Remove all entries."""
        ...


# ---------------------------------------------------------------------------
# Default embedding: deterministic hash-based projection (no deps)
# ---------------------------------------------------------------------------

_DEFAULT_EMBED_DIM = 64


def _hash_embed(text: str, dim: int = _DEFAULT_EMBED_DIM) -> np.ndarray:
    """Deterministic hash-based embedding in [0, 1), L2-normalized."""
    out = np.zeros(dim, dtype=np.float32)
    encoded = text.encode("utf-8")
    for i in range(0, max(len(encoded), 1), 8):
        chunk = encoded[i : i + 8]
        h = hashlib.blake2b(chunk, digest_size=4).digest()
        idx = (i // 8) % dim
        out[idx] = int.from_bytes(h, "little") / 0xFFFFFFFF
    norm = float((out**2).sum()) ** 0.5
    if norm > 0:
        out = out / norm
    return out


# ---------------------------------------------------------------------------
# InMemoryMemory
# ---------------------------------------------------------------------------


@dataclass
class InMemoryMemory:
    """In-memory numpy-backed memory store.

    Deterministic, no external dependencies. Suitable for tests, demos,
    and small pipelines. For larger deployments, swap in a ChromaDB
    backend (roadmap).
    """

    embed_dim: int = _DEFAULT_EMBED_DIM
    embed_fn: Any = None  # callable[[str], np.ndarray]
    _entries: list[MemoryEntry] = field(default_factory=list, repr=False)
    _index: dict[str, MemoryEntry] = field(default_factory=dict, repr=False)

    def _embed(self, text: str) -> np.ndarray:
        if self.embed_fn is not None:
            return np.asarray(self.embed_fn(text), dtype="float32")
        return _hash_embed(text, self.embed_dim)

    def write(self, entry: MemoryEntry) -> str:
        if entry.embedding is None:
            entry.embedding = self._embed(entry.text)
        if entry.id not in self._index:
            self._entries.append(entry)
            self._index[entry.id] = entry
        else:
            # Update existing entry.
            existing = self._index[entry.id]
            existing.text = entry.text
            existing.embedding = entry.embedding
            existing.confidence = entry.confidence
            existing.metadata.update(entry.metadata)
        return entry.id

    def retrieve(
        self,
        query: str,
        k: int = 4,
        *,
        role: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[MemoryEntry]:
        if not self._entries:
            return []
        q = self._embed(query)
        scored: list[tuple[float, MemoryEntry]] = []
        for e in self._entries:
            if role is not None and e.role != role:
                continue
            if e.confidence < min_confidence:
                continue
            if e.embedding is None:
                e.embedding = self._embed(e.text)
            sim = float(np.dot(q, e.embedding))
            scored.append((sim, e))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:k]]

    def count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._index.clear()

    # Convenience aliases
    add = write
    search = retrieve
