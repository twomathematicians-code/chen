"""Tests for the memory store."""

from __future__ import annotations

import numpy as np

from chen.core.memory import MemoryEntry


class TestInMemoryMemory:
    def test_write_returns_id(self, fresh_memory):
        entry = MemoryEntry(text="hello", role="analyst", expert_name="a")
        eid = fresh_memory.write(entry)
        assert isinstance(eid, str)
        assert len(eid) > 0

    def test_count_increments(self, fresh_memory):
        for i in range(3):
            fresh_memory.write(MemoryEntry(text=f"entry {i}", role="analyst", expert_name="a"))
        assert fresh_memory.count() == 3

    def test_clear_empties_store(self, fresh_memory):
        fresh_memory.write(MemoryEntry(text="hello"))
        assert fresh_memory.count() == 1
        fresh_memory.clear()
        assert fresh_memory.count() == 0

    def test_retrieve_returns_relevant_entries(self, fresh_memory):
        fresh_memory.write(MemoryEntry(text="Paris is the capital of France"))
        fresh_memory.write(MemoryEntry(text="The Eiffel Tower is in Paris"))
        fresh_memory.write(MemoryEntry(text="Python is a programming language"))
        results = fresh_memory.retrieve("What is the capital of France?", k=2)
        assert len(results) <= 2
        assert all(isinstance(r, MemoryEntry) for r in results)
        # The "Paris is the capital of France" entry should rank highly.
        assert any("Paris" in r.text for r in results)

    def test_retrieve_empty_store_returns_empty(self, fresh_memory):
        results = fresh_memory.retrieve("anything")
        assert results == []

    def test_retrieve_filters_by_role(self, fresh_memory):
        fresh_memory.write(MemoryEntry(text="entity A", role="analyst"))
        fresh_memory.write(MemoryEntry(text="entity B", role="reasoner"))
        results = fresh_memory.retrieve("entity", k=10, role="analyst")
        assert all(r.role == "analyst" for r in results)

    def test_retrieve_filters_by_min_confidence(self, fresh_memory):
        fresh_memory.write(MemoryEntry(text="low conf", confidence=0.1))
        fresh_memory.write(MemoryEntry(text="high conf", confidence=0.9))
        results = fresh_memory.retrieve("conf", k=10, min_confidence=0.5)
        assert all(r.confidence >= 0.5 for r in results)
        assert len(results) == 1

    def test_write_dedupes_by_id(self, fresh_memory):
        e1 = MemoryEntry(text="hello", role="analyst", expert_name="a")
        e2 = MemoryEntry(text="hello", role="analyst", expert_name="a", confidence=0.5)
        id1 = fresh_memory.write(e1)
        id2 = fresh_memory.write(e2)
        assert id1 == id2
        assert fresh_memory.count() == 1
        # The stored entry should have the updated confidence.
        stored = fresh_memory.retrieve("hello", k=1)[0]
        assert stored.confidence == 0.5

    def test_write_assigns_embedding_if_missing(self, fresh_memory):
        e = MemoryEntry(text="hello", role="analyst", expert_name="a")
        assert e.embedding is None
        fresh_memory.write(e)
        assert e.embedding is not None
        assert isinstance(e.embedding, np.ndarray)


class TestMemoryEntry:
    def test_id_is_deterministic(self):
        e1 = MemoryEntry(text="hello", role="analyst", expert_name="a")
        e2 = MemoryEntry(text="hello", role="analyst", expert_name="a")
        assert e1.id == e2.id

    def test_id_changes_with_text(self):
        e1 = MemoryEntry(text="hello", role="analyst", expert_name="a")
        e2 = MemoryEntry(text="world", role="analyst", expert_name="a")
        assert e1.id != e2.id

    def test_id_changes_with_role(self):
        e1 = MemoryEntry(text="hello", role="analyst", expert_name="a")
        e2 = MemoryEntry(text="hello", role="reasoner", expert_name="a")
        assert e1.id != e2.id

    def test_default_confidence_is_one(self):
        e = MemoryEntry(text="hello")
        assert e.confidence == 1.0

    def test_default_metadata_is_empty_dict(self):
        e = MemoryEntry(text="hello")
        assert e.metadata == {}
