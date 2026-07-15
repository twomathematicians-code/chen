"""Tests for multi-tenant memory isolation."""

from __future__ import annotations

import pytest

from chen.core.memory import InMemoryMemory, MemoryEntry
from chen.persistence.tenant_memory import (
    MultiTenantMemory,
    QuotaExceededError,
)


@pytest.fixture
def mtm():
    return MultiTenantMemory(
        base_factory=InMemoryMemory,
        max_entries_per_tenant=5,
        ttl_seconds=0,
    )


class TestMultiTenantMemory:
    def test_tenants_are_isolated(self, mtm):
        """Tenant A's entries are invisible to Tenant B."""
        mtm.write(MemoryEntry(text="tenant A secret", role="user"), tenant_id="A")
        mtm.write(MemoryEntry(text="tenant B secret", role="user"), tenant_id="B")

        a_results = mtm.retrieve("secret", tenant_id="A")
        b_results = mtm.retrieve("secret", tenant_id="B")

        assert any("tenant A" in e.text for e in a_results)
        assert not any("tenant A" in e.text for e in b_results)
        assert any("tenant B" in e.text for e in b_results)
        assert not any("tenant B" in e.text for e in a_results)

    def test_default_tenant(self, mtm):
        """Without tenant_id, uses the default namespace."""
        mtm.write(MemoryEntry(text="default entry"))
        results = mtm.retrieve("default")
        assert len(results) >= 1

    def test_count_per_tenant(self, mtm):
        mtm.write(MemoryEntry(text="a1"), tenant_id="A")
        mtm.write(MemoryEntry(text="a2"), tenant_id="A")
        mtm.write(MemoryEntry(text="b1"), tenant_id="B")
        assert mtm.count(tenant_id="A") == 2
        assert mtm.count(tenant_id="B") == 1

    def test_quota_enforced(self, mtm):
        """Exceeding max_entries_per_tenant raises QuotaExceededError."""
        for i in range(5):
            mtm.write(MemoryEntry(text=f"entry {i}"), tenant_id="A")
        with pytest.raises(QuotaExceededError):
            mtm.write(MemoryEntry(text="one too many"), tenant_id="A")

    def test_quota_is_per_tenant(self, mtm):
        """Tenant A hitting quota doesn't affect Tenant B."""
        for i in range(5):
            mtm.write(MemoryEntry(text=f"A{i}"), tenant_id="A")
        # B should still be able to write
        mtm.write(MemoryEntry(text="B1"), tenant_id="B")
        assert mtm.count(tenant_id="B") == 1

    def test_clear_tenant(self, mtm):
        mtm.write(MemoryEntry(text="A1"), tenant_id="A")
        mtm.write(MemoryEntry(text="B1"), tenant_id="B")
        mtm.clear(tenant_id="A")
        assert mtm.count(tenant_id="A") == 0
        assert mtm.count(tenant_id="B") == 1

    def test_clear_all(self, mtm):
        mtm.write(MemoryEntry(text="A1"), tenant_id="A")
        mtm.write(MemoryEntry(text="B1"), tenant_id="B")
        mtm.clear()
        assert mtm.count(tenant_id="A") == 0
        assert mtm.count(tenant_id="B") == 0

    def test_list_tenants(self, mtm):
        mtm.write(MemoryEntry(text="A"), tenant_id="tenant-A")
        mtm.write(MemoryEntry(text="B"), tenant_id="tenant-B")
        tenants = mtm.list_tenants()
        assert "tenant-A" in tenants
        assert "tenant-B" in tenants

    def test_total_entries(self, mtm):
        mtm.write(MemoryEntry(text="A1"), tenant_id="A")
        mtm.write(MemoryEntry(text="A2"), tenant_id="A")
        mtm.write(MemoryEntry(text="B1"), tenant_id="B")
        assert mtm.total_entries() == 3

    def test_ttl_garbage_collection(self):
        """Entries older than TTL are garbage collected."""
        mtm = MultiTenantMemory(
            base_factory=InMemoryMemory,
            max_entries_per_tenant=100,
            ttl_seconds=0.1,
        )
        mtm.write(MemoryEntry(text="old entry"), tenant_id="A")
        import time

        time.sleep(0.15)
        # Trigger GC via retrieve
        results = mtm.retrieve("old", tenant_id="A")
        assert len(results) == 0
