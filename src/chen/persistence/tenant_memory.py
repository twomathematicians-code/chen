"""Multi-tenant memory store — wraps any Memory with tenant isolation.

Namespaces all entries by tenant_id so tenants cannot see each other's
data. Enforces per-tenant size quotas and TTL-based garbage collection.

Usage::

    from chen.core.memory import InMemoryMemory
    from chen.persistence.tenant_memory import MultiTenantMemory

    mtm = MultiTenantMemory(
        base_factory=InMemoryMemory,
        max_entries_per_tenant=1000,
        ttl_seconds=3600,
    )

    # Entries are namespaced by tenant_id
    mtm.write(entry, tenant_id="tenant-A")
    mtm.retrieve("query", tenant_id="tenant-A")  # sees tenant-A's entries
    mtm.retrieve("query", tenant_id="tenant-B")  # sees only tenant-B's entries
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from chen.core.memory import Memory, MemoryEntry


@dataclass
class MultiTenantMemory:
    """Wraps a Memory factory with per-tenant isolation.

    Each tenant gets its own Memory instance, completely isolated from
    other tenants. Quotas and TTL prevent any single tenant from
    exhausting memory.

    Attributes:
        base_factory: Callable that returns a new Memory instance
            (e.g. ``InMemoryMemory`` or a ChromaDB-backed store).
        max_entries_per_tenant: Hard cap on entries per tenant.
            0 = unlimited.
        ttl_seconds: Entries older than this are garbage-collected.
            0 = no TTL.
    """

    base_factory: Callable[[], Memory]
    max_entries_per_tenant: int = 10_000
    ttl_seconds: float = 0.0

    _tenants: dict[str, Memory] = field(default_factory=dict, repr=False)
    _tenant_entry_times: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)

    def _get_tenant_memory(self, tenant_id: Optional[str]) -> Memory:  # noqa: UP045
        """Get or create the Memory instance for a tenant."""
        tid = tenant_id or "__default__"
        if tid not in self._tenants:
            self._tenants[tid] = self.base_factory()
            self._tenant_entry_times[tid] = {}
        return self._tenants[tid]

    def _gc_tenant(self, tenant_id: Optional[str]) -> None:  # noqa: UP045
        """Garbage-collect expired entries for a tenant."""
        if self.ttl_seconds <= 0:
            return
        tid = tenant_id or "__default__"
        times = self._tenant_entry_times.get(tid, {})
        now = time.time()
        expired_ids = [eid for eid, t in times.items() if now - t > self.ttl_seconds]
        if expired_ids and tid in self._tenants:
            # In-memory store doesn't support delete-by-id, so we rebuild.
            mem = self._tenants[tid]
            kept = [
                e
                for e in mem._entries  # type: ignore[attr-defined]
                if e.id not in expired_ids
            ]
            mem._entries = kept  # type: ignore[attr-defined]
            mem._index = {e.id: e for e in kept}  # type: ignore[attr-defined]
            for eid in expired_ids:
                times.pop(eid, None)

    def _check_quota(self, tenant_id: Optional[str]) -> None:  # noqa: UP045
        """Raise if tenant has exceeded its entry quota."""
        if self.max_entries_per_tenant <= 0:
            return
        tid = tenant_id or "__default__"
        mem = self._tenants.get(tid)
        if mem is not None and mem.count() >= self.max_entries_per_tenant:
            raise QuotaExceededError(
                f"Tenant '{tid}' has reached the maximum of "
                f"{self.max_entries_per_tenant} memory entries."
            )

    def write(self, entry: MemoryEntry, tenant_id: Optional[str] = None) -> str:  # noqa: UP045
        """Write an entry for a specific tenant."""
        self._gc_tenant(tenant_id)
        self._check_quota(tenant_id)
        mem = self._get_tenant_memory(tenant_id)
        eid = mem.write(entry)
        tid = tenant_id or "__default__"
        self._tenant_entry_times[tid][eid] = time.time()
        return eid

    def retrieve(
        self,
        query: str,
        k: int = 4,
        tenant_id: Optional[str] = None,  # noqa: UP045
        **kwargs: Any,
    ) -> list[MemoryEntry]:
        """Retrieve entries for a specific tenant only."""
        self._gc_tenant(tenant_id)
        mem = self._get_tenant_memory(tenant_id)
        return mem.retrieve(query, k=k, **kwargs)

    def count(self, tenant_id: Optional[str] = None) -> int:  # noqa: UP045
        """Count entries for a tenant."""
        mem = self._get_tenant_memory(tenant_id)
        return mem.count()

    def clear(self, tenant_id: Optional[str] = None) -> None:  # noqa: UP045
        """Clear entries for a tenant (or all tenants if None)."""
        if tenant_id is None:
            for mem in self._tenants.values():
                mem.clear()
            self._tenant_entry_times.clear()
        else:
            tid = tenant_id or "__default__"
            if tid in self._tenants:
                self._tenants[tid].clear()
            self._tenant_entry_times.pop(tid, None)

    def list_tenants(self) -> list[str]:
        """List all tenant IDs that have data."""
        return [t for t in self._tenants.keys() if t != "__default__"]

    def total_entries(self) -> int:
        """Total entries across all tenants."""
        return sum(mem.count() for mem in self._tenants.values())


class QuotaExceededError(Exception):
    """Raised when a tenant exceeds its memory entry quota."""
