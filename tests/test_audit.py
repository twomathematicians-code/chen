"""Tests for the audit log — hash-chain integrity."""

from __future__ import annotations

import json

import pytest

from chen.observability.audit import AuditEntry, AuditLog


@pytest.fixture
def audit_log(tmp_path):
    return AuditLog(path=tmp_path / "audit.log")


class TestAuditEntry:
    def test_compute_hash_is_deterministic(self):
        entry = AuditEntry(
            seq=0,
            timestamp=1234567890.0,
            run_id="test-run",
            prompt_hash="abc",
            output_hash="def",
            tenant_id="",
            prev_hash="",
        )
        h1 = entry.compute_hash()
        h2 = entry.compute_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_changes_with_different_data(self):
        e1 = AuditEntry(
            seq=0,
            timestamp=1.0,
            run_id="a",
            prompt_hash="x",
            output_hash="y",
            tenant_id="",
            prev_hash="",
        )
        e2 = AuditEntry(
            seq=0,
            timestamp=1.0,
            run_id="b",
            prompt_hash="x",
            output_hash="y",
            tenant_id="",
            prev_hash="",
        )
        assert e1.compute_hash() != e2.compute_hash()

    def test_jsonl_roundtrip(self):
        entry = AuditEntry(
            seq=5,
            timestamp=1.0,
            run_id="r",
            prompt_hash="p",
            output_hash="o",
            tenant_id="t",
            prev_hash="prev",
            entry_hash="h",
        )
        line = entry.to_jsonl()
        recovered = AuditEntry.from_jsonl(line)
        assert recovered.seq == 5
        assert recovered.run_id == "r"
        assert recovered.entry_hash == "h"


class TestAuditLog:
    def test_empty_log_verifies(self, audit_log):
        assert audit_log.verify() is True

    def test_record_creates_entry(self, audit_log):
        entry = audit_log.record(run_id="run1", prompt="hello", output="world")
        assert entry.seq == 0
        assert entry.run_id == "run1"
        assert entry.prev_hash == ""
        assert len(entry.entry_hash) == 64

    def test_chain_links_correctly(self, audit_log):
        e1 = audit_log.record("run1", "p1", "o1")
        e2 = audit_log.record("run2", "p2", "o2")
        e3 = audit_log.record("run3", "p3", "o3")
        assert e1.seq == 0
        assert e2.seq == 1
        assert e3.seq == 2
        assert e2.prev_hash == e1.entry_hash
        assert e3.prev_hash == e2.entry_hash

    def test_verify_detects_tampering(self, audit_log, tmp_path):
        audit_log.record("run1", "p1", "o1")
        audit_log.record("run2", "p2", "o2")
        # Tamper with the log file — change a character in the first entry.
        content = audit_log.path.read_text()
        # Find the first entry_hash and flip a character.
        lines = content.strip().split("\n")
        data = json.loads(lines[0])
        data["run_id"] = "TAMPERED"
        lines[0] = json.dumps(data, sort_keys=True)
        audit_log.path.write_text("\n".join(lines) + "\n")
        assert audit_log.verify() is False

    def test_verify_passes_on_intact_chain(self, audit_log):
        for i in range(10):
            audit_log.record(f"run{i}", f"prompt{i}", f"output{i}")
        assert audit_log.verify() is True

    def test_count(self, audit_log):
        assert audit_log.count() == 0
        for i in range(5):
            audit_log.record(f"run{i}", f"p{i}", f"o{i}")
        assert audit_log.count() == 5

    def test_tail(self, audit_log):
        for i in range(10):
            audit_log.record(f"run{i}", f"p{i}", f"o{i}")
        tail = audit_log.tail(3)
        assert len(tail) == 3
        assert tail[-1].run_id == "run9"

    def test_prompt_not_stored_in_plaintext(self, audit_log):
        audit_log.record("run1", "secret prompt data", "secret output")
        content = audit_log.path.read_text()
        assert "secret prompt data" not in content
        assert "secret output" not in content

    def test_tenant_id_recorded(self, audit_log):
        audit_log.record("run1", "p", "o", tenant_id="tenant-A")
        tail = audit_log.tail(1)
        assert tail[0].tenant_id == "tenant-A"

    def test_default_path_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHEN_AUDIT_LOG_PATH", str(tmp_path / "env_audit.log"))
        log = AuditLog.default()
        assert log.path == tmp_path / "env_audit.log"
