"""Tests for the SQLite run store."""

from __future__ import annotations

from pathlib import Path

import pytest

from chen.persistence.run_store import RunRecord, RunStore


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "test.sqlite3")


class TestRunRecord:
    def test_construction(self):
        r = RunRecord(
            run_id="abc123",
            config_hash="hash" * 16,
            prompt="hello",
            output="world",
            phase=1,
            backend="mock",
        )
        assert r.run_id == "abc123"
        assert r.prompt == "hello"
        assert r.output == "world"
        assert r.phase == 1
        assert r.selected_experts == []
        assert r.total_tokens == 0

    def test_to_dict_serializes_experts_as_json(self):
        r = RunRecord(
            run_id="x",
            config_hash="h" * 64,
            prompt="p",
            output="o",
            phase=1,
            backend="mock",
            selected_experts=["a", "b", "c"],
        )
        d = r.to_dict()
        assert isinstance(d["selected_experts"], str)
        assert "a" in d["selected_experts"]


class TestRunStore:
    def test_create_schema_lazily(self, tmp_path):
        path = tmp_path / "subdir" / "runs.sqlite3"
        assert not path.exists()
        s = RunStore(path)
        assert path.exists()
        assert s.count() == 0

    def test_save_and_get(self, store):
        r = RunRecord(
            run_id="run1",
            config_hash="h" * 64,
            prompt="hello",
            output="world",
            phase=1,
            backend="mock",
            selected_experts=["analyst", "synthesizer"],
            total_tokens=100,
            total_cost_usd=0.001,
            total_latency_ms=50.0,
            epu=3.5,
            kv_transfers=2,
        )
        store.save(r)
        assert store.count() == 1

        fetched = store.get("run1")
        assert fetched is not None
        assert fetched.prompt == "hello"
        assert fetched.output == "world"
        assert fetched.total_tokens == 100
        assert fetched.epu == 3.5
        assert fetched.selected_experts == ["analyst", "synthesizer"]

    def test_get_nonexistent_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_save_replaces_on_same_id(self, store):
        r1 = RunRecord(
            run_id="dup",
            config_hash="h" * 64,
            prompt="original",
            output="v1",
            phase=1,
            backend="mock",
        )
        r2 = RunRecord(
            run_id="dup",
            config_hash="h" * 64,
            prompt="updated",
            output="v2",
            phase=2,
            backend="mock",
        )
        store.save(r1)
        store.save(r2)
        assert store.count() == 1
        fetched = store.get("dup")
        assert fetched.prompt == "updated"
        assert fetched.phase == 2

    def test_list_returns_newest_first(self, store):
        import time

        for i in range(5):
            store.save(
                RunRecord(
                    run_id=f"run_{i}",
                    config_hash="h" * 64,
                    prompt=f"prompt_{i}",
                    output=f"output_{i}",
                    phase=1,
                    backend="mock",
                )
            )
            time.sleep(0.01)  # ensure distinct timestamps
        runs = store.list(limit=10)
        assert len(runs) == 5
        # Newest first (run_4 was saved last)
        assert runs[0].run_id == "run_4"

    def test_list_filters_by_phase(self, store):
        for i, phase in enumerate([1, 2, 1, 3, 1]):
            store.save(
                RunRecord(
                    run_id=f"run_{i}",
                    config_hash="h" * 64,
                    prompt=f"p_{i}",
                    output=f"o_{i}",
                    phase=phase,
                    backend="mock",
                )
            )
        phase1_runs = store.list(phase=1)
        assert len(phase1_runs) == 3
        assert all(r.phase == 1 for r in phase1_runs)

    def test_count(self, store):
        assert store.count() == 0
        for i in range(3):
            store.save(
                RunRecord(
                    run_id=f"r{i}",
                    config_hash="h" * 64,
                    prompt="p",
                    output="o",
                    phase=1,
                    backend="mock",
                )
            )
        assert store.count() == 3

    def test_default_path_uses_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHEN_RUN_STORE_PATH", str(tmp_path / "env_default.sqlite3"))
        s = RunStore.default()
        assert s.path == tmp_path / "env_default.sqlite3"
        s.close()

    def test_close(self, store):
        store.close()
        # Operations after close should fail (but we don't test that here
        # because the connection is shared and might still work).
