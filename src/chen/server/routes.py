"""HTTP API routes for CHEN."""

from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel, Field

from chen.backends.mock import MockBackend
from chen.core.expert import Expert, ExpertRole
from chen.core.router import LogisticRouter
from chen.observability.logging import get_logger
from chen.observability.metrics import (
    record_expert_invocation,
    record_kv_transfer,
    record_pipeline_run,
    record_tokens,
)
from chen.persistence.run_store import RunRecord, RunStore
from chen.reproducibility.config_hash import hash_config

_log = get_logger("chen.server.routes")

try:
    from fastapi import APIRouter, HTTPException, Request
except ImportError:
    APIRouter = None  # type: ignore
    HTTPException = None  # type: ignore
    Request = None  # type: ignore


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class InferRequest(BaseModel):
    """Request body for ``POST /v1/infer``."""

    prompt: str = Field(..., description="The prompt to process.", min_length=1)
    phase: int = Field(1, description="Pipeline phase (1, 2, or 3).", ge=1, le=3)
    backend: str = Field("mock", description="Inference backend.")
    max_tokens: int = Field(128, description="Max tokens per expert.", ge=1, le=4096)
    router: str = Field("logistic", description="Router kind (phase 3 only).")
    save_run: bool = Field(True, description="Persist this run to the SQLite store.")


class ExpertMetricsModel(BaseModel):
    expert_name: str
    role: str
    params_m: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    used_kv_cache: bool


class InferResponse(BaseModel):
    """Response body for ``POST /v1/infer``."""

    output: str
    selected_experts: list[str]
    per_expert: list[ExpertMetricsModel]
    total_tokens: int
    total_cost_usd: float
    total_latency_ms: float
    epu: float
    kv_transfers: int
    # NOTE: Use Optional[str], not `str | None`. Pydantic evaluates type
    # annotations at runtime via eval(), and `str | None` (PEP 604) is a
    # SyntaxError on Python 3.9. `from __future__ import annotations` does
    # not help because Pydantic resolves the string annotation at class
    # definition time. The eval_type_backport package is installed as a
    # safety net, but Optional[str] is the correct, portable fix.
    run_id: Optional[str] = None  # noqa: UP045
    config_hash: str


class RunSummary(BaseModel):
    run_id: str
    timestamp: str
    phase: int
    backend: str
    prompt_preview: str
    total_tokens: int
    total_cost_usd: float
    epu: float


class RunDetail(RunSummary):
    config_hash: str
    router: str
    selected_experts: list[str]
    output: str
    total_latency_ms: float
    kv_transfers: int


# ---------------------------------------------------------------------------
# Pipeline construction (server-side)
# ---------------------------------------------------------------------------


def _build_experts(backend: str) -> list[Expert]:
    if backend == "mock":
        return [
            Expert(
                name="analyst",
                role=ExpertRole.ANALYST,
                backend=MockBackend(params_m=3_000, role_hint="analyst"),
            ),
            Expert(
                name="reasoner",
                role=ExpertRole.REASONER,
                backend=MockBackend(params_m=8_000, role_hint="reasoner"),
            ),
            Expert(
                name="coder",
                role=ExpertRole.CODER,
                backend=MockBackend(params_m=7_000, role_hint="coder"),
            ),
            Expert(
                name="synthesizer",
                role=ExpertRole.SYNTHESIZER,
                backend=MockBackend(params_m=3_000, role_hint="synthesizer"),
            ),
        ]
    raise ValueError(f"Backend '{backend}' not supported by server (use 'mock').")


def _make_pipeline(phase: int, router_kind: str, experts: list[Expert], max_tokens: int):
    if phase == 1:
        from chen.phases.phase1_cascade import CascadePipeline

        return CascadePipeline(
            experts=experts,
            sequence=["analyst", "synthesizer"],
            max_tokens_per_expert=max_tokens,
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
    elif phase == 2:
        from chen.phases.phase2_kv_pass import KVPassPipeline

        return KVPassPipeline(
            experts=experts,
            sequence=["analyst", "synthesizer"],
            max_tokens_per_expert=max_tokens,
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
    elif phase == 3:
        from chen.phases.phase3_routing import RoutingPipeline

        router = LogisticRouter.from_experts(experts)
        return RoutingPipeline(
            experts=experts,
            router=router,
            handoff="text",
            max_tokens_per_expert=max_tokens,
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
    raise ValueError(f"Unknown phase: {phase}")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


if APIRouter is not None:
    router = APIRouter()

    @router.post("/infer", response_model=InferResponse, tags=["inference"])
    async def infer(req: InferRequest, request: Request) -> InferResponse:
        """Run a prompt through a CHEN pipeline."""
        t0 = time.perf_counter()
        experts = _build_experts(req.backend)
        pipe = _make_pipeline(req.phase, req.router, experts, req.max_tokens)
        result = pipe.run(req.prompt)
        elapsed = time.perf_counter() - t0

        # Emit metrics
        record_pipeline_run(req.phase)
        for m in result.per_expert:
            record_expert_invocation(m.expert_name, m.role.value)
            record_tokens("input", m.input_tokens)
            record_tokens("output", m.output_tokens)
            if m.used_kv_cache:
                record_kv_transfer("prev", m.expert_name, m.cache_transfer_succeeded)

        config = {
            "phase": req.phase,
            "backend": req.backend,
            "router": req.router,
            "max_tokens": req.max_tokens,
            "sequence": result.selected_experts,
        }
        cfg_hash = hash_config(config)
        run_id = cfg_hash[:16]

        if req.save_run:
            store: RunStore = request.app.state.run_store
            store.save(
                RunRecord(
                    run_id=run_id,
                    config_hash=cfg_hash,
                    prompt=req.prompt,
                    output=result.output,
                    phase=req.phase,
                    backend=req.backend,
                    router=req.router,
                    selected_experts=result.selected_experts,
                    total_tokens=result.metrics.total_tokens,
                    total_cost_usd=result.metrics.total_cost_usd,
                    total_latency_ms=result.metrics.total_latency_ms,
                    epu=result.metrics.epu,
                    kv_transfers=result.metrics.kv_cache_transfers,
                )
            )

        _log.info(
            "infer.complete",
            run_id=run_id,
            elapsed_ms=elapsed * 1000,
            tokens=result.metrics.total_tokens,
        )

        return InferResponse(
            output=result.output,
            selected_experts=result.selected_experts,
            per_expert=[
                ExpertMetricsModel(
                    expert_name=m.expert_name,
                    role=m.role.value,
                    params_m=m.params_m,
                    input_tokens=m.input_tokens,
                    output_tokens=m.output_tokens,
                    latency_ms=m.latency_ms,
                    used_kv_cache=m.used_kv_cache,
                )
                for m in result.per_expert
            ],
            total_tokens=result.metrics.total_tokens,
            total_cost_usd=result.metrics.total_cost_usd,
            total_latency_ms=result.metrics.total_latency_ms,
            epu=result.metrics.epu,
            kv_transfers=result.metrics.kv_cache_transfers,
            run_id=run_id,
            config_hash=cfg_hash,
        )

    @router.get("/runs", response_model=list[RunSummary], tags=["runs"])
    async def list_runs(
        request: Request, limit: int = 50, phase: int | None = None
    ) -> list[RunSummary]:
        """List recent runs, newest first."""
        store: RunStore = request.app.state.run_store
        records = store.list(limit=limit, phase=phase)
        return [
            RunSummary(
                run_id=r.run_id,
                timestamp=r.timestamp,
                phase=r.phase,
                backend=r.backend,
                prompt_preview=r.prompt[:80],
                total_tokens=r.total_tokens,
                total_cost_usd=r.total_cost_usd,
                epu=r.epu,
            )
            for r in records
        ]

    @router.get("/runs/{run_id}", response_model=RunDetail, tags=["runs"])
    async def get_run(run_id: str, request: Request) -> RunDetail:
        """Fetch a specific run by id."""
        store: RunStore = request.app.state.run_store
        r = store.get(run_id)
        if r is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        return RunDetail(
            run_id=r.run_id,
            timestamp=r.timestamp,
            phase=r.phase,
            backend=r.backend,
            prompt_preview=r.prompt[:80],
            total_tokens=r.total_tokens,
            total_cost_usd=r.total_cost_usd,
            epu=r.epu,
            config_hash=r.config_hash,
            router=r.router,
            selected_experts=r.selected_experts,
            output=r.output,
            total_latency_ms=r.total_latency_ms,
            kv_transfers=r.kv_transfers,
        )

else:
    router = None  # type: ignore
