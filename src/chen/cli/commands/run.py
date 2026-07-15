"""``chen run`` — run a single prompt through a CHEN pipeline."""

from __future__ import annotations

import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from chen.backends.mock import MockBackend
from chen.core.expert import Expert, ExpertRole
from chen.core.router import CosineRouter, HybridRouter, LogisticRouter
from chen.observability.logging import configure_logging, get_logger
from chen.persistence.run_store import RunRecord, RunStore
from chen.reproducibility.config_hash import hash_config

console = Console()


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
    elif backend == "hf":
        from chen.backends.hf import HuggingFaceBackend

        return [
            Expert(
                name="analyst",
                role=ExpertRole.ANALYST,
                backend=HuggingFaceBackend(
                    model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", params_m=1_700
                ),
            ),
            Expert(
                name="reasoner",
                role=ExpertRole.REASONER,
                backend=HuggingFaceBackend(model_id="Qwen/Qwen2.5-3B-Instruct", params_m=3_000),
            ),
            Expert(
                name="coder",
                role=ExpertRole.CODER,
                backend=HuggingFaceBackend(model_id="Qwen/Qwen2.5-3B-Instruct", params_m=3_000),
            ),
            Expert(
                name="synthesizer",
                role=ExpertRole.SYNTHESIZER,
                backend=HuggingFaceBackend(
                    model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", params_m=1_700
                ),
            ),
        ]
    raise ValueError(f"Unknown backend: {backend}")


def _make_router(kind: str, experts: list[Expert]):
    if kind == "logistic":
        return LogisticRouter.from_experts(experts)
    if kind == "cosine":
        return CosineRouter.from_experts(experts)
    if kind == "hybrid":
        return HybridRouter.from_experts(experts)
    raise ValueError(f"Unknown router: {kind}")


def _build_pipeline(phase: int, router_kind: str, experts: list[Expert], max_tokens: int):
    if phase == 1:
        from chen.phases.phase1_cascade import CascadePipeline

        return CascadePipeline(
            experts=experts,
            sequence=["analyst", "reasoner", "synthesizer"],
            max_tokens_per_expert=max_tokens,
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
    elif phase == 2:
        from chen.phases.phase2_kv_pass import KVPassPipeline

        return KVPassPipeline(
            experts=experts,
            sequence=["analyst", "reasoner", "synthesizer"],
            max_tokens_per_expert=max_tokens,
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
    elif phase == 3:
        from chen.phases.phase3_routing import RoutingPipeline

        router = _make_router(router_kind, experts)
        return RoutingPipeline(
            experts=experts,
            router=router,
            handoff="text",
            max_tokens_per_expert=max_tokens,
            memory_retrieve_k=0,
            write_intermediate_to_memory=False,
        )
    raise ValueError(f"Unknown phase: {phase}")


def run_command(
    prompt: str,
    phase: int,
    backend: str,
    max_tokens: int,
    router: str,
    save_run: bool,
) -> None:
    """Run a single prompt through a CHEN pipeline."""
    configure_logging(level="INFO")
    log = get_logger(__name__)
    log.info("chen.run.start", prompt=prompt[:80], phase=phase, backend=backend)

    console.print(
        Panel.fit(
            f"[bold]CHEN run[/bold]\n"
            f"phase: [cyan]{phase}[/cyan]  backend: [cyan]{backend}[/cyan]  "
            f"router: [cyan]{router}[/cyan]  max_tokens: [cyan]{max_tokens}[/cyan]",
            border_style="blue",
        )
    )
    console.print(f"[bold]Prompt:[/bold] {prompt!r}\n")

    experts = _build_experts(backend)
    pipe = _build_pipeline(phase, router, experts, max_tokens)

    t0 = time.perf_counter()
    result = pipe.run(prompt)
    elapsed = time.perf_counter() - t0

    # Output panel
    console.print(Panel(result.output, title="Output", border_style="green"))

    # Metrics table
    m_table = Table(title="Per-expert metrics", border_style="dim")
    m_table.add_column("#", style="dim")
    m_table.add_column("expert", style="bold")
    m_table.add_column("role")
    m_table.add_column("params (M)", justify="right")
    m_table.add_column("in tok", justify="right")
    m_table.add_column("out tok", justify="right")
    m_table.add_column("latency (ms)", justify="right")
    m_table.add_column("kv?", justify="center")
    for i, em in enumerate(result.per_expert, 1):
        m_table.add_row(
            str(i),
            em.expert_name,
            em.role.value,
            str(em.params_m),
            str(em.input_tokens),
            str(em.output_tokens),
            f"{em.latency_ms:.2f}",
            "✓" if em.used_kv_cache else "—",
        )
    console.print(m_table)

    # Aggregate
    console.print(
        Panel(
            result.metrics.summary() + f"\nwall_clock: {elapsed * 1000:.1f} ms",
            title="Aggregate metrics",
            border_style="blue",
        )
    )

    if save_run:
        store = RunStore.default()
        config_hash = hash_config(
            {
                "phase": phase,
                "backend": backend,
                "router": router,
                "max_tokens": max_tokens,
                "sequence": result.selected_experts,
            }
        )
        store.save(
            RunRecord(
                run_id=config_hash[:16],
                prompt=prompt,
                output=result.output,
                phase=phase,
                backend=backend,
                router=router,
                selected_experts=result.selected_experts,
                total_tokens=result.metrics.total_tokens,
                total_cost_usd=result.metrics.total_cost_usd,
                total_latency_ms=result.metrics.total_latency_ms,
                epu=result.metrics.epu,
                kv_transfers=result.metrics.kv_cache_transfers,
                config_hash=config_hash,
            )
        )
        console.print(f"\n[green]Run saved to {store.path}[/green]")
        console.print(f"[dim]run_id={config_hash[:16]}  config_hash={config_hash}[/dim]")
