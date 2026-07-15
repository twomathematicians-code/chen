"""Main CLI entrypoint.

Usage:
    chen --help
    chen info
    chen run --prompt "Explain recursion."
    chen run --prompt "..." --phase 2 --backend mock
    chen bench --phase 3 --router logistic
    chen serve --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import typer
from rich.console import Console

from chen import __version__
from chen.cli.commands.bench import bench_command
from chen.cli.commands.info import info_command
from chen.cli.commands.run import run_command
from chen.cli.commands.serve import serve_command

app = typer.Typer(
    name="chen",
    help="Collaborative Heterogeneous Expert Network — distributed inference CLI.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        is_eager=True,
    ),
) -> None:
    """CHEN — Collaborative Heterogeneous Expert Network."""
    if version:
        console.print(f"chen [bold]{__version__}[/bold]")
        raise typer.Exit()


@app.command()
def info() -> None:
    """Print environment, backend, and configuration info."""
    info_command()


@app.command()
def run(
    prompt: str = typer.Option(..., "--prompt", "-p", help="The prompt to process."),
    phase: int = typer.Option(1, "--phase", help="Pipeline phase (1, 2, or 3)."),
    backend: str = typer.Option("mock", "--backend", "-b", help="Inference backend."),
    max_tokens: int = typer.Option(128, "--max-tokens", help="Max tokens per expert."),
    router: str = typer.Option("logistic", "--router", help="Router (phase 3 only)."),
    save_run: bool = typer.Option(
        False, "--save-run", help="Persist this run to the SQLite run store."
    ),
) -> None:
    """Run a single prompt through a CHEN pipeline."""
    run_command(
        prompt=prompt,
        phase=phase,
        backend=backend,
        max_tokens=max_tokens,
        router=router,
        save_run=save_run,
    )


@app.command()
def bench(
    phase: int = typer.Option(1, "--phase", help="Pipeline phase (1, 2, or 3)."),
    router: str = typer.Option("logistic", "--router", help="Router (phase 3 only)."),
    max_tokens: int = typer.Option(64, "--max-tokens", help="Max tokens per expert."),
    task: str = typer.Option("all", "--task", "-t", help="Specific task name or 'all'."),
    baseline_params: int = typer.Option(
        70_000, "--baseline-params", help="Baseline monolith size (M params)."
    ),
) -> None:
    """Run the benchmark suite and print KPI reports."""
    bench_command(
        phase=phase,
        router=router,
        max_tokens=max_tokens,
        task=task,
        baseline_params=baseline_params,
    )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
    backend: str = typer.Option("mock", "--backend", "-b", help="Inference backend."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev)."),
) -> None:
    """Start the CHEN HTTP API server."""
    serve_command(host=host, port=port, backend=backend, reload=reload)


if __name__ == "__main__":
    app()
