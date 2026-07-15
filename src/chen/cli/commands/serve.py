"""``chen serve`` — start the CHEN HTTP API server."""

from __future__ import annotations

import typer
from rich.console import Console

from chen.observability.logging import configure_logging, get_logger

console = Console()


def serve_command(
    host: str,
    port: int,
    backend: str,
    reload: bool,
) -> None:
    """Start the CHEN HTTP API server."""
    configure_logging(level="INFO")
    log = get_logger(__name__)
    log.info("chen.serve.start", host=host, port=port, backend=backend)

    try:
        import uvicorn
    except ImportError as e:
        console.print(
            "[red]Error:[/red] uvicorn not installed. Run: [cyan]pip install -e '.[server]'[/cyan]"
        )
        raise typer.Exit(1) from e

    console.print(
        f"\n[bold blue]CHEN API Server[/bold blue]\n"
        f"  host:     [cyan]{host}[/cyan]\n"
        f"  port:     [cyan]{port}[/cyan]\n"
        f"  backend:  [cyan]{backend}[/cyan]\n"
        f"  docs:     [cyan]http://{host}:{port}/docs[/cyan]\n"
        f"  health:   [cyan]http://{host}:{port}/v1/health[/cyan]\n"
        f"  metrics:  [cyan]http://{host}:{port}/v1/metrics[/cyan]\n"
    )

    uvicorn.run(
        "chen.server.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
