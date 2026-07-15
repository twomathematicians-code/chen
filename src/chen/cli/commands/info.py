"""``chen info`` — print environment, backend, and configuration info."""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from chen import __version__
from chen.backends import list_backends

console = Console()


def _check_package(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, ""


def info_command() -> None:
    """Print environment, backend, and configuration info."""
    console.print(
        Panel.fit(
            f"[bold]CHEN[/bold] — Collaborative Heterogeneous Expert Network\n"
            f"version [cyan]{__version__}[/cyan]",
            border_style="blue",
        )
    )

    # System info
    sys_table = Table(title="System", show_header=False, border_style="dim")
    sys_table.add_column("key", style="bold")
    sys_table.add_column("value")
    sys_table.add_row("python", sys.version.split()[0])
    sys_table.add_row(
        "platform", f"{platform.system()} {platform.release()} ({platform.machine()})"
    )
    sys_table.add_row("git", _git_version())
    console.print(sys_table)

    # CHEN info
    chen_table = Table(title="CHEN", show_header=False, border_style="dim")
    chen_table.add_column("key", style="bold")
    chen_table.add_column("value")
    chen_table.add_row("version", __version__)
    chen_table.add_row("backends", ", ".join(list_backends()))
    console.print(chen_table)

    # Dependencies
    dep_table = Table(title="Dependencies", border_style="dim")
    dep_table.add_column("package", style="bold")
    dep_table.add_column("installed")
    dep_table.add_column("version")
    dep_table.add_column("extra")
    deps: list[tuple[str, str]] = [
        ("numpy", "core"),
        ("pydantic", "core"),
        ("structlog", "core"),
        ("typer", "core"),
        ("rich", "core"),
        ("transformers", "hf"),
        ("torch", "hf"),
        ("tokenizers", "hf"),
        ("accelerate", "hf"),
        ("fastapi", "server"),
        ("uvicorn", "server"),
        ("prometheus_client", "server"),
        ("chromadb", "memory"),
        ("sentence_transformers", "memory"),
        ("pytest", "dev"),
        ("ruff", "dev"),
        ("mypy", "dev"),
        ("hypothesis", "dev"),
    ]
    for name, extra in deps:
        ok, version = _check_package(name)
        dep_table.add_row(
            name,
            "[green]yes[/green]" if ok else "[red]no[/red]",
            version if ok else "—",
            extra,
        )
    console.print(dep_table)

    # Compute backends
    try:
        import torch

        cb_table = Table(title="PyTorch compute backends", show_header=False, border_style="dim")
        cb_table.add_column("key", style="bold")
        cb_table.add_column("value")
        cb_table.add_row("cuda available", str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            cb_table.add_row("gpu", torch.cuda.get_device_name(0))
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            cb_table.add_row("vram (gb)", f"{mem:.1f}")
        mps = getattr(torch.backends, "mps", None)
        cb_table.add_row("mps available", str(mps.is_available() if mps else False))
        cb_table.add_row("cpu threads", str(torch.get_num_threads()))
        console.print(cb_table)
    except ImportError:
        console.print("[dim]PyTorch not installed (install with: pip install -e '.[hf]')[/dim]")

    console.print(
        "\n[dim]Next steps:[/dim]\n"
        "  [cyan]chen run --prompt 'Explain recursion.'[/cyan]\n"
        "  [cyan]chen bench --phase 1[/cyan]\n"
        "  [cyan]chen serve --port 8000[/cyan]\n"
    )


def _git_version() -> str:
    if not shutil.which("git"):
        return "not installed"
    try:
        out = subprocess.run(["git", "--version"], capture_output=True, text=True, check=False)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _dict_to_rows(d: dict[str, Any], table: Table) -> None:
    for k, v in d.items():
        table.add_row(k, str(v))
