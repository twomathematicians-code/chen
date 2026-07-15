"""``chen keys`` — manage encryption keys."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from chen.observability.logging import configure_logging
from chen.security.keys import KeyStore

console = Console()


def keys_app() -> typer.Typer:
    """Create the keys sub-app."""
    app = typer.Typer(
        name="keys",
        help="Manage encryption keys for the security layer.",
        no_args_is_help=True,
        rich_markup_mode="rich",
    )

    @app.command("generate")
    def generate() -> None:
        """Generate a new encryption key (does NOT activate it)."""
        configure_logging(level="WARNING")
        store = KeyStore()
        meta = store.generate_key()
        console.print(f"[green]Generated key:[/green] {meta.key_id}")
        console.print(f"[dim]Status: {meta.status}[/dim]")
        console.print(f"[dim]Keystore: {store.path}[/dim]")
        console.print(
            f"\n[yellow]Note:[/yellow] Key is not active. "
            f"Run [cyan]chen keys activate {meta.key_id}[/cyan] to activate it."
        )

    @app.command("list")
    def list_keys() -> None:
        """List all stored encryption keys."""
        store = KeyStore()
        active = store.get_active()
        active_id = active.key_id if active else None
        keys = store.list_keys()
        if not keys:
            console.print("[yellow]No keys found.[/yellow]")
            console.print("Run [cyan]chen keys generate[/cyan] to create one.")
            return
        table = Table(title="Encryption Keys", border_style="dim")
        table.add_column("Key ID", style="bold")
        table.add_column("Status")
        table.add_column("Created")
        table.add_column("Rotation Of")
        for k in keys:
            status_display = k.status
            if k.key_id == active_id:
                status_display = f"[green]{k.status} (active)[/green]"
            elif k.status == "revoked":
                status_display = f"[red]{k.status}[/red]"
            elif k.status == "retired":
                status_display = f"[yellow]{k.status}[/yellow]"
            table.add_row(
                k.key_id,
                status_display,
                k.created_at,
                k.rotation_of or "—",
            )
        console.print(table)

    @app.command("activate")
    def activate(key_id: str) -> None:
        """Activate a key (retires the previous active key)."""
        store = KeyStore()
        try:
            store.activate(key_id)
            console.print(f"[green]Activated key:[/green] {key_id}")
        except KeyError:
            console.print(f"[red]Error:[/red] key '{key_id}' not found.")
            raise typer.Exit(1) from None

    @app.command("rotate")
    def rotate() -> None:
        """Generate a new key and activate it (retires the old one)."""
        store = KeyStore()
        old = store.get_active()
        new_meta = store.rotate()
        console.print(f"[green]Rotated to new key:[/green] {new_meta.key_id}")
        if old:
            console.print(f"[yellow]Previous key retired:[/yellow] {old.key_id}")
        console.print("\n[dim]The old key is kept for decrypting existing data.[/dim]")

    @app.command("revoke")
    def revoke(key_id: str) -> None:
        """Revoke a key (it can no longer decrypt new data)."""
        store = KeyStore()
        try:
            store.revoke(key_id)
            console.print(f"[red]Revoked key:[/red] {key_id}")
            console.print("[dim]The key is kept for audit but cannot decrypt new data.[/dim]")
        except KeyError:
            console.print(f"[red]Error:[/red] key '{key_id}' not found.")
            raise typer.Exit(1) from None

    @app.command("show")
    def show(key_id: str) -> None:
        """Show details of a specific key (without revealing the key itself)."""
        store = KeyStore()
        meta = store.load(key_id)
        if meta is None:
            console.print(f"[red]Error:[/red] key '{key_id}' not found.")
            raise typer.Exit(1) from None
        console.print(f"[bold]Key ID:[/bold]     {meta.key_id}")
        console.print(f"[bold]Status:[/bold]     {meta.status}")
        console.print(f"[bold]Created:[/bold]    {meta.created_at}")
        console.print(f"[bold]Rotation of:[/bold] {meta.rotation_of or '—'}")
        console.print(f"\n[dim]The master key is stored at {store._key_file(meta.key_id)}[/dim]")
        console.print(
            f"[yellow]Warning:[/yellow] Never share the master key. "
            f"Use [cyan]chen keys export-env {meta.key_id}[/cyan] to get env vars."
        )

    @app.command("export-env")
    def export_env(key_id: str) -> None:
        """Export a key as environment variables (for CI/CD)."""
        store = KeyStore()
        meta = store.load(key_id)
        if meta is None:
            console.print(f"[red]Error:[/red] key '{key_id}' not found.")
            raise typer.Exit(1) from None
        config = meta.to_config()
        env = config.to_env_dict()
        console.print("[dim]# Add these to your environment:[/dim]")
        for k, v in env.items():
            console.print(f"[cyan]export[/cyan] {k}={v}")

    return app
