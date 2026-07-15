"""CHEN command-line interface.

Provides four subcommands:

* ``chen run``     — run a single prompt through a pipeline
* ``chen bench``   — run the benchmark suite
* ``chen serve``   — start the HTTP API server
* ``chen info``    — print environment / configuration info
"""

from __future__ import annotations

from chen.cli.main import app

__all__ = ["app"]
