# CHEN Documentation

Welcome to the CHEN documentation site. CHEN is a distributed inference
architecture that replaces a single monolithic hyper-scale model with a
coordinated garage of specialized, low-parameter models.

## Quick navigation

- **[Architecture](architecture/overview.md)** — system diagrams (Mermaid) and component overview
- **[Math specs](math/specifications.md)** — formal definitions of EPU, cost model, latent nuance score
- **[ADRs](adrs/README.md)** — architecture decision records (why each choice was made)
- **[Operations](operations/README.md)** — deployment, observability, runbooks
- **[Threat model](security/threat-model.md)** — security analysis
- **[Governance](governance.md)** — how decisions are made

## Getting started

See the [README](https://github.com/your-org/chen#readme) on GitHub for
installation and quick-start instructions.

## API reference

Auto-generated API docs are available via mkdocstrings once you run
`mkdocs serve` locally. They cover every public class and function in
`src/chen/`.
