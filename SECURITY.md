# Security Policy

## Supported Versions

CHEN is pre-1.0 software. We will patch security issues only on the latest released version.

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in CHEN, please report it **privately** — do not open a public GitHub issue.

Email: **twomathematicians@gmail.com** (replace with your real address before publishing).

Please include:

1. A description of the vulnerability and its impact.
2. Steps to reproduce, including a minimal code snippet if possible.
3. Your assessment of severity (low / medium / high / critical).
4. Suggested fix, if you have one.

You will receive an acknowledgment within 72 hours. We will coordinate disclosure with you and credit you in the advisory unless you prefer to remain anonymous.

## Scope

The following are **in scope**:

- Vulnerabilities in CHEN's source code that could allow arbitrary code execution, data exfiltration, or denial of service when running CHEN as a library or service.
- Issues with the KV-cache transfer protocol that could allow cache poisoning across experts.
- Router or memory vulnerabilities that could cause unintended expert activation or memory leakage across tenants.

The following are **out of scope**:

- Vulnerabilities in upstream dependencies (transformers, torch, vLLM, llama.cpp) — report those upstream.
- Vulnerabilities that require the attacker to already have arbitrary code execution in the host process.
- Performance issues or cosmetic bugs (open a regular issue instead).

## Safe usage

CHEN is research software. Before deploying it in production:

- Pin to a specific version, not `main`.
- Run the MockBackend in CI to verify determinism.
- If you use the HF backend, ensure your `HUGGING_FACE_HUB_TOKEN` is stored as a secret, not committed.
- The shared memory store (`InMemoryMemory`) is **not** multi-tenant-safe. If you serve multiple tenants, give each pipeline its own `Memory` instance.
- The KV-cache transfer protocol assumes you trust all backends in the pipeline. Do not mix trusted and untrusted backends in the same pipeline.
