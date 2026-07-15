# ADR 0007: CC0 1.0 — public domain dedication

- Status: Accepted
- Date: 2025-01-15
- Deciders: CHEN core team

## Context

CHEN is a research artifact intended to be reproduced, extended, and
built upon. The license choice determines:

- Whether companies can adopt it without legal review.
- Whether academics can cite it without license complications.
- Whether derivative works must be open-sourced.

## Decision

Release CHEN under **CC0 1.0 Universal** — a public domain dedication.
No attribution required, no copyleft, no restrictions.

## Consequences

### Positive

- **Maximum adoption** — no license friction for commercial, academic, or government users.
- **No attribution requirement** — though attribution is appreciated, it's not legally required.
- **Compatible with everything** — CC0 code can be mixed with any other license.
- **Simplifies citation** — academics can cite via `CITATION.cff` without license anxiety.

### Negative

- **No copyleft protection** — a company can take CHEN, improve it, and not share back.
- **No patent grant** — CC0 does not include an explicit patent grant (unlike Apache 2.0). For a pure-software research project this is acceptable.
- **Some jurisdictions don't recognize public domain** — CC0's fallback license kicks in, which is permissive but not identical to "public domain."

### Neutral

- The `CITATION.cff` file exists to encourage (not require) academic attribution.

## Alternatives considered

### Alternative A: Apache 2.0

Permissive, with explicit patent grant.

**Why not:** the patent grant language occasionally triggers legal
review at conservative companies. CC0 is simpler.

### Alternative B: MIT

Permissive, simple, widely understood.

**Why not:** technically requires attribution. For a research artifact
intended to be referenced and built upon freely, CC0 is cleaner.

### Alternative C: GPL v3

Copyleft — derivative works must also be open-sourced.

**Why not:** prevents commercial adoption without open-sourcing the
adopter's code. This narrows the audience for a research artifact
whose goal is to *prove a concept*, not to enforce openness.

## References

- CC0 1.0 full text: https://creativecommons.org/publicdomain/zero/1.0/legalcode
- Choose a License: https://choosealicense.com/licenses/cc0-1.0/
