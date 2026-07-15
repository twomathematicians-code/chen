# Governance

CHEN is an open-source research project released under CC0 1.0. This
document describes how decisions are made and how contributors can
become maintainers.

## 1. Roles

### Contributors

Anyone who opens a PR. Contributors:

- Submit code, docs, tests, or issues.
- Follow the [Code of Conduct](https://github.com/your-org/chen/blob/main/CODE_OF_CONDUCT.md) and
  [Contributing Guide](https://github.com/your-org/chen/blob/main/CONTRIBUTING.md).
- Have their PRs reviewed by a Maintainer before merge.

### Maintainers

Contributors with merge access to `main`. Maintainers:

- Review PRs from contributors.
- Merge approved PRs (their own or others').
- Tag releases.
- Have a `@your-org/chen-maintainers` GitHub team membership.

Current maintainers:

- **@your-org/chen-maintainers** — initial team. Replace with actual
  GitHub usernames when you fork this repo.

### BDFL (Benevolent Dictator For Life)

For v0.x, the project founder has final say on disputed decisions.
This is intended to keep early development fast; it will be replaced
with a maintainer consensus model at v1.0.

## 2. Decision-making process

### Small changes (bug fixes, docs, refactor)

- Open a PR.
- One Maintainer approval required.
- Merge once CI passes.

### Medium changes (new features, new backends, new benchmark tasks)

- Open an issue first describing the change.
- Get a Maintainer to say "go" before opening the PR (avoids wasted work).
- One Maintainer approval required.
- Merge once CI passes.

### Large changes (new phase, breaking API change, new dependency)

- Open an [ADR](adrs/README.md) describing the decision.
- At least two Maintainers must approve the ADR.
- ADR is merged as "Accepted" before implementation begins.
- Implementation PR is reviewed normally.

### Releases

- Maintainer tags `vX.Y.Z` on `main`.
- CI auto-publishes to PyPI and GHCR (see `.github/workflows/release.yml`).
- Release notes are auto-generated from commits.

## 3. How to become a maintainer

1. Contribute 5+ merged PRs of meaningful size (not just typos).
2. Demonstrate understanding of the architecture (review others' PRs constructively).
3. Be nominated by an existing maintainer.
4. Maintainer team votes (lazy consensus — silence = yes).
5. BDFL confirms.

Maintainers can step down at any time. Inactive maintainers (no PRs or
reviews in 6 months) are moved to "Emeritus" status.

## 4. Conflict resolution

1. **Discuss on the PR.** Most conflicts are misunderstandings.
2. **If unresolved, move to a Discussion.** Open a GitHub Discussion
   summarizing the disagreement.
3. **If still unresolved, the BDFL decides.** For v0.x only.
4. **At v1.0+, maintainer majority vote.** BDFL breaks ties.

## 5. Security issues

Security issues are **not** handled in public. See
[https://github.com/your-org/chen/blob/main/SECURITY.md](https://github.com/your-org/chen/blob/main/SECURITY.md) for the private reporting process.

## 6. License & IP

- All contributions are licensed under [CC0 1.0](LICENSE).
- Contributors retain their copyright but waive it under CC0.
- No CLA is required (CC0 is permissive enough).

## 7. Succession

If the BDFL becomes unavailable, the maintainer team selects a
successor by majority vote. If the maintainer team dwindles below 2
active members, the project is marked "maintenance only" until new
maintainers are recruited.
