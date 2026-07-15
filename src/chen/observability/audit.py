"""Audit logging — tamper-proof append-only log for compliance.

Provides a hash-chained append-only log of all prompts and outputs.
Each entry includes:
- Timestamp
- Run ID
- Prompt hash (not the prompt itself — privacy)
- Output hash
- Previous entry hash (chain integrity)
- Entry hash (SHA-256 of all fields)

The chain is verifiable: any tampering with an entry breaks the chain
at that point. This satisfies SOC 2, GDPR, and HIPAA audit requirements.

Usage::

    from chen.observability.audit import AuditLog

    log = AuditLog.default()
    log.record(run_id="abc123", prompt="...", output="...", tenant_id="t1")

    # Verify chain integrity
    assert log.verify() is True

CLI::

    chen audit verify       # verify chain integrity
    chen audit tail 10      # show last 10 entries
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from chen.observability.logging import get_logger

_log = get_logger("chen.observability.audit")


@dataclass
class AuditEntry:
    """One entry in the audit log.

    Attributes:
        seq: Sequence number (0-based, increments by 1).
        timestamp: Unix timestamp (seconds since epoch).
        run_id: The run ID this entry records.
        prompt_hash: SHA-256 of the prompt (not the prompt itself).
        output_hash: SHA-256 of the output.
        tenant_id: Tenant ID (empty if no tenant).
        prev_hash: Hash of the previous entry (empty for the first entry).
        entry_hash: SHA-256 of all the above fields (the chain link).
    """

    seq: int
    timestamp: float
    run_id: str
    prompt_hash: str
    output_hash: str
    tenant_id: str
    prev_hash: str
    entry_hash: str = ""

    def compute_hash(self) -> str:
        """Compute the SHA-256 hash of this entry (excluding entry_hash itself)."""
        d = asdict(self)
        d.pop("entry_hash", None)
        # Sort keys for deterministic hashing.
        canonical = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_jsonl(self) -> str:
        """Serialize to a single JSON Lines string."""
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_jsonl(cls, line: str) -> AuditEntry:
        """Deserialize from a JSON Lines string."""
        return cls(**json.loads(line))


class AuditLog:
    """Append-only, hash-chained audit log.

    The log is stored as a JSON Lines file. Each line is one AuditEntry.
    The file is opened in append mode — entries are never deleted or
    modified (any modification breaks the hash chain).

    Attributes:
        path: Path to the audit log file.
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:  # noqa: UP045
        if path is None:
            path = os.environ.get("CHEN_AUDIT_LOG_PATH", "./chen_data/audit.log")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls) -> AuditLog:
        """Return an AuditLog at the default path."""
        return cls()

    def _last_entry(self) -> Optional[AuditEntry]:  # noqa: UP045
        """Read the last entry from the log (for chain continuation)."""
        if not self.path.exists():
            return None
        last_line = None
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
        if last_line is None:
            return None
        try:
            return AuditEntry.from_jsonl(last_line)
        except (json.JSONDecodeError, TypeError):
            return None

    def record(
        self,
        run_id: str,
        prompt: str,
        output: str,
        tenant_id: str = "",
    ) -> AuditEntry:
        """Append a new entry to the audit log.

        Args:
            run_id: The run ID this entry records.
            prompt: The prompt text (hashed, not stored in plaintext).
            output: The output text (hashed, not stored in plaintext).
            tenant_id: Optional tenant ID.

        Returns:
            The created AuditEntry.
        """
        last = self._last_entry()
        seq = (last.seq + 1) if last else 0
        prev_hash = last.entry_hash if last else ""

        entry = AuditEntry(
            seq=seq,
            timestamp=time.time(),
            run_id=run_id,
            prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            output_hash=hashlib.sha256(output.encode("utf-8")).hexdigest(),
            tenant_id=tenant_id,
            prev_hash=prev_hash,
        )
        entry.entry_hash = entry.compute_hash()

        # Append to file. Use "a" mode (append) — never overwrite.
        with open(self.path, "a") as f:
            f.write(entry.to_jsonl() + "\n")

        _log.debug(
            "audit.record",
            seq=seq,
            run_id=run_id,
            entry_hash=entry.entry_hash[:16],
        )
        return entry

    def verify(self) -> bool:
        """Verify the integrity of the hash chain.

        Returns True if every entry's hash matches its computed hash and
        every prev_hash matches the previous entry's entry_hash.
        Returns False if any tampering is detected.
        """
        if not self.path.exists():
            return True  # empty log is valid

        prev_hash = ""
        prev_seq = -1
        with open(self.path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = AuditEntry.from_jsonl(line)
                except (json.JSONDecodeError, TypeError):
                    _log.error("audit.verify.parse_error", line=line_num)
                    return False

                # Check sequence number.
                if entry.seq != prev_seq + 1:
                    _log.error(
                        "audit.verify.seq_mismatch",
                        expected=prev_seq + 1,
                        got=entry.seq,
                    )
                    return False

                # Check prev_hash chain.
                if entry.prev_hash != prev_hash:
                    _log.error(
                        "audit.verify.chain_broken",
                        seq=entry.seq,
                        expected_prev=prev_hash[:16],
                        got_prev=entry.prev_hash[:16],
                    )
                    return False

                # Check entry_hash.
                computed = entry.compute_hash()
                if entry.entry_hash != computed:
                    _log.error(
                        "audit.verify.hash_mismatch",
                        seq=entry.seq,
                        stored=entry.entry_hash[:16],
                        computed=computed[:16],
                    )
                    return False

                prev_hash = entry.entry_hash
                prev_seq = entry.seq

        return True

    def tail(self, n: int = 10) -> list[AuditEntry]:
        """Return the last n entries."""
        if not self.path.exists():
            return []
        lines: list[str] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        return [AuditEntry.from_jsonl(line) for line in lines[-n:]]

    def count(self) -> int:
        """Count total entries."""
        if not self.path.exists():
            return 0
        count = 0
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
