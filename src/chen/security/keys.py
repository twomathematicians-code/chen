"""Key management — local keystore for encryption keys.

Provides a simple file-based keystore for managing encryption keys
with rotation support. For production deployments, replace this with
an HSM, KMS, or HashiCorp Vault integration.

The keystore stores keys as JSON files in a directory:

    ~/.chen/keys/
    ├── active.json          ← pointer to the current key
    ├── key_<key_id>.json    ← key metadata + encrypted master key
    └── key_<key_id>.json

Each key file contains:
    {
        "key_id": "a1b2c3d4e5f67890",
        "created_at": "2025-07-15T10:00:00Z",
        "status": "active",          # active | retired | revoked
        "master_key_b64": "...",     # base64-encoded 32-byte key
        "rotation_of": null          # key_id this replaces, or null
    }
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from chen.security.config import EncryptionConfig


@dataclass
class KeyMetadata:
    """Metadata for a stored encryption key.

    Attributes:
        key_id: Unique identifier (hex string).
        created_at: ISO 8601 timestamp.
        status: "active", "retired", or "revoked".
        master_key_b64: Base64-encoded 32-byte AES-256 key.
        rotation_of: Key ID this key replaces, or None.
    """

    key_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "active"
    master_key_b64: str = ""
    rotation_of: Optional[str] = None  # noqa: UP045

    def to_config(self) -> EncryptionConfig:
        """Convert to an EncryptionConfig."""
        return EncryptionConfig(
            master_key=base64.b64decode(self.master_key_b64),
            key_id=self.key_id,
        )

    @classmethod
    def from_config(
        cls,
        config: EncryptionConfig,
        rotation_of: Optional[str] = None,  # noqa: UP045
    ) -> KeyMetadata:
        """Create metadata from an EncryptionConfig."""
        return cls(
            key_id=config.key_id,
            master_key_b64=base64.b64encode(config.master_key).decode("ascii"),
            rotation_of=rotation_of,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> KeyMetadata:
        return cls(**d)


class KeyStore:
    """File-based keystore for encryption keys.

    Attributes:
        path: Directory where key files are stored.
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:  # noqa: UP045
        if path is None:
            path = os.environ.get(
                "CHEN_KEYSTORE_PATH",
                str(Path.home() / ".chen" / "keys"),
            )
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _key_file(self, key_id: str) -> Path:
        return self.path / f"key_{key_id}.json"

    @property
    def active_file(self) -> Path:
        return self.path / "active.json"

    def generate_key(self, rotation_of: Optional[str] = None) -> KeyMetadata:  # noqa: UP045
        """Generate a new key and store it. Does NOT activate it."""
        config = EncryptionConfig.generate()
        meta = KeyMetadata.from_config(config, rotation_of=rotation_of)
        self._save(meta)
        return meta

    def _save(self, meta: KeyMetadata) -> None:
        """Save key metadata to disk."""
        # Set restrictive permissions on the key file (owner read/write only).
        key_path = self._key_file(meta.key_id)
        key_path.write_text(json.dumps(meta.to_dict(), indent=2))
        try:
            key_path.chmod(0o600)
        except OSError:
            # Windows doesn't support Unix permissions; ignore.
            pass

    def load(self, key_id: str) -> Optional[KeyMetadata]:  # noqa: UP045
        """Load a key by ID. Returns None if not found."""
        path = self._key_file(key_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return KeyMetadata.from_dict(data)

    def list_keys(self) -> list[KeyMetadata]:
        """List all stored keys."""
        keys: list[KeyMetadata] = []
        for p in sorted(self.path.glob("key_*.json")):
            try:
                data = json.loads(p.read_text())
                keys.append(KeyMetadata.from_dict(data))
            except (json.JSONDecodeError, TypeError):
                continue
        return keys

    def get_active(self) -> Optional[KeyMetadata]:  # noqa: UP045
        """Return the currently active key, or None."""
        if not self.active_file.exists():
            return None
        try:
            data = json.loads(self.active_file.read_text())
            key_id = data.get("active_key_id")
            if key_id:
                return self.load(key_id)
        except (json.JSONDecodeError, KeyError):
            pass
        return None

    def activate(self, key_id: str) -> None:
        """Set a key as the active key."""
        meta = self.load(key_id)
        if meta is None:
            raise KeyError(f"key '{key_id}' not found in keystore at {self.path}")
        # Mark all other keys as retired.
        for k in self.list_keys():
            if k.key_id != key_id and k.status == "active":
                k.status = "retired"
                self._save(k)
        meta.status = "active"
        self._save(meta)
        self.active_file.write_text(json.dumps({"active_key_id": key_id}, indent=2))

    def rotate(self) -> KeyMetadata:
        """Generate a new key and activate it. The old key is retired
        but kept for decrypting existing data."""
        current = self.get_active()
        rotation_of = current.key_id if current else None
        new_meta = self.generate_key(rotation_of=rotation_of)
        self.activate(new_meta.key_id)
        return new_meta

    def revoke(self, key_id: str) -> None:
        """Revoke a key. Revoked keys cannot decrypt new data but are
        kept for audit purposes."""
        meta = self.load(key_id)
        if meta is None:
            raise KeyError(f"key '{key_id}' not found")
        meta.status = "revoked"
        self._save(meta)

    def get_active_config(self) -> EncryptionConfig:
        """Return the active EncryptionConfig. Generates one if none exists."""
        meta = self.get_active()
        if meta is None:
            meta = self.generate_key()
            self.activate(meta.key_id)
        return meta.to_config()

    def get_decryptor_keys(self) -> list[EncryptionConfig]:
        """Return all non-revoked keys for the decryptor (enables rotation)."""
        return [m.to_config() for m in self.list_keys() if m.status != "revoked"]
