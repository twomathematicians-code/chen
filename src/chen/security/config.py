"""Encryption configuration and security settings.

Defines the cryptographic parameters used throughout CHEN's security
layer. All parameters are FIPS-140-2 compliant.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Cryptographic constants (do not change without a migration plan)
# ---------------------------------------------------------------------------

# AES-256-GCM: 256-bit key, 96-bit nonce, 128-bit tag.
# These are the NIST-recommended parameters (SP 800-38D).
KEY_SIZE_BYTES = 32  # 256 bits
NONCE_SIZE_BYTES = 12  # 96 bits
TAG_SIZE_BYTES = 16  # 128 bits

# HKDF (RFC 5869) for key derivation.
HKDF_INFO = b"chen-encryption-v1"
HKDF_HASH = "sha256"

# Default block size for cryptic stream mode (4 KB).
DEFAULT_BLOCK_SIZE_BYTES = 4096


@dataclass(frozen=True)
class EncryptionConfig:
    """Master encryption configuration.

    Holds the master key and parameters for deriving per-session and
    per-block encryption keys. The master key should be stored in a
    secure key management system (HSM, KMS, Vault) — not in source code.

    Attributes:
        master_key: 32-byte AES-256 master key. If None, a new random
            key is generated. NEVER hardcode this in source.
        key_id: Human-readable identifier for this key (for rotation
            tracking). Auto-generated if empty.
        block_size: Block size in bytes for cryptic stream mode.
        algorithm: Encryption algorithm name (always "AES-256-GCM").
    """

    master_key: bytes = field(default_factory=lambda: os.urandom(KEY_SIZE_BYTES))
    key_id: str = field(default_factory=lambda: os.urandom(8).hex())
    block_size: int = DEFAULT_BLOCK_SIZE_BYTES
    algorithm: str = "AES-256-GCM"

    def __post_init__(self) -> None:
        if len(self.master_key) != KEY_SIZE_BYTES:
            raise ValueError(
                f"master_key must be exactly {KEY_SIZE_BYTES} bytes "
                f"({KEY_SIZE_BYTES * 8} bits), got {len(self.master_key)}."
            )
        if self.block_size < 64:
            raise ValueError(f"block_size must be at least 64 bytes, got {self.block_size}.")

    @classmethod
    def generate(cls, key_id: Optional[str] = None) -> EncryptionConfig:  # noqa: UP045
        """Generate a new config with a random master key."""
        return cls(
            master_key=os.urandom(KEY_SIZE_BYTES),
            key_id=key_id or os.urandom(8).hex(),
        )

    @classmethod
    def from_env(cls) -> EncryptionConfig:
        """Load config from environment variables.

        Reads:
            CHEN_MASTER_KEY: Base64-encoded 32-byte master key.
            CHEN_KEY_ID: Key identifier for rotation tracking.

        Raises:
            ValueError: If CHEN_MASTER_KEY is set but invalid.
        """
        import base64

        key_b64 = os.environ.get("CHEN_MASTER_KEY", "")
        if not key_b64:
            # No env var — generate a new key (development mode).
            return cls.generate()
        try:
            master_key = base64.b64decode(key_b64)
        except Exception as e:
            raise ValueError(f"CHEN_MASTER_KEY is not valid base64: {e}") from e
        key_id = os.environ.get("CHEN_KEY_ID", "")
        return cls(master_key=master_key, key_id=key_id)

    def to_env_dict(self) -> dict[str, str]:
        """Return env vars that would reproduce this config."""
        import base64

        return {
            "CHEN_MASTER_KEY": base64.b64encode(self.master_key).decode("ascii"),
            "CHEN_KEY_ID": self.key_id,
        }

    def derive_session_key(self, session_id: bytes) -> bytes:
        """Derive a per-session key from the master key via HKDF.

        Args:
            session_id: Unique session identifier (e.g. pipeline run ID).

        Returns:
            A 32-byte derived key.
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        kdf = HKDF(
            algorithm=hashes.SHA256(),
            length=KEY_SIZE_BYTES,
            salt=session_id,
            info=HKDF_INFO,
        )
        return kdf.derive(self.master_key)


@dataclass(frozen=True)
class SecuritySettings:
    """Global security settings for a CHEN deployment.

    Controls which components are encrypted, key rotation behavior, and
    zero-knowledge routing.

    Attributes:
        enabled: Master switch for encryption. If False, all crypto is
            bypassed (development mode only).
        encrypt_prompts: Encrypt prompts before sending to external backends.
        encrypt_responses: Decrypt responses from external backends.
        encrypt_kv_cache: Encrypt KV-caches stored in memory or on disk.
        encrypt_memory_entries: Encrypt shared memory (RAG+) entries.
        zero_knowledge_routing: If True, the router operates on encrypted
            metadata only — it never sees prompt content.
        trusted_backends: Set of backend names that are considered local
            and trusted (encryption is skipped for these).
    """

    enabled: bool = True
    encrypt_prompts: bool = True
    encrypt_responses: bool = True
    encrypt_kv_cache: bool = True
    encrypt_memory_entries: bool = True
    zero_knowledge_routing: bool = False
    trusted_backends: frozenset[str] = frozenset({"mock"})
    config: Optional[EncryptionConfig] = None  # noqa: UP045

    @classmethod
    def disabled(cls) -> SecuritySettings:
        """Return a settings object with encryption disabled (dev mode)."""
        return cls(enabled=False)

    def is_trusted(self, backend_name: str) -> bool:
        """Return True if the backend is in the trusted set (no encryption)."""
        return backend_name in self.trusted_backends
