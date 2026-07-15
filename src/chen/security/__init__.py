"""CHEN Security & Encryption Layer.

Provides end-to-end encryption for prompts, responses, KV-caches, and
memory entries flowing through CHEN pipelines. Designed for deployments
where sensitive data (financial, government, healthcare) must never
appear in plaintext on external or cloud-hosted LLMs.

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │               Trust boundary (local)                 │
    │                                                      │
    │  User ──► Encryptor ──► [local private LLM]         │
    │                            │                         │
    │                            ▼ (plaintext KV-cache)   │
    │  User ◄── Decryptor ◄── [local private LLM]         │
    │                                                      │
    │  Encryptor ──► [encrypted blocks] ──► External LLM  │
    │  Decrypter ◄── [encrypted blocks] ◄── External LLM  │
    └─────────────────────────────────────────────────────┘

Key features:
- AES-256-GCM authenticated encryption (FIPS-compliant).
- Per-session ephemeral keys derived from a master key via HKDF.
- Block-level "cryptic stream" mode for streaming data in transit.
- Encrypted KV-cache storage (at-rest encryption for latent memory).
- Encrypted shared memory entries (at-rest encryption for RAG+ store).
- Key rotation with backward-compatible decryption of old data.
- Zero-knowledge routing: the router sees only encrypted metadata.

Use cases:
- **Financial firms**: trade data, client PII, internal research never
  leaves the trust boundary in plaintext.
- **Government**: classified prompts processed on-premises; only
  encrypted blocks transit to cloud models for heavy reasoning.
- **Secure communications**: end-to-end encrypted LLM pipelines.

Usage::

    from chen.security import EncryptionConfig, Encryptor, EncryptedBackend
    from chen.backends.mock import MockBackend

    # Generate a master key (store securely — e.g. HSM, KMS, vault)
    config = EncryptionConfig.generate()
    encryptor = Encryptor(config)

    # Wrap any backend — prompts/responses are now encrypted at the boundary
    encrypted_backend = EncryptedBackend(
        inner=MockBackend(params_m=3_000),
        encryptor=encryptor,
    )
"""

from __future__ import annotations

from chen.security.backends import EncryptedBackend
from chen.security.config import EncryptionConfig, SecuritySettings
from chen.security.crypto import (
    CryptoError,
    Decryptor,
    EncryptedBlock,
    Encryptor,
)
from chen.security.keys import KeyMetadata, KeyStore
from chen.security.stream import CrypticStream, StreamBlock

__all__ = [
    "EncryptionConfig",
    "SecuritySettings",
    "Encryptor",
    "Decryptor",
    "EncryptedBlock",
    "CryptoError",
    "EncryptedBackend",
    "KeyStore",
    "KeyMetadata",
    "CrypticStream",
    "StreamBlock",
]
