"""Core cryptographic operations: AES-256-GCM encryption and decryption.

This module is the only place that directly touches the ``cryptography``
library. All other CHEN security modules go through :class:`Encryptor`
and :class:`Decryptor`.

Design choices:
- **AES-256-GCM**: authenticated encryption — protects confidentiality
  AND integrity. Tampered ciphertext is rejected.
- **Per-block nonce**: each encrypted block gets a fresh random nonce.
  Nonce reuse is the #1 failure mode in GCM; we eliminate it by never
  reusing.
- **HKDF key derivation**: per-session keys are derived from the master
  key via HKDF-SHA256, so compromising one session's key does not
  compromise the master.
- **Key ID in ciphertext**: each encrypted block carries the key_id
  used to encrypt it, enabling transparent key rotation — old data is
  decrypted with the old key, new data with the new key.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Optional

from chen.security.config import (
    NONCE_SIZE_BYTES,
    TAG_SIZE_BYTES,
    EncryptionConfig,
)


class CryptoError(Exception):
    """Base class for all cryptographic errors.

    Raised when encryption/decryption fails (wrong key, tampered
    ciphertext, corrupted data). Never reveals the specific cause to
    avoid side-channel information leakage.
    """


@dataclass
class EncryptedBlock:
    """A single encrypted block of data.

    The wire format is::

        [key_id_len: 1 byte][key_id: N bytes][nonce: 12 bytes]
        [ciphertext: M bytes][tag: 16 bytes]

    The key_id is included so the decryptor knows which key to use
    (enabling key rotation). The nonce is unique per block. The GCM
    authentication tag is appended by the encryption library.

    Attributes:
        key_id: Identifier of the key used to encrypt this block.
        nonce: 12-byte random nonce (unique per encryption).
        ciphertext: The encrypted data (same length as plaintext for GCM).
        tag: 16-byte GCM authentication tag.
    """

    key_id: str
    nonce: bytes
    ciphertext: bytes
    tag: bytes

    def __post_init__(self) -> None:
        if len(self.nonce) != NONCE_SIZE_BYTES:
            raise CryptoError(f"nonce must be {NONCE_SIZE_BYTES} bytes, got {len(self.nonce)}")
        if len(self.tag) != TAG_SIZE_BYTES:
            raise CryptoError(f"tag must be {TAG_SIZE_BYTES} bytes, got {len(self.tag)}")

    def serialize(self) -> bytes:
        """Serialize to a byte string for storage or transmission."""
        key_id_bytes = self.key_id.encode("utf-8")
        if len(key_id_bytes) > 255:
            raise CryptoError("key_id too long (max 255 bytes)")
        header = struct.pack("B", len(key_id_bytes))
        return header + key_id_bytes + self.nonce + self.ciphertext + self.tag

    @classmethod
    def deserialize(cls, data: bytes) -> EncryptedBlock:
        """Deserialize from a byte string."""
        if len(data) < 1 + NONCE_SIZE_BYTES + TAG_SIZE_BYTES:
            raise CryptoError("data too short to be an EncryptedBlock")
        offset = 0
        key_id_len = struct.unpack("B", data[offset : offset + 1])[0]
        offset += 1
        if len(data) < 1 + key_id_len + NONCE_SIZE_BYTES + TAG_SIZE_BYTES:
            raise CryptoError("data too short for declared key_id length")
        key_id = data[offset : offset + key_id_len].decode("utf-8")
        offset += key_id_len
        nonce = data[offset : offset + NONCE_SIZE_BYTES]
        offset += NONCE_SIZE_BYTES
        tag_start = len(data) - TAG_SIZE_BYTES
        ciphertext = data[offset:tag_start]
        tag = data[tag_start:]
        return cls(key_id=key_id, nonce=nonce, ciphertext=ciphertext, tag=tag)


@dataclass
class Encryptor:
    """Encrypts data using AES-256-GCM with per-session key derivation.

    Usage::

        config = EncryptionConfig.generate()
        enc = Encryptor(config)
        block = enc.encrypt(b"secret data", session_id=b"session-1")
        # block.ciphertext is opaque — safe to send to external LLMs

    Attributes:
        config: The master encryption configuration.
        _session_keys: Cache of derived session keys (session_id -> key).
    """

    config: EncryptionConfig
    _session_keys: dict[bytes, bytes] = field(default_factory=dict, repr=False)

    def _get_session_key(self, session_id: bytes) -> bytes:
        """Get or derive the key for a given session."""
        if session_id not in self._session_keys:
            self._session_keys[session_id] = self.config.derive_session_key(session_id)
        return self._session_keys[session_id]

    def encrypt(self, plaintext: bytes, session_id: bytes) -> EncryptedBlock:
        """Encrypt ``plaintext`` under the key derived for ``session_id``.

        Args:
            plaintext: The data to encrypt.
            session_id: Unique session identifier for key derivation.

        Returns:
            An :class:`EncryptedBlock` safe to store or transmit.

        Raises:
            CryptoError: If encryption fails.
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        if not isinstance(plaintext, (bytes, bytearray)):
            raise CryptoError("plaintext must be bytes")
        key = self._get_session_key(bytes(session_id))
        nonce = os.urandom(NONCE_SIZE_BYTES)
        aesgcm = AESGCM(key)
        # AESGCM.encrypt returns ciphertext + tag concatenated.
        ct_and_tag = aesgcm.encrypt(nonce, bytes(plaintext), associated_data=None)
        ciphertext = ct_and_tag[:-TAG_SIZE_BYTES]
        tag = ct_and_tag[-TAG_SIZE_BYTES:]
        return EncryptedBlock(
            key_id=self.config.key_id,
            nonce=nonce,
            ciphertext=ciphertext,
            tag=tag,
        )

    def encrypt_text(self, text: str, session_id: bytes) -> EncryptedBlock:
        """Convenience: encrypt a string."""
        return self.encrypt(text.encode("utf-8"), session_id)

    def encrypt_stream(
        self,
        data: bytes,
        session_id: bytes,
        block_size: Optional[int] = None,  # noqa: UP045
    ) -> list[EncryptedBlock]:
        """Encrypt ``data`` as a sequence of blocks (cryptic stream mode).

        Each block is independently encrypted with a fresh nonce. The
        blocks can be streamed to an external system one at a time
        without waiting for the full dataset.

        Args:
            data: The data to encrypt.
            session_id: Session identifier for key derivation.
            block_size: Block size in bytes (defaults to config.block_size).

        Returns:
            A list of :class:`EncryptedBlock` objects.
        """
        bs = block_size or self.config.block_size
        blocks: list[EncryptedBlock] = []
        for i in range(0, len(data), bs):
            chunk = data[i : i + bs]
            blocks.append(self.encrypt(chunk, session_id))
        return blocks


@dataclass
class Decryptor:
    """Decrypts :class:`EncryptedBlock` objects.

    Supports key rotation: if the decryptor has multiple keys (old + new),
    it picks the right one based on the block's ``key_id``.

    Usage::

        config = EncryptionConfig.generate()
        enc = Encryptor(config)
        block = enc.encrypt(b"secret", session_id=b"s1")

        dec = Decryptor(config)
        plaintext = dec.decrypt(block, session_id=b"s1")
        assert plaintext == b"secret"

    Attributes:
        keys: Mapping from key_id to (EncryptionConfig) — supports rotation.
        _session_keys: Cache of derived session keys.
    """

    keys: dict[str, EncryptionConfig] = field(default_factory=dict)
    _session_keys: dict[tuple[str, bytes], bytes] = field(default_factory=dict, repr=False)

    def __init__(self, config: EncryptionConfig) -> None:
        self.keys = {config.key_id: config}
        self._session_keys = {}

    def add_key(self, config: EncryptionConfig) -> None:
        """Register an additional key (for rotation)."""
        self.keys[config.key_id] = config

    def _get_session_key(self, key_id: str, session_id: bytes) -> bytes:
        cache_key = (key_id, bytes(session_id))
        if cache_key not in self._session_keys:
            config = self.keys.get(key_id)
            if config is None:
                raise CryptoError(f"unknown key_id: {key_id}")
            self._session_keys[cache_key] = config.derive_session_key(session_id)
        return self._session_keys[cache_key]

    def decrypt(self, block: EncryptedBlock, session_id: bytes) -> bytes:
        """Decrypt an :class:`EncryptedBlock`.

        Args:
            block: The encrypted block.
            session_id: The session ID used during encryption.

        Returns:
            The original plaintext bytes.

        Raises:
            CryptoError: If decryption fails (wrong key, tampered data,
                unknown key_id).
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        key = self._get_session_key(block.key_id, session_id)
        # Reconstruct the ciphertext+tag blob that AESGCM expects.
        ct_and_tag = block.ciphertext + block.tag
        aesgcm = AESGCM(key)
        try:
            return aesgcm.decrypt(block.nonce, ct_and_tag, associated_data=None)
        except Exception as e:
            # Never reveal the specific error — could leak information.
            raise CryptoError("decryption failed (wrong key or tampered data)") from e

    def decrypt_text(self, block: EncryptedBlock, session_id: bytes) -> str:
        """Convenience: decrypt to a string."""
        return self.decrypt(block, session_id).decode("utf-8")

    def decrypt_stream(self, blocks: list[EncryptedBlock], session_id: bytes) -> bytes:
        """Decrypt a sequence of blocks and concatenate."""
        return b"".join(self.decrypt(b, session_id) for b in blocks)
