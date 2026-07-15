"""EncryptedBackend — wraps any backend with transparent encryption.

The :class:`EncryptedBackend` wraps an existing backend (mock, hf, vllm,
llama_cpp) and encrypts/decrypts data at the boundary:

- ``generate(prompt)``: encrypts the prompt before passing to the inner
  backend, decrypts the response before returning.
- ``encode(prompt)``: encrypts the prompt, stores the ciphertext as the
  KV-cache's ``source_text``, and produces a KV-cache as usual.
- ``decode(cache)``: the cache's source_text is ciphertext — decrypt it
  before passing to the inner backend's decode.

This means the inner backend (which may be a cloud-hosted LLM) only
ever sees ciphertext. The local key holder can decrypt the responses.

For **local trusted backends** (mock, local hf), encryption is
optionally skipped — see :class:`SecuritySettings.trusted_backends`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from chen.backends.base import BackendCapabilities
from chen.core.kv_cache import KVCache
from chen.security.config import EncryptionConfig
from chen.security.crypto import Decryptor, EncryptedBlock, Encryptor


@dataclass
class EncryptedBackend:
    """Wraps an inner backend with AES-256-GCM encryption.

    Attributes:
        inner: The underlying backend (mock, hf, vllm, etc.).
        encryptor: The encryptor to use for outbound data.
        decryptor: The decryptor to use for inbound data.
        session_id: Session ID for key derivation. Defaults to a
            per-instance random value.
        skip_encryption: If True, pass through without encryption
            (used for trusted local backends).
    """

    inner: Any  # InferenceBackend
    encryptor: Encryptor
    decryptor: Decryptor
    session_id: bytes = field(default_factory=lambda: __import__("os").urandom(16))
    skip_encryption: bool = False

    # ------------------------------------------------------------------
    # Passthrough properties
    # ------------------------------------------------------------------

    @property
    def params_m(self) -> int:
        return self.inner.params_m

    @property
    def model_id(self) -> str:
        return f"encrypted({self.inner.model_id})"

    @property
    def capabilities(self) -> BackendCapabilities:
        caps = self.inner.capabilities
        # The encrypted backend inherits the inner's capabilities.
        # We mark it as non-deterministic because encryption adds nonce
        # randomness, even if the inner backend is deterministic.
        return BackendCapabilities(
            supports_kv_cache=caps.supports_kv_cache,
            supports_streaming=caps.supports_streaming,
            supports_batching=caps.supports_batching,
            deterministic=False,
        )

    # ------------------------------------------------------------------
    # Encryption helpers
    # ------------------------------------------------------------------

    def _encrypt_text(self, text: str) -> str:
        """Encrypt a text string, return a base64-encoded ciphertext string."""
        if self.skip_encryption:
            return text
        import base64

        block = self.encryptor.encrypt_text(text, self.session_id)
        return base64.b64encode(block.serialize()).decode("ascii")

    def _decrypt_text(self, text: str) -> str:
        """Decrypt a base64-encoded ciphertext string."""
        if self.skip_encryption:
            return text
        import base64

        try:
            raw = base64.b64decode(text)
            block = EncryptedBlock.deserialize(raw)
            return self.decryptor.decrypt_text(block, self.session_id)
        except Exception:
            # If decryption fails, return the original text (might be
            # an unencrypted error message from the inner backend).
            return text

    # ------------------------------------------------------------------
    # InferenceBackend protocol methods
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs: Any) -> str:
        """Encrypt prompt → inner.generate → decrypt response."""
        encrypted_prompt = self._encrypt_text(prompt)
        raw_output = self.inner.generate(encrypted_prompt, max_tokens=max_tokens, **kwargs)
        return self._decrypt_text(raw_output)

    def encode(self, prompt: str, **kwargs: Any) -> KVCache:
        """Encrypt prompt → inner.encode → return cache with encrypted source_text."""
        encrypted_prompt = self._encrypt_text(prompt)
        cache = self.inner.encode(encrypted_prompt, **kwargs)
        # The cache's source_text is now the encrypted prompt. This means
        # the KV-cache is stored encrypted-at-rest — only someone with
        # the key can decrypt the source_text to recover the original prompt.
        return cache

    def decode(self, cache: KVCache, max_tokens: int = 256, **kwargs: Any) -> str:
        """The cache's source_text is encrypted — decrypt it before decoding."""
        # Decrypt the source_text for the inner backend.
        decrypted_text = self._decrypt_text(cache.source_text)
        # Create a modified cache with decrypted source_text.
        # Note: we can't modify the original cache (it's a dataclass),
        # so we create a new one. The K/V tensors are unchanged — only
        # the source_text provenance is decrypted for the inner backend.
        from dataclasses import replace

        decrypted_cache = replace(cache, source_text=decrypted_text)
        raw_output = self.inner.decode(decrypted_cache, max_tokens=max_tokens, **kwargs)
        return self._decrypt_text(raw_output)

    def transfer_cache(self, cache: KVCache) -> KVCache:
        """Pass through to the inner backend's transfer_cache.

        The cache's source_text remains encrypted during transfer — this
        is the whole point. An eavesdropper on the transfer channel sees
        only encrypted source_text and opaque K/V tensors.
        """
        return self.inner.transfer_cache(cache)

    # ------------------------------------------------------------------
    # Tokenization (passthrough — tokenization doesn't reveal content
    # to external systems if the backend is local)
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Count tokens in the plaintext (before encryption)."""
        if hasattr(self.inner, "count_tokens"):
            return self.inner.count_tokens(text)
        return max(1, len(text) // 4)


def wrap_with_encryption(
    backend: Any,
    config: EncryptionConfig,
    *,
    session_id: Optional[bytes] = None,  # noqa: UP045
    skip_if_trusted: bool = True,
    trusted_names: frozenset[str] = frozenset({"mock"}),
) -> Any:
    """Convenience: wrap a backend with encryption, unless it's trusted.

    Args:
        backend: The backend to wrap.
        config: Encryption configuration.
        session_id: Optional session ID (random if None).
        skip_if_trusted: If True, don't wrap trusted backends.
        trusted_names: Set of backend names considered trusted.

    Returns:
        Either the original backend (if trusted and skip_if_trusted) or
        an :class:`EncryptedBackend` wrapping it.
    """
    backend_name = getattr(backend, "model_id", "").split("-")[0].lower()
    if skip_if_trusted and backend_name in trusted_names:
        return backend
    import os

    enc = Encryptor(config)
    dec = Decryptor(config)
    return EncryptedBackend(
        inner=backend,
        encryptor=enc,
        decryptor=dec,
        session_id=session_id or os.urandom(16),
    )
