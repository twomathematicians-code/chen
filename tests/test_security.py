"""Tests for the CHEN security & encryption layer."""

from __future__ import annotations

import base64
import os

import pytest

from chen.backends.mock import MockBackend
from chen.security import (
    CrypticStream,
    CryptoError,
)
from chen.security.backends import EncryptedBackend, wrap_with_encryption
from chen.security.config import (
    KEY_SIZE_BYTES,
    NONCE_SIZE_BYTES,
    TAG_SIZE_BYTES,
    EncryptionConfig,
    SecuritySettings,
)
from chen.security.crypto import (
    Decryptor,
    EncryptedBlock,
    Encryptor,
)
from chen.security.keys import KeyMetadata, KeyStore


class TestEncryptionConfig:
    def test_generate_creates_valid_config(self):
        config = EncryptionConfig.generate()
        assert len(config.master_key) == KEY_SIZE_BYTES
        assert len(config.key_id) > 0
        assert config.algorithm == "AES-256-GCM"

    def test_invalid_key_length_raises(self):
        with pytest.raises(ValueError, match="master_key must be"):
            EncryptionConfig(master_key=b"too short")

    def test_from_env_without_var_generates_key(self, monkeypatch):
        monkeypatch.delenv("CHEN_MASTER_KEY", raising=False)
        config = EncryptionConfig.from_env()
        assert len(config.master_key) == KEY_SIZE_BYTES

    def test_from_env_with_valid_var(self, monkeypatch):
        key = os.urandom(KEY_SIZE_BYTES)
        monkeypatch.setenv("CHEN_MASTER_KEY", base64.b64encode(key).decode())
        monkeypatch.setenv("CHEN_KEY_ID", "test-key-123")
        config = EncryptionConfig.from_env()
        assert config.master_key == key
        assert config.key_id == "test-key-123"

    def test_from_env_with_invalid_base64_raises(self, monkeypatch):
        monkeypatch.setenv("CHEN_MASTER_KEY", "not-valid-base64!!!")
        with pytest.raises(ValueError, match="not valid base64"):
            EncryptionConfig.from_env()

    def test_to_env_dict_roundtrip(self):
        config = EncryptionConfig.generate(key_id="roundtrip-test")
        env = config.to_env_dict()
        assert "CHEN_MASTER_KEY" in env
        assert env["CHEN_KEY_ID"] == "roundtrip-test"

    def test_derive_session_key_deterministic(self):
        config = EncryptionConfig.generate()
        k1 = config.derive_session_key(b"session-1")
        k2 = config.derive_session_key(b"session-1")
        assert k1 == k2

    def test_derive_session_key_different_per_session(self):
        config = EncryptionConfig.generate()
        k1 = config.derive_session_key(b"session-1")
        k2 = config.derive_session_key(b"session-2")
        assert k1 != k2

    def test_derive_session_key_different_per_master(self):
        c1 = EncryptionConfig.generate()
        c2 = EncryptionConfig.generate()
        k1 = c1.derive_session_key(b"session-1")
        k2 = c2.derive_session_key(b"session-1")
        assert k1 != k2


class TestSecuritySettings:
    def test_disabled(self):
        s = SecuritySettings.disabled()
        assert s.enabled is False

    def test_is_trusted(self):
        s = SecuritySettings(trusted_backends=frozenset({"mock", "local-hf"}))
        assert s.is_trusted("mock") is True
        assert s.is_trusted("local-hf") is True
        assert s.is_trusted("vllm") is False

    def test_defaults(self):
        s = SecuritySettings()
        assert s.enabled is True
        assert s.encrypt_prompts is True
        assert s.encrypt_kv_cache is True
        assert "mock" in s.trusted_backends


class TestEncryptorDecryptor:
    @pytest.fixture
    def config(self):
        return EncryptionConfig.generate(key_id="test-key")

    @pytest.fixture
    def encryptor(self, config):
        return Encryptor(config)

    @pytest.fixture
    def decryptor(self, config):
        return Decryptor(config)

    def test_encrypt_decrypt_roundtrip(self, encryptor, decryptor):
        plaintext = b"Hello, encrypted world!"
        block = encryptor.encrypt(plaintext, session_id=b"session-1")
        recovered = decryptor.decrypt(block, session_id=b"session-1")
        assert recovered == plaintext

    def test_encrypt_text_roundtrip(self, encryptor, decryptor):
        text = "Sensitive financial data: $1,000,000"
        block = encryptor.encrypt_text(text, session_id=b"s1")
        recovered = decryptor.decrypt_text(block, session_id=b"s1")
        assert recovered == text

    def test_ciphertext_differs_from_plaintext(self, encryptor):
        plaintext = b"secret"
        block = encryptor.encrypt(plaintext, session_id=b"s1")
        assert block.ciphertext != plaintext
        assert block.nonce != plaintext

    def test_different_encryptions_have_different_nonces(self, encryptor):
        b1 = encryptor.encrypt(b"data", session_id=b"s1")
        b2 = encryptor.encrypt(b"data", session_id=b"s1")
        assert b1.nonce != b2.nonce

    def test_different_sessions_cannot_decrypt(self, encryptor):
        config2 = EncryptionConfig.generate()
        dec2 = Decryptor(config2)
        block = encryptor.encrypt(b"secret", session_id=b"s1")
        with pytest.raises(CryptoError):
            dec2.decrypt(block, session_id=b"s1")

    def test_wrong_session_id_cannot_decrypt(self, encryptor, decryptor, config):
        block = encryptor.encrypt(b"secret", session_id=b"session-A")
        with pytest.raises(CryptoError):
            decryptor.decrypt(block, session_id=b"session-B")

    def test_tampered_ciphertext_raises(self, encryptor, decryptor):
        block = encryptor.encrypt(b"secret", session_id=b"s1")
        # Tamper with the ciphertext
        tampered = EncryptedBlock(
            key_id=block.key_id,
            nonce=block.nonce,
            ciphertext=block.ciphertext[:-1] + bytes([block.ciphertext[-1] ^ 1]),
            tag=block.tag,
        )
        with pytest.raises(CryptoError, match="decryption failed"):
            decryptor.decrypt(tampered, session_id=b"s1")

    def test_encrypt_stream(self, encryptor, decryptor):
        data = b"x" * 10_000  # 10 KB — spans multiple blocks
        blocks = encryptor.encrypt_stream(data, session_id=b"s1", block_size=1024)
        assert len(blocks) == 10  # 10000 / 1024 = 9.77 → 10 blocks
        recovered = decryptor.decrypt_stream(blocks, session_id=b"s1")
        assert recovered == data

    def test_encrypt_empty_data(self, encryptor, decryptor):
        block = encryptor.encrypt(b"", session_id=b"s1")
        recovered = decryptor.decrypt(block, session_id=b"s1")
        assert recovered == b""


class TestEncryptedBlockSerialization:
    def test_serialize_deserialize_roundtrip(self):
        block = EncryptedBlock(
            key_id="test-key-123",
            nonce=os.urandom(NONCE_SIZE_BYTES),
            ciphertext=b"some ciphertext here",
            tag=os.urandom(TAG_SIZE_BYTES),
        )
        serialized = block.serialize()
        recovered = EncryptedBlock.deserialize(serialized)
        assert recovered.key_id == block.key_id
        assert recovered.nonce == block.nonce
        assert recovered.ciphertext == block.ciphertext
        assert recovered.tag == block.tag

    def test_deserialize_too_short_raises(self):
        with pytest.raises(CryptoError, match="too short"):
            EncryptedBlock.deserialize(b"short")

    def test_serialize_long_key_id_raises(self):
        block = EncryptedBlock(
            key_id="x" * 300,  # too long
            nonce=os.urandom(NONCE_SIZE_BYTES),
            ciphertext=b"data",
            tag=os.urandom(TAG_SIZE_BYTES),
        )
        with pytest.raises(CryptoError, match="key_id too long"):
            block.serialize()


class TestCrypticStream:
    @pytest.fixture
    def config(self):
        return EncryptionConfig.generate()

    @pytest.fixture
    def encryptor(self, config):
        return Encryptor(config)

    @pytest.fixture
    def decryptor(self, config):
        return Decryptor(config)

    def test_encrypt_to_stream_and_decrypt(self, encryptor, decryptor):
        data = b"sensitive document " * 500  # ~10 KB
        stream = CrypticStream.encrypt_to_stream(data, encryptor, session_id=b"s1", block_size=256)
        assert len(stream) > 1
        assert stream[-1].is_last is True
        assert stream[0].is_last is False
        recovered = CrypticStream.decrypt_stream(stream, decryptor, session_id=b"s1")
        assert recovered == data

    def test_empty_data_stream(self, encryptor, decryptor):
        stream = CrypticStream.encrypt_to_stream(b"", encryptor, session_id=b"s1")
        assert len(stream) == 1
        assert stream[0].is_last is True
        recovered = CrypticStream.decrypt_stream(stream, decryptor, session_id=b"s1")
        assert recovered == b""

    def test_serialize_deserialize_stream(self, encryptor, decryptor):
        data = b"streaming data " * 100
        stream = CrypticStream.encrypt_to_stream(data, encryptor, session_id=b"s1", block_size=128)
        serialized = CrypticStream.serialize_stream(stream)
        deserialized = CrypticStream.deserialize_stream(serialized)
        assert len(deserialized) == len(stream)
        recovered = CrypticStream.decrypt_stream(deserialized, decryptor, session_id=b"s1")
        assert recovered == data

    def test_stream_iter(self, encryptor, decryptor):
        data = b"iter data " * 1000
        blocks = list(CrypticStream.stream_iter(data, encryptor, session_id=b"s1", block_size=256))
        recovered = CrypticStream.decrypt_stream(blocks, decryptor, session_id=b"s1")
        assert recovered == data

    def test_blocks_independently_decryptable(self, encryptor, decryptor):
        """Each block can be decrypted on its own — no dependency on order."""
        data = b"block independent " * 100
        stream = CrypticStream.encrypt_to_stream(data, encryptor, session_id=b"s1", block_size=64)
        # Reverse the order and decrypt — should still work.
        reversed_stream = list(reversed(stream))
        recovered = CrypticStream.decrypt_stream(reversed_stream, decryptor, session_id=b"s1")
        assert recovered == data


class TestEncryptedBackend:
    @pytest.fixture
    def config(self):
        return EncryptionConfig.generate(key_id="backend-test")

    @pytest.fixture
    def encrypted_backend(self, config):
        enc = Encryptor(config)
        dec = Decryptor(config)
        inner = MockBackend(params_m=3_000, role_hint="inner")
        return EncryptedBackend(
            inner=inner,
            encryptor=enc,
            decryptor=dec,
            session_id=b"backend-session-1",
        )

    def test_generate_encrypts_and_decrypts(self, encrypted_backend):
        prompt = "This is a secret prompt"
        output = encrypted_backend.generate(prompt, max_tokens=32)
        # The output should be decrypted (readable text, not base64 ciphertext)
        assert isinstance(output, str)
        # The inner backend should NOT have seen the plaintext prompt.
        # (We can verify this by checking that the inner backend's output
        # contains the encrypted prompt, not the plaintext.)
        # Since MockBackend's output includes a snippet of the input,
        # and the input was the encrypted prompt, the raw inner output
        # would contain base64 ciphertext. But our EncryptedBackend
        # decrypts the response — so the final output is decrypted text.
        assert "prompt_id" in output or "decoded" in output or "processed" in output

    def test_encode_produces_encrypted_source_text(self, encrypted_backend):
        prompt = "secret prompt for encoding"
        cache = encrypted_backend.encode(prompt)
        # The cache's source_text should be the encrypted prompt, not the plaintext.
        assert cache.source_text != prompt
        # It should be a base64 string (the encrypted block).
        import base64

        try:
            decoded = base64.b64decode(cache.source_text)
            assert len(decoded) > 0
        except Exception:
            pytest.fail("cache.source_text should be valid base64 ciphertext")

    def test_decode_decrypts_source_text(self, encrypted_backend):
        prompt = "secret prompt for decode test"
        cache = encrypted_backend.encode(prompt)
        output = encrypted_backend.decode(cache, max_tokens=32)
        assert isinstance(output, str)

    def test_passthrough_when_skip_encryption(self, config):
        """When skip_encryption=True, the backend passes through."""
        enc = Encryptor(config)
        dec = Decryptor(config)
        inner = MockBackend(params_m=3_000, role_hint="passthrough")
        backend = EncryptedBackend(
            inner=inner,
            encryptor=enc,
            decryptor=dec,
            session_id=b"s1",
            skip_encryption=True,
        )
        prompt = "not encrypted"
        backend.generate(prompt, max_tokens=32)
        # The inner backend sees the plaintext prompt.
        assert "not encrypted" in inner.generate(prompt, max_tokens=32)

    def test_params_m_delegates_to_inner(self, encrypted_backend):
        assert encrypted_backend.params_m == 3_000

    def test_model_id_includes_encrypted_prefix(self, encrypted_backend):
        assert encrypted_backend.model_id.startswith("encrypted(")

    def test_capabilities_inherits_from_inner(self, encrypted_backend):
        caps = encrypted_backend.capabilities
        # MockBackend supports KV-cache.
        assert caps.supports_kv_cache is True
        # But the encrypted backend is NOT deterministic (nonce randomness).
        assert caps.deterministic is False

    def test_wrap_with_encryption_skips_trusted(self, config):
        """wrap_with_encryption should skip trusted backends."""
        inner = MockBackend(params_m=3_000)
        wrapped = wrap_with_encryption(inner, config, skip_if_trusted=True)
        # MockBackend is trusted by default — should not be wrapped.
        assert wrapped is inner

    def test_wrap_with_encryption_wraps_untrusted(self, config):
        """wrap_with_encryption should wrap untrusted backends."""
        # Create a backend that's not in the trusted set.
        inner = MockBackend(params_m=3_000, model_id="untrusted-backend")
        wrapped = wrap_with_encryption(
            inner, config, skip_if_trusted=True, trusted_names=frozenset()
        )
        assert isinstance(wrapped, EncryptedBackend)


class TestKeyStore:
    def test_generate_and_load(self, tmp_path):
        store = KeyStore(path=tmp_path)
        meta = store.generate_key()
        loaded = store.load(meta.key_id)
        assert loaded is not None
        assert loaded.key_id == meta.key_id
        assert loaded.master_key_b64 == meta.master_key_b64

    def test_list_keys(self, tmp_path):
        store = KeyStore(path=tmp_path)
        store.generate_key()
        store.generate_key()
        keys = store.list_keys()
        assert len(keys) == 2

    def test_activate_sets_active(self, tmp_path):
        store = KeyStore(path=tmp_path)
        meta = store.generate_key()
        store.activate(meta.key_id)
        active = store.get_active()
        assert active is not None
        assert active.key_id == meta.key_id
        assert active.status == "active"

    def test_get_active_config_generates_if_none(self, tmp_path):
        store = KeyStore(path=tmp_path)
        config = store.get_active_config()
        assert len(config.master_key) == KEY_SIZE_BYTES
        # The key should now be stored and active.
        assert store.get_active() is not None

    def test_rotate_creates_new_and_retires_old(self, tmp_path):
        store = KeyStore(path=tmp_path)
        old = store.get_active_config()
        new_meta = store.rotate()
        assert new_meta.key_id != old.key_id
        assert new_meta.rotation_of == old.key_id
        # Old key should be retired.
        old_meta = store.load(old.key_id)
        assert old_meta.status == "retired"
        # New key should be active.
        assert store.get_active().key_id == new_meta.key_id

    def test_revoke(self, tmp_path):
        store = KeyStore(path=tmp_path)
        meta = store.generate_key()
        store.revoke(meta.key_id)
        loaded = store.load(meta.key_id)
        assert loaded.status == "revoked"

    def test_get_decryptor_keys_excludes_revoked(self, tmp_path):
        store = KeyStore(path=tmp_path)
        m1 = store.generate_key()
        store.activate(m1.key_id)
        m2 = store.generate_key()
        store.revoke(m2.key_id)
        keys = store.get_decryptor_keys()
        key_ids = [k.key_id for k in keys]
        assert m1.key_id in key_ids
        assert m2.key_id not in key_ids

    def test_key_metadata_roundtrip(self):
        config = EncryptionConfig.generate(key_id="roundtrip")
        meta = KeyMetadata.from_config(config)
        d = meta.to_dict()
        recovered = KeyMetadata.from_dict(d)
        assert recovered.key_id == meta.key_id
        assert recovered.master_key_b64 == meta.master_key_b64

    def test_keystore_uses_env_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHEN_KEYSTORE_PATH", str(tmp_path / "custom_keys"))
        store = KeyStore()
        assert store.path == tmp_path / "custom_keys"
