"""Cryptic stream — block-level encrypted data stream for transit.

The cryptic stream mode splits data into fixed-size blocks, encrypts
each independently with a fresh nonce, and serializes them into a
stream that can be sent to external LLMs or stored on untrusted media.

The external system sees only opaque ciphertext blocks — it cannot read
the content but can:
- Route blocks based on metadata (key_id, block index).
- Store blocks on disk.
- Forward blocks to another system.
- Reorder blocks (each is independently decryptable).

The stream format is a sequence of length-prefixed EncryptedBlock
serializations::

    [4-byte big-endian length][EncryptedBlock.serialize()]
    [4-byte big-endian length][EncryptedBlock.serialize()]
    ...

This format is self-delimiting — the receiver doesn't need to know the
total length in advance, enabling true streaming.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

from chen.security.crypto import Decryptor, EncryptedBlock, Encryptor


@dataclass
class StreamBlock:
    """One block in a cryptic stream.

    Attributes:
        index: Block position in the stream (0-based).
        block: The encrypted block.
        is_last: True if this is the final block in the stream.
    """

    index: int
    block: EncryptedBlock
    is_last: bool


class CrypticStream:
    """A stream of encrypted blocks.

    Usage — encrypting::

        enc = Encryptor(config)
        stream = CrypticStream.encrypt_to_stream(
            data=b"large sensitive document...",
            encryptor=enc,
            session_id=b"session-1",
            block_size=4096,
        )
        # `stream` is a list of StreamBlock objects — send them one at a time

    Usage — decrypting::

        dec = Decryptor(config)
        data = CrypticStream.decrypt_stream(stream, dec, session_id=b"session-1")
    """

    @staticmethod
    def encrypt_to_stream(
        data: bytes,
        encryptor: Encryptor,
        session_id: bytes,
        block_size: Optional[int] = None,  # noqa: UP045
    ) -> list[StreamBlock]:
        """Encrypt data into a list of :class:`StreamBlock` objects."""
        bs = block_size or encryptor.config.block_size
        blocks: list[StreamBlock] = []
        if not data:
            # Empty input → single empty block marked as last.
            block = encryptor.encrypt(b"", session_id)
            blocks.append(StreamBlock(index=0, block=block, is_last=True))
            return blocks
        for i in range(0, len(data), bs):
            chunk = data[i : i + bs]
            block = encryptor.encrypt(chunk, session_id)
            is_last = (i + bs) >= len(data)
            blocks.append(StreamBlock(index=i // bs, block=block, is_last=is_last))
        return blocks

    @staticmethod
    def decrypt_stream(
        blocks: list[StreamBlock],
        decryptor: Decryptor,
        session_id: bytes,
    ) -> bytes:
        """Decrypt a list of :class:`StreamBlock` objects back to plaintext."""
        # Sort by index to handle potential reordering.
        sorted_blocks = sorted(blocks, key=lambda b: b.index)
        return b"".join(decryptor.decrypt(sb.block, session_id) for sb in sorted_blocks)

    @staticmethod
    def serialize_stream(blocks: list[StreamBlock]) -> bytes:
        """Serialize a stream to a single byte string for transmission.

        Format: sequence of [4-byte length][block_bytes][1-byte is_last_flag].
        """
        out = bytearray()
        for sb in blocks:
            block_bytes = sb.block.serialize()
            out.extend(struct.pack(">I", len(block_bytes)))
            out.extend(block_bytes)
            out.extend(struct.pack("B", 1 if sb.is_last else 0))
        return bytes(out)

    @staticmethod
    def deserialize_stream(data: bytes) -> list[StreamBlock]:
        """Deserialize a byte string back to a list of StreamBlocks."""
        blocks: list[StreamBlock] = []
        offset = 0
        idx = 0
        while offset < len(data):
            if offset + 4 > len(data):
                break
            block_len = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4
            if offset + block_len + 1 > len(data):
                break
            block_bytes = data[offset : offset + block_len]
            offset += block_len
            is_last = struct.unpack("B", data[offset : offset + 1])[0]
            offset += 1
            block = EncryptedBlock.deserialize(block_bytes)
            blocks.append(StreamBlock(index=idx, block=block, is_last=bool(is_last)))
            idx += 1
        return blocks

    @staticmethod
    def stream_iter(
        data: bytes,
        encryptor: Encryptor,
        session_id: bytes,
        block_size: Optional[int] = None,  # noqa: UP045
    ) -> Iterator[StreamBlock]:
        """Yield encrypted blocks one at a time (true streaming).

        Useful for large datasets that don't fit in memory — each block
        is produced lazily as the consumer requests it.
        """
        bs = block_size or encryptor.config.block_size
        total = len(data)
        for i in range(0, max(total, 1), bs):
            chunk = data[i : i + bs] if i < total else b""
            block = encryptor.encrypt(chunk, session_id)
            is_last = (i + bs) >= total or i == 0 and total == 0
            yield StreamBlock(index=i // bs, block=block, is_last=is_last)
            if is_last:
                break
