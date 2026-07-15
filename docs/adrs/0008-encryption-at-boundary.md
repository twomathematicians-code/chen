# ADR 0008: Encryption-at-boundary for confidential LLM inference

- Status: Accepted
- Date: 2025-07-15
- Deciders: CHEN core team

## Context

CHEN's core architecture routes prompts through a pipeline of small,
specialized models. In regulated industries — financial services,
government, healthcare, legal — the prompt content itself is often
confidential:

- **Financial firms**: trade signals, client PII, internal research,
  pre-release earnings data.
- **Government**: classified prompts, citizen data, intelligence
  queries.
- **Healthcare**: patient records, diagnoses, treatment plans (HIPAA,
  GDPR Article 9).
- **Legal**: attorney-client privileged communications, case strategy.

Sending this data to a cloud-hosted LLM (OpenAI, Anthropic, or even a
self-hosted model on shared infrastructure) creates compliance and
security risks. The data could be:

1. **Intercepted** in transit (network eavesdropping).
2. **Logged** by the cloud provider (for abuse monitoring or training).
3. **Stored** in the provider's prompt cache or training data.
4. **Subpoenaed** by a government with jurisdiction over the provider.
5. **Leaked** in a breach of the provider's infrastructure.

CHEN's architecture already supports local inference (MockBackend,
HuggingFaceBackend on local GPUs). But users want the *option* to
offload heavy reasoning to cloud models while keeping their data
confidential — "compute on encrypted data" for LLMs.

## Decision

CHEN will implement an **encryption-at-boundary** layer that encrypts
all data crossing the trust boundary between local and external
systems. The design has five components:

### 1. AES-256-GCM authenticated encryption

All encryption uses AES-256 in GCM mode (NIST SP 800-38D). GCM provides
both confidentiality (encryption) and integrity (authentication tag) —
tampered ciphertext is detected and rejected. This is the same algorithm
approved by NSA for TOP SECRET data (CNSSP-15).

### 2. Per-session key derivation via HKDF-SHA256

The master key (32 bytes, stored in a keystore or HSM) is never used
directly for encryption. Instead, per-session keys are derived via
HKDF-SHA256 with a session-specific salt. This means:

- Compromising one session's key does not compromise the master key.
- Sessions are cryptographically isolated from each other.
- Key derivation is deterministic (same master + same session → same key).

### 3. Cryptic stream mode for block-level encryption

Large data (documents, code, long prompts) is split into fixed-size
blocks (default 4 KB). Each block is encrypted independently with a
fresh random nonce. The blocks form a "cryptic stream" that can be:

- Streamed to an external system one block at a time.
- Stored on untrusted media (cloud storage, shared filesystems).
- Reordered in transit (each block is independently decryptable).
- Routed based on metadata (key_id, block index) without decryption.

The wire format is self-delimiting: `[4-byte length][block bytes][1-byte
is_last_flag]`. The receiver doesn't need to know the total length in
advance.

### 4. EncryptedBackend wrapper

A transparent wrapper (`EncryptedBackend`) that sits between the
pipeline and any backend. It:

- Encrypts prompts before passing to the inner backend.
- Decrypts responses before returning to the pipeline.
- Encrypts the `source_text` field of KV-caches (at-rest encryption).
- Carries the `key_id` in each encrypted block (for key rotation).

Trusted local backends (MockBackend, local HuggingFaceBackend) can be
configured to skip encryption via `SecuritySettings.trusted_backends`.

### 5. Key rotation with backward compatibility

Each encrypted block carries a `key_id`. The decryptor maintains a map
of `key_id → EncryptionConfig` and picks the right key for each block.
This enables:

- **Zero-downtime rotation**: generate a new key, activate it, old data
  is still decryptable with the retired key.
- **Revocation**: mark a key as revoked — it can no longer decrypt new
  data, but old data remains accessible (audit requirement).
- **Multiple active keys**: during a rotation window, both old and new
  keys are active.

Keys are stored in a file-based keystore (`~/.chen/keys/`) with
restrictive permissions (0o600). For production, this should be
replaced with an HSM, AWS KMS, GCP KMS, or HashiCorp Vault integration.

## Consequences

### Positive

- **Compliance**: enables CHEN deployment in regulated industries
  where plaintext data cannot leave the trust boundary.
- **Zero-knowledge external processing**: external LLMs see only
  ciphertext — they cannot log, store, or leak the prompt content.
- **At-rest encryption**: KV-caches and memory entries are encrypted on
  disk, protecting against physical theft or disk disposal attacks.
- **Key rotation**: security teams can rotate keys on a schedule without
  disrupting operations or losing access to historical data.
- **Auditability**: each encrypted block carries a `key_id`, so the
  audit trail shows which key was used for which data.
- **Performance**: AES-256-GCM is hardware-accelerated on modern CPUs
  (AES-NI). Encryption overhead is typically <1% of inference latency.

### Negative

- **External LLMs cannot process ciphertext meaningfully**: an external
  cloud LLM receiving encrypted prompts cannot generate useful text
  output. This limits the "offload to cloud" use case to:
  - Routing and storage (the external system sees opaque blocks).
  - Homomorphic-style computation (future research — not supported today).
  - The external LLM must be within the trust boundary OR the local
    model must do the actual inference.
- **Key management burden**: the operator must securely store and
  rotate keys. Losing the master key means losing all encrypted data.
- **Latency overhead**: each encryption/decryption operation adds ~0.1ms.
  For high-throughput pipelines, this is negligible, but it's non-zero.
- **No search on encrypted data**: the shared memory store cannot
  perform semantic search on encrypted entries (embeddings would need
  to be computed on plaintext). This limits zero-knowledge RAG+ to
  keyword-based retrieval on encrypted metadata.

### Neutral

- The `cryptography` library is now a core dependency (not optional).
  It's the most widely-used Python crypto library (PyCA, audited).
- The encryption layer is opt-in via `SecuritySettings.enabled`. Users
  who don't need encryption pay no overhead.

## Alternatives considered

### Alternative A: TLS-only (transport encryption)

Use HTTPS/TLS for all communication with external LLMs, but don't
encrypt the data itself.

**Why not:** TLS protects data in transit, but not at rest. The cloud
provider still sees plaintext prompts and can log/store them. TLS also
doesn't protect against insider threats at the provider. CHEN's
encryption-at-boundary protects data even if the provider's
infrastructure is compromised.

### Alternative B: Homomorphic encryption (FHE)

Use fully homomorphic encryption to perform inference directly on
ciphertext.

**Why not:** FHE is 1000–10000× slower than plaintext computation.
Current FHE schemes cannot run a 7B parameter transformer in
reasonable time. CHEN's approach (encrypt at boundary, decrypt
locally) is practical today. FHE is a research direction for v0.4+.

### Alternative C: Differential privacy

Add noise to prompts/responses to prevent reconstruction of individual
data points.

**Why not:** DP reduces output quality (the privacy-utility tradeoff).
It also doesn't protect against direct data exfiltration. DP is
complementary, not a replacement for encryption.

### Alternative D: Secure enclaves (SGX, TrustZone)

Run the external LLM inside a hardware secure enclave that guarantees
confidentiality.

**Why not:** Enclaves require specific hardware (Intel SGX, AMD SEV,
ARM TrustZone) and are not universally available. They also have
side-channel vulnerabilities (Spectre, Meltdown, Foreshadow). CHEN's
approach works on any hardware. Enclave support is a future
optimization, not a replacement.

### Alternative E: Token-level encryption

Encrypt each token individually rather than the whole prompt.

**Why not:** Token-level encryption leaks the token count (an
information-theoretic leak). Block-level encryption (CHEN's approach)
pads blocks to a fixed size, leaking only the block count — much less
information.

## References

- [NIST SP 800-38D](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38d.pdf) — GCM specification
- [RFC 5869](https://datatracker.ietf.org/doc/html/rfc5869) — HKDF
- [PyCA cryptography](https://cryptography.io/) — Python crypto library
- [CNSSP-15](https://www.cnss.gov/CNSS/issuances/Policies/) — NSA encryption standards
