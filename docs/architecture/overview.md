# System Architecture

This document gives a visual overview of CHEN's architecture. For the
prose explanation, see [`ARCHITECTURE.md`](https://github.com/your-org/chen/blob/main/ARCHITECTURE.md). For
formal definitions of KPIs and cost models, see
[`math/specifications.md`](../math/specifications.md).

## 1. High-level topology

```mermaid
graph TB
    subgraph "Client"
        User[User / CLI / HTTP Client]
    end
    subgraph "CHEN"
        API[FastAPI server<br/>or CLI]
        Router[Latent State Router]
        Pipeline[Pipeline orchestrator]
        Memory[Shared External Memory<br/>RAG+]
        RunStore[SQLite run store<br/>for reproducibility]
    end
    subgraph "Expert garage"
        E1[Analyst<br/>3B]
        E2[Reasoner<br/>8B]
        E3[Coder<br/>7B]
        E4[Synthesizer<br/>3B]
    end
    subgraph "Backends"
        B1[MockBackend]
        B2[HuggingFaceBackend]
        B3[vLLM Backend]
        B4[llama.cpp Backend]
    end

    User -->|prompt| API
    API --> Pipeline
    Pipeline --> Router
    Router -->|selected experts| Pipeline
    Pipeline <-->|retrieve / write| Memory
    Pipeline -->|invoke| E1
    Pipeline -->|invoke| E2
    Pipeline -->|invoke| E3
    Pipeline -->|invoke| E4
    Pipeline -->|persist run| RunStore
    E1 -.-> B1
    E1 -.-> B2
    E2 -.-> B2
    E3 -.-> B3
    E4 -.-> B4
```

## 2. Pipeline data flow (Phase 2 — KV-cache passing)

```mermaid
sequenceDiagram
    participant U as User
    participant P as Pipeline
    participant R as Router
    participant M as Memory
    participant E1 as Expert 1<br/>(Analyst)
    participant E2 as Expert 2<br/>(Reasoner)
    participant E3 as Expert 3<br/>(Synthesizer)

    U->>P: run(prompt)
    P->>R: route(prompt, experts)
    R-->>P: [analyst, reasoner, synthesizer]
    P->>M: retrieve(prompt, k=4)
    M-->>P: context entries

    P->>E1: encode(prompt + context)
    E1-->>P: KVCache_1
    Note over E1: Analyst runs prefill,<br/>produces KV-cache.

    P->>E2: transfer_cache(KVCache_1)
    E2-->>P: KVCache_1' (adapted)
    P->>E2: decode(KVCache_1')
    E2-->>P: output_2, KVCache_2
    Note over E2: Reasoner decodes<br/>from transferred cache.

    P->>E3: transfer_cache(KVCache_2)
    E3-->>P: KVCache_2' (adapted)
    P->>E3: decode(KVCache_2')
    E3-->>P: final_output
    Note over E3: Synthesizer produces<br/>natural-language output.

    P->>M: write(final_output, role="synthesizer")
    P->>P: aggregate metrics
    P-->>U: PipelineResult
```

## 3. Backend abstraction

```mermaid
classDiagram
    class InferenceBackend {
        <<protocol>>
        +params_m: int
        +capabilities: BackendCapabilities
        +generate(prompt, max_tokens) str
        +encode(prompt) KVCache
        +decode(cache, max_tokens) str
        +transfer_cache(cache) KVCache
    }

    class MockBackend {
        +params_m: int
        +model_id: str
        +n_layers: int
        +n_heads: int
        +head_dim: int
        +seed: int
        +role_hint: str
    }

    class HuggingFaceBackend {
        +model_id: str
        +device: str
        +dtype: str
        +trust_remote_code: bool
    }

    class VLLMBackend {
        +model_id: str
        +tensor_parallel_size: int
        +gpu_memory_utilization: float
    }

    class LlamaCppBackend {
        +model_path: str
        +n_ctx: int
        +n_gpu_layers: int
    }

    InferenceBackend <|.. MockBackend
    InferenceBackend <|.. HuggingFaceBackend
    InferenceBackend <|.. VLLMBackend
    InferenceBackend <|.. LlamaCppBackend
```

## 4. Router decision flow

```mermaid
flowchart TD
    Start([Prompt arrives]) --> Extract[Extract features<br/>keyword density<br/>length bucket<br/>code/math indicators]
    Extract --> Score[Score each role<br/>logit = bias + sum weights × features]
    Score --> Sigmoid[Apply sigmoid<br/>score = 1 / 1 + exp-logit]
    Sigmoid --> Filter[Filter roles<br/>with score ≥ threshold]
    Filter --> Sort[Sort by score<br/>descending]
    Sort --> Cap[Cap at max_activation]
    Cap --> Force[Force force_last_role<br/>to be last]
    Force --> Return([Return ordered expert list])

    Filter -->|none pass threshold| Fallback[Fallback:<br/>pick top-1 or synthesizer]
    Fallback --> Force
```

## 5. Deployment topology

```mermaid
graph TB
    subgraph "Development"
        DevLaptop[Developer laptop<br/>venv + MockBackend]
        DevLaptop -->|chen run| CLI[CLI app]
        DevLaptop -->|chen serve :8000| LocalAPI[Local API]
    end

    subgraph "CI / CD"
        GitHub[GitHub Actions]
        GitHub -->|on push| Lint[ruff + mypy]
        GitHub -->|on push| Test[pytest 3.9-3.12<br/>Ubuntu + macOS + Windows]
        GitHub -->|on tag| Publish[PyPI + GHCR]
        GitHub -->|on push to main| Docs[MkDocs to GH Pages]
    end

    subgraph "Production single-node"
        Server[Docker container<br/>chen serve :8000]
        Server -->|read/write| SQLite[(SQLite<br/>runs.sqlite3)]
        Server -->|optional| Chroma[(ChromaDB)]
        Server -->|optional| HF[HuggingFace Hub<br/>weight download]
        Prometheus[Prometheus] -->|scrape /v1/metrics| Server
    end

    subgraph "Production multi-node"
        LB[Load balancer]
        LB --> Node1[CHEN node 1]
        LB --> Node2[CHEN node 2]
        LB --> Node3[CHEN node 3]
        Node1 -.->|shared| Postgres[(Postgres<br/>runs table)]
        Node2 -.->|shared| Postgres
        Node3 -.->|shared| Postgres
    end
```

## 6. Reproducibility flow

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as CLI / API
    participant Rep as Reproducibility
    participant Pipe as Pipeline
    participant Store as RunStore (SQLite)

    U->>CLI: chen run --prompt "..." --save-run
    CLI->>Rep: hash_config({phase, backend, router, ...})
    Rep-->>CLI: config_hash (SHA-256)
    CLI->>Rep: seed_everything(42)
    CLI->>Pipe: run(prompt)
    Pipe-->>CLI: PipelineResult
    CLI->>Store: save(RunRecord)
    Note over Store: run_id = config_hash[:16]<br/>timestamp = now()<br/>KPIs, output, config all persisted
    CLI-->>U: output + run_id

    Note over U,Store: Later: chen runs the same config_hash<br/>→ same run_id → idempotent save
```

## 7. State machine: KV-cache transfer

```mermaid
stateDiagram-v2
    [*] --> Encode: expert.invoke(prompt)
    Encode --> HasCache: encode succeeded
    Encode --> TextMode: encode failed

    HasCache --> Transfer: next expert.invoke(cache=...)
    Transfer --> Transferred: same-family (no-op)
    Transfer --> ReEncoded: shape mismatch (re-encode)
    Transfer --> Failed: incompatible (IncompatibleCacheError)

    Transferred --> Decode
    ReEncoded --> Decode
    Failed --> TextFallback: pipeline catches error
    TextFallback --> TextMode

    TextMode --> Generate: expert.invoke(prompt=text)
    Decode --> HasOutput
    Generate --> HasOutput

    HasOutput --> [*]
```
