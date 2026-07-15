# Performance Debugging Runbook

## Symptoms

- CHEN is slower than expected (latency > 5s per query).
- CHEN is using more memory than expected.
- CHEN is using more GPU than expected.

## Step 1: Profile a single request

```bash
# Run with debug logging
CHEN_LOG_LEVEL=DEBUG chen run --prompt "Explain recursion." --phase 1

# Or via the API with timing
time curl -X POST http://localhost:8000/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain recursion.","phase":1}'
```

Look for:
- Which expert took the longest? (per-expert latency in the metrics)
- Was the latency in `encode` (prefill) or `decode` (generation)?
- Was there a KV-cache transfer failure that triggered text fallback?

## Step 2: Check the bottleneck

| Bottleneck | Symptom | Fix |
|------------|---------|-----|
| **CPU** (HF backend on CPU) | 100% CPU, slow generation | Add GPU, or use smaller models, or use llama.cpp with quantization |
| **GPU memory** (OOM) | `torch.cuda.OutOfMemoryError` | Reduce `max_tokens`, use smaller model, or use vLLM with PagedAttention |
| **Disk I/O** (model loading) | First request slow, subsequent fast | Pre-load models at startup, or warm the cache |
| **Network** (HF download) | First run slow | Pre-download models to the Docker image or a volume |
| **Lock contention** (SQLite) | Concurrent requests queue | Use per-process SQLite, or migrate to Postgres |

## Step 3: Use the right backend

| Use case | Recommended backend | Why |
|----------|---------------------|-----|
| Tests, demos, CPU dev | `mock` | Deterministic, no model downloads, <1ms |
| Single-user CPU inference | `llama_cpp` | Quantized models, CPU-friendly |
| Single-user GPU inference | `hf` | Direct torch, full KV-cache access |
| Multi-user GPU inference | `vllm` | PagedAttention, batching, streaming |
| Edge / Mac | `llama_cpp` | MPS support, low memory |

## Step 4: Reduce token count

The biggest lever for latency is `max_tokens`. Most queries don't need
128 tokens of output — try 64 or 32.

```bash
chen run --prompt "..." --max-tokens 32
```

## Step 5: Reduce expert count (Phase 3)

The router's `max_activation` defaults to 3. Lower it to 2 to save cost
at the risk of quality:

```python
from chen.core.router import LogisticRouter
router = LogisticRouter.from_experts(experts, max_activation=2)
```

## Step 6: Benchmark regression

If CHEN was fast and is now slow, run the benchmark suite and compare:

```bash
chen bench --phase 1 > bench-current.txt
# Compare to a baseline:
diff bench-baseline.txt bench-current.txt
```

Look for:
- EPU dropping (more params invoked per query → routing regression).
- Latency increasing per expert (model loading overhead).
- KV transfer failures (architecture mismatch).

## Step 7: Profiling tools

### Python profiler

```bash
python -m cProfile -o profile.out -m chen run --prompt "test"
python -p profile.out  # or use snakeviz
```

### PyTorch profiler (HF backend)

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    result = pipe.run("test")
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Memory profiler

```bash
pip install memory-profiler
python -m memory_profiler examples/run_phase1.py
```

## Step 8: When to give up

If you've tried everything and CHEN is still slow:

1. You may be hitting a fundamental limit of small models — upgrade to a larger model.
2. Your workload may not suit CHEN's architecture — if every query needs all experts, you're paying routing overhead for no benefit.
3. Open a Discussion on GitHub with your profiling output — the community may have ideas.
