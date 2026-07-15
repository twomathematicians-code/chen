# Benchmark Results

This directory contains benchmark results for CHEN against standard datasets.

## Available Benchmarks

| Task | Description | Samples (subset) | Full dataset |
|------|-------------|-------------------|--------------|
| `mmlu` | Massive Multitask Language Understanding | 5 | 14,042 ([cais/mmlu](https://huggingface.co/datasets/cais/mmlu)) |
| `humaneval` | Code generation | 5 | 164 ([openai_humaneval](https://huggingface.co/datasets/openai_humaneval)) |
| `gsm8k` | Grade School Math | 5 | 8,500 ([openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)) |
| `math_arithmetic` | Simple arithmetic | 5 | — |
| `code_python_basics` | Basic Python coding | 5 | — |
| `qa_factual` | Factual Q&A | 5 | — |
| `summarization` | Passage summarization | 2 | — |
| `reasoning_logical` | Multi-step reasoning | 5 | — |

## Running Benchmarks

### Quick run (MockBackend, <1s)

```bash
chen bench --phase 1
chen bench --phase 3 --router logistic
```

### Real model benchmarks (requires HF backend)

```bash
pip install -e ".[hf]"
python examples/run_real_models.py --tier cpu --phase 1
```

### Specific dataset

```bash
chen bench --task mmlu --phase 1
chen bench --task humaneval --phase 1
chen bench --task gsm8k --phase 1
```

### Full benchmark suite

```bash
python examples/run_benchmarks.py --phase 1 --baseline-params 70000
```

## Reproduction Configs

Each benchmark task in `src/chen/benchmarks/tasks.py` includes:
- A `name` for CLI/API reference
- A `description` of the task
- A `samples` list of `(prompt, expected_answer)` tuples
- A `grader` function that scores the output (0.0 to 1.0)
- A `tags` set for filtering

The subset samples are representative — for full benchmark runs, load the complete datasets from HuggingFace and pass them to `BenchmarkRunner`.

## Comparison: CHEN vs. 70B Baseline

> **Note:** The table below will be populated with real numbers after running the benchmarks with the HuggingFace backend. The MockBackend produces deterministic but meaningless output — use it only to verify the harness works.

### Expected results (based on architecture analysis)

| Benchmark | 70B Baseline (expected) | CHEN Phase 1 (target) | CHEN Phase 3 (target) | EPU Target |
|-----------|--------------------------|----------------------|----------------------|------------|
| MMLU | 75-80% | 60-70% | 65-75% | > 3.0 |
| HumanEval | 65-75% | 50-60% | 55-65% | > 3.0 |
| GSM8K | 80-85% | 65-75% | 70-80% | > 3.0 |
| Cost/1M tokens | $0.42 | $0.05-0.10 | $0.03-0.08 | — |
| Latency p50 | 2.0s | 0.5-1.0s | 0.3-0.8s | — |

### How to reproduce

1. Install CHEN with HF backend:
   ```bash
   pip install -e ".[hf]"
   ```

2. Set up the expert models (defaults use open, non-gated models):
   ```bash
   export CHEN_HF_ANALYST_MODEL=HuggingFaceTB/SmolLM2-1.7B-Instruct
   export CHEN_HF_REASONER_MODEL=Qwen/Qwen2.5-3B-Instruct
   export CHEN_HF_SYNTHESIZER_MODEL=HuggingFaceTB/SmolLM2-1.7B-Instruct
   ```

3. Run the benchmarks:
   ```bash
   python examples/run_benchmarks.py --phase 1 --backend hf
   python examples/run_benchmarks.py --phase 3 --backend hf --router logistic
   ```

4. Compare against a 70B baseline:
   ```bash
   # Run the same prompts through a 70B model (e.g. via Together AI or local vLLM)
   # and compare accuracy + cost.
   ```

## KPI Definitions

See [`docs/math/specifications.md`](../math/specifications.md) for formal definitions of:
- **EPU** (Effective Parameter Utilization)
- **Cost per 1M tokens**
- **Latency-to-Accuracy Ratio**
