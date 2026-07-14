---
name: Bug report
about: Report a defect in CHEN
title: "[bug] "
labels: bug
assignees: ''
---

## Describe the bug

<!-- A clear and concise description of what the bug is. -->

## To reproduce

```python
# Minimal code snippet that reproduces the issue
```

Steps:
1. ...
2. ...
3. ...

## Expected behavior

<!-- What you expected to happen. -->

## Actual behavior

<!-- What actually happened, including stack traces / error messages. -->

```
Paste error output here
```

## Environment

- CHEN version: `python -c "import chen; print(chen.__version__)"`
- Python version: `python --version`
- OS: [e.g. Ubuntu 22.04, macOS 14.2, Windows 11]
- Backend: [mock / hf / vllm / llama_cpp]
- If using the HF backend:
  - `transformers` version:
  - `torch` version:
  - CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- GPU model (if applicable):

## Additional context

<!-- Anything else that would help diagnose the issue. -->
