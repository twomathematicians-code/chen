# Pull request template

## Description

<!-- Briefly describe what this PR changes and why. -->

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Test improvement
- [ ] Benchmark / KPI change

## Related issues

<!-- "Closes #123" or "Relates to #456". Leave blank if none. -->

## How was this tested?

- [ ] `make test-fast` passes locally
- [ ] `make lint` passes
- [ ] `make typecheck` passes (or N/A)
- [ ] New tests added for new functionality
- [ ] MockBackend-based tests added (no GPU required)

## Checklist

- [ ] My code follows the style guidelines (ruff, line length 100)
- [ ] I have run `make format` on my code
- [ ] I have added type hints to all new public APIs
- [ ] I have updated the CHANGELOG / README if needed
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or my feature works
- [ ] New and existing unit tests pass locally with `make test-fast`

## Notes for reviewers

<!-- Anything reviewers should pay attention to, or context that would help them review. -->
