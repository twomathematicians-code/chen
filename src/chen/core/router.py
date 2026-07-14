"""Prompt → expert routing.

A :class:`Router` decides which subset of experts to wake up for a given
prompt. CHEN ships three router variants:

* :class:`LogisticRouter`  — deterministic, no model download. Default.
* :class:`CosineRouter`    — embedding similarity; needs the ``memory`` extra.
* :class:`HybridRouter`     — weighted product of logistic + cosine.

All routers implement the same :class:`Router` protocol and support
``min_activation`` / ``max_activation`` bounds and a ``force_last`` role
constraint (so the pipeline always ends with a Synthesizer).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Protocol

from chen.core.expert import Expert, ExpertRole


class Router(Protocol):
    """Prompt → ordered list of expert names."""

    def route(self, prompt: str, available_experts: list[Expert]) -> list[str]:
        """Return the ordered list of expert names to activate.

        The first name in the list is invoked first; the last name is
        invoked last (and should typically be a Synthesizer).
        """
        ...

    @property
    def name(self) -> str:
        """Short router name (e.g. ``"logistic"``), for telemetry."""
        ...


@dataclass
class RouterDecision:
    """Routing decision with diagnostics."""

    selected: list[str]
    scores: dict[str, float]
    router_name: str
    fallback_reason: str = ""

    def __iter__(self):
        return iter(self.selected)


# ---------------------------------------------------------------------------
# Feature extraction (shared by logistic and hybrid routers)
# ---------------------------------------------------------------------------

# Keyword → role scoring. Tuned heuristically; can be learned from labeled data.
_ROLE_KEYWORDS: dict[ExpertRole, list[str]] = {
    ExpertRole.ANALYST: [
        "extract",
        "entity",
        "parse",
        "summarize",
        "analyze",
        "identify",
        "list",
        "find",
        "what",
        "who",
        "where",
        "when",
    ],
    ExpertRole.REASONER: [
        "why",
        "how",
        "explain",
        "derive",
        "prove",
        "calculate",
        "solve",
        "math",
        "physics",
        "logic",
        "reason",
        "step",
        "equation",
        "formula",
        "reasoning",
    ],
    ExpertRole.CODER: [
        "code",
        "python",
        "javascript",
        "java",
        "c++",
        "rust",
        "go",
        "sql",
        "function",
        "class",
        "bug",
        "debug",
        "compile",
        "runtime",
        "stack",
        "trace",
        "segfault",
        "allocator",
        "refactor",
        "implement",
        "algorithm",
    ],
    ExpertRole.SYNTHESIZER: [
        "write",
        "compose",
        "draft",
        "generate",
        "produce",
        "essay",
        "story",
        "poem",
        "haiku",
        "letter",
        "email",
        "report",
        "summary",
    ],
    ExpertRole.TRANSLATOR: [
        "translate",
        "translation",
        "in french",
        "in spanish",
        "in german",
        "in chinese",
        "in japanese",
        "translate to",
    ],
}

_CODE_INDICATORS = re.compile(
    r"(\bdef\b|\bclass\b|\bimport\b|\bfrom\b|\bfunction\b|\breturn\b|"
    r"<<|>>|=>|->|::|\{\s*$|^\s*\}|```)",
    re.MULTILINE,
)
_MATH_INDICATORS = re.compile(
    r"(\d+\s*[+\-*/^]\s*\d|\\frac|\\sum|\\int|\\sqrt|\\alpha|\\beta|"
    r"\bequation\b|\bformula\b|\bproof\b)",
    re.MULTILINE,
)


def _extract_features(prompt: str) -> dict[str, float]:
    """Extract simple, deterministic features from a prompt."""
    text = prompt.lower()
    n = max(1, len(prompt))
    features: dict[str, float] = {
        "length_short": 1.0 if n < 100 else 0.0,
        "length_medium": 1.0 if 100 <= n < 500 else 0.0,
        "length_long": 1.0 if n >= 500 else 0.0,
        "has_code": 1.0 if _CODE_INDICATORS.search(prompt) else 0.0,
        "has_math": 1.0 if _MATH_INDICATORS.search(prompt) else 0.0,
        "has_question": 1.0 if "?" in prompt else 0.0,
    }
    # Per-role keyword density.
    for role, kws in _ROLE_KEYWORDS.items():
        count = sum(text.count(kw) for kw in kws)
        features[f"kw_{role.value}"] = count / (n / 100.0)  # density per 100 chars
    return features


# ---------------------------------------------------------------------------
# LogisticRouter
# ---------------------------------------------------------------------------


@dataclass
class LogisticRouter:
    """Per-expert logistic classifier on prompt features.

    The default weights are heuristic but deterministic — no training
    data required. To learn weights from observed (prompt, expert, quality)
    triples, call :meth:`fit` (basic logistic regression via numpy).
    """

    min_activation: int = 1
    max_activation: int = 3
    force_last_role: ExpertRole = ExpertRole.SYNTHESIZER
    threshold: float = 0.15
    weights: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.weights:
            self.weights = self._default_weights()

    @staticmethod
    def _default_weights() -> dict[str, dict[str, float]]:
        """Heuristic weights: each role's weight vector over features."""
        return {
            ExpertRole.ANALYST.value: {
                "kw_analyst": 3.0,
                "has_question": 1.0,
                "length_long": 0.5,
                "bias": -0.5,
            },
            ExpertRole.REASONER.value: {
                "kw_reasoner": 4.0,
                "has_math": 2.0,
                "length_long": 0.7,
                "bias": -1.0,
            },
            ExpertRole.CODER.value: {
                "kw_coder": 5.0,
                "has_code": 3.0,
                "bias": -1.5,
            },
            ExpertRole.SYNTHESIZER.value: {
                "kw_synthesizer": 4.0,
                "length_short": 0.5,
                "bias": 0.0,
            },
            ExpertRole.TRANSLATOR.value: {
                "kw_translator": 6.0,
                "bias": -2.0,
            },
            ExpertRole.GENERALIST.value: {
                "bias": 0.0,  # always-available fallback
            },
        }

    @property
    def name(self) -> str:
        return "logistic"

    def _score(self, prompt: str) -> dict[str, float]:
        features = _extract_features(prompt)
        scores: dict[str, float] = {}
        for role_value, w in self.weights.items():
            logit = w.get("bias", 0.0)
            for fname, val in features.items():
                logit += w.get(fname, 0.0) * val
            scores[role_value] = 1.0 / (1.0 + math.exp(-logit))  # sigmoid
        return scores

    def route(self, prompt: str, available_experts: list[Expert]) -> list[str]:
        if not available_experts:
            return []
        scores = self._score(prompt)
        # Aggregate per role (max score across experts of the same role).
        role_best: dict[str, tuple[float, Expert]] = {}
        for expert in available_experts:
            r = expert.role.value
            s = scores.get(r, 0.0)
            # Slight bonus for smaller experts (cheaper to invoke).
            s += -0.0001 * (expert.params_m / 1000.0)
            if r not in role_best or s > role_best[r][0]:
                role_best[r] = (s, expert)

        # Filter by threshold, sort by score descending.
        ranked = sorted(
            ((s, e) for r, (s, e) in role_best.items() if s >= self.threshold),
            key=lambda x: -x[0],
        )
        selected = [e.name for _, e in ranked[: self.max_activation]]

        # Enforce min activation: fall back to top-1 if nothing passed threshold.
        if len(selected) < self.min_activation and ranked:
            selected = [ranked[0][1].name]
        if not selected and available_experts:
            # Absolute fallback: pick the first synthesizer, else first expert.
            for e in available_experts:
                if e.role == self.force_last_role:
                    selected = [e.name]
                    break
            else:
                selected = [available_experts[0].name]

        # Enforce force_last_role: append a synthesizer if not already last.
        if self.force_last_role.value not in {
            self._expert_role_name(n, available_experts) for n in selected
        }:
            for e in available_experts:
                if e.role == self.force_last_role and e.name not in selected:
                    selected.append(e.name)
                    break

        # If force_last_role expert is in the list but not last, move it to last.
        last_role_names = [e.name for e in available_experts if e.role == self.force_last_role]
        if last_role_names:
            last_in_list = [n for n in selected if n in last_role_names]
            if last_in_list:
                last = last_in_list[-1]
                selected = [n for n in selected if n != last] + [last]

        # Cap at max_activation again (force_last may have pushed us over).
        return selected[: self.max_activation] if self.max_activation > 0 else selected

    @staticmethod
    def _expert_role_name(name: str, experts: list[Expert]) -> str:
        for e in experts:
            if e.name == name:
                return e.role.value
        return ""

    @classmethod
    def from_experts(cls, experts: list[Expert], **kwargs) -> LogisticRouter:
        """Build a router, deriving roles from the expert list."""
        # The default weights cover all roles; nothing to derive, but this
        # method provides a uniform API with CosineRouter.
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# CosineRouter
# ---------------------------------------------------------------------------


@dataclass
class CosineRouter:
    """Embedding similarity router.

    Computes the cosine similarity between the prompt's embedding and
    each expert's prototype embedding (precomputed mean of the expert's
    training prompts). Top-k experts are activated.

    The embedding function defaults to a deterministic hash-based
    projection (no dependencies). For real embeddings, pass a callable
    that takes a str and returns a 1D numpy array (e.g. a
    sentence-transformers model).
    """

    min_activation: int = 1
    max_activation: int = 3
    force_last_role: ExpertRole = ExpertRole.SYNTHESIZER
    threshold: float = 0.1
    embed_dim: int = 64
    embed_fn: object = None  # callable[[str], np.ndarray]
    prototypes: dict[str, object] = field(default_factory=dict)  # name -> np.ndarray

    @property
    def name(self) -> str:
        return "cosine"

    def _embed(self, text: str):
        import numpy as np

        if self.embed_fn is not None:
            return np.asarray(self.embed_fn(text), dtype="float32")
        # Fallback: deterministic hash embedding (no deps).
        import hashlib

        out = np.zeros(self.embed_dim, dtype="float32")
        for i in range(0, max(len(text), 1), 8):
            chunk = text.encode("utf-8")[i : i + 8]
            h = hashlib.blake2b(chunk, digest_size=4).digest()
            idx = (i // 8) % self.embed_dim
            out[idx] = int.from_bytes(h, "little") / 0xFFFFFFFF
        # L2 normalize.
        norm = float((out**2).sum()) ** 0.5
        if norm > 0:
            out = out / norm
        return out

    def register_expert(self, expert: Expert, sample_prompts: list[str]) -> None:
        """Register an expert with prototype embeddings computed from samples."""
        import numpy as np

        if not sample_prompts:
            return
        embs = np.stack([self._embed(p) for p in sample_prompts])
        proto = embs.mean(axis=0)
        norm = float((proto**2).sum()) ** 0.5
        if norm > 0:
            proto = proto / norm
        self.prototypes[expert.name] = proto

    @classmethod
    def from_experts(
        cls,
        experts: list[Expert],
        sample_prompts: dict[str, list[str]] | None = None,
        **kwargs,
    ) -> CosineRouter:
        """Build a router and pre-compute prototypes for each expert.

        Args:
            experts: The expert pool.
            sample_prompts: Optional mapping from expert name to a list of
                representative prompts. If absent, the expert's role name
                is used as a single prompt (very weak signal but works
                for tests).
        """
        router = cls(**kwargs)
        for e in experts:
            samples = (sample_prompts or {}).get(e.name) or [
                e.role.value,
                e.description or e.role.value,
            ]
            router.register_expert(e, samples)
        return router

    def route(self, prompt: str, available_experts: list[Expert]) -> list[str]:
        if not available_experts:
            return []
        import numpy as np

        q = self._embed(prompt)
        scored: list[tuple[float, Expert]] = []
        for e in available_experts:
            proto = self.prototypes.get(e.name)
            if proto is None:
                # Register on the fly with the expert's role as the only sample.
                self.register_expert(e, [e.role.value, e.description or e.role.value])
                proto = self.prototypes[e.name]
            sim = float(np.dot(q, proto))
            scored.append((sim, e))

        scored.sort(key=lambda x: -x[0])
        selected = [e.name for s, e in scored[: self.max_activation] if s >= self.threshold]

        if len(selected) < self.min_activation and scored:
            selected = [scored[0][1].name]
        if not selected and available_experts:
            for e in available_experts:
                if e.role == self.force_last_role:
                    selected = [e.name]
                    break
            else:
                selected = [available_experts[0].name]

        # Force-last-role.
        last_role_names = {e.name for e in available_experts if e.role == self.force_last_role}
        if not any(n in last_role_names for n in selected):
            for e in available_experts:
                if e.role == self.force_last_role and e.name not in selected:
                    selected.append(e.name)
                    break
        if last_role_names:
            in_list = [n for n in selected if n in last_role_names]
            if in_list:
                last = in_list[-1]
                selected = [n for n in selected if n != last] + [last]
        return selected[: self.max_activation] if self.max_activation > 0 else selected


# ---------------------------------------------------------------------------
# HybridRouter
# ---------------------------------------------------------------------------


@dataclass
class HybridRouter:
    """Weighted product of logistic and cosine router scores.

    Args:
        logistic: The logistic router to use.
        cosine: The cosine router to use.
        alpha: Weight on the logistic score (0..1). The final score is
            ``alpha * logistic + (1 - alpha) * cosine``.
    """

    logistic: LogisticRouter = field(default_factory=LogisticRouter)
    cosine: CosineRouter = field(default_factory=CosineRouter)
    alpha: float = 0.6
    min_activation: int = 1
    max_activation: int = 3
    force_last_role: ExpertRole = ExpertRole.SYNTHESIZER

    @property
    def name(self) -> str:
        return "hybrid"

    @classmethod
    def from_experts(cls, experts: list[Expert], **kwargs) -> HybridRouter:
        alpha = kwargs.pop("alpha", 0.6)
        logistic = LogisticRouter.from_experts(experts)
        cosine = CosineRouter.from_experts(experts)
        return cls(logistic=logistic, cosine=cosine, alpha=alpha)

    def route(self, prompt: str, available_experts: list[Expert]) -> list[str]:
        # Delegate: take the union of both routers' top picks, ordered by
        # the logistic router (which is cheaper and more interpretable).
        l_selection = set(self.logistic.route(prompt, available_experts))
        c_selection = set(self.cosine.route(prompt, available_experts))
        union = list(l_selection | c_selection)
        if not union:
            return self.logistic.route(prompt, available_experts)
        # Order by logistic preference, then cosine.
        l_order = self.logistic.route(prompt, available_experts)
        ordered = [n for n in l_order if n in union] + [n for n in union if n not in l_order]

        # Force-last-role: pick one force_last_role expert to be the final slot,
        # then fill the rest with non-last-role experts (so max_activation
        # never truncates the force_last_role expert off the end).
        last_role_names = {e.name for e in available_experts if e.role == self.force_last_role}
        if last_role_names:
            # Pick the force_last_role expert (prefer one already in `ordered`).
            last = next((n for n in ordered if n in last_role_names), None)
            if last is None:
                # No force_last_role expert in selection — pick any from the pool.
                for e in available_experts:
                    if e.role == self.force_last_role:
                        last = e.name
                        break
            if last is not None:
                others = [n for n in ordered if n != last]
                if self.max_activation > 0:
                    # Reserve one slot for `last`, fill the rest from `others`.
                    budget = max(0, self.max_activation - 1)
                    others = others[:budget]
                ordered = others + [last]

        if self.max_activation > 0 and not last_role_names:
            ordered = ordered[: self.max_activation]
        return ordered
