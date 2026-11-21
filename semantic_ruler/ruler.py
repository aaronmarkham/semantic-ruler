"""Core maths of a semantic ruler (Abonyi & Tombor, Array 32 (2026) 101249).

Pure numpy, no I/O. Equation numbers refer to the paper.

    delta_{i,b} = E(x+_{i,b}) - E(x-_{i,b})                          (2)
    v_b = mean_i delta_{i,b};  v* = normalize(sum_b v_b / ||v_b||)   (3)
    s(x) = E(x) . v*                                                 (4)
    p(x) = clip((s - Q05) / (Q95 - Q05), 0, 1)                       (5)
    U(p) = p^gamma  or  4p(1-p)                                      (7), (9)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

Blocks = Mapping[str, np.ndarray]  # block name -> (n, d) embedding matrix


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return v / n


@dataclass(frozen=True)
class Ruler:
    """A calibrated concept direction: the measurement instrument."""

    concept: str
    vector: np.ndarray  # unit vector, shape (d,)
    q_low: float  # raw score mapped to p = 0
    q_high: float  # raw score mapped to p = 1
    embedding_model: str
    blocks: tuple[str, ...] = ()

    def raw(self, embeddings: np.ndarray) -> np.ndarray:
        """Eq. 4: projection onto the concept vector."""
        return np.atleast_2d(embeddings) @ self.vector

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Eq. 5: calibrated intensity p in [0, 1]."""
        span = self.q_high - self.q_low
        if abs(span) < 1e-12:
            return np.full(np.atleast_2d(embeddings).shape[0], 0.5)
        return np.clip((self.raw(embeddings) - self.q_low) / span, 0.0, 1.0)


def paired_deltas(neg: Blocks, pos: Blocks) -> dict[str, np.ndarray]:
    """Eq. 2: per-pair difference vectors, grouped by block."""
    if set(neg) != set(pos):
        raise ValueError("neg and pos must have the same blocks")
    out = {}
    for b in neg:
        if neg[b].shape != pos[b].shape:
            raise ValueError(f"block {b!r}: neg/pos shapes differ")
        out[b] = pos[b] - neg[b]
    return out


def concept_vector(deltas: Blocks) -> np.ndarray:
    """Eq. 3: mean within each block, then normalised mean across blocks.

    Normalising each block vector first gives every block an equal vote,
    whatever the magnitude of its shifts.
    """
    return _unit(sum(_unit(d.mean(axis=0)) for d in deltas.values()))


def anchor_cohesion(neg: Blocks, pos: Blocks) -> dict[str, float]:
    """Median cosine of each anchor embedding to its block centroid.

    This appears to be the paper's block diagnostic: it reports 0.86-0.91 (Gemini)
    and 0.80-0.85 (OpenAI) for Case Study I. It measures how tightly a block's
    texts cluster by topic, not how parallel the concept shifts are.
    """
    out = {}
    for b in neg:
        a = np.vstack([neg[b], pos[b]])
        out[b] = float(np.median((a @ _unit(a.mean(axis=0))) / np.linalg.norm(a, axis=1)))
    return out


def block_coherence(deltas: Blocks) -> dict[str, float]:
    """Median cosine of each pair's delta to its block's mean delta.

    A stricter diagnostic than the paper's: how parallel the neg-to-pos shifts are.
    """
    out = {}
    for b, d in deltas.items():
        centre = _unit(d.mean(axis=0))
        norms = np.linalg.norm(d, axis=1)
        out[b] = float(np.median((d @ centre) / np.where(norms == 0, 1.0, norms)))
    return out


def fit_direction(
    concept: str,
    direction: np.ndarray,
    anchor_embeddings: np.ndarray,
    embedding_model: str,
    blocks: tuple[str, ...] = (),
    quantiles: tuple[float, float] = (0.05, 0.95),
) -> Ruler:
    """Calibrate any direction against the anchor score distribution (Eq. 5)."""
    v = _unit(np.asarray(direction, dtype=np.float64))
    raw = anchor_embeddings @ v
    q_low, q_high = (float(q) for q in np.quantile(raw, quantiles))
    return Ruler(concept, v, q_low, q_high, embedding_model, blocks)


def fit_ruler(
    concept: str,
    neg: Blocks,
    pos: Blocks,
    embedding_model: str,
    quantiles: tuple[float, float] = (0.05, 0.95),
) -> Ruler:
    """Build the blocked, paired-difference ruler and calibrate it on its anchors."""
    deltas = paired_deltas(neg, pos)
    anchors = np.vstack([*neg.values(), *pos.values()])
    return fit_direction(
        concept, concept_vector(deltas), anchors, embedding_model, tuple(neg), quantiles
    )


# --- baselines (Section 4.1) -------------------------------------------------


def centroid_direction(neg: Blocks, pos: Blocks) -> np.ndarray:
    """Difference of pooled class centroids, ignoring pairing and blocks."""
    return _unit(
        np.vstack(list(pos.values())).mean(axis=0) - np.vstack(list(neg.values())).mean(axis=0)
    )


def random_direction(dim: int, seed: int = 0) -> np.ndarray:
    return _unit(np.random.default_rng(seed).standard_normal(dim))


# --- utility mappings (Section 3.4) ------------------------------------------


def power_utility(p: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """Eq. 7: monotone preference. gamma > 1 convex, gamma < 1 concave."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    return np.asarray(p, dtype=np.float64) ** gamma


def parabolic_utility(p: np.ndarray) -> np.ndarray:
    """Eq. 9: preference with an internal optimum at p = 0.5."""
    p = np.asarray(p, dtype=np.float64)
    return 4.0 * p * (1.0 - p)


# --- validation metrics (Section 3.5) ----------------------------------------


def _rank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=ranks) / counts)[inv]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(target: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    """Agreement between requested intensity (lambda) and calibrated score p."""
    t = np.asarray(target, dtype=np.float64)
    m = np.asarray(measured, dtype=np.float64)
    r = _pearson(t, m)
    return {
        "n": int(len(t)),
        "pearson_r": r,
        "r2": r * r,
        "spearman_rho": _pearson(_rank(t), _rank(m)),
        "rmse": float(np.sqrt(np.mean((m - t) ** 2))),
        "mae": float(np.mean(np.abs(m - t))),
    }
