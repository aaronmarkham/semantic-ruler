import numpy as np
import pytest

from semantic_ruler.ruler import (
    Ruler,
    block_coherence,
    centroid_direction,
    concept_vector,
    evaluate,
    fit_ruler,
    paired_deltas,
    parabolic_utility,
    power_utility,
)

DIM = 64


def synthetic(n=8, blocks=4, noise=0.05, seed=0):
    """Embeddings = block offset + pair content + lambda * concept direction + noise."""
    rng = np.random.default_rng(seed)
    direction = np.zeros(DIM)
    direction[0] = 1.0
    neg, pos = {}, {}
    for b in range(blocks):
        offset = rng.standard_normal(DIM) * 3.0
        offset[0] = 0.0
        content = rng.standard_normal((n, DIM))
        content[:, 0] = 0.0
        base = offset + content
        neg[f"b{b}"] = base + rng.standard_normal((n, DIM)) * noise
        pos[f"b{b}"] = base + direction * (b + 1.0) + rng.standard_normal((n, DIM)) * noise
    return direction, neg, pos


def test_concept_vector_recovers_planted_direction():
    direction, neg, pos = synthetic()
    v = concept_vector(paired_deltas(neg, pos))
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert v @ direction > 0.99


def test_pairing_cancels_content_that_centroids_keep():
    # Few pairs + large per-pair content: unpaired centroids of *different* samples
    # would keep the content noise; paired deltas cancel it exactly.
    direction, neg, pos = synthetic(n=3, noise=0.0)
    v = concept_vector(paired_deltas(neg, pos))
    assert v @ direction > 0.999999
    # With matched pairs the pooled centroid difference is also clean (paper: parity).
    assert centroid_direction(neg, pos) @ direction > 0.99


def test_blocks_get_equal_votes():
    d1, d2 = np.eye(DIM)[0], np.eye(DIM)[1]
    deltas = {"small": np.tile(d1 * 0.1, (4, 1)), "large": np.tile(d2 * 10.0, (4, 1))}
    v = concept_vector(deltas)
    assert np.isclose(v[0], v[1])


def test_scores_are_calibrated_and_monotone():
    direction, neg, pos = synthetic()
    ruler = fit_ruler("c", neg, pos, "test-model")
    assert ruler.blocks == ("b0", "b1", "b2", "b3")
    lam = np.linspace(-1.0, 6.0, 30)
    p = ruler.score(lam[:, None] * direction[None, :])
    assert p.min() == 0.0 and p.max() == 1.0
    assert np.all(np.diff(p) >= 0)
    assert ruler.score(np.vstack(list(neg.values()))).mean() < 0.2
    assert ruler.score(np.vstack(list(pos.values()))).mean() > 0.4


def test_degenerate_calibration_does_not_divide_by_zero():
    ruler = Ruler("c", np.eye(DIM)[0], 0.3, 0.3, "m")
    assert np.allclose(ruler.score(np.ones((2, DIM))), 0.5)


def test_block_coherence_is_high_for_parallel_shifts():
    _, neg, pos = synthetic(noise=0.01)
    assert all(c > 0.95 for c in block_coherence(paired_deltas(neg, pos)).values())


def test_mismatched_blocks_rejected():
    _, neg, pos = synthetic()
    pos.pop("b0")
    with pytest.raises(ValueError):
        paired_deltas(neg, pos)


def test_evaluate():
    t = np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
    perfect = evaluate(t, t)
    assert perfect["pearson_r"] == pytest.approx(1.0)
    assert perfect["spearman_rho"] == pytest.approx(1.0)
    assert perfect["rmse"] == 0.0
    off = evaluate(t, 1.0 - t)
    assert off["pearson_r"] == pytest.approx(-1.0)
    assert off["mae"] == pytest.approx(np.mean(np.abs(1 - 2 * t)))
    assert np.isnan(evaluate(t, np.full(6, 0.5))["pearson_r"])


def test_utilities():
    assert power_utility(np.array([0.4]), 2.0)[0] == pytest.approx(0.16)  # paper's example
    assert parabolic_utility(np.array([0.0, 0.5, 1.0])).tolist() == [0.0, 1.0, 0.0]
    with pytest.raises(ValueError):
        power_utility(np.array([0.5]), 0.0)
