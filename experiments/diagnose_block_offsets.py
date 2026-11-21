"""Diagnose the r = 0.853 gap using only stored shards + cached embeddings (no LLM calls)."""
import itertools
import sys

import numpy as np
from spiritwriter.fabric.store import ShardStore

from semantic_ruler import store as shards
from semantic_ruler.concepts import SUSTAINABILITY as C
from semantic_ruler.embed import CachedEmbedder, OpenAIEmbedder
from semantic_ruler.ruler import (
    _unit, block_coherence, concept_vector, evaluate, fit_direction, fit_ruler, paired_deltas,
)

root = sys.argv[1]
store = ShardStore(f"{root}/data/store")
emb = CachedEmbedder(OpenAIEmbedder(), f"{root}/.cache/embeddings")
np.set_printoptions(precision=3, suppress=True)

neg, pos, lv = {}, {}, {}
for b in C.blocks:
    _, pairs = shards.load_anchors(store, C, b)
    e = emb.embed([t for p in pairs for t in p])
    neg[b.name], pos[b.name] = e[0::2], e[1::2]
    _, levels = shards.load_levels(store, C, b)
    lv[b.name] = {lam: emb.embed(texts) for lam, texts in levels.items()}
print("cache hits/misses:", emb.hits, emb.misses)
names = list(neg)
deltas = paired_deltas(neg, pos)
ruler = fit_ruler(C.name, neg, pos, emb.model)
v = ruler.vector

T = np.vstack([lv[b][lam] for b in names for lam in sorted(lv[b])])
lam = np.array([l for b in names for l in sorted(lv[b]) for _ in lv[b][l]])
blk = np.array([b for b in names for l in sorted(lv[b]) for _ in lv[b][l]])

print("\n1. cosine between block concept vectors v_b")
vb = {b: _unit(d.mean(axis=0)) for b, d in deltas.items()}
for a, b in itertools.combinations(names, 2):
    print(f"   {a:<24} {b:<24} {vb[a] @ vb[b]:.3f}")
print("   cos(v_b, v*):", {b: round(float(vb[b] @ v), 3) for b in names})
print("   mean |delta| per block:", {b: round(float(np.linalg.norm(d, axis=1).mean()), 3) for b, d in deltas.items()})

print("\n2. coherence definitions")
print("   delta -> block mean delta (mine):", {b: round(c, 3) for b, c in block_coherence(deltas).items()})
for b in names:
    A = np.vstack([neg[b], pos[b]])
    cen = _unit(A.mean(axis=0))
    cos = (A @ cen) / np.linalg.norm(A, axis=1)
    print(f"   anchor embedding -> block centroid  {b:<24} median {np.median(cos):.3f}")
for b in names:
    for lab, A in (("neg", neg[b]), ("pos", pos[b])):
        cen = _unit(A.mean(axis=0)); cos = (A @ cen) / np.linalg.norm(A, axis=1)
        print(f"   {lab} embeddings -> {lab} centroid       {b:<24} median {np.median(cos):.3f}")

print("\n3. raw projection s = E.v* : anchors and level texts, by block")
print(f"   {'block':<24} {'neg':>7} {'pos':>7} | " + " ".join(f"l={l:.1f}" for l in sorted(lv[names[0]])))
for b in names:
    row = " ".join(f"{(lv[b][l] @ v).mean():>5.3f}" for l in sorted(lv[b]))
    print(f"   {b:<24} {(neg[b] @ v).mean():>7.3f} {(pos[b] @ v).mean():>7.3f} | {row}")
print(f"   global calibration: [{ruler.q_low:.3f}, {ruler.q_high:.3f}]")

print("\n4. per-block linear fit of p on lambda (p = a + b*lambda)")
p = ruler.score(T)
for b in names:
    s = blk == b
    slope, icpt = np.polyfit(lam[s], p[s], 1)
    print(f"   {b:<24} a={icpt:+.3f} b={slope:.3f} r={np.corrcoef(lam[s], p[s])[0,1]:.3f}")

print("\n5. where is the error? (raw scores, no clipping)")
s_raw = T @ v
print(f"   pooled raw r                       {np.corrcoef(lam, s_raw)[0,1]:.3f}")
s_c = s_raw.copy()
for b in names:
    s_c[blk == b] -= s_raw[blk == b].mean()
print(f"   after removing each block's mean   {np.corrcoef(lam, s_c)[0,1]:.3f}")
cells = [(b, l) for b in names for l in sorted(lv[b])]
cm = np.array([(lv[b][l] @ v).mean() for b, l in cells]); cl = np.array([l for _, l in cells])
print(f"   cell means (24 block x level)      {np.corrcoef(cl, cm)[0,1]:.3f}")
within = np.mean([np.var(lv[b][l] @ v) for b, l in cells])
print(f"   variance: total {s_raw.var():.5f}, within-cell (3 variants) {within:.5f}")


def variant(name, scores):
    m = evaluate(lam, scores)
    print(f"   {name:<46} r={m['pearson_r']:.3f} rho={m['spearman_rho']:.3f} RMSE={m['rmse']:.3f} MAE={m['mae']:.3f}")


print("\n6. variants (all calibrated on anchors only)")
variant("baseline (global calibration)", p)

pb = np.empty_like(p)
for b in names:
    r_b = fit_direction(C.name, v, np.vstack([neg[b], pos[b]]), emb.model)
    pb[blk == b] = r_b.score(T[blk == b])
variant("per-block calibration (block known at scoring)", pb)

# project the between-block subspace out of v
cents = np.vstack([np.vstack([neg[b], pos[b]]).mean(axis=0) for b in names])
Bc = cents - cents.mean(axis=0)
q, _ = np.linalg.qr(Bc.T)  # orthonormal basis of block-centroid subspace
q = q[:, : len(names) - 1]
v_orth = _unit(v - q @ (q.T @ v))
print(f"   (fraction of v* inside block-centroid subspace: {np.linalg.norm(q.T @ v):.3f}; cos(v*, v_orth)={v @ v_orth:.3f})")
anchors = np.vstack([*neg.values(), *pos.values()])
variant("v* with block-centroid subspace projected out", fit_direction(C.name, v_orth, anchors, emb.model).score(T))

# also project that subspace out of the embeddings being scored (equivalent to above for dot product) - skip
# median-based calibration bounds: neg median -> 0, pos median -> 1
allneg, allpos = np.vstack(list(neg.values())), np.vstack(list(pos.values()))
lo, hi = np.median(allneg @ v), np.median(allpos @ v)
variant("calibrate on neg/pos medians instead of Q05/Q95", np.clip((T @ v - lo) / (hi - lo), 0, 1))
lo, hi = (allneg @ v).mean(), (allpos @ v).mean()
variant("calibrate on neg/pos means", np.clip((T @ v - lo) / (hi - lo), 0, 1))

# first principal component of deltas instead of mean
allD = np.vstack(list(deltas.values()))
_, _, vt = np.linalg.svd(allD, full_matrices=False)
pc = vt[0] * np.sign(vt[0] @ v)
variant("first singular vector of all deltas", fit_direction(C.name, pc, anchors, emb.model).score(T))

# logistic-regression-free "whitened" direction: LDA-ish with shrinkage on delta covariance
D = allD - allD.mean(axis=0)
for shrink in (1.0, 0.1):
    # Woodbury-free: solve in the span of D
    G = D @ D.T / len(D)
    lam_reg = shrink * np.trace(G) / len(D)
    mu = allD.mean(axis=0)
    # (S + lam I)^-1 mu  with S = D^T D / n
    alpha = np.linalg.solve(G + lam_reg * np.eye(len(D)), D @ mu / len(D))
    w = (mu - D.T @ alpha) / lam_reg
    variant(f"shrinkage-whitened mean delta (shrink={shrink})", fit_direction(C.name, _unit(w), anchors, emb.model).score(T))

print("\n7. leave-one-block-out: vector + calibration from 3 blocks, scored on the 4th")
loo = np.empty_like(p)
for b in names:
    n3 = {k: neg[k] for k in names if k != b}; p3 = {k: pos[k] for k in names if k != b}
    r3 = fit_ruler(C.name, n3, p3, emb.model)
    s = blk == b
    loo[s] = r3.score(T[s])
    m = evaluate(lam[s], loo[s])
    print(f"   held out {b:<24} r={m['pearson_r']:.3f} RMSE={m['rmse']:.3f}  cos(v_3, v_b)={r3.vector @ vb[b]:.3f}")
variant("pooled leave-one-block-out", loo)

print("\n8. learning curve: pairs per block")
rng = np.random.default_rng(0)
for n in (1, 2, 4, 6, 8):
    rs = []
    for _ in range(20 if n < 8 else 1):
        idx = {b: rng.choice(8, n, replace=False) for b in names}
        rr = fit_ruler(C.name, {b: neg[b][idx[b]] for b in names}, {b: pos[b][idx[b]] for b in names}, emb.model)
        rs.append(np.corrcoef(lam, rr.score(T))[0, 1])
    print(f"   n={n}: r mean {np.mean(rs):.3f} (min {np.min(rs):.3f}, max {np.max(rs):.3f})")

print("\n9. do level-0.0 / level-1.0 texts land where the anchors do? (calibrated p)")
for b in names:
    print(f"   {b:<24} anchors neg {ruler.score(neg[b]).mean():.2f} pos {ruler.score(pos[b]).mean():.2f} | level0.0 {ruler.score(lv[b][0.0]).mean():.2f} level1.0 {ruler.score(lv[b][1.0]).mean():.2f}")
