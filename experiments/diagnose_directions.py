"""Follow-up: is a better linear direction available, and what confounds the anchor one?"""
import json
import sys

import numpy as np
from spiritwriter.fabric.store import ShardStore

from semantic_ruler import store as shards
from semantic_ruler.concepts import SUSTAINABILITY as C
from semantic_ruler.embed import CachedEmbedder, OpenAIEmbedder
from semantic_ruler.ruler import _unit, concept_vector, evaluate, fit_direction

root = sys.argv[1]
store = ShardStore(f"{root}/data/store")
emb = CachedEmbedder(OpenAIEmbedder(), f"{root}/.cache/embeddings")
ruler = shards.load_ruler(store, C, emb.model)
v = ruler.vector
exp = json.load(open(f"{root}/out/experiments.json", encoding="utf-8"))
J, K = exp["judged"], exp["continuum"]
names = [b.name for b in C.blocks]

T = emb.embed([r["text"] for r in J]); lam = np.array([r["lambda"] for r in J])
blk = np.array([r["block"] for r in J]); judge = np.array([r["judge"] for r in J])
p = ruler.score(T)

pairs = {b.name: shards.load_anchors(store, C, b)[1] for b in C.blocks}
neg = {b: emb.embed([n for n, _ in pairs[b]]) for b in names}
pos = {b: emb.embed([q for _, q in pairs[b]]) for b in names}
anchors = np.vstack([*neg.values(), *pos.values()])
wc = lambda t: len(t.split())


def show(name, target, score):
    m = evaluate(np.asarray(target), np.asarray(score))
    print(f"  {name:<58} r={m['pearson_r']:.3f} rho={m['spearman_rho']:.3f} RMSE={m['rmse']:.3f}")


print("1. length confound")
nl = [wc(n) for b in names for n, _ in pairs[b]]; pl = [wc(q) for b in names for _, q in pairs[b]]
print(f"  anchor words: neg mean {np.mean(nl):.0f}, pos mean {np.mean(pl):.0f}")
tl = np.array([wc(r["text"]) for r in J])
print("  level-text words by level:", {l: int(tl[lam == l].mean()) for l in np.unique(lam)})
print("  level-text words by block:", {b: int(tl[blk == b].mean()) for b in names})
res = p - judge
print(f"  corr(ruler - judge, word count) = {np.corrcoef(res, tl)[0,1]:.3f}")
print("  mean (ruler - judge) by block:", {b: round(float(res[blk == b].mean()), 3) for b in names})

print("\n2. supervised upper bound: ridge on level-text embeddings, leave-one-block-out")
for alpha in (0.01, 0.1, 1.0):
    pred = np.empty_like(lam)
    for b in names:
        tr, te = blk != b, blk == b
        mu, ym = T[tr].mean(axis=0), lam[tr].mean()
        X = T[tr] - mu
        a = np.linalg.solve(X @ X.T + alpha * np.eye(tr.sum()), lam[tr] - ym)
        pred[te] = (T[te] - mu) @ (X.T @ a) + ym
    show(f"ridge alpha={alpha} (trained on 3 blocks' level texts)", lam, np.clip(pred, 0, 1))
mu = T.mean(axis=0); X = T - mu
w = X.T @ np.linalg.solve(X @ X.T + 0.1 * np.eye(len(lam)), lam - lam.mean())
print(f"  cos(ridge direction fit on all 72, v*) = {_unit(w) @ v:.3f}")

print("\n3. graded-anchor rulers built from the matched-continuum set, tested on the independent 72")
KE = emb.embed([r["text"] for r in K]); kl = np.array([r["lambda"] for r in K])
kb = np.array([r["block"] for r in K]); ks = np.array([r["scenario"] for r in K])
slopes, ends = {}, {}
for b in names:
    S, E = [], []
    for s in np.unique(ks[kb == b]):
        m = (kb == b) & (ks == s)
        x = kl[m] - kl[m].mean()
        S.append((x[:, None] * KE[m]).sum(axis=0) / (x**2).sum())  # per-dimension regression slope
        E.append(KE[m][np.argmax(kl[m])] - KE[m][np.argmin(kl[m])])
    slopes[b], ends[b] = np.vstack(S), np.vstack(E)
for name, d in (("continuum endpoints only (3 pairs/block)", ends), ("continuum regression slope (6 levels x 3 scenarios)", slopes)):
    vv = concept_vector(d)
    print(f"  cos(v*, this) = {v @ vv:.3f}")
    show(f"{name}, calibrated on original anchors", lam, fit_direction(C.name, vv, anchors, emb.model).score(T))
    show(f"{name}, calibrated on continuum texts", lam, fit_direction(C.name, vv, KE, emb.model).score(T))
# combine: original deltas + slopes
both = {b: np.vstack([pos[b] - neg[b], slopes[b]]) for b in names}
show("original 8 pairs + continuum slopes", lam, fit_direction(C.name, concept_vector(both), anchors, emb.model).score(T))

print("\n4. reverse check: original ruler v* on continuum texts, calibrated on continuum's own range")
kp = fit_direction(C.name, v, KE, emb.model).score(KE)
show("v* on continuum, quantile-calibrated on continuum (in-sample bounds)", kl, kp)

print("\n5. ruler vs judge agreement by block (original ruler)")
for b in names:
    m = blk == b
    print(f"  {b:<24} ruler~lambda r={np.corrcoef(lam[m], p[m])[0,1]:.3f}  judge~lambda r={np.corrcoef(lam[m], judge[m])[0,1]:.3f}  ruler~judge r={np.corrcoef(judge[m], p[m])[0,1]:.3f}")
