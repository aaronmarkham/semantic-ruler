"""Run-to-run variance of the ruler.

Generates K fresh anchor sets per generator model (same prompt, new draws) and one
extra Sonnet test set, each in its own shard store under data/variance/, then
scores every ruler on every test set. Resumable: existing stores are reused.
"""
import asyncio
import itertools
import json
import sys

import numpy as np
from spiritwriter.fabric.store import ShardStore
from spiritwriter.llm import AnthropicProvider
from spiritwriter.secrets import configure

from semantic_ruler import store as shards
from semantic_ruler.concepts import SUSTAINABILITY as C
from semantic_ruler.embed import KEYCHAIN_SERVICE, CachedEmbedder, OpenAIEmbedder
from semantic_ruler.generate import generate_level_texts, generate_pairs
from semantic_ruler.ruler import evaluate, fit_ruler

root = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
MODELS = {"sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}
LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
configure(service_name=KEYCHAIN_SERVICE)
emb = CachedEmbedder(OpenAIEmbedder(), f"{root}/.cache/embeddings")
sem = asyncio.Semaphore(4)
failures = []


async def limited(coro, label):
    async with sem:
        try:
            return await coro
        except Exception as e:  # keep going; report at the end
            failures.append(f"{label}: {e}")
            return None


async def anchors_job(name, k, block):
    st = ShardStore(f"{root}/data/variance/{name}-anchors-{k}")
    if shards.load_anchors(st, C, block):
        return
    llm = AnthropicProvider(model=MODELS[name])
    pairs = await limited(generate_pairs(llm, C, block, 8, MODELS[name]), f"{name}-anchors-{k}/{block.name}")
    if pairs:
        shards.save_anchors(st, C, block, pairs, generator=MODELS[name])


async def levels_job(name, k, block):
    st = ShardStore(f"{root}/data/variance/{name}-test-{k}")
    if shards.load_levels(st, C, block):
        return
    llm = AnthropicProvider(model=MODELS[name])
    res = await asyncio.gather(*[
        limited(generate_level_texts(llm, C, block, lam, 3, MODELS[name]), f"{name}-test-{k}/{block.name}/{lam}")
        for lam in LEVELS])
    if all(res):
        shards.save_levels(st, C, block, dict(zip(LEVELS, res)), generator=MODELS[name])


async def generate():
    jobs = [anchors_job(n, k, b) for n in MODELS for k in range(1, K + 1) for b in C.blocks]
    jobs += [levels_job("sonnet", 1, b) for b in C.blocks]
    await asyncio.gather(*jobs)


asyncio.run(generate())
if failures:
    print("FAILURES:\n  " + "\n  ".join(failures))


def load_anchor_emb(path):
    st = ShardStore(path)
    neg, pos = {}, {}
    for b in C.blocks:
        found = shards.load_anchors(st, C, b)
        if not found:
            return None
        e = emb.embed([t for p in found[1] for t in p])
        neg[b.name], pos[b.name] = e[0::2], e[1::2]
    return neg, pos


def load_test(path):
    st = ShardStore(path)
    texts, lam = [], []
    for b in C.blocks:
        found = shards.load_levels(st, C, b)
        if not found:
            return None
        for l, ts in found[1].items():
            texts += ts; lam += [l] * len(ts)
    return emb.embed(texts), np.array(lam)


anchor_paths = {"sonnet": [f"{root}/data/store"], "haiku": [f"{root}/data/store-haiku"]}
for n in MODELS:
    anchor_paths[n] += [f"{root}/data/variance/{n}-anchors-{k}" for k in range(1, K + 1)]
anchor_sets = {n: [a for a in map(load_anchor_emb, ps) if a] for n, ps in anchor_paths.items()}
rulers = {n: [fit_ruler(C.name, neg, pos, emb.model) for neg, pos in sets] for n, sets in anchor_sets.items()}
tests = {"sonnet test A": load_test(f"{root}/data/store"), "sonnet test B": load_test(f"{root}/data/variance/sonnet-test-1"),
         "haiku test": load_test(f"{root}/data/store-haiku")}
tests = {k: v for k, v in tests.items() if v}
print(f"rulers: { {n: len(r) for n, r in rulers.items()} }, test sets: {list(tests)}, embedding cache misses: {emb.misses}")

out = {}
print("\n1. Pearson r of each ruler (rows: anchor draws; first row is the original run)")
for n, rs in rulers.items():
    print(f"  {n} anchors      " + "  ".join(f"{t:>14}" for t in tests))
    mat = np.array([[evaluate(lam, r.score(T))["pearson_r"] for T, lam in tests.values()] for r in rs])
    rm = np.array([[evaluate(lam, r.score(T))["rmse"] for T, lam in tests.values()] for r in rs])
    for i, row in enumerate(mat):
        print(f"    draw {i}          " + "  ".join(f"{x:>14.3f}" for x in row))
    print("    mean            " + "  ".join(f"{x:>14.3f}" for x in mat.mean(axis=0)))
    print("    sd              " + "  ".join(f"{x:>14.3f}" for x in mat.std(axis=0, ddof=1)))
    print("    min / max       " + "  ".join(f"{a:>6.3f}/{b:<7.3f}" for a, b in zip(mat.min(axis=0), mat.max(axis=0))))
    print("    mean RMSE       " + "  ".join(f"{x:>14.3f}" for x in rm.mean(axis=0)))
    out[n] = {"tests": list(tests), "r": mat.tolist(), "rmse": rm.tolist()}

print("\n2. cosine between rulers")
for n, rs in rulers.items():
    cs = [a.vector @ b.vector for a, b in itertools.combinations(rs, 2)]
    print(f"  within {n}: mean {np.mean(cs):.3f} (min {np.min(cs):.3f}, max {np.max(cs):.3f})")
cs = [a.vector @ b.vector for a in rulers["sonnet"] for b in rulers["haiku"]]
print(f"  sonnet vs haiku: mean {np.mean(cs):.3f} (min {np.min(cs):.3f}, max {np.max(cs):.3f})")

print("\n3. pooled rulers (all draws of a model together, 8 x draws pairs per block)")
for n, sets in anchor_sets.items():
    neg = {b.name: np.vstack([s[0][b.name] for s in sets]) for b in C.blocks}
    pos = {b.name: np.vstack([s[1][b.name] for s in sets]) for b in C.blocks}
    r = fit_ruler(C.name, neg, pos, emb.model)
    print(f"  {n} pooled ({len(sets)} draws): " + "  ".join(
        f"{t}: r={evaluate(lam, r.score(T))['pearson_r']:.3f}" for t, (T, lam) in tests.items()))
neg = {b.name: np.vstack([s[0][b.name] for sets in anchor_sets.values() for s in sets]) for b in C.blocks}
pos = {b.name: np.vstack([s[1][b.name] for sets in anchor_sets.values() for s in sets]) for b in C.blocks}
r = fit_ruler(C.name, neg, pos, emb.model)
print("  both models pooled: " + "  ".join(f"{t}: r={evaluate(lam, r.score(T))['pearson_r']:.3f}" for t, (T, lam) in tests.items()))

print("\n4. test-set sampling: bootstrap 95% interval of r for the original Sonnet ruler (2000 resamples of texts)")
rng = np.random.default_rng(0)
for t, (T, lam) in tests.items():
    p = rulers["sonnet"][0].score(T)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(lam), len(lam))
        bs.append(np.corrcoef(lam[i], p[i])[0, 1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  {t}: r={np.corrcoef(lam, p)[0,1]:.3f}  [{lo:.3f}, {hi:.3f}]")

json.dump(out, open(f"{root}/out/variance.json", "w"), indent=1)
