"""Same service, different facts: larger corpus, several anchor draws, intervals.

Corpus: the hand-picked pages (corpus.json) plus the systematic crawl
(corpus_crawl.json), de-duplicated by section text. Services that were not in
the hand-picked analysis form the confirmation set.

    python experiments/aws_intents/analyze_confirm.py . [n_draws]
"""

import asyncio
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from spiritwriter.fabric.shard import AtomKind, DecayClass, MemoryShard, ShardAtom
from spiritwriter.llm import AnthropicProvider
from spiritwriter.secrets import configure

sys.path.insert(0, str(Path(__file__).parent))
from concepts_aws import CONTRASTS  # noqa: E402
from facts import facts  # noqa: E402
from make_draws import draw_store  # noqa: E402

from semantic_ruler import store as shards  # noqa: E402
from semantic_ruler.embed import KEYCHAIN_SERVICE, CachedEmbedder, OpenAIEmbedder  # noqa: E402
from semantic_ruler.generate import judge_texts  # noqa: E402
from semantic_ruler.ruler import _unit, fit_ruler  # noqa: E402

root = Path(sys.argv[1]).resolve()
N_DRAWS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
here = root / "experiments" / "aws_intents"
MODEL = "claude-sonnet-5-5"
MAX_CHARS, MIN_WORDS = 6000, 30
SKIP_JOBS = {"s3-access-logs-permissions"}
rng = np.random.default_rng(0)

configure(service_name=KEYCHAIN_SERVICE)
emb = CachedEmbedder(OpenAIEmbedder(), root / ".cache" / "embeddings")
llm = AnthropicProvider(model=MODEL)

# --- sections ----------------------------------------------------------------
rows, seen = [], set()
for source in ("corpus.json", "corpus_crawl.json"):
    for job in json.load(open(here / source, encoding="utf-8")):
        if job["job"] in SKIP_JOBS:
            continue
        for sec in job["sections"]:
            f = facts(sec)
            if not f["n_s3_allow"]:
                continue
            kinds = [k for k, on in (("S", f["bucket_policy_service"]), ("A", f["bucket_policy_aws_principal"]),
                                     ("I", f["identity_policy_s3"]), ("P", f["public_read"])) if on]
            text = (sec["heading"] + "\n" + "\n".join(sec["prose"]))[:MAX_CHARS]
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if len(text.split()) < MIN_WORDS or len(kinds) != 1 or key in seen:
                continue
            seen.add(key)
            rows.append({"service": job["service"], "label": kinds[0], "text": text, "hand_picked": source == "corpus.json"})
texts = [r["text"] for r in rows]
lab = np.array([r["label"] for r in rows])
svc = np.array([r["service"] for r in rows])
old_services = {r["service"] for r in rows if r["hand_picked"]}
n = len(rows)
E = emb.embed(texts)
print(f"{n} sections, {len(set(svc))} services ({len(old_services)} in the hand-picked analysis, "
      f"{len(set(svc) - old_services)} new); labels " + ", ".join(f"{k}={int((lab == k).sum())}" for k in "SAIP"))


def wilson(k, m, z=1.96):
    p = k / m
    c = (p + z * z / (2 * m)) / (1 + z * z / m)
    h = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / (1 + z * z / m)
    return c - h, c + h


# --- 1. raw similarity -------------------------------------------------------
print("\n1. Raw embedding similarity")
En = E / np.linalg.norm(E, axis=1, keepdims=True)
S = En @ En.T
for name, subset in (("all services", np.ones(n, bool)), ("new services only", ~np.isin(svc, list(old_services)))):
    wins = total = 0
    for i in np.flatnonzero(subset):
        a = (svc == svc[i]) & (lab != lab[i])
        b = (svc != svc[i]) & (lab == lab[i])
        if a.any() and b.any():
            total += 1
            wins += S[i][a].max() > S[i][b].max()
    lo, hi = wilson(wins, total)
    print(f"  {name:<18} same-service/different-facts neighbour is the closer one for {wins}/{total} sections "
          f"({wins / total:.0%}, 95% interval {lo:.0%} to {hi:.0%})")
hit = sum(lab[np.argmax(np.where(np.arange(n) != i, S[i], -np.inf))] == lab[i] for i in range(n))
print(f"  1-NN label accuracy {hit / n:.2f} (majority baseline {max((lab == k).mean() for k in 'SAIP'):.2f})")


# --- 2. within-service ordering ----------------------------------------------
def per_service(score, pos_l, neg_l):
    """service -> (share of pos/neg pairs ordered correctly, number of pairs)."""
    out = {}
    for s in sorted(set(svc)):
        p = np.flatnonzero((svc == s) & (lab == pos_l))
        q = np.flatnonzero((svc == s) & (lab == neg_l))
        if len(p) and len(q):
            d = score[p][:, None] - score[q][None, :]
            out[s] = (float(((d > 0) + 0.5 * (d == 0)).mean()), len(p) * len(q))
    return out


def summarise(acc: dict[str, float], pairs: dict[str, int]):
    """Mean over services with a cluster-bootstrap interval, a sign test, and the pooled share."""
    names = list(acc)
    vals = np.array([acc[s] for s in names])
    boot = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    above, below = int((vals > 0.5).sum()), int((vals < 0.5).sum())
    m = above + below
    p_sign = sum(math.comb(m, k) for k in range(above, m + 1)) / 2**m if m else float("nan")
    w = np.array([pairs[s] for s in names])
    return {"services": len(names), "pairs": int(w.sum()), "mean": float(vals.mean()), "lo": float(lo), "hi": float(hi),
            "above": above, "below": below, "p_sign": p_sign, "pooled": float((vals * w).sum() / w.sum())}


def show(label, st):
    print(f"      {label:<26} {st['services']:>3} services, {st['pairs']:>4} pairs: mean {st['mean']:.2f} "
          f"[{st['lo']:.2f}, {st['hi']:.2f}]  above/below chance {st['above']}/{st['below']} (sign test p={st['p_sign']:.4f})  pooled {st['pooled']:.2f}")


def groups(acc, pairs):
    out = {"all services": summarise(acc, pairs)}
    for name, keep in (("new services", lambda s: s not in old_services), ("hand-picked services", lambda s: s in old_services),
                       ("all except S3", lambda s: s != "S3")):
        sub = {s: v for s, v in acc.items() if keep(s)}
        if len(sub) >= 2:
            out[name] = summarise(sub, pairs)
    return out


async def judge_all(concept, idx):
    slot_ref = f"{concept.slug}/judge/variants"
    store = draw_store(root, 0)
    shard = store.resolve_ref(slot_ref)
    known = {a.entity.split(":", 1)[1]: float(a.value) for a in shard.atoms} if shard else {}
    todo = [i for i in idx if shards.text_key(texts[i]) not in known]
    if todo:
        sem = asyncio.Semaphore(4)

        async def one(batch):
            async with sem:
                try:
                    return await judge_texts(llm, concept, [texts[i] for i in batch], MODEL)
                except Exception as e:  # leave these unjudged
                    print(f"      judge batch failed ({e}); {len(batch)} sections left unjudged")
                    return None

        batches = [todo[k:k + 6] for k in range(0, len(todo), 6)]
        for batch, res in zip(batches, await asyncio.gather(*[one(b) for b in batches])):
            if res:
                known.update({shards.text_key(texts[i]): s for i, s in zip(batch, res)})
        new = MemoryShard(
            atoms=[ShardAtom(text=f"Judge score {v:.2f}", kind=AtomKind.FACT, entity=f"judged:{k}", key="score", value=repr(float(v)))
                   for k, v in sorted(known.items())],
            scope=f"ruler:{concept.slug}", origin=shards.ORIGIN, decay_class=DecayClass.PERMANENT,
            tags=[f"{concept.name} judge scores: variants"], meta={"generator": MODEL})
        store.set_ref(slot_ref, store.put(new).shard_id)
    return np.array([known.get(shards.text_key(texts[i]), np.nan) for i in range(n)])


results = {"n_sections": n, "n_services": len(set(svc)), "contrasts": {}}
print("\n2. Ordering two sections of the same service that differ in facts")
for name, (pos_l, neg_l, concept) in CONTRASTS.items():
    idx = [i for i in range(n) if lab[i] in (pos_l, neg_l)]
    pair_services = [s for s in sorted(set(svc)) if ((svc == s) & (lab == pos_l)).any() and ((svc == s) & (lab == neg_l)).any()]
    in_pairs = [i for i in idx if svc[i] in pair_services]
    print(f"\n  {name}")

    # generated rulers, one per anchor draw
    draw_acc = []
    for k in range(N_DRAWS):
        st = draw_store(root, k)
        loaded = {b.name: shards.load_anchors(st, concept, b) for b in concept.blocks}
        if not all(loaded.values()):
            continue
        neg = {b: emb.embed([x for x, _ in v[1]]) for b, v in loaded.items()}
        pos = {b: emb.embed([y for _, y in v[1]]) for b, v in loaded.items()}
        draw_acc.append(per_service(E @ fit_ruler(concept.name, neg, pos, emb.model).vector, pos_l, neg_l))
    pairs = {s: v[1] for s, v in draw_acc[0].items()}
    mean_acc = {s: float(np.mean([d[s][0] for d in draw_acc])) for s in pairs}
    by_draw = [np.mean([d[s][0] for s in pairs]) for d in draw_acc]
    by_draw_new = [np.mean([d[s][0] for s in pairs if s not in old_services]) for d in draw_acc]
    print(f"    ruler from generated pairs ({len(draw_acc)} anchor draws; mean over services per draw: "
          + ", ".join(f"{x:.2f}" for x in by_draw) + f"; sd {np.std(by_draw, ddof=1):.3f})")
    g = groups(mean_acc, pairs)
    for label, st in g.items():
        show(label, st)
    print(f"      new services per draw: " + ", ".join(f"{x:.2f}" for x in by_draw_new))
    results["contrasts"][name] = {"generated": g, "per_draw_mean": [float(x) for x in by_draw], "per_service": mean_acc, "pairs": pairs}

    # natural ruler: other services' real fact-change vectors
    fam = {s: _unit(E[(svc == s) & (lab == pos_l)].mean(axis=0) - E[(svc == s) & (lab == neg_l)].mean(axis=0)) for s in pair_services}
    nat_acc = {}
    for s in pair_services:
        d = _unit(np.sum([v for k2, v in fam.items() if k2 != s], axis=0))
        nat_acc[s] = per_service(E @ d, pos_l, neg_l)[s][0]
    print("    ruler from other services' real sections (leave one service out)")
    g = groups(nat_acc, pairs)
    for label, st in g.items():
        show(label, st)
    results["contrasts"][name]["natural"] = g
    cs = [float(a @ b) for a, b in itertools.combinations(fam.values(), 2)]
    print(f"      cosine between services' fact-change vectors: mean {np.mean(cs):.3f} (5th-95th percentile {np.percentile(cs, 5):.3f} to {np.percentile(cs, 95):.3f})")

    # LLM judge
    j = asyncio.run(judge_all(concept, in_pairs))
    ok = [s for s in pair_services if not np.isnan(j[(svc == s) & np.isin(lab, [pos_l, neg_l])]).any()]
    jacc_full = per_service(np.nan_to_num(j, nan=0.5), pos_l, neg_l)
    jacc = {s: jacc_full[s][0] for s in ok}
    print(f"    LLM judge ({len(ok)} of {len(pair_services)} services fully judged)")
    g = groups(jacc, pairs)
    for label, st in g.items():
        show(label, st)
    results["contrasts"][name]["judge"] = g

    # random directions
    r = [np.mean([v[0] for v in per_service(E @ _unit(rng.standard_normal(E.shape[1])), pos_l, neg_l).values()]) for _ in range(200)]
    print(f"    random direction: mean {np.mean(r):.2f} (2.5th-97.5th percentile {np.percentile(r, 2.5):.2f} to {np.percentile(r, 97.5):.2f})")
    results["contrasts"][name]["random"] = [float(np.mean(r)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]

json.dump(results, open(here / "results_confirm.json", "w"), indent=1)
print(f"\nembedding cache hits/misses: {emb.hits}/{emb.misses}")
