"""The hard case: same service, different facts.

Unit: a doc *section* (heading + prose, code removed) whose own policy JSON
grants S3 access to exactly one kind of principal:

    S  bucket policy names an AWS service principal
    A  bucket policy names an AWS account, role or user (typically cross-account)
    I  identity-based policy on a role or user (no Principal element)
    P  bucket policy allows anyone to read

Questions:
  1. Do raw embeddings put a section nearer to the same service with different
     facts than to another service with the same facts?
  2. Within one service, can a ruler order two sections that differ in facts?

    python experiments/aws_intents/analyze_variants.py .
"""

import asyncio
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from spiritwriter.fabric.store import ShardStore
from spiritwriter.llm import AnthropicProvider
from spiritwriter.secrets import configure

sys.path.insert(0, str(Path(__file__).parent))
from facts import facts  # noqa: E402

from semantic_ruler import store as shards  # noqa: E402
from semantic_ruler.concepts import Block, Concept  # noqa: E402
from semantic_ruler.embed import KEYCHAIN_SERVICE, CachedEmbedder, OpenAIEmbedder  # noqa: E402
from semantic_ruler.generate import generate_pairs, judge_texts  # noqa: E402
from semantic_ruler.ruler import _unit, fit_ruler  # noqa: E402

root = Path(sys.argv[1]).resolve()
here = root / "experiments" / "aws_intents"
MODEL = "claude-sonnet-5-5"
MAX_CHARS, MIN_WORDS, N_PAIRS = 6000, 30, 8
SKIP_JOBS = {"s3-access-logs-permissions"}  # same page as s3-server-access-logs

from concepts_aws import CONTRASTS as _ALL  # noqa: E402

CONTRASTS = {k: _ALL[k] for k in ("S vs I", "A vs I", "S vs A")}

configure(service_name=KEYCHAIN_SERVICE)
emb = CachedEmbedder(OpenAIEmbedder(), root / ".cache" / "embeddings")
store = ShardStore(root / "data" / "aws-intents" / "store")
llm = AnthropicProvider(model=MODEL)
sem = asyncio.Semaphore(4)

# --- sections and their policy-derived label ---------------------------------
rows = []
mixed = 0
for job in json.load(open(here / "corpus.json", encoding="utf-8")):
    if job["job"] in SKIP_JOBS:
        continue
    for i, sec in enumerate(job["sections"]):
        f = facts(sec)
        if not f["n_s3_allow"]:
            continue
        kinds = [k for k, on in (("S", f["bucket_policy_service"]), ("A", f["bucket_policy_aws_principal"]),
                                 ("I", f["identity_policy_s3"]), ("P", f["public_read"])) if on]
        text = (sec["heading"] + "\n" + "\n".join(sec["prose"]))[:MAX_CHARS]
        if len(text.split()) < MIN_WORDS:
            continue
        if len(kinds) != 1:
            mixed += 1
            continue
        rows.append({"job": job["job"], "service": job["service"], "section": i, "heading": sec["heading"],
                     "label": kinds[0], "text": text})
texts = [r["text"] for r in rows]
lab = np.array([r["label"] for r in rows])
svc = np.array([r["service"] for r in rows])
E = emb.embed(texts)
En = E / np.linalg.norm(E, axis=1, keepdims=True)
S = En @ En.T
n = len(rows)
print(f"{n} sections with one kind of principal ({mixed} mixed sections set aside), {len(set(svc))} services")
print("  labels: " + ", ".join(f"{k}={int((lab == k).sum())}" for k in "SAIP"))

# --- 1. does service beat facts in raw similarity? ---------------------------
print("\n1. Raw embedding similarity")
iu = np.triu_indices(n, 1)
ss, sl = svc[iu[0]] == svc[iu[1]], lab[iu[0]] == lab[iu[1]]
for name, m in (("same service, same facts", ss & sl), ("same service, different facts", ss & ~sl),
                ("different service, same facts", ~ss & sl), ("different service, different facts", ~ss & ~sl)):
    print(f"  {name:<36} mean cosine {S[iu][m].mean():.3f}  ({int(m.sum())} pairs)")
wins = total = 0
for i in range(n):
    a = (svc == svc[i]) & (lab != lab[i])  # same service, different facts
    b = (svc != svc[i]) & (lab == lab[i])  # different service, same facts
    if a.any() and b.any():
        total += 1
        wins += S[i][a].max() > S[i][b].max()
print(f"  nearest 'same service, different facts' section is closer than the nearest "
      f"'different service, same facts' one for {wins}/{total} sections ({wins / total:.0%})")


def knn_acc(mask_fn):
    hit = cnt = 0
    for i in range(n):
        m = mask_fn(i)
        m[i] = False
        if not m.any():
            continue
        cnt += 1
        hit += lab[np.argmax(np.where(m, S[i], -np.inf))] == lab[i]
    return hit / cnt, cnt


maj = max((lab == k).mean() for k in "SAIP")
for name, fn in (("any other section", lambda i: np.ones(n, bool)),
                 ("another service only", lambda i: svc != svc[i]),
                 ("same service only", lambda i: svc == svc[i])):
    acc, cnt = knn_acc(fn)
    print(f"  1-NN label accuracy, neighbour from {name:<22} {acc:.2f} (n={cnt}; majority {maj:.2f})")


# --- 2. ordering same-service variants ---------------------------------------
async def limited(coro):
    async with sem:
        return await coro


async def anchors(concept):
    todo = [b for b in concept.blocks if not shards.load_anchors(store, concept, b)]
    fresh = await asyncio.gather(*[limited(generate_pairs(llm, concept, b, N_PAIRS, MODEL)) for b in todo])
    for b, pairs in zip(todo, fresh):
        shards.save_anchors(store, concept, b, pairs, generator=MODEL)
    return {b.name: shards.load_anchors(store, concept, b)[1] for b in concept.blocks}


async def judge(concept, idx):
    """Judge scores for sections idx, cached by text hash under a per-concept ref."""
    slot = Block("variants", "")
    known = shards.load_judgements(store, concept, slot)
    todo = [i for i in idx if shards.text_key(texts[i]) not in known]
    if todo:
        batches = [todo[k:k + 6] for k in range(0, len(todo), 6)]
        res = await asyncio.gather(*[limited(judge_texts(llm, concept, [texts[i] for i in b], MODEL)) for b in batches])
        allscores = {shards.text_key(texts[i]): s for b, r in zip(batches, res) for i, s in zip(b, r)}
        # shard atoms are keyed by text hash; merge with what was there
        merged = {**known, **allscores}
        from spiritwriter.fabric.shard import AtomKind, DecayClass, MemoryShard, ShardAtom
        shard = MemoryShard(
            atoms=[ShardAtom(text=f"Judge score {v:.2f}", kind=AtomKind.FACT, entity=f"judged:{k}", key="score", value=repr(float(v)))
                   for k, v in sorted(merged.items())],
            scope=f"ruler:{concept.slug}", origin=shards.ORIGIN, decay_class=DecayClass.PERMANENT,
            tags=[f"{concept.name} judge scores: variants"], meta={"generator": MODEL})
        store.set_ref(f"{concept.slug}/judge/{slot.slug}", store.put(shard).shard_id)
        known = merged
    return {i: known[shards.text_key(texts[i])] for i in idx}


def pair_stats(score, pos_l, neg_l):
    """Within-service pair ordering accuracy, per service and pooled; plus global AUC."""
    per, right, pairs = {}, 0.0, 0
    for s in sorted(set(svc)):
        p = [i for i in range(n) if svc[i] == s and lab[i] == pos_l and i in score]
        q = [i for i in range(n) if svc[i] == s and lab[i] == neg_l and i in score]
        if p and q:
            c = sum((score[a] > score[b]) + 0.5 * (score[a] == score[b]) for a, b in itertools.product(p, q))
            per[s] = (c / (len(p) * len(q)), len(p) * len(q))
            right += c
            pairs += len(p) * len(q)
    P = [score[i] for i in score if lab[i] == pos_l]
    Q = [score[i] for i in score if lab[i] == neg_l]
    auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in P for b in Q]))
    return per, (np.mean([v[0] for v in per.values()]) if per else float("nan")), (right / pairs if pairs else float("nan")), pairs, auc


async def run_contrast(name, pos_l, neg_l, concept):
    idx = [i for i in range(n) if lab[i] in (pos_l, neg_l)]
    pairs_by_block, jscore = await asyncio.gather(anchors(concept), judge(concept, idx))
    neg = {b: emb.embed([x for x, _ in p]) for b, p in pairs_by_block.items()}
    pos = {b: emb.embed([y for _, y in p]) for b, p in pairs_by_block.items()}
    ruler = fit_ruler(concept.name, neg, pos, emb.model)
    r_all = ruler.raw(E)
    out = {"ruler from generated pairs": {i: float(r_all[i]) for i in idx}, "LLM judge": jscore}

    # natural ruler: paired differences between real sections of other services
    fam = {}
    for s in sorted(set(svc)):
        p = [i for i in idx if svc[i] == s and lab[i] == pos_l]
        q = [i for i in idx if svc[i] == s and lab[i] == neg_l]
        if p and q:
            fam[s] = _unit(E[p].mean(axis=0) - E[q].mean(axis=0))
    nat = {}
    for s in fam:
        others = [v for k, v in fam.items() if k != s]
        if others:
            d = _unit(np.sum(others, axis=0))
            nat.update({i: float(E[i] @ d) for i in idx if svc[i] == s})
    out["ruler from other services' real sections"] = nat
    rng = np.random.default_rng(0)
    rnd = [pair_stats({i: float(E[i] @ _unit(rng.standard_normal(E.shape[1]))) for i in idx}, pos_l, neg_l)[1] for _ in range(100)]
    deltas = list(fam.values())
    dcos = [float(a @ b) for a, b in itertools.combinations(deltas, 2)]
    return name, pos_l, neg_l, out, float(np.mean(rnd)), dcos, {s: float(v @ ruler.vector) for s, v in fam.items()}


async def main():
    return await asyncio.gather(*[run_contrast(k, *v) for k, v in CONTRASTS.items()])


results = {}
print("\n2. Ordering two sections of the same service that differ in facts")
for name, pos_l, neg_l, out, rnd, dcos, align in asyncio.run(main()):
    first = pair_stats(out["LLM judge"], pos_l, neg_l)
    print(f"\n  {name}: {first[3]} same-service pairs across {len(first[0])} services "
          f"({', '.join(f'{s} {v[1]}' for s, v in first[0].items())})")
    print(f"    {'method':<44} {'mean over services':>19} {'pooled pairs':>13} {'all-section AUC':>16}")
    results[name] = {}
    for method, score in out.items():
        per, mean_s, pooled, pairs, auc = pair_stats(score, pos_l, neg_l)
        print(f"    {method:<44} {mean_s:>19.2f} {pooled:>13.2f} {auc:>16.2f}")
        results[name][method] = {"mean_over_services": float(mean_s), "pooled": float(pooled), "auc": auc,
                                 "per_service": {s: v[0] for s, v in per.items()}}
    print(f"    {'random direction (mean of 100)':<44} {rnd:>19.2f}")
    if dcos:
        print(f"    cosine between services' real fact-change vectors: mean {np.mean(dcos):.3f} (min {min(dcos):.3f}, max {max(dcos):.3f}, {len(dcos)} pairs)")
    print("    cosine of each service's real fact-change vector with the generated ruler: "
          + ", ".join(f"{s} {v:.2f}" for s, v in align.items()))
    per = results[name]["ruler from generated pairs"]["per_service"]
    print("    generated ruler by service: " + ", ".join(f"{s} {v:.2f}" for s, v in per.items()))

json.dump({"n_sections": n, "labels": {k: int((lab == k).sum()) for k in "SAIP"}, "contrasts": results},
          open(here / "results_variants.json", "w"), indent=1)
print(f"\nembedding cache hits/misses: {emb.hits}/{emb.misses}")
