"""Two experiments on the gap:
A. blind LLM judge on the existing 72 level texts (paper's LLM-as-judge baseline)
B. matched continuum: each scenario written at all six levels in one call
Raw outputs are saved to <root>/out/experiments.json."""
import asyncio
import json
import random
import sys

import numpy as np
from spiritwriter.fabric.store import ShardStore
from spiritwriter.llm import AnthropicProvider
from spiritwriter.secrets import configure

from semantic_ruler import store as shards
from semantic_ruler.concepts import SUSTAINABILITY as C
from semantic_ruler.embed import KEYCHAIN_SERVICE, CachedEmbedder, OpenAIEmbedder
from semantic_ruler.generate import _query_json
from semantic_ruler.ruler import evaluate

root = sys.argv[1]
MODEL = "claude-sonnet-5-5"
configure(service_name=KEYCHAIN_SERVICE)
llm = AnthropicProvider(model=MODEL)
store = ShardStore(f"{root}/data/store")
emb = CachedEmbedder(OpenAIEmbedder(), f"{root}/.cache/embeddings")
ruler = shards.load_ruler(store, C, emb.model)
LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

JUDGE = """\
You are rating {subject}s for {scale_name}.
Scale: 0.0 = {neg}; 1.0 = {pos}. Use the full continuous range and two decimals.
Rate each numbered description independently, on its own merits.

{items}

Respond with JSON only, in exactly this shape, one entry per description id:
{{"scores": [{{"id": 1, "score": 0.00}}]}}
"""

CONTINUUM = """\
Context: {context} Domain: {block} ({focus}).
Task: Invent {k} different production scenarios (different component and process setup each).
For EACH scenario write six versions of the same {subject}, at {scale_name} levels
lambda = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, where 0.0 = {neg} and 1.0 = {pos}.

Requirements:
- All six versions of a scenario keep the same component, functional purpose and product requirements.
- They differ strictly in {dimensions}, in evenly spaced steps from 0.0 to 1.0.
- Each version is 7-10 sentences, similar length, professional neutral tone, concrete and operational.
- Do not mention lambda, the scale, or any numeric rating in the text.

Respond with JSON only, in exactly this shape:
{{"scenarios": [{{"levels": [{{"lambda": 0.0, "text": "..."}}]}}]}}
"""


async def judge_block(block):
    _, levels = shards.load_levels(store, C, block)
    items = [(lam, t) for lam in sorted(levels) for t in levels[lam]]
    random.Random(block.name).shuffle(items)  # blind to level order
    body = "\n\n".join(f"[{i + 1}] {t}" for i, (_, t) in enumerate(items))
    for attempt in range(3):
        data = await _query_json(llm, JUDGE.format(
            subject=C.subject, scale_name=C.scale_name, neg=C.neg_label, pos=C.pos_label, items=body), MODEL)
        by_id = {int(d["id"]): float(d["score"]) for d in data["scores"]}
        if sorted(by_id) == list(range(1, len(items) + 1)):
            break
    else:
        raise RuntimeError(f"judge ids wrong for {block.name}: {sorted(by_id)}")
    scores = [by_id[i + 1] for i in range(len(items))]
    return [{"block": block.name, "lambda": lam, "text": t, "judge": s} for (lam, t), s in zip(items, scores)]


async def continuum_block(block, k=3):
    data = await _query_json(llm, CONTINUUM.format(
        context=C.context, block=block.name, focus=block.focus, k=k, subject=C.subject,
        scale_name=C.scale_name, neg=C.neg_label, pos=C.pos_label, dimensions=C.dimensions), MODEL)
    rows = []
    for si, sc in enumerate(data["scenarios"][:k]):
        lv = sorted(sc["levels"], key=lambda d: float(d["lambda"]))
        assert [round(float(d["lambda"]), 1) for d in lv] == LEVELS, (block.name, si)
        rows += [{"block": block.name, "scenario": si, "lambda": float(d["lambda"]), "text": d["text"].strip()} for d in lv]
    return rows


async def main():
    res = await asyncio.gather(*[judge_block(b) for b in C.blocks], *[continuum_block(b) for b in C.blocks])
    n = len(C.blocks)
    return [r for part in res[:n] for r in part], [r for part in res[n:] for r in part]


judged, cont = asyncio.run(main())
json.dump({"judged": judged, "continuum": cont}, open(f"{root}/out/experiments.json", "w", encoding="utf-8"), indent=1)


def show(name, lam, score):
    m = evaluate(np.array(lam), np.array(score))
    print(f"  {name:<44} n={m['n']:<3} r={m['pearson_r']:.3f} rho={m['spearman_rho']:.3f} RMSE={m['rmse']:.3f} MAE={m['mae']:.3f}")


print("A. existing independent level texts (72)")
lam = [r["lambda"] for r in judged]
jp = [r["judge"] for r in judged]
rp = ruler.score(emb.embed([r["text"] for r in judged]))
show("ruler vs requested lambda", lam, rp)
show("LLM judge vs requested lambda", lam, jp)
show("ruler vs LLM judge", jp, rp)
for b in C.blocks:
    s = [i for i, r in enumerate(judged) if r["block"] == b.name]
    print(f"    {b.name:<24} judge mean by level: " + " ".join(
        f"{np.mean([jp[i] for i in s if judged[i]['lambda'] == l]):.2f}" for l in LEVELS)
        + "   ruler: " + " ".join(f"{np.mean([rp[i] for i in s if judged[i]['lambda'] == l]):.2f}" for l in LEVELS))

print("\nB. matched continuum texts (same scenario at six levels)")
lam = [r["lambda"] for r in cont]
rp = ruler.score(emb.embed([r["text"] for r in cont]))
show("ruler vs requested lambda", lam, rp)
for b in C.blocks:
    s = [i for i, r in enumerate(cont) if r["block"] == b.name]
    m = evaluate(np.array([lam[i] for i in s]), rp[s])
    print(f"    {b.name:<24} r={m['pearson_r']:.3f} RMSE={m['rmse']:.3f}  mean p by level: " + " ".join(
        f"{np.mean([rp[i] for i in s if cont[i]['lambda'] == l]):.2f}" for l in LEVELS))
per = []
for b in C.blocks:
    for si in range(3):
        s = [i for i, r in enumerate(cont) if r["block"] == b.name and r["scenario"] == si]
        if len(s) == 6:
            per.append(np.corrcoef([lam[i] for i in s], rp[s])[0, 1])
print(f"  per-scenario r (n={len(per)}): mean {np.mean(per):.3f}, min {np.min(per):.3f}")
print("\nembedding cache hits/misses:", emb.hits, emb.misses)
