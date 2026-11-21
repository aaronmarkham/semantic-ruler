"""Cross-generator transfer: each model's ruler on the other model's test texts (cached embeddings)."""
import sys

import numpy as np
from spiritwriter.fabric.store import ShardStore

from semantic_ruler import store as shards
from semantic_ruler.concepts import SUSTAINABILITY as C
from semantic_ruler.embed import CachedEmbedder, OpenAIEmbedder
from semantic_ruler.ruler import evaluate

root = sys.argv[1]
emb = CachedEmbedder(OpenAIEmbedder(), f"{root}/.cache/embeddings")
data = {}
for name, d in (("sonnet", "data/store"), ("haiku", "data/store-haiku")):
    st = ShardStore(f"{root}/{d}")
    texts, lam = [], []
    for b in C.blocks:
        for l, ts in shards.load_levels(st, C, b)[1].items():
            texts += ts; lam += [l] * len(ts)
    data[name] = (shards.load_ruler(st, C, emb.model), emb.embed(texts), np.array(lam))
print(f"cos(sonnet ruler, haiku ruler) = {data['sonnet'][0].vector @ data['haiku'][0].vector:.3f}")
for rn in data:
    for tn in data:
        m = evaluate(data[tn][2], data[rn][0].score(data[tn][1]))
        print(f"  {rn:>6} ruler on {tn:>6} texts: r={m['pearson_r']:.3f} RMSE={m['rmse']:.3f}")
print("cache misses:", emb.misses)
