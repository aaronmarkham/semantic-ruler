"""Do intent rulers separate AWS bucket jobs by how access is granted?

Embedded text: each doc page's bucket-related prose, code blocks removed.
Ground truth: facts.json, derived from the policy JSON only (see facts.py).

    python experiments/aws_intents/analyze.py .
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from spiritwriter.fabric.store import ShardStore
from spiritwriter.secrets import configure

from semantic_ruler import store as shards
from semantic_ruler.concepts import Block, Concept
from semantic_ruler.embed import KEYCHAIN_SERVICE, CachedEmbedder, OpenAIEmbedder
from semantic_ruler.generate import generate_pairs, judge_texts
from semantic_ruler.llm_plain import PlainTextProvider
from semantic_ruler.ruler import _unit, fit_direction, fit_ruler

root = Path(sys.argv[1]).resolve()
here = root / "experiments" / "aws_intents"
MODEL = "claude-sonnet-5-5"
MAX_CHARS = 6000
N_PAIRS = 8

DOC_CONTEXT = (
    "Documentation for a cloud provider's object storage. Use a fictional service name; "
    "do not name any real cloud service other than the storage service and IAM."
)
MECHANISM = Concept(
    name="Bucket access granted by service-principal bucket policy",
    scale_name="reliance on a bucket policy that names a service principal",
    subject="setup guide passage about giving a service access to a storage bucket",
    neg_label=(
        "access is granted only through an IAM role that the service assumes; the role's "
        "identity-based policy lists the bucket permissions and the bucket policy is not changed"
    ),
    pos_label=(
        "access is granted only through the bucket's resource-based policy, which names the "
        "service's service principal; no IAM role is created"
    ),
    dimensions=(
        "how access is granted (which principal is named and which policy carries the "
        "permissions); the data flow, i.e. what is written to or read from the bucket, "
        "must be identical"
    ),
    context=DOC_CONTEXT,
    blocks=(
        Block("Log delivery", "a service delivers log files into the bucket"),
        Block("Analytics ingestion", "a service reads input data and writes query or job results"),
        Block("Media and document processing", "a service reads source files and writes processed output"),
        Block("Backup and replication", "a service copies objects for durability or disaster recovery"),
    ),
)
EXPOSURE = Concept(
    name="Public exposure of bucket contents",
    scale_name="public exposure",
    subject="setup guide passage about who can read the objects in a storage bucket",
    neg_label=(
        "the bucket is private: public access is blocked and only named principals in the "
        "owner's account can read objects"
    ),
    pos_label=(
        "the bucket's objects are publicly readable by anyone on the internet without "
        "authentication, through a bucket policy that allows everyone to get objects"
    ),
    dimensions="who is allowed to read the objects; the content and purpose must be identical",
    context=DOC_CONTEXT,
    blocks=(
        Block("Website assets", "HTML, CSS, JavaScript and images for a site"),
        Block("Dataset distribution", "data files shared with consumers"),
        Block("Software downloads", "installers and release artifacts"),
        Block("Media library", "audio and video files"),
    ),
)

configure(service_name=KEYCHAIN_SERVICE)
emb = CachedEmbedder(OpenAIEmbedder(), root / ".cache" / "embeddings")
store = ShardStore(root / "data" / "aws-intents" / "store")
# The judge prompts embed fetched doc text, so use the provider that has no tools.
llm = PlainTextProvider(MODEL)

corpus = json.load(open(here / "corpus.json", encoding="utf-8"))
facts = json.load(open(here / "facts.json", encoding="utf-8"))
jobs = [j for j in corpus if facts[j["job"]]["n_s3_allow"] > 0]


def page_text(job) -> str:
    out, n = [], 0
    for para in job["bucket_prose"]:
        if n + len(para) > MAX_CHARS:
            break
        out.append(para)
        n += len(para)
    return "\n".join(out)


texts = [page_text(j) for j in jobs]
names = [j["job"] for j in jobs]
F = [facts[n] for n in names]
y_mech = np.array([f["bucket_policy_service"] for f in F])
y_pub = np.array([f["public_read"] for f in F])
cls = np.array(["public" if f["public_read"] else "service-principal" if f["bucket_policy_service"] else "role" for f in F])
E = emb.embed(texts)
print(f"{len(jobs)} jobs with S3 grants in their policy JSON; classes: " + ", ".join(f"{c}={int((cls == c).sum())}" for c in np.unique(cls)))
print(f"mean prose length {np.mean([len(t) for t in texts]):.0f} chars")


def auc(score, y):
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


print("\n1. Raw embeddings")
En = E / np.linalg.norm(E, axis=1, keepdims=True)
S = En @ En.T
iu = np.triu_indices(len(jobs), 1)
same = cls[iu[0]] == cls[iu[1]]
print(f"  pairwise cosine: all {S[iu].mean():.3f} (min {S[iu].min():.3f}, max {S[iu].max():.3f}); same class {S[iu][same].mean():.3f}; different class {S[iu][~same].mean():.3f}")


def loo_knn(M, k):
    hits = 0
    for i in range(len(M)):
        s = M[i].copy()
        s[i] = -np.inf
        nn = np.argsort(-s)[:k]
        vals, counts = np.unique(cls[nn], return_counts=True)
        hits += vals[np.argmax(counts)] == cls[i]
    return hits / len(M)


maj = max((cls == c).mean() for c in np.unique(cls))
print(f"  leave-one-out class accuracy: 1-NN {loo_knn(S, 1):.2f}, 3-NN {loo_knn(S, 3):.2f} (majority baseline {maj:.2f})")
Ec = E - E.mean(axis=0)
Ecn = Ec / np.linalg.norm(Ec, axis=1, keepdims=True)
Sc = Ecn @ Ecn.T
print(f"  after subtracting the corpus centroid (cosine of residuals): 1-NN {loo_knn(Sc, 1):.2f}, 3-NN {loo_knn(Sc, 3):.2f}; "
      f"same class {Sc[iu][same].mean():.3f}, different class {Sc[iu][~same].mean():.3f}")

# supervised reference: leave-one-out difference of class centroids
sup = np.empty(len(jobs))
for i in range(len(jobs)):
    m = np.ones(len(jobs), bool); m[i] = False
    d = E[m & y_mech].mean(axis=0) - E[m & ~y_mech].mean(axis=0)
    sup[i] = E[i] @ _unit(d)
print(f"  supervised reference (leave-one-out centroid difference on the real docs): mechanism AUC {auc(sup, y_mech):.3f}")


async def anchors(concept):
    out = {}
    todo = [b for b in concept.blocks if not shards.load_anchors(store, concept, b)]
    fresh = await asyncio.gather(*[generate_pairs(llm, concept, b, N_PAIRS, MODEL) for b in todo])
    for b, pairs in zip(todo, fresh):
        shards.save_anchors(store, concept, b, pairs, generator=MODEL)
    for b in concept.blocks:
        out[b.name] = shards.load_anchors(store, concept, b)[1]
    return out


async def judge(concept):
    known = {}
    for b in concept.blocks[:1]:
        known.update(shards.load_judgements(store, concept, b))
    if all(shards.text_key(t) in known for t in texts):
        return np.array([known[shards.text_key(t)] for t in texts])
    batches = [texts[i:i + 6] for i in range(0, len(texts), 6)]
    res = await asyncio.gather(*[judge_texts(llm, concept, b, MODEL) for b in batches])
    scores = [s for r in res for s in r]
    shards.save_judgements(store, concept, concept.blocks[0], dict(zip(texts, scores)), generator=MODEL)
    return np.array(scores)


async def main():
    return await asyncio.gather(anchors(MECHANISM), anchors(EXPOSURE), judge(MECHANISM), judge(EXPOSURE))


a_mech, a_pub, j_mech, j_pub = asyncio.run(main())
results = {"jobs": names, "class": cls.tolist()}
print("\n2. Intent rulers built from generated counterfactual pairs (fictional services), scored on the real doc prose")
for label, concept, pairs_by_block, y, jscore in (
    ("mechanism (service-principal bucket policy vs IAM role)", MECHANISM, a_mech, y_mech, j_mech),
    ("exposure (public read vs private)", EXPOSURE, a_pub, y_pub, j_pub),
):
    neg = {b: emb.embed([n for n, _ in p]) for b, p in pairs_by_block.items()}
    pos = {b: emb.embed([q for _, q in p]) for b, p in pairs_by_block.items()}
    ruler = fit_ruler(concept.name, neg, pos, emb.model)
    shards.save_ruler(store, concept, ruler, [])
    p = ruler.score(E)
    anchors_all = np.vstack([*neg.values(), *pos.values()])
    cos_pos = fit_direction(concept.name, emb.embed([concept.pos_label])[0], anchors_all, emb.model).score(E)
    rnd = np.mean([auc(E @ _unit(np.random.default_rng(s).standard_normal(E.shape[1])), y) for s in range(200)])
    print(f"\n  {label}: {int(y.sum())} positive of {len(y)}")
    print(f"    ruler AUC                      {auc(p, y):.3f}")
    print(f"    cosine to positive-label text  {auc(cos_pos, y):.3f}")
    print(f"    random direction (mean of 200) {rnd:.3f}")
    print(f"    LLM judge reading the prose    {auc(jscore, y):.3f}")
    print(f"    ruler vs judge Pearson r       {np.corrcoef(p, jscore)[0, 1]:.3f}")
    results[label] = {"ruler": p.tolist(), "judge": jscore.tolist(), "truth": y.tolist()}
    order = np.argsort(-p)
    print("    job                               ruler  judge  truth")
    for i in order:
        print(f"    {names[i]:<34} {p[i]:>5.2f}  {jscore[i]:>5.2f}  {'yes' if y[i] else 'no':<3}  {cls[i]}")

json.dump(results, open(here / "results.json", "w"), indent=1)
print(f"\nembedding cache hits/misses: {emb.hits}/{emb.misses}")
