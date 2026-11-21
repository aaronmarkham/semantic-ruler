"""Join the two threads: do ordering surprises concentrate where a page's
permission mechanism differs from that of similar-looking pages?

Per page:
- mechanism: set of S (bucket policy names a service principal), A (bucket policy
  names an account/role/user), I (identity policy), P (public read), from the
  policy JSON only (aws_intents/facts.py).
- neighbours: the K most similar pages from other services, by raw embedding of
  the page prose (code removed). "Looks like the same job."
- mismatch (facts): mean Jaccard distance between the page's mechanism set and
  each neighbour's.
- mismatch (ruler): |ruler score - mean neighbour ruler score|, where the ruler
  is the S-vs-I direction built from other services' real sections.
- outcome: permission prerequisites (bucket policy, IAM role, identity policy,
  KMS key policy) that are detours, i.e. surprises whose steps are not on the
  page; averaged over the extraction runs in stability.py.

    python experiments/doc_ordering/join_intents.py . [k]
"""

import json
import sys
from pathlib import Path

import numpy as np
from spiritwriter.secrets import configure

root = Path(sys.argv[1]).resolve()
K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
here = root / "experiments" / "doc_ordering"
aws = root / "experiments" / "aws_intents"
sys.path.insert(0, str(aws))
from facts import facts  # noqa: E402

from semantic_ruler.embed import KEYCHAIN_SERVICE, CachedEmbedder, OpenAIEmbedder  # noqa: E402
from semantic_ruler.ruler import _unit  # noqa: E402

MAX_CHARS = 6000
PERMISSION = {"bucket_policy", "iam_role", "iam_identity_policy", "kms_key_policy"}
rng = np.random.default_rng(0)
configure(service_name=KEYCHAIN_SERVICE)
emb = CachedEmbedder(OpenAIEmbedder(), root / ".cache" / "embeddings")


def kinds(f) -> set:
    return {k for k, on in (("S", f["bucket_policy_service"]), ("A", f["bucket_policy_aws_principal"]),
                            ("I", f["identity_policy_s3"]), ("P", f["public_read"])) if on}


# --- pages and sections --------------------------------------------------------
pages, sections, seen = [], [], set()
for source in ("corpus.json", "corpus_crawl.json"):
    for job in json.load(open(aws / source, encoding="utf-8")):
        url = job["url"].split("#")[0]
        if url in seen:
            continue
        seen.add(url)
        mech, prose = set(), []
        for sec in job["sections"]:
            f = facts(sec)
            k = kinds(f) if f["n_s3_allow"] else set()
            mech |= k
            text = (sec["heading"] + "\n" + "\n".join(sec["prose"]))[:MAX_CHARS]
            prose.append(text)
            if len(k) == 1 and len(text.split()) >= 30:
                sections.append({"service": job["service"], "label": next(iter(k)), "text": text})
        if mech:
            pages.append({"url": url, "id": job["job"], "service": job["service"], "mech": mech, "text": "\n".join(prose)[:MAX_CHARS]})
print(f"{len(pages)} pages with S3 grants in their policy JSON, {len({p['service'] for p in pages})} services")

E = emb.embed([p["text"] for p in pages])
En = E / np.linalg.norm(E, axis=1, keepdims=True)
S = En @ En.T
svc = np.array([p["service"] for p in pages])
Es = emb.embed([s["text"] for s in sections])
s_svc = np.array([s["service"] for s in sections])
s_lab = np.array([s["label"] for s in sections])


def natural_direction(exclude: str):
    """S-vs-I direction from real sections of services other than `exclude`."""
    deltas = []
    for s in set(s_svc) - {exclude}:
        a, b = Es[(s_svc == s) & (s_lab == "S")], Es[(s_svc == s) & (s_lab == "I")]
        if len(a) and len(b):
            deltas.append(_unit(a.mean(axis=0) - b.mean(axis=0)))
    return _unit(np.mean(deltas, axis=0))


dirs = {s: natural_direction(s) for s in set(svc)}
score_own = np.array([E[i] @ dirs[svc[i]] for i in range(len(pages))])  # page scored without its own service
for i, p in enumerate(pages):
    other = np.flatnonzero(svc != svc[i])
    nn = other[np.argsort(-S[i][other])[:K]]
    p["neighbours"] = [pages[j]["id"] for j in nn]
    p["mismatch_facts"] = float(np.mean([1 - len(p["mech"] & pages[j]["mech"]) / len(p["mech"] | pages[j]["mech"]) for j in nn]))
    v = dirs[svc[i]]
    p["ruler"] = float(score_own[i])
    p["mismatch_ruler"] = float(abs(E[i] @ v - np.mean(E[nn] @ v)))
    p["neighbour_sim"] = float(S[i][nn].mean())

# sanity: does the page-level ruler score track the page's facts at all?
only_s = np.array([p["mech"] == {"S"} for p in pages])
only_i = np.array([p["mech"] == {"I"} for p in pages])
d = score_own[only_s][:, None] - score_own[only_i][None, :]
print(f"check: page-level ruler orders S-only above I-only pages with AUC {((d > 0) + 0.5 * (d == 0)).mean():.2f} "
      f"({int(only_s.sum())} S-only, {int(only_i.sum())} I-only)")

# --- outcome from the extraction runs ------------------------------------------
runs = {"1": json.load(open(here / "extractions.json", encoding="utf-8"))}
rp = here / "extractions_runs.json"
if rp.exists():
    runs.update(json.load(open(rp, encoding="utf-8")))
runs = {k: v for k, v in runs.items() if len(v) >= 0.9 * len(runs["1"])}


def surprise(q) -> bool:
    need, intro = q.get("needed_at_block"), q.get("introduced_at_block")
    return intro is None or (need is not None and intro >= need)


by_url = {p["url"]: p for p in pages}
rows = []
for url in runs["1"]:
    ds = [r[url] for r in runs.values() if url in r]
    p = by_url.get(url)
    if p is None or sum(bool(d.get("is_procedure")) for d in ds) * 2 <= len(ds):
        continue
    perm = [[q for q in d["prerequisites"] if q["type"] in PERMISSION] for d in ds]
    rows.append({**{k: p[k] for k in ("id", "service", "mismatch_facts", "mismatch_ruler", "ruler", "neighbour_sim")}, "mech": "".join(sorted(p["mech"])),
                 "n_mech": len(p["mech"]),
                 "perm_prereqs": float(np.mean([len(x) for x in perm])),
                 "perm_surprises": float(np.mean([sum(surprise(q) for q in x) for x in perm])),
                 "perm_detours": float(np.mean([sum(surprise(q) and q.get("instructions") != "on_page" for q in x) for x in perm])),
                 "perm_off_page": float(np.mean([sum(q.get("instructions") == "other_page" for q in x) for x in perm])),
                 "all_detours": float(np.mean([sum(surprise(q) and q.get("instructions") != "on_page" for q in d["prerequisites"]) for d in ds]))})
print(f"{len(rows)} procedure pages (majority of {len(runs)} extraction runs) joined to mechanism facts")


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    # average ranks for ties
    def avg(v, r):
        out = r.astype(float)
        for u in np.unique(v):
            m = v == u
            out[m] = r[m].mean()
        return out
    rx, ry = avg(np.asarray(x), rx), avg(np.asarray(y), ry)
    return float(np.corrcoef(rx, ry)[0, 1])


def test(x, y, n_perm=10000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = spearman(x, y)
    perm = np.array([spearman(x, rng.permutation(y)) for _ in range(n_perm)])
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 1 and len(set(y[i])) > 1:
            boot.append(spearman(x[i], y[i]))
    return r, float((np.abs(perm) >= abs(r)).mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


results = {"k": K, "pages": len(rows), "runs": sorted(runs), "tests": {}}
print(f"\nSpearman correlation across pages (two-sided permutation p, bootstrap 95% interval); K={K} neighbours")
for xk, xl in (("mismatch_facts", "mechanism differs from neighbours (facts)"), ("mismatch_ruler", "mechanism differs from neighbours (ruler)"),
               ("n_mech", "number of mechanisms on the page"), ("neighbour_sim", "similarity to neighbours (control)")):
    for yk, yl in (("perm_detours", "permission detours"), ("perm_surprises", "permission surprises"), ("perm_off_page", "permission steps off-page"), ("all_detours", "all detours")):
        r, pv, lo, hi = test([w[xk] for w in rows], [w[yk] for w in rows])
        results["tests"][f"{xk} ~ {yk}"] = {"spearman": r, "p": pv, "lo": lo, "hi": hi}
        print(f"  {xl:<42} vs {yl:<26} r = {r:+.2f}  p = {pv:.3f}  [{lo:+.2f}, {hi:+.2f}]")
r, pv, lo, hi = test([w["mismatch_facts"] for w in rows], [w["mismatch_ruler"] for w in rows])
print(f"  facts mismatch vs ruler mismatch: r = {r:+.2f}  p = {pv:.3f}  [{lo:+.2f}, {hi:+.2f}]")
results["tests"]["mismatch_facts ~ mismatch_ruler"] = {"spearman": r, "p": pv, "lo": lo, "hi": hi}

print("\nBy the page's own mechanism (mean per page)")
print("  mechanism  pages  perm prereqs  perm surprises  perm detours  perm off-page")
results["by_mechanism"] = {}
for m in sorted({w["mech"] for w in rows}, key=lambda m: -sum(w["mech"] == m for w in rows)):
    g = [w for w in rows if w["mech"] == m]
    vals = [float(np.mean([w[k] for w in g])) for k in ("perm_prereqs", "perm_surprises", "perm_detours", "perm_off_page")]
    results["by_mechanism"][m] = {"pages": len(g), "perm_prereqs": vals[0], "perm_surprises": vals[1], "perm_detours": vals[2], "perm_off_page": vals[3]}
    print(f"  {m:<9}  {len(g):>5}  {vals[0]:>12.2f}  {vals[1]:>14.2f}  {vals[2]:>12.2f}  {vals[3]:>13.2f}")

print("\nHighest facts mismatch")
for w in sorted(rows, key=lambda w: -w["mismatch_facts"])[:6]:
    print(f"  {w['mismatch_facts']:.2f}  {w['mech']:<3} detours {w['perm_detours']:.1f}  {w['service']}: {w['id'][:60]}")
results["rows"] = rows
json.dump(results, open(here / "join_results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"\nembedding cache hits/misses: {emb.hits}/{emb.misses}")
