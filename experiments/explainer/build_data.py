"""Collect the numbers the interactive explainer needs into one JSON file.

Embeddings are too large to ship, so the ruler section exports projections
instead: each text's dot product with each block's unit direction, plus the
Gram matrix of those directions. That is enough to rebuild the ruler exactly in
the browser for any subset of blocks and any calibration quantiles.

    python experiments/explainer/build_data.py .
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from spiritwriter.fabric.store import ShardStore
from spiritwriter.secrets import configure

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "experiments" / "aws_intents"))
from concepts_aws import CONTRASTS  # noqa: E402
from facts import facts  # noqa: E402

from semantic_ruler import store as shards  # noqa: E402
from semantic_ruler.concepts import SUSTAINABILITY as C  # noqa: E402
from semantic_ruler.embed import KEYCHAIN_SERVICE, CachedEmbedder, OpenAIEmbedder  # noqa: E402
from semantic_ruler.ruler import _unit, fit_ruler  # noqa: E402

configure(service_name=KEYCHAIN_SERVICE)
emb = CachedEmbedder(OpenAIEmbedder(), root / ".cache" / "embeddings")
aws = root / "experiments" / "aws_intents"
ordering = root / "experiments" / "doc_ordering"
out = {}


def r4(a):
    return [round(float(x), 4) for x in a]


def snippet(t: str, n: int = 150) -> str:
    t = " ".join(t.split())
    return t[:n] + ("…" if len(t) > n else "")


# --- 1. the paper's ruler --------------------------------------------------------
blocks = [b.name for b in C.blocks]
sonnet = ShardStore(root / "data" / "store")
levels, judge = [], {}
for bi, b in enumerate(C.blocks):
    judge.update(shards.load_judgements(sonnet, C, b) or {})
    for lam, texts in sorted(shards.load_levels(sonnet, C, b)[1].items()):
        levels += [{"b": bi, "lam": float(lam), "text": t} for t in texts]
E_lv = emb.embed([x["text"] for x in levels])
ruler = {"blocks": blocks, "concept": C.name, "neg": C.neg_label, "pos": C.pos_label, "generators": {},
         "levels": [{"b": x["b"], "lam": x["lam"], "judge": judge.get(shards.text_key(x["text"])), "text": snippet(x["text"])} for x in levels]}
for gen, path in (("Sonnet", "store"), ("Haiku", "store-haiku")):
    st = ShardStore(root / "data" / path)
    V, anchors = [], []
    for bi, b in enumerate(C.blocks):
        pairs = shards.load_anchors(st, C, b)[1]
        neg, pos = emb.embed([n for n, _ in pairs]), emb.embed([p for _, p in pairs])
        V.append(_unit((pos - neg).mean(axis=0)))
        anchors.append((bi, neg, pos))
    V = np.array(V)
    A = np.vstack([np.vstack([n, p]) for _, n, p in anchors])
    ruler["generators"][gen] = {
        "gram": [r4(row) for row in V @ V.T],
        "anchors": [r4(row) for row in A @ V.T],          # every anchor text x block direction
        "anchor_block": [bi for bi, n, p in anchors for _ in range(len(n) + len(p))],
        "anchor_pole": [pole for _, n, p in anchors for pole in [0] * len(n) + [1] * len(p)],
        "levels": [r4(row) for row in E_lv @ V.T],
    }
out["ruler"] = ruler
print(f"ruler: {len(levels)} level texts, {len(judge)} judged, generators {list(ruler['generators'])}")

# --- 2. same service, different facts ----------------------------------------------
MAX_CHARS, MIN_WORDS = 6000, 30
rows, seen = [], set()
for source in ("corpus.json", "corpus_crawl.json"):
    for job in json.load(open(aws / source, encoding="utf-8")):
        if job["job"] == "s3-access-logs-permissions":
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
            rows.append({"service": job["service"], "label": kinds[0], "text": text, "heading": sec["heading"]})
E = emb.embed([r["text"] for r in rows])
lab = np.array([r["label"] for r in rows])
svc = np.array([r["service"] for r in rows])
En = E / np.linalg.norm(E, axis=1, keepdims=True)
intents = {"n_sections": len(rows), "n_services": len(set(svc)), "contrasts": {}}
draw0 = ShardStore(root / "data" / "aws-intents" / "store")
for name, (pos_l, neg_l, concept) in CONTRASTS.items():
    if name not in ("S vs I", "A vs I"):
        continue
    pair_services = [s for s in sorted(set(svc)) if ((svc == s) & (lab == pos_l)).any() and ((svc == s) & (lab == neg_l)).any()]
    loaded = {b.name: shards.load_anchors(draw0, concept, b)[1] for b in concept.blocks}
    neg = {b: emb.embed([x for x, _ in v]) for b, v in loaded.items()}
    pos = {b: emb.embed([y for _, y in v]) for b, v in loaded.items()}
    gen = E @ fit_ruler(concept.name, neg, pos, emb.model).vector
    fam = {s: _unit(E[(svc == s) & (lab == pos_l)].mean(axis=0) - E[(svc == s) & (lab == neg_l)].mean(axis=0)) for s in pair_services}
    centroid = _unit(E.mean(axis=0))
    pts = []
    for s in pair_services:
        d = _unit(np.sum([v for k, v in fam.items() if k != s], axis=0))
        for i in np.flatnonzero((svc == s) & np.isin(lab, [pos_l, neg_l])):
            pts.append({"s": pair_services.index(s), "pos": int(lab[i] == pos_l), "gen": round(float(gen[i]), 4), "nat": round(float(E[i] @ d), 4),
                        "raw": round(float(En[i] @ centroid), 4), "h": snippet(rows[i]["heading"], 70)})
    intents["contrasts"][name] = {"services": pair_services, "pos": pos_l, "neg": neg_l, "points": pts}
    print(f"intents {name}: {len(pair_services)} services, {len(pts)} sections")
out["intents"] = intents

# --- 3. detours ------------------------------------------------------------------
det = json.load(open(ordering / "detours.json", encoding="utf-8"))["rows"]
TYPES = ["s3_bucket", "bucket_policy", "iam_role", "iam_identity_policy", "kms_key", "kms_key_policy",
         "other_aws_resource", "account_or_service_setting", "tool_or_access"]
INST = ["on_page", "other_page", "none"]
pages, items = [], []
for r in det:
    pages.append({"service": r["service"], "mech": r["mech"], "goal": snippet(r["goal"], 110), "url": r["url"]})
    for it in r["items"]:
        items.append([len(pages) - 1, TYPES.index(it["type"]), -1 if it["needed"] is None else it["needed"],
                      -1 if it["introduced"] is None else it["introduced"], INST.index(it["instructions"]),
                      int("announced in a prerequisites section" in it.get("doubts", [])) + 2 * int("need placed at the top of the page" in it.get("doubts", []))])
out["detours"] = {"types": TYPES, "instructions": INST, "pages": pages, "items": items}
print(f"detours: {len(pages)} pages, {len(items)} prerequisites")

# --- 4. dead ends ------------------------------------------------------------------
de = json.load(open(ordering / "dead_ends.json", encoding="utf-8"))
out["dead_ends"] = {"rows": [{"type": r["type"], "detour": r["detour"], "o": r["outcomes"]["1"], "hop2": r.get("hop2")} for r in de["rows"]],
                    "agreement": 0.98}
print(f"dead ends: {len(de['rows'])} links, hop2 on {sum(bool(r.get('hop2')) for r in de['rows'])}")

# --- 5. the WAF chain and its checklist ------------------------------------------------
chain = json.load(open(ordering / "chain_waf.json", encoding="utf-8"))
out["chain"] = {"start": chain["start"], "tree": [
    {"url": n["url"], "depth": n["depth"], "via": snippet(n["via"] or "", 120), "title": n["title"], "procedure": bool(n["is_procedure"]),
     "pre": [{"what": snippet(q["what"], 130), "type": q["type"], "inst": q.get("instructions"), "link": (q.get("link") or "").split("#")[0] or None}
             for q in sorted(n["prerequisites"], key=lambda q: q.get("needed_at_block") or 0)]} for n in chain["tree"]]}
re_res = json.load(open(ordering / "reorder_results.json", encoding="utf-8"))
out["chain"]["checklist"] = [{"what": snippet(i["what"], 130), "type": i["type"], "optional": i["optional"], "automatic": i["automatic"]} for i in re_res["chain"]["items"]]

# --- 6. the join -----------------------------------------------------------------------
join = json.load(open(ordering / "join_results.json", encoding="utf-8"))
out["join"] = {"rows": [{"x": round(w["mismatch_facts"], 3), "y": round(w["perm_detours"], 3), "mech": w["mech"], "service": w["service"]} for w in join["rows"]],
               "r": join["tests"]["mismatch_facts ~ perm_detours"]}

dest = root / "experiments" / "explainer" / "data.json"
dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"wrote {dest} ({dest.stat().st_size / 1024:.0f} KB); embedding cache hits/misses {emb.hits}/{emb.misses}")

# A self-contained copy for local preview (the published page fetches data.json instead).
page = (root / "experiments" / "explainer" / "explainer.html").read_text(encoding="utf-8")
inline = "<script>window.__DATA__ = " + dest.read_text(encoding="utf-8").replace("</", "<\/") + ";</script>\n"
preview = root / "out" / "explainer-preview.html"
preview.parent.mkdir(exist_ok=True)
preview.write_text(page.replace("<script>\n(async", inline + "<script>\n(async", 1), encoding="utf-8")
print(f"preview: {preview}")
