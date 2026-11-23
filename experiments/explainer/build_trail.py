"""Collect one record per how-to page for the choose-your-own-detour page.

Joins three result files: detours.json (prerequisites and where each is
mentioned), dead_ends.json (what the linked page turned out to hold, and the
second hop) and stability_all.json (how many of three readings called each
prerequisite a surprise). Standard library only.

    python -I experiments/explainer/build_trail.py .
"""

import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1]).resolve()
ordering = root / "experiments" / "doc_ordering"
here = root / "experiments" / "explainer"

TYPES = ["s3_bucket", "bucket_policy", "iam_role", "iam_identity_policy", "kms_key", "kms_key_policy",
         "other_aws_resource", "account_or_service_setting", "tool_or_access"]
INST = ["on_page", "other_page", "none"]
OUTCOMES = ["arrives", "partial", "passed on", "no steps", "dead link"]


def load(name):
    return json.load(open(ordering / name, encoding="utf-8"))


def snippet(t, n=140):
    t = " ".join((t or "").split())
    return t[:n].rstrip() + ("…" if len(t) > n else "")


detours, dead, stab = load("detours.json"), load("dead_ends.json"), load("stability_all.json")
checked = {(r["page"], r["what"]): r for r in dead["rows"]}
pages, joined, voted, n_items = [], 0, 0, 0
for r in detours["rows"]:
    votes = {i["what"]: i for i in stab["consensus"].get(r["url"], {}).get("items", [])}
    items = []
    for it in sorted(r["items"], key=lambda i: (i["needed"] is None, i["needed"] or 0)):
        n_items += 1
        c = checked.get((r["url"], it["what"]))
        v = votes.get(it["what"])
        joined += c is not None
        voted += v is not None
        items.append({
            "w": snippet(it["what"]), "t": TYPES.index(it["type"]), "k": INST.index(it["instructions"]),
            "n": -1 if it["needed"] is None else it["needed"], "i": -1 if it["introduced"] is None else it["introduced"],
            "d": {"": 0, "sent": 1, "alone": 2}[it["kind"]], "q": len(it.get("doubts", [])),
            "o": OUTCOMES.index(c["outcomes"]["1"]) if c else -1,
            "h": OUTCOMES.index(c["hop2"]) if c and c.get("hop2") else -1,
            "l": (it.get("link") or "") if it["instructions"] == "other_page" else "",
            "v": [v["surprise_votes"], v["runs"]] if v else None,
        })
    pages.append({"u": r["url"], "s": r["service"], "m": r["mech"], "g": snippet(r["goal"], 170), "items": items})

# Deep trails: trees traced by chains.py, where each linked page's own prerequisites are followed too.
chains = []
for start, tree in (load("chains.json") if (ordering / "chains.json").exists() else {}).items():
    index = {n["url"]: k for k, n in enumerate(tree)}
    aliases = {n["redirected_from"]: k for k, n in enumerate(tree) if n.get("redirected_from")}
    child = {(n["parent"], n["via"]): k for k, n in enumerate(tree) if n["parent"]}
    deepest = max(n["depth"] for n in tree)
    nodes = []
    for n in tree:
        items = []
        for q in sorted(n["prerequisites"], key=lambda q: (q["needed_at_block"] is None, q["needed_at_block"] or 0)):
            c, why = child.get((n["url"], q["what"]), -1), 0
            if q["instructions"] == "other_page" and c < 0:   # why the link was not followed
                link = q.get("link")
                why = 4 if not link else 1 if link in index or link in aliases else 2 if "link is not on the page" in q["doubts"] else 3
            items.append({"w": snippet(q["what"]), "t": TYPES.index(q["type"]), "k": INST.index(q["instructions"]),
                          "n": -1 if q["needed_at_block"] is None else q["needed_at_block"],
                          "d": {"": 0, "sent": 1, "alone": 2}[q["detour"]], "c": c, "x": why})
        nodes.append({"u": n["url"], "t": snippet(n["title"], 90), "p": bool(n["is_procedure"]), "r": bool(n.get("redirected_from")),
                      "depth": n["depth"], "items": items})
    svc = next((r["service"] for r in detours["rows"] if r["url"] == start), "WAF")
    chains.append({"s": svc, "g": snippet(tree[0]["goal"] or tree[0]["title"], 170), "nodes": nodes})
    print(f"chain {svc}: {len(nodes)} pages, depth {deepest}, {sum(len(n['items']) for n in nodes)} prerequisites "
          f"({len(nodes[0]['items'])} on the start page)")

out = {"types": TYPES, "outcomes": OUTCOMES, "pages": pages, "chains": chains,
       "runs": [round(stab["per_run"][k]["detour"], 4) for k in stab["runs"]],
       "agreement": 0.98}
dest = here / "trail.json"
dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

dets = [sum(i["d"] > 0 for i in p["items"]) for p in pages]
ends = [sum(i["o"] >= 2 for i in p["items"]) for p in pages]
print(f"{len(pages)} pages, {n_items} prerequisites; {joined} joined to a link check, {voted} to run votes")
print(f"pages with a detour {sum(d > 0 for d in dets)}, with a dead end {sum(e > 0 for e in ends)}")
print("stops per page:", sorted(Counter(d + e for d, e in zip(dets, ends)).items()))
print("off-page without a check:", sum(i["k"] == 1 and i["o"] < 0 for p in pages for i in p["items"]))
print(f"wrote {dest} ({dest.stat().st_size / 1024:.0f} KB)")

page = here / "adventure.html"
if page.exists():
    inline = "<script>window.__DATA__ = " + dest.read_text(encoding="utf-8").replace("</", "<\\/") + ";</script>\n"
    preview = root / "out" / "adventure-preview.html"
    preview.parent.mkdir(exist_ok=True)
    head = '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{margin:0}</style></head><body>\n'
    preview.write_text(head + page.read_text(encoding="utf-8").replace("<script>\n(async", inline + "<script>\n(async", 1) + "</body></html>", encoding="utf-8")
    print(f"preview: {preview}")
