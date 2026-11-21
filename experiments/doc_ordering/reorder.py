"""Reorder prerequisites into a front-loaded checklist. No LLM: reads the saved extractions.

Two outputs:
1. For the sampled procedure pages: how much a checklist placed before step 1
   would change, i.e. how many surprise prerequisites are detours (the steps
   are not on the page) rather than walk-throughs.
2. For a traced chain (chain.py): one checklist for the whole job, flattened
   from the tree in do-order (a page's own prerequisites before the thing that
   page creates), with duplicates merged and optional items marked.

    python experiments/doc_ordering/reorder.py .
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1]).resolve()
here = root / "experiments" / "doc_ordering"

# Assumed do-order between types on one page: access first, then the things
# policies refer to, then the policies. This is a fixed convention, not learned.
RANK = {t: i for i, t in enumerate([
    "tool_or_access", "account_or_service_setting", "iam_identity_policy", "kms_key", "s3_bucket",
    "other_aws_resource", "iam_role", "bucket_policy", "kms_key_policy"])}
OPTIONAL = re.compile(r"\boptional|\bonly (if|needed|when|required)|\brequired only", re.I)
AUTOMATIC = re.compile(r"(created|added|attached) automatically", re.I)
WHERE = {"on_page": "steps on the page", "other_page": "steps on another page", "none": "no steps given"}


def flags(q) -> str:
    out = [w for w, rx in (("conditional", OPTIONAL), ("may be automatic", AUTOMATIC)) if rx.search(q["what"])]
    return f" [{', '.join(out)}]" if out else ""


# --- 1. per-page: what a checklist before step 1 would change ------------------
# A surprise (as in backpedal.py) is a prerequisite the page mentions only at the
# block where it is needed, later, or never. A checklist removes the surprise but
# not the work, so split surprises by where the steps are: a surprise whose steps
# are on the page is a walk-through, not a detour.
ext = json.load(open(here / "extractions.json", encoding="utf-8"))
pages = [d for d in ext.values() if d.get("is_procedure") and d.get("prerequisites")]
n_pre, clean, kinds, per_page = 0, 0, Counter(), []
for d in pages:
    sur = []
    for q in d["prerequisites"]:
        n_pre += 1
        need, intro = q.get("needed_at_block"), q.get("introduced_at_block")
        if intro is None or (need is not None and intro >= need):
            sur.append(q)
            kinds[q.get("instructions")] += 1
    detours = [q for q in sur if q.get("instructions") != "on_page"]
    clean += not detours
    per_page.append({"id": d["id"], "service": d["service"], "prerequisites": len(d["prerequisites"]), "surprises": len(sur), "detours": len(detours),
                     "checklist": [{"what": q["what"], "type": q["type"], "instructions": q.get("instructions"), "link": q.get("link")}
                                   for q in sorted(d["prerequisites"], key=lambda q: RANK.get(q["type"], 9))]})
n_sur = sum(kinds.values())
n_det = n_sur - kinds["on_page"]
print(f"{len(pages)} procedure pages, {n_pre} prerequisites, {n_sur} surprises ({n_sur / n_pre:.0%})")
print(f"  surprises by where the steps are: on the page {kinds['on_page']}, another page {kinds['other_page']}, nowhere {kinds['none']}")
print(f"  detours (surprise and the steps are not on the page): {n_det} ({n_det / n_pre:.0%} of prerequisites)")
print(f"  pages with no detours, where a checklist changes nothing: {clean} of {len(pages)}")
worst = sorted(per_page, key=lambda p: -p["detours"])[:5]
print("  pages a checklist would change most: " + "; ".join(f"{p['id'][:40]} ({p['detours']} of {p['prerequisites']})" for p in worst))

# --- 2. chain: one checklist for the whole job ---------------------------------
chain = json.load(open(here / "chain_waf.json", encoding="utf-8"))
nodes = chain["tree"]
by_via = {}
for nd in nodes:
    by_via.setdefault((nd["depth"], nd["via"]), nd)


def child_of(parent, q):
    """The page opened for prerequisite q of parent, if chain.py followed it."""
    return by_via.get((parent["depth"] + 1, q["what"]))


def hub_children(parent):
    return [nd for nd in nodes if nd["depth"] == parent["depth"] + 1 and (nd["via"] or "").startswith("the S3 option on this hub page")] \
        if not parent["prerequisites"] else []


items, resolved, merged = [], [], 0


def walk(node, inherited: set, optional: bool):
    """Emit node's prerequisites in do-order. inherited: types the parent page already lists."""
    global merged
    own = sorted(node["prerequisites"], key=lambda q: (RANK.get(q["type"], 9), q.get("needed_at_block") or 0))
    types_here = {q["type"] for q in own}
    for q in own:
        # A child page asking for "an existing bucket" when its parent already lists a
        # bucket is the same item; keep the parent's, which carries the job's constraints.
        if q["type"] in inherited and q.get("instructions") != "other_page":
            merged += 1
            continue
        opt = optional or bool(OPTIONAL.search(q["what"]))
        child = child_of(node, q)
        if child:
            walk(child, types_here, opt)
            if hub_children(child):
                # the page it led to is a list of options; the option taken supplies it
                resolved.append(q["what"])
                continue
        items.append({"what": q["what"], "type": q["type"], "optional": opt, "automatic": bool(AUTOMATIC.search(q["what"])),
                      "instructions": q.get("instructions"), "link": q.get("link") if q.get("instructions") == "other_page" else None,
                      "from_page": node["title"], "from_url": node["url"], "depth": node["depth"], "traced": bool(child)})
    for h in hub_children(node):
        walk(h, types_here, optional)


walk(nodes[0], set(), False)
required = [i for i in items if not i["optional"]]
print(f"\nChain from {chain['start'].rsplit('/', 1)[-1]}: {len(nodes)} pages, {chain['prerequisites']} prerequisites as met by the reader")
print(f"  checklist: {len(items)} items after merging {merged} duplicates; {len(required)} required, {len(items) - len(required)} conditional")
print(f"  {len(resolved)} prerequisite replaced by the S3 option of a hub page")
print("  by type: " + ", ".join(f"{k} {v}" for k, v in Counter(i["type"] for i in required).most_common()))

md = [f"# Before you start: {nodes[0]['title']}", "",
      f"Generated by `reorder.py` from `chain_waf.json` ({len(nodes)} pages). Do the items in order; each one comes after the things it depends on.", "",
      "## Required", ""]
for part, rows in (("req", required), ("opt", [i for i in items if i["optional"]])):
    if part == "opt" and rows:
        md += ["", "## Only in some setups", ""]
    for k, i in enumerate(rows, 1):
        src = f"[{i['from_page'][:70]}]({i['from_url']})"
        steps = f"steps: {i['link']}" if i["link"] else WHERE.get(i["instructions"], "?")
        md.append(f"{k}. **{i['type'].replace('_', ' ')}**: {i['what']}{' (may be created for you)' if i['automatic'] else ''}")
        md.append(f"   - asked for by {src}{', ' + str(i['depth']) + ' pages from the start' if i['depth'] else ''}; {steps}")
md += ["", f"Then follow [{nodes[0]['title']}]({nodes[0]['url']})."]
(here / "checklist_waf.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n" + "\n".join(md))

json.dump({"pages": len(pages), "prerequisites": n_pre, "surprises": n_sur, "surprises_by_instructions": dict(kinds), "detours": n_det,
           "pages_without_detours": clean, "per_page": per_page,
           "chain": {"start": chain["start"], "pages": len(nodes), "items": items, "merged_duplicates": merged, "resolved_by_hub": resolved}},
          open(here / "reorder_results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
