"""Run the prerequisite extraction again and measure agreement between runs.

Run 1 is extractions.json (the run every published number came from). This
script adds runs 2..N over the same pages into extractions_runs.json, then
compares every pair of runs field by field.

    python experiments/doc_ordering/stability.py . [n_runs]        # extract, then compare
    python experiments/doc_ordering/stability.py . [n_runs] --compare-only
"""

import asyncio
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extract import PROMPT, TYPES  # noqa: E402
from stream import parse, render  # noqa: E402

root = Path(sys.argv[1]).resolve()
N_RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
COMPARE_ONLY = "--compare-only" in sys.argv
MODEL = "claude-sonnet-5-5"
here = root / "experiments" / "doc_ordering"
aws = root / "experiments" / "aws_intents"
cache = root / "data" / "aws-docs-cache"
runs_path = here / "extractions_runs.json"
TOL = 3  # blocks; two runs "agree" on a position if within this many blocks

run1 = json.load(open(here / "extractions.json", encoding="utf-8"))
runs = json.load(open(runs_path, encoding="utf-8")) if runs_path.exists() else {}
runs["1"] = run1


def files_by_url() -> dict:
    out = {}
    for j in json.load(open(aws / "corpus.json", encoding="utf-8")):
        out.setdefault(j["url"].split("#")[0], (cache / f"{j['job']}.html", j["title"]))
    for j in json.load(open(aws / "corpus_crawl.json", encoding="utf-8")):
        out.setdefault(j["url"].split("#")[0], (cache / "crawl" / (re.sub(r"[^A-Za-z0-9]+", "-", j["job"])[:150] + ".html"), j["title"]))
    return out


async def extract_runs():
    from semantic_ruler.generate import _parse_json
    from semantic_ruler.llm_plain import PlainTextProvider

    llm = PlainTextProvider(MODEL)
    files = files_by_url()
    sem = asyncio.Semaphore(4)

    async def one(run: str, url: str):
        if url in runs.setdefault(run, {}):
            return
        path, title = files[url]
        blocks = parse(path.read_text(encoding="utf-8"), url)
        body, n_shown = render(blocks)
        async with sem:
            err = None
            for _ in range(2):
                try:
                    data = _parse_json(await llm.query(PROMPT.format(types=", ".join(TYPES), title=title[:120], body=body)))
                    if isinstance(data.get("prerequisites"), list):
                        runs[run][url] = {"id": run1[url]["id"], "service": run1[url]["service"], "n_blocks": len(blocks), "n_shown": n_shown, **data}
                        return
                except Exception as e:
                    err = e
            print(f"  run {run}: extraction failed for {run1[url]['id']}: {err}")

    await asyncio.gather(*[one(str(r), u) for r in range(2, N_RUNS + 1) for u in run1])
    assert llm.last_tools in (None, []), f"LLM session had tools: {llm.last_tools}"
    runs_path.write_text(json.dumps({k: v for k, v in runs.items() if k != "1"}, indent=1, ensure_ascii=False), encoding="utf-8")


if not COMPARE_ONLY:
    asyncio.run(extract_runs())


# --- compare -------------------------------------------------------------------
def toks(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


def match(a: list, b: list) -> list:
    """Greedy one-to-one matching of prerequisites: same type, best word overlap first."""
    cands = []
    for i, p in enumerate(a):
        for j, q in enumerate(b):
            if p["type"] == q["type"]:
                tp, tq = toks(p["what"]), toks(q["what"])
                cands.append((len(tp & tq) / max(1, len(tp | tq)), i, j))
    used_a, used_b, pairs = set(), set(), []
    for s, i, j in sorted(cands, reverse=True):
        # a lone prerequisite of its type on both sides is the same item whatever the wording
        lone = sum(p["type"] == a[i]["type"] for p in a) == 1 and sum(q["type"] == b[j]["type"] for q in b) == 1
        if i not in used_a and j not in used_b and (s >= 0.2 or lone):
            used_a.add(i)
            used_b.add(j)
            pairs.append((a[i], b[j]))
    return pairs


def surprise(q) -> bool:
    need, intro = q.get("needed_at_block"), q.get("introduced_at_block")
    return intro is None or (need is not None and intro >= need)


def close(x, y) -> bool:
    return (x is None and y is None) or (x is not None and y is not None and abs(x - y) <= TOL)


ids = sorted(k for k in runs if len(runs[k]) >= 0.9 * len(run1))
print(f"runs compared: {ids}; pages per run: {[len(runs[k]) for k in ids]}")
agree = defaultdict(list)
for ra, rb in itertools.combinations(ids, 2):
    for url in run1:
        a, b = runs[ra].get(url), runs[rb].get(url)
        if not a or not b:
            continue
        agree["page: is_procedure"].append(bool(a.get("is_procedure")) == bool(b.get("is_procedure")))
        agree["page: main_procedure_block within 3"].append(close(a.get("main_procedure_block"), b.get("main_procedure_block")))
        pa, pb = a["prerequisites"], b["prerequisites"]
        pairs = match(pa, pb)
        if pa or pb:
            agree["page: share of prerequisites found by both runs"].append(2 * len(pairs) / (len(pa) + len(pb)))
        for p, q in pairs:
            agree["item: instructions (on page / other page / none)"].append(p.get("instructions") == q.get("instructions"))
            agree["item: off-page or not"].append((p.get("instructions") == "other_page") == (q.get("instructions") == "other_page"))
            agree["item: has steps anywhere or not"].append((p.get("instructions") == "none") == (q.get("instructions") == "none"))
            agree["item: needed_at within 3 blocks"].append(close(p.get("needed_at_block"), q.get("needed_at_block")))
            agree["item: introduced_at within 3 blocks"].append(close(p.get("introduced_at_block"), q.get("introduced_at_block")))
            agree["item: surprise or not"].append(surprise(p) == surprise(q))
            if p.get("instructions") == q.get("instructions") == "other_page":
                agree["item: same link, when both say other page"].append((p.get("link") or "").split("#")[0] == (q.get("link") or "").split("#")[0])

print("\nAgreement between pairs of runs")
for k, v in agree.items():
    print(f"  {k:<52} {np.mean(v):.2f}  (n={len(v)})")


print("\nHeadline measures, per run (procedure pages as each run classed them)")
per_run = {}
for r in ids:
    pages = [d for d in runs[r].values() if d.get("is_procedure") and d.get("prerequisites")]
    pre = [q for d in pages for q in d["prerequisites"]]
    det = [surprise(q) and q.get("instructions") != "on_page" for q in pre]
    per_run[r] = {"procedure_pages": len(pages), "prerequisites": len(pre), "surprise": float(np.mean([surprise(q) for q in pre])),
                  "off_page": float(np.mean([q.get("instructions") == "other_page" for q in pre])),
                  "no_steps": float(np.mean([q.get("instructions") == "none" for q in pre])), "detour": float(np.mean(det)),
                  "pages_sent_elsewhere": sum(any(q.get("instructions") == "other_page" for q in d["prerequisites"]) for d in pages)}
    m = per_run[r]
    print(f"  run {r}: {m['procedure_pages']} pages, {m['prerequisites']} prerequisites, surprise {m['surprise']:.0%}, off-page {m['off_page']:.0%}, "
          f"no steps {m['no_steps']:.0%}, detour {m['detour']:.0%}, pages sent elsewhere {m['pages_sent_elsewhere']}")

print("\nBy type across runs (surprise rate; off-page rate; count)")
for t in TYPES[:6]:
    cells = []
    for r in ids:
        pre = [q for d in runs[r].values() if d.get("is_procedure") for q in d["prerequisites"] if q["type"] == t]
        cells.append(f"{np.mean([surprise(q) for q in pre]):.0%}; {np.mean([q.get('instructions') == 'other_page' for q in pre]):.0%}; {len(pre)}" if pre else "-")
    print(f"  {t:<22} " + "   ".join(cells))

# consensus: prerequisites found by every run, with majority values
consensus = {}
for url in run1:
    if not all(url in runs[r] for r in ids):
        continue
    base = runs[ids[0]][url]
    items = []
    for p in base["prerequisites"]:
        group = [p]
        for r in ids[1:]:
            m = [q for a, q in match([p], runs[r][url]["prerequisites"])]
            if m:
                group.append(m[0])
        if len(group) == len(ids):
            inst = Counter(q.get("instructions") for q in group).most_common(1)[0]
            items.append({"what": p["what"], "type": p["type"], "instructions": inst[0], "instructions_unanimous": inst[1] == len(ids),
                          "surprise_votes": sum(surprise(q) for q in group), "runs": len(ids)})
    consensus[url] = {"id": base["id"], "service": base["service"], "procedure_votes": sum(bool(runs[r][url].get("is_procedure")) for r in ids), "items": items}
n_items = sum(len(c["items"]) for c in consensus.values())
n_all = np.mean([sum(len(runs[r][u]["prerequisites"]) for u in consensus) for r in ids])
print(f"\nConsensus: {n_items} prerequisites found by all {len(ids)} runs (runs average {n_all:.0f} each)")
its = [i for c in consensus.values() if c["procedure_votes"] * 2 > len(ids) for i in c["items"]]
print(f"  on majority-procedure pages: {len(its)} items; surprise unanimous yes {sum(i['surprise_votes'] == len(ids) for i in its)}, "
      f"unanimous no {sum(i['surprise_votes'] == 0 for i in its)}, split {sum(0 < i['surprise_votes'] < len(ids) for i in its)}")
print(f"  instructions unanimous: {sum(i['instructions_unanimous'] for i in its)} of {len(its)}")
json.dump({"runs": ids, "agreement": {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in agree.items()}, "per_run": per_run, "consensus": consensus},
          open(here / "stability.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
