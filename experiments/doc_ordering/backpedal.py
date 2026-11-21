"""Measure how much a how-to page makes the reader backpedal.

For each procedural page an LLM lists the prerequisites of the page's job, the
block where the reader first needs each one, the block where the page first
says so, and where the instructions for it live. Extraction prompts contain
fetched doc text, so they go through the text-only provider (no tools).

    python experiments/doc_ordering/backpedal.py . [max_pages] [per_service]
"""

import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extract import PROMPT, TYPES  # noqa: E402
from stream import parse, policy_summary, render  # noqa: E402

from semantic_ruler.generate import _parse_json  # noqa: E402
from semantic_ruler.llm_plain import PlainTextProvider  # noqa: E402

root = Path(sys.argv[1]).resolve()
MAX_PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 60
PER_SERVICE = int(sys.argv[3]) if len(sys.argv) > 3 else 2
MODEL = "claude-sonnet-5-5"
aws = root / "experiments" / "aws_intents"
cache = root / "data" / "aws-docs-cache"
out_path = root / "experiments" / "doc_ordering" / "extractions.json"
FORCE = ["waf-logs", "static-website-tutorial", "cloudfront-oac-origin", "alb-access-logs", "cloudtrail-trail-logs"]
def slug(job: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", job)[:150]


def load_pages():
    pages = []
    for j in json.load(open(aws / "corpus.json", encoding="utf-8")):
        pages.append({"id": j["job"], "service": j["service"], "url": j["url"], "title": j["title"], "file": cache / f"{j['job']}.html"})
    for j in json.load(open(aws / "corpus_crawl.json", encoding="utf-8")):
        pages.append({"id": j["job"], "service": j["service"], "url": j["url"], "title": j["title"], "file": cache / "crawl" / f"{slug(j['job'])}.html"})
    seen, out = set(), []
    for p in pages:
        u = p["url"].split("#")[0]
        if u in seen or not p["file"].exists():
            continue
        seen.add(u)
        p["blocks"] = parse(p["file"].read_text(encoding="utf-8"), u)
        out.append(p)
    return out


def procedural(p) -> bool:
    heads = " | ".join(b["text"] for b in p["blocks"] if b["kind"] == "h").lower()
    return bool(re.search(r"\bstep \d|prerequisite|\bto (create|enable|configure|set up|attach|add)\b", heads)) or \
        bool(re.match(r"(tutorial|enabl|configur|creat|set(ting)? up|export|send|publish|stor)", p["title"].lower()))


def first_policy_block(p, resource_based: bool) -> int | None:
    for i, b in enumerate(p["blocks"]):
        if b["kind"] == "code":
            s = policy_summary(b["text"]) or ""
            if "s3" in s.split("actions:")[-1] and (("resource-based" in s) == resource_based):
                return i
    return None


pages = load_pages()
proc = [p for p in pages if procedural(p)]
chosen, per = [], Counter()
for p in sorted(proc, key=lambda p: (p["id"] not in FORCE, p["service"], p["id"])):
    if p["id"] in FORCE or (per[p["service"]] < PER_SERVICE and len(chosen) < MAX_PAGES):
        chosen.append(p)
        per[p["service"]] += 1
print(f"{len(pages)} cached pages with S3 grants; {len(proc)} look procedural; analysing {len(chosen)} from {len(per)} services")

done = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {}
llm = PlainTextProvider(MODEL)


async def extract(p, sem):
    if p["url"] in done:
        return
    body, n_shown = render(p["blocks"])
    async with sem:
        for _ in range(2):
            try:
                data = _parse_json(await llm.query(PROMPT.format(types=", ".join(TYPES), title=p["title"][:120], body=body)))
                if isinstance(data.get("prerequisites"), list):
                    done[p["url"]] = {"id": p["id"], "service": p["service"], "n_blocks": len(p["blocks"]), "n_shown": n_shown, **data}
                    return
            except Exception as e:  # retry once, then report
                err = e
        print(f"  extraction failed for {p['id']}: {err}")


async def main():
    sem = asyncio.Semaphore(4)
    await asyncio.gather(*[extract(p, sem) for p in chosen])


asyncio.run(main())
out_path.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
assert llm.last_tools in (None, []), f"LLM session had tools: {llm.last_tools}"

# --- measure -----------------------------------------------------------------
by_url = {p["url"]: p for p in chosen}
rows, checks = [], Counter()
for url, d in done.items():
    p = by_url.get(url)
    if p is None or not d.get("prerequisites"):
        continue
    main_block = d.get("main_procedure_block")
    page_links = {l["url"] for b in p["blocks"] if b["kind"] == "p" for l in b["links"]}
    pre = []
    for q in d["prerequisites"]:
        intro, inst = q.get("introduced_at_block"), q.get("instructions")
        for key in ("needed_at_block", "introduced_at_block", "instructions_block"):
            v = q.get(key)
            checks["block index out of range"] += v is not None and not (0 <= v < d["n_blocks"])
        if inst == "other_page":
            checks["off-page links total"] += 1
            checks["off-page link not found on page"] += (q.get("link") or "").split("#")[0] not in page_links
        need = q.get("needed_at_block")
        # surprise: the page mentions it only at the point of need, later, or never
        late = intro is None or (need is not None and intro >= need)
        pre.append({**q, "late": late, "off_page": inst == "other_page", "unexplained": inst == "none"})
    # deterministic cross-check: where the policy JSON actually sits
    for typ, rb in (("bucket_policy", True), ("iam_identity_policy", False)):
        pos = first_policy_block(p, rb)
        for q in pre:
            if q["type"] == typ and q["instructions"] == "on_page" and q.get("instructions_block") is not None and pos is not None:
                checks["policy prerequisites with JSON on page"] += 1
                checks["instructions block within 25 blocks before the JSON"] += 0 <= pos - q["instructions_block"] <= 25
    rows.append({"id": d["id"], "service": d["service"], "goal": d["goal"], "procedure": bool(d.get("is_procedure")), "main": main_block, "n_blocks": d["n_blocks"], "pre": pre})

n = len(rows)
allpre = [q for r in rows for q in r["pre"]]
print(f"\n{n} procedural pages analysed, {len(allpre)} prerequisites ({len(allpre) / n:.1f} per page)")
print("  by type: " + ", ".join(f"{k} {v}" for k, v in Counter(q["type"] for q in allpre).most_common()))
print(f"  surprises (mentioned only at the point of need, later, or never): {sum(q['late'] for q in allpre)} ({np.mean([q['late'] for q in allpre]):.0%})")
print(f"  instructions on another page: {sum(q['off_page'] for q in allpre)} ({np.mean([q['off_page'] for q in allpre]):.0%})")
print(f"  no instructions at all: {sum(q['unexplained'] for q in allpre)} ({np.mean([q['unexplained'] for q in allpre]):.0%})")
lines = [len({q.get('link') for q in r['pre'] if q['off_page']}) + sum(q['late'] and not q['off_page'] for q in r['pre']) for r in rows]
print(f"  open lines of investigation per page (distinct other pages + on-page surprises): "
      f"median {np.median(lines):.0f}, mean {np.mean(lines):.1f}, max {max(lines)}; pages with 3 or more: {sum(x >= 3 for x in lines)} of {n}")
print("  surprise or off-page, by type: " + ", ".join(
    f"{t} {np.mean([q['late'] or q['off_page'] for q in allpre if q['type'] == t]):.0%}" for t, _ in Counter(q['type'] for q in allpre).most_common(6)))
print("\nConsistency checks on the extraction")
for k, v in checks.items():
    print(f"  {k}: {v}")

print("\nWorst pages by open lines")
for r, x in sorted(zip(rows, lines), key=lambda t: -t[1])[:8]:
    print(f"  {x}  {r['service']}: {r['goal'][:90]}")
for want in FORCE[:2]:
    for r in rows:
        if r["id"] == want:
            print(f"\n{r['service']} - {r['goal']}\n  main procedure starts at block {r['main']} of {r['n_blocks']}")
            for q in sorted(r["pre"], key=lambda q: q.get("needed_at_block") or 0):
                print(f"  needed at {q.get('needed_at_block')}, introduced at {q.get('introduced_at_block')}, "
                      f"{q['instructions']:<10} {'SURPRISE ' if q['late'] else ''}{q['type']}: {q['what'][:80]}")
json.dump({"pages": n, "prerequisites": len(allpre), "open_lines": lines, "checks": dict(checks),
           "rows": [{k: v for k, v in r.items()} for r in rows]},
          open(root / "experiments" / "doc_ordering" / "results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
