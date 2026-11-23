"""Trace prerequisite trees for several how-to pages: each off-page prerequisite
is followed to its page, whose own prerequisites are followed in turn.

Start pages are the procedure pages with the most off-page prerequisites in
detours.json, one per service, plus the WAF logging page used elsewhere.
Finished trees are kept, so a rerun only traces what is missing.

    python experiments/doc_ordering/chains.py . [n_starts] [max_depth] [max_pages]
"""

import asyncio
import json
import sys
from pathlib import Path

from semantic_ruler.llm_plain import PlainTextProvider
from semantic_ruler.ordering import Fetcher, trace

root = Path(sys.argv[1]).resolve()
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
MAX_DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MAX_PAGES = int(sys.argv[4]) if len(sys.argv) > 4 else 12
here = root / "experiments" / "doc_ordering"
dest = here / "chains.json"
done = json.load(open(dest, encoding="utf-8")) if dest.exists() else {}

rows = json.load(open(here / "detours.json", encoding="utf-8"))["rows"]
ranked = sorted(rows, key=lambda r: -sum(i["instructions"] == "other_page" for i in r["items"]))
starts, seen = ["https://docs.aws.amazon.com/waf/latest/developerguide/logging-s3.html"], {"WAF"}
for r in ranked:
    if r["service"] not in seen and len(starts) < N:
        seen.add(r["service"])
        starts.append(r["url"])

llm = PlainTextProvider("claude-sonnet-5-5")
fetch = Fetcher(root / "data" / "aws-docs-cache" / "chain")


async def main():
    for url in starts:
        if url in done:
            continue
        tree = await trace(llm, url, fetch, MAX_DEPTH, MAX_PAGES)
        if not tree:
            print(f"nothing traced from {url}")
            continue
        done[url] = [{"url": n.report.url, "depth": n.depth, "parent": n.parent, "via": n.via, "title": n.report.title,
                      "goal": n.report.goal, "is_procedure": n.report.is_procedure, "redirected_from": n.report.redirected_from,
                      "prerequisites": n.report.prerequisites} for n in tree]
        dest.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
        pre = sum(len(n.report.prerequisites) for n in tree)
        print(f"{len(tree):2d} pages, depth {max(n.depth for n in tree)}, {pre:2d} prerequisites, "
              f"{sum(len(n.report.detours) for n in tree):2d} detours  {url}", flush=True)


asyncio.run(main())
assert llm.last_tools in (None, []), f"LLM session had tools: {llm.last_tools}"
