"""Follow one job's prerequisites across pages: the "open lines of investigation" tree.

Starts at a how-to page, extracts its prerequisites, and follows every
prerequisite whose instructions live on another docs page, up to a depth limit.

    python experiments/doc_ordering/chain.py . <start_url> [max_depth] [max_pages]
"""

import asyncio
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract import PROMPT, TYPES  # noqa: E402
from stream import parse, render  # noqa: E402

from semantic_ruler.generate import _parse_json  # noqa: E402
from semantic_ruler.llm_plain import PlainTextProvider  # noqa: E402

root = Path(sys.argv[1]).resolve()
START = sys.argv[2].split("#")[0]
MAX_DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MAX_PAGES = int(sys.argv[4]) if len(sys.argv) > 4 else 14
cache = root / "data" / "aws-docs-cache" / "chain"
cache.mkdir(parents=True, exist_ok=True)
store = root / "experiments" / "doc_ordering" / "chain_extractions.json"
done = json.load(open(store, encoding="utf-8")) if store.exists() else {}
llm = PlainTextProvider("claude-sonnet-5-5")


def fetch(url: str) -> str:
    dest = cache / (hashlib.sha256(url.encode()).hexdigest()[:20] + ".html")
    if not dest.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; docs-reader)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_text(r.read().decode("utf-8", "replace"), encoding="utf-8")
        time.sleep(0.3)
    return dest.read_text(encoding="utf-8")


async def extract(url: str) -> dict | None:
    if url in done:
        return done[url]
    try:
        blocks = parse(fetch(url), url)
    except Exception as e:
        print(f"  could not fetch {url}: {e}")
        return None
    title = next((b["text"] for b in blocks if b["kind"] == "h"), url)
    body, _ = render(blocks)
    for _ in range(2):
        try:
            data = _parse_json(await llm.query(PROMPT.format(types=", ".join(TYPES), title=title[:120], body=body)))
            if isinstance(data.get("prerequisites"), list):
                links = {l["url"] for b in blocks if b["kind"] == "p" for l in b["links"]}
                done[url] = {"title": title, "n_blocks": len(blocks), "links_on_page": sorted(links), **data}
                return done[url]
        except Exception as e:
            err = e
    print(f"  extraction failed for {url}: {err}")
    return None


async def main():
    tree, visited, frontier = [], {START}, [(START, 0, None)]
    while frontier and len(visited) <= MAX_PAGES:
        level, frontier = frontier, []
        results = await asyncio.gather(*[extract(u) for u, _, _ in level])
        for (url, depth, via), d in zip(level, results):
            if d is None:
                continue
            tree.append((url, depth, via, d))
            if depth >= MAX_DEPTH:
                continue
            if not d["prerequisites"]:
                # Hub page: it lists options rather than prerequisites. This study is
                # about S3, so take the S3 branch: same-guide links that mention S3.
                guide = url.rsplit("/", 1)[0]
                for b in parse(fetch(url), url):
                    for link in b.get("links", []) if b["kind"] == "p" else []:
                        if link["url"].startswith(guide) and re.search(r"\bs3\b|bucket|simple storage", link["text"] + link["url"], re.I) \
                                and link["url"] not in visited and len(visited) < MAX_PAGES:
                            visited.add(link["url"])
                            frontier.append((link["url"], depth + 1, f"the S3 option on this hub page ({link['text'][:40]})"))
            for q in d["prerequisites"]:
                link = (q.get("link") or "").split("#")[0]
                # follow only links that really are on the page
                if q.get("instructions") == "other_page" and link in d["links_on_page"] and link not in visited and len(visited) < MAX_PAGES:
                    visited.add(link)
                    frontier.append((link, depth + 1, q["what"]))
    return tree


tree = asyncio.run(main())
store.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")
assert llm.last_tools in (None, []), f"LLM session had tools: {llm.last_tools}"

total = surprises = 0
print(f"\nPrerequisite tree from {START}\n")
for url, depth, via, d in tree:
    pad = "    " * depth
    if via:
        print(f"{pad}^ opened for: {via[:70]}")
    print(f"{pad}PAGE: {d['title'][:90]}")
    print(f"{pad}  ({'step-by-step procedure' if d.get('is_procedure') else 'reference page'}; {url.rsplit('/', 1)[-1]})")
    for q in sorted(d["prerequisites"], key=lambda q: q.get("needed_at_block") or 0):
        need, intro = q.get("needed_at_block"), q.get("introduced_at_block")
        surprise = intro is None or (need is not None and intro >= need)
        total += 1
        surprises += surprise
        where = {"on_page": "steps on this page", "other_page": "steps on another page", "none": "no steps given"}.get(q.get("instructions"), "?")
        print(f"{pad}  - [{q['type']}] {q['what'][:95]}")
        print(f"{pad}      {where}{'; mentioned only when needed' if surprise else ''}")
pages = len(tree)
depth = max(d for _, d, _, _ in tree)
print(f"\n{pages} pages open, {depth} levels deep, {total} prerequisites, {surprises} mentioned only at the point of need")
json.dump({"start": START, "pages": pages, "depth": depth, "prerequisites": total, "surprises": surprises,
           "tree": [{"url": u, "depth": dp, "via": v, "title": d["title"], "is_procedure": d.get("is_procedure"),
                     "prerequisites": d["prerequisites"]} for u, dp, v, d in tree]},
          open(root / "experiments" / "doc_ordering" / "chain_waf.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
