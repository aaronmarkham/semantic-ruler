"""Detours as a page metric, on every cached page.

A detour is a prerequisite that the page (a) mentions only at the block where
it is needed, later, or never, and (b) gives no steps for on the page. Two kinds:
"sent elsewhere" (the page links to another page for the steps) and "left
alone" (no steps and no link).

This script extends the extraction from the 60 pages of run 1 to every cached
page (extractions_all.json), then reports the metric by page, prerequisite
type, service and permission mechanism. The 60 original pages produced two
exploratory findings (join_intents.py); here they are tested on the pages that
were not part of that sample.

    python experiments/doc_ordering/detours.py .                  # extract, then analyse
    python experiments/doc_ordering/detours.py . --analyse-only
"""

import asyncio
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
ANALYSE_ONLY = "--analyse-only" in sys.argv
MODEL = "claude-sonnet-5-5"
here = root / "experiments" / "doc_ordering"
aws = root / "experiments" / "aws_intents"
cache = root / "data" / "aws-docs-cache"
all_path = here / "extractions_all.json"
PERMISSION = {"bucket_policy", "iam_role", "iam_identity_policy", "kms_key_policy"}
rng = np.random.default_rng(0)
sys.path.insert(0, str(aws))
from facts import facts  # noqa: E402

run1 = json.load(open(here / "extractions.json", encoding="utf-8"))
done = json.load(open(all_path, encoding="utf-8")) if all_path.exists() else {}

# --- pages -----------------------------------------------------------------------
pages, seen = {}, set()
for source in ("corpus.json", "corpus_crawl.json"):
    for j in json.load(open(aws / source, encoding="utf-8")):
        url = j["url"].split("#")[0]
        f = cache / f"{j['job']}.html" if source == "corpus.json" else cache / "crawl" / (re.sub(r"[^A-Za-z0-9]+", "-", j["job"])[:150] + ".html")
        if url in seen or not f.exists():
            continue
        seen.add(url)
        mech = set()
        for sec in j["sections"]:
            fc = facts(sec)
            if fc["n_s3_allow"]:
                mech |= {k for k, on in (("S", fc["bucket_policy_service"]), ("A", fc["bucket_policy_aws_principal"]),
                                         ("I", fc["identity_policy_s3"]), ("P", fc["public_read"])) if on}
        pages[url] = {"id": j["job"], "service": j["service"], "title": j["title"], "file": f, "mech": "".join(sorted(mech))}


async def extract_all():
    from semantic_ruler.generate import _parse_json
    from semantic_ruler.llm_plain import PlainTextProvider

    llm = PlainTextProvider(MODEL)
    sem = asyncio.Semaphore(4)
    n_done = 0

    async def one(url):
        nonlocal n_done
        p = pages[url]
        blocks = parse(p["file"].read_text(encoding="utf-8"), url)
        body, n_shown = render(blocks)
        async with sem:
            err = None
            for _ in range(2):
                try:
                    data = _parse_json(await llm.query(PROMPT.format(types=", ".join(TYPES), title=p["title"][:120], body=body)))
                    if isinstance(data.get("prerequisites"), list):
                        done[url] = {"id": p["id"], "service": p["service"], "n_blocks": len(blocks), "n_shown": n_shown, **data}
                        break
                except Exception as e:
                    err = e
            else:
                print(f"  extraction failed for {p['id']}: {err}")
            n_done += 1
            if n_done % 20 == 0:  # save as we go so an interrupted run keeps its work
                all_path.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")

    todo = [u for u in pages if u not in run1 and u not in done]
    print(f"{len(pages)} cached pages; {len(run1)} in run 1; extracting {len(todo)} more")
    await asyncio.gather(*[one(u) for u in todo])
    assert llm.last_tools in (None, []), f"LLM session had tools: {llm.last_tools}"
    all_path.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding="utf-8")


if not ANALYSE_ONLY:
    asyncio.run(extract_all())


# --- the metric --------------------------------------------------------------------
def surprise(q) -> bool:
    need, intro = q.get("needed_at_block"), q.get("introduced_at_block")
    return intro is None or (need is not None and intro >= need)


def kind(q) -> str:
    """'' if not a detour, else 'sent' or 'alone'."""
    if not surprise(q) or q.get("instructions") == "on_page":
        return ""
    return "sent" if q.get("instructions") == "other_page" else "alone"


PRE_SECTION = re.compile(r"prerequisite|before you (begin|start)|requirement|what you.ll need|getting ready", re.I)


def doubts(q, blocks) -> list:
    """Reasons to doubt that a detour is a real surprise; checked without an LLM."""
    out = []
    need, intro = q.get("needed_at_block"), q.get("introduced_at_block")
    k = intro if intro is not None else need
    if k is not None and 0 <= k < len(blocks):
        heading = next((b["text"] for b in reversed(blocks[:k + 1]) if b["kind"] == "h"), "")
        if PRE_SECTION.search(heading) or PRE_SECTION.search(blocks[k]["text"][:200]):
            out.append("announced in a prerequisites section")
    if need is not None and intro is not None and need <= 3 < intro:
        out.append("need placed at the top of the page")
    return out


rows = []
for url, d in {**done, **run1}.items():
    if url not in pages or not d.get("is_procedure"):
        continue
    pre = d["prerequisites"]
    blocks = parse(pages[url]["file"].read_text(encoding="utf-8"), url)
    for q in pre:
        q["doubts"] = doubts(q, blocks) if kind(q) else []
    perm = [q for q in pre if q["type"] in PERMISSION]
    rows.append({"url": url, "id": d["id"], "service": d["service"], "goal": d.get("goal", ""), "mech": pages[url]["mech"], "new": url not in run1,
                 "n": len(pre), "detours": sum(bool(kind(q)) for q in pre), "sent": sum(kind(q) == "sent" for q in pre),
                 "alone": sum(kind(q) == "alone" for q in pre), "n_perm": len(perm), "perm_detours": sum(bool(kind(q)) for q in perm),
                 "perm_off_page": sum(q.get("instructions") == "other_page" for q in perm),
                 "items": [{"type": q["type"], "what": q["what"], "kind": kind(q), "instructions": q.get("instructions"),
                            "needed": q.get("needed_at_block"), "introduced": q.get("introduced_at_block"), "link": q.get("link"),
                            "doubts": q["doubts"]} for q in pre]})
n_all = len([u for u in {**done, **run1} if u in pages])
new = [r for r in rows if r["new"]]
print(f"\n{n_all} pages extracted; {len(rows)} are procedures ({len(new)} outside the first sample), from {len({r['service'] for r in rows})} services")


def boot(rs, f, n_boot=2000):
    """Point value and 95% interval for f(rows), resampling services."""
    by = defaultdict(list)
    for r in rs:
        by[r["service"]].append(r)
    names = list(by)
    vals = []
    for _ in range(n_boot):
        pick = [x for s in rng.choice(len(names), len(names)) for x in by[names[s]]]
        v = f(pick)
        if v == v:
            vals.append(v)
    return f(rs), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def rate(num, den):
    return lambda rs: sum(r[num] for r in rs) / max(1, sum(r[den] for r in rs)) if sum(r[den] for r in rs) else float("nan")


def share(pred):
    return lambda rs: float(np.mean([pred(r) for r in rs])) if rs else float("nan")


def fmt(t):
    return f"{t[0]:.0%} [{t[1]:.0%}, {t[2]:.0%}]"


out = {"pages_extracted": n_all, "procedures": len(rows), "new_procedures": len(new)}
print("\n1. The metric (95% intervals resample services)")
for label, rs in (("all procedure pages", rows), ("first sample", [r for r in rows if not r["new"]]), ("pages outside it", new)):
    tot = sum(r["n"] for r in rs)
    m = {"pages": len(rs), "prerequisites": tot, "detour_rate": boot(rs, rate("detours", "n")), "sent": boot(rs, rate("sent", "n")),
         "alone": boot(rs, rate("alone", "n")), "pages_with_detour": boot(rs, share(lambda r: r["detours"] > 0)),
         "pages_with_2plus": boot(rs, share(lambda r: r["detours"] >= 2))}
    out[label] = m
    print(f"  {label}: {len(rs)} pages, {tot} prerequisites ({tot / len(rs):.1f} per page)")
    print(f"    detours per prerequisite {fmt(m['detour_rate'])}: sent elsewhere {fmt(m['sent'])}, left alone {fmt(m['alone'])}")
    print(f"    pages with at least one detour {fmt(m['pages_with_detour'])}; with two or more {fmt(m['pages_with_2plus'])}")
print("  detours per page: " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(r["detours"] for r in rows).items())))

det = [i for r in rows for i in r["items"] if i["kind"]]
how = Counter("never mentioned" if i["introduced"] is None else "mentioned at the block where it is needed" if i["introduced"] == i["needed"]
              else "mentioned after it is needed" for i in det)
dc = Counter(x for i in det for x in i["doubts"])
firm = sum(not i["doubts"] for i in det)
n_pre_all = sum(r["n"] for r in rows)
out["checks"] = {"detours": len(det), "how": dict(how), "doubts": dict(dc), "without_doubts": firm, "rate_without_doubts": firm / n_pre_all}
print(f"  how the {len(det)} detours arise: " + ", ".join(f"{k} {v}" for k, v in how.most_common()))
print("  doubtful: " + ", ".join(f"{k} {v}" for k, v in dc.items()) + f"; without those, {firm} detours ({firm / n_pre_all:.0%} of prerequisites)")

print("\n2. By prerequisite type (all procedure pages)")
print("  type                         count  detour  sent  alone")
out["by_type"] = {}
for t in TYPES:
    its = [i for r in rows for i in r["items"] if i["type"] == t]
    if its:
        v = [np.mean([bool(i["kind"]) for i in its]), np.mean([i["kind"] == "sent" for i in its]), np.mean([i["kind"] == "alone" for i in its])]
        out["by_type"][t] = {"count": len(its), "detour": float(v[0]), "sent": float(v[1]), "alone": float(v[2])}
        print(f"  {t:<28} {len(its):>5}  {v[0]:>6.0%}  {v[1]:>4.0%}  {v[2]:>5.0%}")


# --- 3. confirmation of the two exploratory findings, on new pages only ---------------
def perm_test(a, b, n_perm=20000):
    """Two-sided permutation test on the difference of means between two lists of per-page values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        hits += abs(pool[:len(a)].mean() - pool[len(a):].mean()) >= abs(obs) - 1e-12
    return float(obs), hits / n_perm


print("\n3. Mechanism and detours")
out["mechanism"] = {}
for label, rs in (("pages outside the first sample (confirmation)", new), ("all procedure pages", rows)):
    withperm = [r for r in rs if r["n_perm"] and r["mech"]]
    multi = [r["perm_off_page"] / r["n_perm"] for r in withperm if len(r["mech"]) > 1]
    single = [r["perm_off_page"] / r["n_perm"] for r in withperm if len(r["mech"]) == 1]
    s_pages = [r for r in rs if r["mech"] == "S"]
    i_pages = [r for r in rs if r["mech"] == "I"]
    print(f"  {label}")
    res = {}
    if multi and single:
        d, p = perm_test(multi, single)
        res["multi_vs_single_off_page"] = {"multi": float(np.mean(multi)), "n_multi": len(multi), "single": float(np.mean(single)), "n_single": len(single), "p": p}
        print(f"    a. share of permission prerequisites sent off-page: {np.mean(multi):.0%} on {len(multi)} multi-mechanism pages, "
              f"{np.mean(single):.0%} on {len(single)} single-mechanism pages (difference {d:+.0%}, p = {p:.3f})")
    if s_pages and i_pages:
        for key, name in (("perm_off_page", "sends the reader away for a permission prerequisite"), ("perm_detours", "has a permission detour"), ("detours", "has any detour")):
            sa, ia = [r[key] > 0 for r in s_pages], [r[key] > 0 for r in i_pages]
            d, p = perm_test(sa, ia)
            res[f"S_vs_I_{key}"] = {"S": int(sum(sa)), "n_S": len(sa), "I": int(sum(ia)), "n_I": len(ia), "p": p}
            print(f"    b. {name}: {sum(sa)} of {len(sa)} service-principal pages, {sum(ia)} of {len(ia)} identity-policy pages (p = {p:.3f})")
    out["mechanism"][label] = res
print("  by mechanism, all procedure pages: mechanism  pages  prereqs/page  detours/page  share with a detour")
for m, c in Counter(r["mech"] or "(none)" for r in rows).most_common():
    g = [r for r in rows if (r["mech"] or "(none)") == m]
    print(f"    {m:<7} {c:>4}  {np.mean([r['n'] for r in g]):>5.1f}  {np.mean([r['detours'] for r in g]):>5.2f}  {np.mean([r['detours'] > 0 for r in g]):>5.0%}")

print("\n4. Services with three or more procedure pages, by detours per prerequisite")
by = defaultdict(list)
for r in rows:
    by[r["service"]].append(r)
svc = sorted(((s, g) for s, g in by.items() if len(g) >= 3), key=lambda t: -rate("detours", "n")(t[1]))
out["by_service"] = {s: {"pages": len(g), "detour_rate": rate("detours", "n")(g)} for s, g in svc}
for s, g in svc[:6] + [("...", None)] + svc[-4:]:
    print(f"  {s}" if g is None else f"  {s:<28} {len(g)} pages  {rate('detours', 'n')(g):.0%} of {sum(r['n'] for r in g)} prerequisites")

print("\n5. Pages with the most detours")
for r in sorted(rows, key=lambda r: -r["detours"])[:8]:
    print(f"  {r['detours']} of {r['n']} ({r['sent']} sent, {r['alone']} alone)  {r['service']}: {r['goal'][:85]}")
out["rows"] = rows
json.dump(out, open(here / "detours.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
