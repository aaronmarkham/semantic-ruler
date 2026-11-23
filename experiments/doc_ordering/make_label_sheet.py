"""Build a blind label sheet for validating the detour metric.

Draws prerequisites from detours.json, half that the metric calls detours and
half that it does not, at most one per page, shuffled. The sheet does not show
the metric's verdict; the answer key goes to label_key.json.

Outputs (the sheet holds short doc excerpts, so it goes under out/, which is
not committed):
    out/labels/label_sheet.html   open in a browser, click through, download labels.json
    out/labels/label_sheet.csv    same items for a spreadsheet, if preferred
    out/labels/detour-label-sheet.html   the same sheet as a claude.ai artifact page; answers save to its store
    experiments/doc_ordering/label_key.json

    python -I experiments/doc_ordering/make_label_sheet.py . [n_items] [seed]
"""

import csv
import json
import random
import re
import sys
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).parent))
from stream import parse  # noqa: E402

root = Path(sys.argv[1]).resolve()
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 7
EXCERPT = 280
here = root / "experiments" / "doc_ordering"
aws = root / "experiments" / "aws_intents"
cache = root / "data" / "aws-docs-cache"
out_dir = root / "out" / "labels"
out_dir.mkdir(parents=True, exist_ok=True)

files = {}
for source in ("corpus.json", "corpus_crawl.json"):
    for j in json.load(open(aws / source, encoding="utf-8")):
        f = cache / f"{j['job']}.html" if source == "corpus.json" else cache / "crawl" / (re.sub(r"[^A-Za-z0-9]+", "-", j["job"])[:150] + ".html")
        files.setdefault(j["url"].split("#")[0], f)

rows = json.load(open(here / "detours.json", encoding="utf-8"))["rows"]
rng = random.Random(SEED)
pool = {True: [], False: []}
for r in rows:
    for k, it in enumerate(r["items"]):
        if it["needed"] is not None:
            pool[bool(it["kind"])].append((r, k))

chosen, used = [], set()
for want in (True, False):
    cands = pool[want][:]
    rng.shuffle(cands)
    picked = 0
    for r, k in cands:
        if picked == N // 2:
            break
        if r["url"] in used:
            continue
        used.add(r["url"])
        chosen.append((r, k))
        picked += 1
rng.shuffle(chosen)

items, key = [], {}
for n, (r, k) in enumerate(chosen, 1):
    it = r["items"][k]
    blocks = parse(files[r["url"]].read_text(encoding="utf-8"), r["url"])
    b = blocks[it["needed"]] if 0 <= it["needed"] < len(blocks) else {"text": ""}
    text = re.sub(r"\s+", " ", b["text"]).strip()
    section = next((x["text"] for x in reversed(blocks[:it["needed"] + 1]) if x["kind"] == "h"), "")
    # text fragment so the browser jumps to the spot; a few leading words is enough
    words = [w for w in text.split()[:6] if "<" not in w and "(" not in w]
    jump = r["url"] + ("#:~:text=" + quote(" ".join(words)) if len(words) >= 4 else "")
    iid = f"L{n:02d}"
    items.append({"id": iid, "service": r["service"], "goal": r["goal"], "url": r["url"], "jump": jump, "type": it["type"].replace("_", " "),
                  "what": it["what"], "section": section[:120], "where": f"{round(100 * it['needed'] / max(1, len(blocks)))}% down the page",
                  "excerpt": text[:EXCERPT] + ("..." if len(text) > EXCERPT else "")})
    key[iid] = {"url": r["url"], "item_index": k, "what": it["what"], "type": it["type"], "metric_kind": it["kind"], "metric_detour": bool(it["kind"]),
                "instructions": it["instructions"], "needed": it["needed"], "introduced": it["introduced"], "doubts": it.get("doubts", [])}

(here / "label_key.json").write_text(json.dumps({"seed": SEED, "n": len(items), "items": key}, indent=1, ensure_ascii=False), encoding="utf-8")

with open(out_dir / "label_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["id", "service", "job", "prerequisite", "type", "page", "section where it is needed", "position", "excerpt",
                "detour (yes / no / not a prerequisite / unsure)", "steps (on page / other page / none)", "warned earlier (yes / no)", "note"])
    for i in items:
        w.writerow([i["id"], i["service"], i["goal"], i["what"], i["type"], i["url"], i["section"], i["where"], i["excerpt"], "", "", "", ""])

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Detour label sheet</title>
<style>
:root{--bg:#f6f5f1;--card:#fff;--ink:#1d1c1a;--mute:#6b6861;--line:#dedbd3;--accent:#1f5fa8;--yes:#a8431f;--no:#2c6e49;--chip:#eeece6}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--card:#201f1d;--ink:#ecebe7;--mute:#a09d95;--line:#383632;--accent:#7fb0ea;--yes:#e58a66;--no:#7cc79b;--chip:#2b2a27}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
.bar{max-width:820px;margin:0 auto;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.bar h1{font-size:16px;margin:0;flex:1 1 auto}.count{font-variant-numeric:tabular-nums;color:var(--mute)}
progress{width:140px;height:8px;accent-color:var(--accent)}
main{max-width:820px;margin:0 auto;padding:16px}
.intro{color:var(--mute);margin:4px 0 20px}.intro b{color:var(--ink)}
details.def{margin:0 0 20px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px}
details.def summary{cursor:pointer;font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 16px}
.card.done{border-left:4px solid var(--accent)}
.meta{display:flex;gap:8px;align-items:center;flex-wrap:wrap;color:var(--mute);font-size:13px}
.chip{background:var(--chip);border-radius:999px;padding:1px 9px;color:var(--ink)}
.goal{margin:8px 0 2px;color:var(--mute)}.what{font-size:17px;font-weight:600;margin:2px 0 10px;text-wrap:balance}
.where{border-left:3px solid var(--line);padding:2px 0 2px 12px;margin:0 0 12px;color:var(--mute)}
.where q{color:var(--ink);quotes:none;display:block;overflow-wrap:anywhere}
a{color:var(--accent)}
.q{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}.q span{min-width:150px;color:var(--mute);font-size:13px}
button{font:inherit;border:1px solid var(--line);background:transparent;color:var(--ink);border-radius:7px;padding:5px 12px;cursor:pointer}
button:hover{border-color:var(--accent)}button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
button.on[data-v=yes]{background:var(--yes);border-color:var(--yes)}button.on[data-v=no]{background:var(--no);border-color:var(--no)}
input.note{flex:1 1 240px;font:inherit;padding:5px 9px;border:1px solid var(--line);border-radius:7px;background:transparent;color:var(--ink)}
.primary{background:var(--accent);border-color:var(--accent);color:#fff}
footer{max-width:820px;margin:0 auto;padding:8px 16px 48px;color:var(--mute)}
textarea{width:100%;height:120px;font:12px/1.4 ui-monospace,Consolas,monospace;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px}
</style></head><body>
<header><div class="bar"><h1>Detour label sheet</h1><span class="count" id="count"></span><progress id="prog" value="0" max="1"></progress>
<button class="primary" id="dl">Download labels.json</button></div></header>
<main>
<p class="intro">__N__ prerequisites from AWS doc pages. For each one, open the page and decide whether a reader doing the job would hit a <b>detour</b>.
Only the first question is required. Answers save in this browser as you go.</p>
<details class="def" open><summary>What counts</summary>
<p><b>Detour: yes</b> when both are true: the page does not tell the reader about the prerequisite until the point it is needed (or later, or never), and the page gives no steps for it, so the reader must stop and work it out or go to another page.</p>
<p><b>Detour: no</b> when the page warns about it before it is needed, or gives the steps for it on the page.</p>
<p><b>Not a prerequisite</b> when it is not something the reader must already have or set up for this job (for example, it is the job's own main resource, or it does not apply).</p>
<p>The quoted text is where the extraction placed the need. It is a pointer, not a verdict; it may be wrong.</p></details>
<div id="list"></div>
</main>
<footer><p>When finished, use <b>Download labels.json</b> and save it to <code>experiments/doc_ordering/labels.json</code>, or copy the text below.</p>
<textarea id="dump" readonly></textarea></footer>
<script>
const ITEMS = __DATA__;
const KEY = "detour-labels-seed__SEED__";
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const QS = [
  ["detour", "Detour?", [["yes", "Yes"], ["no", "No"], ["not_prereq", "Not a prerequisite"], ["unsure", "Unsure"]]],
  ["steps", "Steps for it (optional)", [["on_page", "On this page"], ["other_page", "Link to another page"], ["none", "None"]]],
  ["warned", "Warned earlier (optional)", [["yes", "Yes"], ["no", "No"]]],
];
const el = (t, c, txt) => { const e = document.createElement(t); if (c) e.className = c; if (txt != null) e.textContent = txt; return e; };
function save() {
  try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {}
  const done = ITEMS.filter(i => labels[i.id] && labels[i.id].detour).length;
  document.getElementById("count").textContent = done + " of " + ITEMS.length;
  document.getElementById("prog").value = done / ITEMS.length;
  document.getElementById("dump").value = JSON.stringify(labels, null, 1);
}
const list = document.getElementById("list");
for (const it of ITEMS) {
  const card = el("section", "card"); card.id = it.id;
  const meta = el("div", "meta");
  meta.append(el("span", null, it.id), el("span", "chip", it.service), el("span", "chip", it.type));
  const link = el("a", null, "Open the page at this spot"); link.href = it.jump; link.target = "_blank"; link.rel = "noopener";
  meta.append(link);
  const where = el("div", "where");
  where.append(el("span", null, "Needed in “" + (it.section || "top of page") + "”, " + it.where + ":"), el("q", null, it.excerpt));
  card.append(meta, el("p", "goal", "Job: " + it.goal), el("p", "what", it.what), where);
  for (const [name, label, opts] of QS) {
    const row = el("div", "q"); row.append(el("span", null, label));
    for (const [v, text] of opts) {
      const b = el("button", null, text); b.type = "button"; b.dataset.v = v;
      if (labels[it.id] && labels[it.id][name] === v) b.classList.add("on");
      b.onclick = () => {
        labels[it.id] = labels[it.id] || {};
        const off = labels[it.id][name] === v;
        if (off) delete labels[it.id][name]; else labels[it.id][name] = v;
        row.querySelectorAll("button").forEach(x => x.classList.toggle("on", !off && x === b));
        card.classList.toggle("done", !!labels[it.id].detour); save();
      };
      row.append(b);
    }
    card.append(row);
  }
  const nrow = el("div", "q"); nrow.append(el("span", null, "Note (optional)"));
  const note = el("input", "note"); note.value = (labels[it.id] && labels[it.id].note) || "";
  note.oninput = () => { labels[it.id] = labels[it.id] || {}; labels[it.id].note = note.value; save(); };
  nrow.append(note); card.append(nrow);
  card.classList.toggle("done", !!(labels[it.id] && labels[it.id].detour));
  list.append(card);
}
document.getElementById("dl").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(labels, null, 1)], { type: "application/json" }));
  a.download = "labels.json"; a.click();
};
save();
</script></body></html>
"""
page = HTML.replace("__DATA__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/")).replace("__N__", str(len(items))).replace("__SEED__", str(SEED))
(out_dir / "label_sheet.html").write_text(page, encoding="utf-8")
# Phone-friendly version that saves answers to the artifact's own store (see label_artifact.html).
data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
(out_dir / "detour-label-sheet.html").write_text((here / "label_artifact.html").read_text(encoding="utf-8").replace("__DATA__", data), encoding="utf-8")
print(f"{len(items)} items from {len(used)} pages ({sum(v['metric_detour'] for v in key.values())} the metric calls detours)")
print(f"sheet: {out_dir / 'label_sheet.html'}\ncsv:   {out_dir / 'label_sheet.csv'}\nkey:   {here / 'label_key.json'}")
