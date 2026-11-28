"""Collect the drift page's numbers from recorded pipeline runs.

Each file in experiments/doc_drift/results/*-runs.json holds repeated runs of
experiments/doc_drift/run.py on one source: the extracted graphs, the
regenerated text, the probes, every score and, when the run had a repair
pass, the repaired draft and its scores. The files in results/cohort/ are
the same thing for a set of procedure pages run two times each. This
summarises them (median and range over runs, one representative run shown
in full, the cohort as per-page medians and rates across pages) into
drift-explainer.json, which the page fetches. Standard library only.

    python -I experiments/explainer/build_drift.py .
"""

import json
import random
import re
import statistics
import sys
from pathlib import Path

CONDITION_WORDS = re.compile(
    r"\b(only (?:when|if)|if|unless|when|whenever|before|after|until|requires?|required|must|provided that|"
    r"except|otherwise|as long as|depends? on|in order to|however|but)\b", re.I)


def graph_measures(orig):
    """Granularity and conditional load from the stored graph alone (older runs did not record them)."""
    facts, edges = orig["facts"], orig["edges"]
    n = len(facts)
    nxt = {}
    for e in edges:
        for a in e["antecedents"]:
            nxt.setdefault(a, []).append(e["consequent"])
    best = {}

    def depth(node, seen):
        if node in best:
            return best[node]
        d = 1 + max((depth(c, seen | {node}) for c in nxt.get(node, []) if c not in seen), default=0)
        best[node] = d
        return d

    widest = max((len(e["antecedents"]) for e in edges if e["kind"] == "requires"), default=0)
    longest = max((depth(f["id"], frozenset()) for f in facts), default=0)
    epa = len(edges) / n if n else 0.0
    scaled = [min(1.0, epa), min(1.0, widest / 4.0), min(1.0, (longest - 1) / 4.0) if longest else 0.0]
    return {"n_atoms": n, "n_edges": len(edges), "folded": (sum(bool(CONDITION_WORDS.search(f["text"])) for f in facts) / n) if n else None,
            "edges_per_atom": epa, "widest_requires": widest, "longest_chain": longest, "load": sum(scaled) / len(scaled)}


def spearman_ci(xs, ys, B=2000, seed=7):
    """Rank correlation with a bootstrap interval; n is small, so the interval is the point."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    def rho(a, b):
        ra, rb = rank(a), rank(b); ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
        num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)); den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
        return num / den if den else 0.0

    n = len(xs)
    if n < 4:
        return None
    rng = random.Random(seed); vals = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        a, b = [xs[i] for i in idx], [ys[i] for i in idx]
        if len(set(a)) > 1 and len(set(b)) > 1:
            vals.append(rho(a, b))
    vals.sort()
    return {"rho": r3(rho(xs, ys)), "lo": r3(vals[int(0.025 * len(vals))]), "hi": r3(vals[int(0.975 * len(vals)) - 1]), "n": n}

root = Path(sys.argv[1]).resolve()
results = root / "experiments" / "doc_drift" / "results"
dest = root / "experiments" / "explainer" / "drift-explainer.json"

SCORES = ("f1", "precision", "recall", "cbr", "tbr", "probes", "drift")
STRUCT = ("prereq_recall", "orig_detour_rate", "regen_detour_rate", "detour_rate_delta", "intro_position_mae", "structural_loss")


def r3(x):
    return None if x is None or x != x else round(float(x), 3)


def spread(values):
    vs = [v for v in values if v is not None and v == v]
    return {"median": r3(statistics.median(vs)), "min": r3(min(vs)), "max": r3(max(vs)), "n": len(vs)} if vs else None


def scores_of(s):
    """Flatten one scored pass (the first draft or the repaired one) into one row."""
    a, c, e, d, st = s["atomic_f1"], s["cond_bridge_recall"], s["entailment"], s["drift"], s.get("structural") or {}
    tb = s.get("text_bridge") or {}
    return {
        "tbr": r3(tb.get("rate")), "tbr_yes": tb.get("n_yes"), "tbr_n": tb.get("n_edges"), "tbr_unknown": tb.get("n_unknown"), "tbr_lost": tb.get("lost"),
        "orig": s.get("orig"), "regen": s.get("regen"),
        "f1": r3(a["f1"]), "precision": r3(a["precision"]), "recall": r3(a["recall"]), "unsupported": a.get("unsupported"),
        "cbr": r3(c.get("cbr")), "cbr_by_kind": {k: r3(v) for k, v in c.get("by_kind", {}).items()}, "lost": c.get("lost"),
        "probes": r3(e.get("rate")), "n_probes": e.get("n_probes", 0), "n_correct": e.get("n_correct"),
        "drift": r3(d["drift"]), "losses": {k: r3(v) for k, v in d.items() if k != "drift"},
        "structural": {k: (r3(v) if isinstance(v, float) else v) for k, v in st.items()},
    }


def summarise(runs, source, path=Path('')):
    rows = []
    for r in runs:
        row = {"run": r["run"], **scores_of(r["scores"])}
        if r.get("repair"):
            row["repair"] = {**scores_of(r["repair"]), "fed_back": r["repair"].get("fed_back")}
        rows.append(row)
    med = statistics.median(x["drift"] for x in rows)
    rep = min(rows, key=lambda x: abs(x["drift"] - med))["run"]
    shown = next(r for r in runs if r["run"] == rep)
    first = runs[0]["scores"]
    gm = graph_measures(runs[0]["orig"])
    # Older runs did not record sentence counts; for the passages the text is beside the results.
    text_file = {"lambda": "lambda-logging.txt", "s3public": "s3-public-access.txt"}.get(path.stem.replace("-runs", ""))
    if text_file and not (first.get("granularity") or {}).get("n_sentences") and (results / text_file).exists():
        sents = [x for x in re.split(r"(?<=[.!?])\s+|\n{2,}", (results / text_file).read_text(encoding="utf-8")) if x.strip()]
        gm["n_sentences"] = len(sents); gm["atoms_per_sentence"] = gm["n_atoms"] / len(sents) if sents else None
    out = {
        "source": source,
        "atomizer": first.get("atomizer", "sentence"),
        "granularity": {**gm, **{k: v for k, v in (first.get("granularity") or {}).items()}},
        "load": {**{k: gm[k] for k in ("edges_per_atom", "widest_requires", "longest_chain", "load")}, **{k: v for k, v in (first.get("load") or {}).items()}},
        "n_runs": len(rows),
        "runs": rows,
        "spread": {k: spread(x[k] for x in rows) for k in SCORES},
        "probes_total": [sum(x["n_correct"] or 0 for x in rows), sum(x["n_probes"] for x in rows)],
        "structural_applicable": any(x["structural"].get("applicable") for x in rows),
        "procedure_votes": [x["structural"].get("procedure_votes") for x in rows],
        "structural_spread": {k: spread(x["structural"].get(k) for x in rows) for k in STRUCT
                              if any(x["structural"].get(k) is not None for x in rows)},
        "shown": rep,
        "orig": shown["orig"],
        "regen": shown["regen"],
        "regen_doc": shown["regen_doc"],
        "probes": shown.get("probes"),
        "structural_reports": shown.get("structural_reports"),
    }
    repaired = [x["repair"] for x in rows if x.get("repair")]
    if repaired:
        out["repair"] = {
            "n_runs": len(repaired),
            "spread": {k: spread(x[k] for x in repaired) for k in SCORES},
            "probes_total": [sum(x["n_correct"] or 0 for x in repaired), sum(x["n_probes"] for x in repaired)],
            "structural_spread": {k: spread(x["structural"].get(k) for x in repaired) for k in STRUCT
                                  if any(x["structural"].get(k) is not None for x in repaired)},
            "shown": shown["repair"] and {k: v for k, v in shown["repair"].items() if k not in ("regen",)},
        }
    return out


def page_names():
    """slug -> "Service · title" from the detour set, for pages the page script has no name for."""
    try:
        rows = json.loads((root / "experiments" / "doc_ordering" / "detours.json").read_text(encoding="utf-8"))["rows"]
    except OSError:
        return {}
    return {r["url"].split("/")[-1].replace(".html", ""): f'{r["service"]} · {r.get("goal", "")[:60]}' for r in rows}


NAMES = page_names()


def cohort(paths):
    """Per-page medians for a set of pages, then rates across the pages."""
    pages = []
    for path in paths:
        runs = json.loads(path.read_text(encoding="utf-8"))
        s = summarise(runs, runs[0].get("source", path.stem), path)
        page = {"slug": path.stem.replace("-runs", ""), "name": NAMES.get(path.stem.replace("-runs", "")), "source": s["source"], "n_runs": s["n_runs"], "shown": s["shown"],
                "atomizer": s["atomizer"], "granularity": s["granularity"], "load": s["load"],
                "n_facts": s["orig"] and len(s["orig"]["facts"]), "n_edges": s["orig"] and len(s["orig"]["edges"]),
                "scores": {k: (v["median"] if v else None) for k, v in s["spread"].items()},
                "probes_total": s["probes_total"], "structural_applicable": s["structural_applicable"],
                "procedure_lost_runs": sum(bool(x["structural"].get("procedure_lost")) for x in s["runs"]),
                "procedure_restored_runs": sum(bool(x["structural"].get("procedure_lost")) and x.get("repair") and not x["repair"]["structural"].get("procedure_lost") for x in s["runs"]),
                "structural": {k: (v["median"] if v else None) for k, v in s["structural_spread"].items()},
                "detours_rose_runs": sum((x["structural"].get("detour_rate_delta") or 0) > 0 for x in s["runs"]),
                "n_prereqs": [x["structural"].get("n_orig_prereqs") for x in s["runs"]][0]}
        if "repair" in s:
            rp = s["repair"]
            page["repair"] = {"scores": {k: (v["median"] if v else None) for k, v in rp["spread"].items()},
                              "probes_total": rp["probes_total"],
                              "structural": {k: (v["median"] if v else None) for k, v in rp["structural_spread"].items()},
                              "fed_back": [len(x["repair"]["fed_back"]["edges"]) if x.get("repair") else None for x in s["runs"]],
                              "fed_back_prereqs": [len(x["repair"]["fed_back"]["prerequisites"]) if x.get("repair") else None for x in s["runs"]]}
        pages.append(page)
    if not pages:
        return None
    n = len(pages)
    applicable = [p for p in pages if p["structural_applicable"]]
    out = {
        "n_pages": n,
        "n_runs": sum(p["n_runs"] for p in pages),
        "pages": pages,
        "across_pages": {k: spread(p["scores"].get(k) for p in pages) for k in SCORES},
        "probes_total": [sum(p["probes_total"][0] for p in pages), sum(p["probes_total"][1] for p in pages)],
        "structural": {
            "pages_applicable": len(applicable),
            "pages_procedure_lost": sum(p["procedure_lost_runs"] > 0 for p in applicable),
            "pages_detours_rose": sum(p["detours_rose_runs"] > 0 for p in applicable),
            "pages_detours_rose_every_run": sum(p["detours_rose_runs"] == p["n_runs"] for p in applicable),
            **{k: spread(p["structural"].get(k) for p in applicable) for k in STRUCT},
        },
    }
    repaired = [p for p in pages if p.get("repair")]
    if repaired:
        out["repair"] = {
            "n_pages": len(repaired),
            "across_pages": {k: spread(p["repair"]["scores"].get(k) for p in repaired) for k in SCORES},
            "probes_total": [sum(p["repair"]["probes_total"][0] for p in repaired), sum(p["repair"]["probes_total"][1] for p in repaired)],
            "structural": {k: spread(p["repair"]["structural"].get(k) for p in repaired if p["structural_applicable"]) for k in STRUCT},
            "pages_drift_fell": sum((p["repair"]["scores"]["drift"] or 1) < (p["scores"]["drift"] or 0) for p in repaired),
            "pages_cbr_rose": sum((p["repair"]["scores"]["cbr"] or 0) > (p["scores"]["cbr"] or 0) for p in repaired),
            "pages_tbr_rose": sum((p["repair"]["scores"].get("tbr") or 0) > (p["scores"].get("tbr") or 0) for p in repaired if p["scores"].get("tbr") is not None),
            "pages_tbr_scored": sum(p["scores"].get("tbr") is not None for p in repaired),
            "pages_detours_fell": sum((p["repair"]["structural"].get("regen_detour_rate") or 0) < (p["structural"].get("regen_detour_rate") or 0)
                                      for p in repaired if p["structural_applicable"]),
            "pages_procedure_restored": sum(p["procedure_restored_runs"] > 0 for p in repaired),
            "pages_procedure_lost_after_repair": sum(p["procedure_lost_runs"] > 0 and p["procedure_restored_runs"] < p["procedure_lost_runs"] for p in repaired),
        }
    return out


def correlate(sources):
    """Does conditional load predict what the round trip loses? One point per source."""
    pts = [{"name": n, "load": s["load"]["load"], "folded": s["granularity"]["folded"], "drift": s["spread"]["drift"]["median"],
            "cbr_loss": 1 - s["spread"]["cbr"]["median"] if s["spread"]["cbr"] else None,
            "tbr_loss": 1 - s["spread"]["tbr"]["median"] if s["spread"].get("tbr") else None,
            "probe_loss": 1 - s["probes_total"][0] / s["probes_total"][1] if s["probes_total"][1] else None,
            "structural_loss": s["structural_spread"].get("structural_loss", {}).get("median") if s["structural_applicable"] else None}
           for n, s in sources.items()]
    out = {"points": pts}
    for y in ("drift", "cbr_loss", "tbr_loss", "probe_loss", "structural_loss"):
        pairs = [(p["load"], p[y]) for p in pts if p["load"] is not None and p[y] is not None]
        out[y] = spearman_ci([a for a, _ in pairs], [b for _, b in pairs]) if len(pairs) >= 4 else None
    return out


def instrument_agreement(folder):
    """Across every scored run under a folder: rank correlations between graph bridge recall, text bridge
    recall and the probe rate, first drafts and repairs pooled. Says which instruments agree."""
    xs = []
    for path in sorted(list(folder.glob("*-runs.json")) + list((folder / "cohort").glob("*-runs.json")) + list((folder / "holdout").glob("*-runs.json"))):
        for r in json.loads(path.read_text(encoding="utf-8")):
            for s in (r["scores"], r.get("repair")):
                if s and s.get("text_bridge") and s["text_bridge"].get("rate") is not None:
                    xs.append((s["cond_bridge_recall"].get("cbr"), s["text_bridge"]["rate"], s["entailment"].get("rate")))
    xs = [x for x in xs if all(v is not None and v == v for v in x)]
    if len(xs) < 4:
        return None
    return {"n": len(xs),
            "graph_vs_text": spearman_ci([a for a, _, _ in xs], [b for _, b, _ in xs]),
            "text_vs_probes": spearman_ci([b for _, b, _ in xs], [c for _, _, c in xs]),
            "graph_vs_probes": spearman_ci([a for a, _, _ in xs], [c for _, _, c in xs]),
            "repair_delta": None}


def build(folder):
    """Scenarios and cohort for one atomizer's results folder."""
    o = {"scenarios": {}}
    for path in sorted(folder.glob("*-runs.json")):
        name = path.stem.replace("-runs", "")
        runs = json.loads(path.read_text(encoding="utf-8"))
        s = o["scenarios"][name] = summarise(runs, runs[0].get("source", name), path)
        print(f"{folder.name}/{name}: {s['n_runs']} runs, {s['atomizer']} atomizer, folded {s['granularity']['folded']:.2f}, load {s['load']['load']:.2f}, "
              f"drift median {s['spread']['drift']['median']}, f1 {s['spread']['f1']['median']}, cbr {s['spread']['cbr']['median']}, "
              f"probes {s['probes_total'][0]}/{s['probes_total'][1]}, text bridge {s['spread'].get('tbr') and s['spread']['tbr']['median']}, structural {'yes' if s['structural_applicable'] else 'no'}"
              + (f", repaired drift {s['repair']['spread']['drift']['median']}" if "repair" in s else ""))
    for name in ("cohort", "holdout"):
        co = cohort(sorted((folder / name).glob("*-runs.json"))) if (folder / name).exists() else None
        if not co:
            continue
        o[name] = co
        st = co["structural"]
        print(f"{folder.name}/{name}: {co['n_pages']} pages, {co['n_runs']} runs; drift median {co['across_pages']['drift']['median']}; "
              f"detours rose on {st['pages_detours_rose']} (every run on {st['pages_detours_rose_every_run']}), procedure lost on {st['pages_procedure_lost']}"
              + (f"; repair: drift fell on {co['repair']['pages_drift_fell']}, cbr rose on {co['repair']['pages_cbr_rose']}, tbr rose on {co['repair'].get('pages_tbr_rose')}, detours fell on {co['repair']['pages_detours_fell']}" if "repair" in co else ""))
    # every source as one point, for the load correlation: the singles plus each cohort page
    everything = dict(o["scenarios"])
    for path in sorted(list((folder / "cohort").glob("*-runs.json")) + list((folder / "holdout").glob("*-runs.json"))):
        runs = json.loads(path.read_text(encoding="utf-8"))
        everything[path.stem.replace("-runs", "")] = summarise(runs, runs[0].get("source", path.stem), path)
    o["load_correlation"] = correlate(everything)
    o["instruments"] = instrument_agreement(folder)
    if o["instruments"]:
        print(f"{folder.name}/instruments: n={o['instruments']['n']} graph~text {o['instruments']['graph_vs_text']}, text~probes {o['instruments']['text_vs_probes']}, graph~probes {o['instruments']['graph_vs_probes']}")
    lc = o["load_correlation"]
    print(f"{folder.name}/load: {len(lc['points'])} sources; rho(load, drift) {lc['drift']}, rho(load, cbr loss) {lc['cbr_loss']}, rho(load, structural loss) {lc['structural_loss']}")
    return o


out = build(results)
if (results / "strict").exists():
    out["strict"] = build(results / "strict")
dest.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"wrote {dest} ({dest.stat().st_size / 1024:.0f} KB)")
