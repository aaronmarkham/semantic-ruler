"""Re-grade the probes of recorded runs, in place, keeping the replies.

Earlier grading read the reply's prefix, so "Not enough information" passed as a no, and the
replies were not kept. This puts each run's stored question set to the first draft and to the
repaired draft again, grades each reply by its first word (see grade_probe_reply), keeps the
replies, and recomputes the drift composite. The stored set is the last one the run generated,
which is the repaired draft's when the run had a repair pass; using one set for both drafts
also makes the before/after comparison paired. Skips runs already re-graded.

    python -I experiments/doc_drift/regrade_probes.py <runs.json> [<runs.json> ...]
"""
import asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from semantic_ruler.drift import PROBE_PROMPT, drift_score, grade_probe_reply
from semantic_ruler.llm_plain import PlainTextProvider
from spiritwriter.secrets import configure
from semantic_ruler.embed import KEYCHAIN_SERVICE

MODEL = "claude-sonnet-5-5"
configure(service_name=KEYCHAIN_SERVICE)
llm = PlainTextProvider(MODEL)


async def grade(doc, probes):
    correct, n_valid, replies = 0, 0, []
    for probe in probes:
        q, expected = probe.get("q", ""), str(probe.get("answer", "")).strip().lower()
        if not q or expected not in ("yes", "no"):
            replies.append(None)
            continue
        n_valid += 1
        reply = await llm.query(PROBE_PROMPT.format(doc=doc, question=q), model=MODEL)
        replies.append(reply.strip()[:120])
        if grade_probe_reply(reply) == expected:
            correct += 1
    if not n_valid:
        return {"rate": float("nan"), "n_probes": 0, "n_generated": len(probes), "replies": replies, "regraded": True}
    return {"rate": correct / n_valid, "n_probes": n_valid, "n_correct": correct, "n_generated": len(probes),
            "expected": [str(p.get("answer", "")).strip().lower() for p in probes], "replies": replies, "regraded": True}


for arg in sys.argv[1:]:
    path = Path(arg)
    runs = json.loads(path.read_text(encoding="utf-8"))
    for r in runs:
        probes = r.get("probes") or []
        if not probes:
            continue
        for key, doc in (("scores", r["regen_doc"]), ("repair", (r.get("repair") or {}).get("regen_doc"))):
            s = r.get(key)
            if not s or doc is None or s.get("entailment", {}).get("regraded"):
                continue
            old = s["entailment"].get("rate")
            s["entailment_before_regrade"] = s["entailment"]
            s["entailment"] = asyncio.run(grade(doc, probes))
            st = s.get("structural") or {}
            s["drift"] = drift_score(s["atomic_f1"], s["cond_bridge_recall"], s["entailment"], st if st.get("applicable") else None)
            print(f"{path.name[:40]:40s} run {r['run']} {key:6s}: probes {old} -> {s['entailment'].get('rate')} ({s['entailment'].get('n_correct')}/{s['entailment'].get('n_probes')})", flush=True)
        path.write_text(json.dumps(runs, indent=1, ensure_ascii=False), encoding="utf-8")
print("regraded")
