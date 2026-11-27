"""Add text bridge recall to recorded runs, in place (this is how results/ got its text_bridge fields): for each run, score the first draft and the repaired draft
from the stored original graph and texts. Skips runs that already have it.

    python -I experiments/doc_drift/rescore_bridge.py <runs.json> [<runs.json> ...]
"""
import asyncio, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from semantic_ruler.drift import bridge_entailment
from semantic_ruler.factgraph import EdgeKind, FactAtom, FactEdge, FactGraph
from semantic_ruler.llm_plain import PlainTextProvider
from spiritwriter.secrets import configure
from semantic_ruler.embed import KEYCHAIN_SERVICE

MODEL = "claude-sonnet-5-5"
configure(service_name=KEYCHAIN_SERVICE)
llm = PlainTextProvider(MODEL)


def graph(d):
    return FactGraph(facts=[FactAtom(text=f["text"], fact_id=f["id"]) for f in d["facts"]],
                     edges=[FactEdge(kind=EdgeKind(e["kind"]), antecedents=e["antecedents"], consequent=e["consequent"], confidence=e["confidence"]) for e in d["edges"]])


for arg in sys.argv[1:]:
    path = Path(arg)
    runs = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for r in runs:
        g = graph(r["orig"])
        if "text_bridge" not in r["scores"]:
            r["scores"]["text_bridge"] = asyncio.run(bridge_entailment(g, r["regen_doc"], llm, model=MODEL)); changed = True
        if r.get("repair") and "text_bridge" not in r["repair"]:
            r["repair"]["text_bridge"] = asyncio.run(bridge_entailment(g, r["repair"]["regen_doc"], llm, model=MODEL)); changed = True
        tb, rb = r["scores"].get("text_bridge", {}), (r.get("repair") or {}).get("text_bridge", {})
        print(f"{path.name[:40]:40s} run {r['run']}: text bridge {tb.get('rate', float('nan')):.2f} (graph {r['scores']['cond_bridge_recall'].get('cbr', float('nan')):.2f})"
              + (f" -> repaired {rb.get('rate', float('nan')):.2f} (graph {r['repair']['cond_bridge_recall'].get('cbr', float('nan')):.2f})" if rb else ""), flush=True)
        if changed:
            path.write_text(json.dumps(runs, indent=1, ensure_ascii=False), encoding="utf-8")
print("rescored")
