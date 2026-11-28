"""Run the drift round-trip several times on one source and record everything: the graphs,
the regenerated text, the probes, every score, the structural reports and the repair pass.
This is what produced experiments/doc_drift/results/.

    python -I experiments/doc_drift/record.py <url-or-file> <n_runs> <out.json> [repair|repair-all]

Environment: ATOMIZER=strict for the strict extractor (default sentence); STORE names the
shard-store folder under .cache/ (default drift-store; use another for another atomizer).
Needs a Claude model for extraction and an OpenAI key for embeddings, like run.py.
"""
import asyncio, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "doc_drift"))
import run as R  # the pipeline module
from semantic_ruler import CachedEmbedder, OpenAIEmbedder
from semantic_ruler.embed import KEYCHAIN_SERVICE
from semantic_ruler.llm_plain import PlainTextProvider
from spiritwriter.fabric.store import ShardStore
from spiritwriter.secrets import configure
from semantic_ruler import store as shards

SOURCE, N, OUT = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
REPAIR = len(sys.argv) > 4 and sys.argv[4].startswith("repair")
SCOPE = "all" if len(sys.argv) > 4 and sys.argv[4] == "repair-all" else "lost"
MODEL = "claude-sonnet-5-5"
configure(service_name=KEYCHAIN_SERVICE)


class Logged(PlainTextProvider):
    def __init__(self, model):
        super().__init__(model); self.log = []
    async def query(self, prompt, **kw):
        out = await super().query(prompt, **kw)
        text = out[0] if isinstance(out, tuple) else out
        self.log.append({"prompt": prompt, "reply": text})
        return out


graphs, structs = {}, []
_extract, _struct = R.extract_facts, R.analyse_structural
async def capturing_extract(llm, text, model=None, **kw):
    g = await _extract(llm, text, model, **kw); graphs[text] = g; return g
async def capturing_struct(llm, body, **kw):
    s = await _struct(llm, body, **kw); structs.append(s); return s
R.extract_facts, R.analyse_structural = capturing_extract, capturing_struct

def dump(g):
    return {"facts": [{"id": f.fact_id, "text": f.text} for f in g.facts],
            "edges": [{"kind": e.kind.value, "antecedents": e.antecedents, "consequent": e.consequent, "confidence": e.confidence} for e in g.edges]}

def dump_struct(s):
    if s is None: return None
    return {"is_procedure": bool(s.is_procedure), "title": getattr(s, "title", None), "goal": getattr(s, "goal", None), "n_blocks": getattr(s, "n_blocks", None),
            "prerequisites": [dict(q) for q in s.prerequisites], "detours": [dict(q) if isinstance(q, dict) else str(q) for q in s.detours]}

body, is_html = R._fetch(SOURCE, ROOT / ".cache" / "pages")
source_url = SOURCE if SOURCE.startswith("http") else f"file://{SOURCE}"
embedder = CachedEmbedder(OpenAIEmbedder("text-embedding-3-large"), str(ROOT / ".cache" / "embeddings"))
store = ShardStore(ROOT / ".cache" / os.environ.get("STORE", "drift-store"))
out = []
for i in range(N):
    llm = Logged(MODEL); graphs.clear(); structs.clear()
    res = asyncio.run(R.run_drift(body, source_url, llm, embedder, store, is_html=is_html, model=MODEL,
                                  n_probes=20, regenerate=False, run_structural=True, repair=REPAIR, repair_scope=SCOPE, atomizer=os.environ.get("ATOMIZER", "sentence"), log=lambda *a: None))
    regen_graph = graphs.get(res["regen_doc"])
    atom = os.environ.get("ATOMIZER", "sentence")
    orig_graph = shards.load_fact_graph(store, (f"orig:{res['source_hash']}" if atom == "sentence" else f"orig-{atom}:{res['source_hash']}"))
    probes = None
    for q in llm.log:
        if '"questions"' in q["reply"]:
            try:
                s = q["reply"].find("{"); probes = json.JSONDecoder().raw_decode(q["reply"], s)[0].get("questions")
            except Exception:
                pass
    # run_drift reads the original several times (majority vote), then the regeneration once per repeat.
    votes = (res["structural"] or {}).get("procedure_votes") or []
    orig_reads, regen_reads = structs[:len(votes)], structs[len(votes):]
    repair = None
    if REPAIR and "repair" in res:
        rp = res["repair"]; rg = graphs.get(rp["regen_doc"])
        repair = {k: v for k, v in rp.items() if k != "regen_doc"}
        repair.update({"regen_doc": rp["regen_doc"], "regen": dump(rg) if rg else None,
                       "structural_report": dump_struct(regen_reads[1]) if len(regen_reads) > 1 else None})
    orig_rep = next((s for s in orig_reads if s and s.is_procedure), None) if sum(votes) * 2 > len(votes) else (orig_reads[0] if orig_reads else None)
    out.append({"run": i, "source": SOURCE, "scores": {k: v for k, v in res.items() if k not in ("regen_doc", "repair")}, "regen_doc": res["regen_doc"],
                "orig": dump(orig_graph), "regen": dump(regen_graph) if regen_graph else None, "probes": probes,
                "structural_reports": [dump_struct(orig_rep), dump_struct(regen_reads[0]) if regen_reads else None],
                "orig_reads": [dump_struct(s) for s in orig_reads], "n_queries": len(llm.log), "repair": repair})
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    a, c, e, d, st = res["atomic_f1"], res["cond_bridge_recall"], res["entailment"], res["drift"], res["structural"]
    print(f"run {i}: orig {res['orig']['n_facts']}f/{res['orig']['n_edges']}e regen {res['regen']['n_facts']}f/{res['regen']['n_edges']}e  "
          f"F1 {a['f1']:.3f} (P {a['precision']:.2f} R {a['recall']:.2f})  CBR {c['cbr']:.2f}  probes {e.get('rate')} ({e.get('n_probes')})  "
          f"drift {d['drift']:.3f}  structural {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()})}", flush=True)
print("done")
