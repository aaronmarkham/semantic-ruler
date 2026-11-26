"""Semantic drift: how much is lost when docs are distilled to a fact store and regenerated?

Pipeline for a single page:
  1. Fetch or read the source text
  2. LLM extracts a FactGraph (atoms + conditional edges)        [G_orig]
  3. Regenerate prose from the fact atoms only (edges discarded) [regen_doc]
  4. Re-extract a FactGraph from the regenerated prose           [G_regen]
  5. Score drift:
     - Micro: atomic F1, conditional bridge recall, entailment probes
     - Macro: structural drift (prerequisite recall, detour rate delta) — procedure pages only

Usage:
    python -I experiments/doc_drift/run.py <url-or-file>
    python -I experiments/doc_drift/run.py <url-or-file> --mock
    python -I experiments/doc_drift/run.py <url-or-file> --json
    python -I experiments/doc_drift/run.py <url-or-file> --no-structural
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html as html_lib
import json
import re
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Package bootstrap: load semantic_ruler without requiring an installed editable
# wheel.  Run with `python -I` from the repo root so this path resolution works.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from semantic_ruler import CachedEmbedder, HashEmbedder, OpenAIEmbedder
from semantic_ruler.drift import (
    analyse_structural,
    atomic_f1,
    cond_bridge_recall,
    drift_score,
    entailment_rate,
    structural_drift,
)
from semantic_ruler.embed import KEYCHAIN_SERVICE
from semantic_ruler.factgraph import FactGraph, extract_facts, regenerate_doc
from semantic_ruler import store as shards
from spiritwriter.fabric.store import ShardStore


def _fetch(url_or_path: str, cache_dir: Path) -> tuple[str, bool]:
    """Return (body, is_html).

    For URLs: returns raw HTML (cached). For local files: returns content as-is
    and detects HTML by the presence of an <html> tag.
    """
    p = Path(url_or_path)
    if p.exists():
        body = p.read_text(encoding="utf-8")
        is_html = "<html" in body[:500].lower()
        return body, is_html

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url_or_path.encode()).hexdigest()[:16]
    cached = cache_dir / f"{key}.html"
    if cached.exists():
        return cached.read_text(encoding="utf-8"), True
    with urllib.request.urlopen(url_or_path, timeout=30) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    cached.write_text(body, encoding="utf-8")
    return body, True


def _html_to_text(body: str) -> str:
    text = html_lib.unescape(re.sub(r"<[^>]+>", " ", body))
    return re.sub(r"\s{3,}", "\n\n", text).strip()


async def run_drift(
    source_body: str,
    source_url: str,
    llm,
    embedder,
    shard_store: ShardStore,
    *,
    is_html: bool = False,
    model: str | None,
    n_probes: int,
    regenerate: bool,
    run_structural: bool = True,
    log=print,
) -> dict:
    source_text = _html_to_text(source_body) if is_html else source_body
    source_hash = hashlib.sha256(source_text.encode()).hexdigest()[:24]

    # 1. Extract orig graph (cached unless --regenerate) ----------------------
    orig_graph = None if regenerate else shards.load_fact_graph(shard_store, f"orig:{source_hash}")
    if orig_graph is None:
        log("Extracting facts from source...")
        orig_graph = await extract_facts(llm, source_text, model)
        shards.save_fact_graph(shard_store, f"orig:{source_hash}", orig_graph, label="orig")
        log(f"  {len(orig_graph.facts)} facts, {len(orig_graph.edges)} edges")
    else:
        log(f"Loaded orig graph from store: {len(orig_graph.facts)} facts, {len(orig_graph.edges)} edges")

    # 2. Regenerate prose from atoms only (edges deliberately not passed) -----
    atoms_only = FactGraph(facts=orig_graph.facts, edges=[])
    log("Regenerating doc from fact atoms...")
    regen_doc = await regenerate_doc(llm, atoms_only, model)

    # 3. Re-extract from regen ------------------------------------------------
    log("Re-extracting facts from regenerated doc...")
    regen_graph = await extract_facts(llm, regen_doc, model)
    shards.save_fact_graph(shard_store, f"regen:{source_hash}", regen_graph, label="regen")
    log(f"  {len(regen_graph.facts)} facts, {len(regen_graph.edges)} edges")

    # 4. Micro drift metrics --------------------------------------------------
    log("Scoring drift...")
    a = atomic_f1(orig_graph, regen_graph, embedder)
    c = cond_bridge_recall(orig_graph, regen_graph, embedder)
    e = await entailment_rate(orig_graph, regen_doc, llm, n_probes=n_probes, model=model)

    # 5. Structural drift (procedure pages only) ------------------------------
    s: dict = {"applicable": False}
    if run_structural:
        log("Running structural analysis...")
        orig_struct = await analyse_structural(llm, source_body, url=source_url, is_html=is_html)
        if orig_struct and orig_struct.is_procedure:
            log(f"  procedure page: {len(orig_struct.prerequisites)} prerequisites, "
                f"{len(orig_struct.detours)} detours in orig")
            regen_struct = await analyse_structural(llm, regen_doc, title="Regenerated Documentation")
            if regen_struct:
                s = structural_drift(orig_struct, regen_struct)
                log(f"  regen: {len(regen_struct.prerequisites)} prerequisites, "
                    f"{len(regen_struct.detours)} detours")
        else:
            log("  not a procedure page — structural layer skipped")

    d = drift_score(a, c, e, s if s.get("applicable") else None)

    return {
        "source_hash": source_hash,
        "orig": {"n_facts": len(orig_graph.facts), "n_edges": len(orig_graph.edges)},
        "regen": {"n_facts": len(regen_graph.facts), "n_edges": len(regen_graph.edges)},
        "atomic_f1": a,
        "cond_bridge_recall": c,
        "entailment": e,
        "structural": s,
        "drift": d,
        "regen_doc": regen_doc,
    }


def _fmt(result: dict) -> str:
    d = result["drift"]
    a = result["atomic_f1"]
    c = result["cond_bridge_recall"]
    e = result["entailment"]
    s = result.get("structural", {})

    def pct(v):
        try:
            f = float(v)
            return "n/a" if f != f else f"{f:.1%}"
        except (TypeError, ValueError):
            return "n/a"

    def delta(v):
        try:
            f = float(v)
            return "n/a" if f != f else f"{f:+.1%}"
        except (TypeError, ValueError):
            return "n/a"

    lines = [
        "",
        f"Source:  {result['source_hash']}",
        f"  orig   {int(result['orig']['n_facts'])} facts, {int(result['orig']['n_edges'])} edges",
        f"  regen  {int(result['regen']['n_facts'])} facts, {int(result['regen']['n_edges'])} edges",
        "",
        f"Atomic F1 .............. {pct(a.get('f1'))}",
        f"  recall                 {pct(a.get('recall'))}",
        f"  precision              {pct(a.get('precision'))}",
        "",
        f"Cond. bridge recall .... {pct(c.get('cbr'))}  ({int(c.get('n_edges', 0))} edges)",
    ]
    for kind, v in sorted(c.get("by_kind", {}).items()):
        lines.append(f"  {kind:<20} {pct(v)}")

    lines += [
        "",
        f"Entailment probes ...... {pct(e.get('rate'))}  ({int(e.get('n_probes', 0))} probes)",
    ]

    if s.get("applicable"):
        lines += ["", "Structural (procedure)"]
        if s.get("procedure_lost"):
            lines.append("  regen no longer identified as a procedure  [complete structural loss]")
        else:
            lines += [
                f"  prereq recall          {pct(s.get('prereq_recall'))}  "
                f"({int(s.get('n_orig_prereqs', 0))} → {int(s.get('n_regen_prereqs', 0))})",
                f"  detour rate            orig {pct(s.get('orig_detour_rate'))}  "
                f"regen {pct(s.get('regen_detour_rate'))}  "
                f"delta {delta(s.get('detour_rate_delta'))}",
                f"  intro position MAE     {pct(s.get('intro_position_mae'))}  (normalized by n_blocks)",
            ]
    else:
        lines.append("\nStructural ............. n/a (not a procedure page)")

    lines += [
        "",
        f"Drift score ............ {pct(d.get('drift'))}  (0=perfect, 1=complete loss)",
    ]
    for k in ("atomic_loss", "cbr_loss", "entailment_loss", "structural_loss"):
        if k in d:
            lines.append(f"  {k:<20} {pct(d[k])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="doc-drift",
        description="Measure semantic drift through a doc → fact store → regen round-trip.",
    )
    ap.add_argument("source", help="URL or local file path of the documentation")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--embedding-model", default="text-embedding-3-large")
    ap.add_argument("--probes", type=int, default=20, help="entailment probes to generate")
    ap.add_argument("--store", type=Path, default=Path("data/drift-store"))
    ap.add_argument("--cache", type=Path, default=Path(".cache/pages"))
    ap.add_argument("--regenerate", action="store_true", help="ignore cached graphs")
    ap.add_argument("--no-structural", action="store_true", help="skip the structural layer")
    ap.add_argument("--mock", action="store_true", help="offline: no LLM or embedder calls")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args(argv)

    source_body, is_html = _fetch(args.source, args.cache)
    source_url = args.source if args.source.startswith("http") else f"file://{args.source}"

    if args.mock:
        from semantic_ruler.mock import mock_generator
        llm = mock_generator()
        model = None
        embedder = HashEmbedder()
        store_dir = args.store.parent / "mock-drift-store"
    else:
        from spiritwriter.secrets import configure
        from semantic_ruler.llm_plain import PlainTextProvider

        configure(service_name=KEYCHAIN_SERVICE)
        llm = PlainTextProvider(args.model)
        model = args.model
        embedder = CachedEmbedder(OpenAIEmbedder(args.embedding_model), ".cache/embeddings")
        store_dir = args.store

    result = asyncio.run(run_drift(
        source_body,
        source_url,
        llm,
        embedder,
        ShardStore(store_dir),
        is_html=is_html,
        model=model if not args.mock else None,
        n_probes=args.probes,
        regenerate=args.regenerate,
        run_structural=not args.no_structural,
    ))

    if args.as_json:
        out = {k: v for k, v in result.items() if k != "regen_doc"}
        print(json.dumps(out, indent=2))
    else:
        print(_fmt(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
