"""Semantic drift metrics for doc → fact store → regeneration round-trips.

Three complementary metrics:

  atomic_f1          fraction of original facts matched in regen by embedding similarity
  cond_bridge_recall fraction of conditional edges from orig that survive in regen
  entailment_rate    fraction of multi-fact probes answered correctly from regen doc
  drift_score        weighted composite (0 = perfect preservation, 1 = complete loss)

CBR carries double weight because it directly captures compositional loss —
the structural failure mode that atomic F1 and entailment probes can both
undercount when a regen doc paraphrases facts correctly but drops their
logical connectives.
"""

from __future__ import annotations

import html as _html_lib
import json
import math
import re

import numpy as np

from .embed import Embedder
from .factgraph import EdgeKind, FactGraph


PROBE_GEN_PROMPT = """\
Generate {n} yes/no questions that require knowing multiple facts simultaneously.
Each question must be unanswerable from any single fact alone — it needs the
combination to give a definite yes or no answer.

Facts:
{facts}

Conditional relationships:
{edges}

Respond with JSON only:
{{"questions": [{{"q": "...", "answer": "yes"}}]}}
"""

PROBE_PROMPT = """\
Answer using ONLY the information in the document below.
If the document does not contain enough to answer, reply "unknown".

Document:
{doc}

Question: {question}
Answer yes, no, or unknown:"""


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity: (m, d) × (n, d) → (m, n)."""
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-9)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-9)
    return a_n @ b_n.T


def atomic_f1(
    orig: FactGraph,
    regen: FactGraph,
    embedder: Embedder,
    threshold: float = 0.85,
) -> dict[str, float]:
    """Embedding-based atomic fact recall/precision/F1.

    A regen fact "matches" an orig fact when cosine similarity >= threshold.
    Recall measures coverage; precision measures hallucination rate.
    """
    result: dict[str, float] = {
        "n_orig": float(len(orig.facts)),
        "n_regen": float(len(regen.facts)),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    if not orig.facts or not regen.facts:
        return result
    sim = _cosine_sim(
        embedder.embed([f.text for f in orig.facts]),
        embedder.embed([f.text for f in regen.facts]),
    )
    result["recall"] = float(np.mean(sim.max(axis=1) >= threshold))
    result["precision"] = float(np.mean(sim.max(axis=0) >= threshold))
    p, r = result["precision"], result["recall"]
    result["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return result


def cond_bridge_recall(
    orig: FactGraph,
    regen: FactGraph,
    embedder: Embedder,
    threshold: float = 0.85,
) -> dict[str, float]:
    """Fraction of orig conditional edges whose structure survives in regen.

    Facts are matched across graphs by embedding similarity. An edge is
    "preserved" when regen has an edge of the same kind where every antecedent
    and the consequent map to matched regen facts. Scores are weighted by the
    edge's extraction confidence so low-confidence edges don't dominate CBR.
    """
    if not orig.edges:
        return {"cbr": float("nan"), "n_edges": 0, "by_kind": {}}
    if not orig.facts or not regen.facts:
        return {"cbr": 0.0, "n_edges": len(orig.edges), "by_kind": {}}

    orig_ids = [f.fact_id for f in orig.facts]
    regen_ids = [f.fact_id for f in regen.facts]
    sim = _cosine_sim(
        embedder.embed([f.text for f in orig.facts]),
        embedder.embed([f.text for f in regen.facts]),
    )

    # Map each orig fact_id to its best-matching regen fact_id (None if below threshold)
    orig_to_regen: dict[str, str | None] = {}
    for i, oid in enumerate(orig_ids):
        j = int(np.argmax(sim[i]))
        orig_to_regen[oid] = regen_ids[j] if sim[i, j] >= threshold else None

    # Index of edges present in the regen graph, expressed in regen fact IDs
    regen_keys: set[tuple] = {
        (e.kind, frozenset(e.antecedents), e.consequent) for e in regen.edges
    }

    by_kind: dict[str, list[float]] = {}
    for edge in orig.edges:
        mapped_ants = [orig_to_regen.get(a) for a in edge.antecedents]
        mapped_con = orig_to_regen.get(edge.consequent)
        preserved = (
            all(a is not None for a in mapped_ants)
            and mapped_con is not None
            and (edge.kind, frozenset(mapped_ants), mapped_con) in regen_keys  # type: ignore[arg-type]
        )
        by_kind.setdefault(edge.kind.value, []).append(edge.confidence if preserved else 0.0)

    total_conf = sum(e.confidence for e in orig.edges)
    all_weighted = [s for scores in by_kind.values() for s in scores]
    cbr = sum(all_weighted) / total_conf if total_conf > 0 else float("nan")
    return {
        "cbr": cbr,
        "n_edges": float(len(orig.edges)),
        "by_kind": {k: float(np.mean(v)) for k, v in by_kind.items()},
    }


async def entailment_rate(
    orig: FactGraph,
    regen_doc: str,
    llm,
    n_probes: int = 20,
    model: str | None = None,
) -> dict[str, float]:
    """Generate multi-fact probes from orig_graph, test them against regen_doc.

    Probes are generated from REQUIRES edges (2+ antecedents) because those are
    the conditional bridges most at risk of being lost in distillation. Falls back
    to all edges when no REQUIRES edges exist.
    """
    if not orig.edges:
        return {"rate": float("nan"), "n_probes": 0}

    target_edges = [
        e for e in orig.edges if e.kind == EdgeKind.REQUIRES and len(e.antecedents) >= 2
    ] or orig.edges

    facts_text = "\n".join(f"[{f.fact_id}] {f.text}" for f in orig.facts)
    edges_text = "\n".join(
        f"{e.kind.value}: [{', '.join(e.antecedents)}] → {e.consequent}"
        for e in target_edges[:20]
    )
    kwargs = {"model": model} if model else {}
    gen_reply = await llm.query(
        PROBE_GEN_PROMPT.format(
            n=min(n_probes, max(len(target_edges) * 2, 5)),
            facts=facts_text,
            edges=edges_text,
        ),
        **kwargs,
    )
    try:
        start = gen_reply.find("{")
        probes = json.JSONDecoder().raw_decode(gen_reply, start)[0].get("questions", [])
    except (ValueError, json.JSONDecodeError):
        return {"rate": float("nan"), "n_probes": 0, "error": "probe generation failed"}

    if not probes:
        return {"rate": float("nan"), "n_probes": 0}

    correct = 0
    for probe in probes:
        q = probe.get("q", "")
        expected = probe.get("answer", "").strip().lower()
        if not q or expected not in ("yes", "no"):
            continue
        reply = (await llm.query(PROBE_PROMPT.format(doc=regen_doc, question=q), **kwargs)).strip().lower()
        if reply.startswith(expected) or f" {expected}" in reply[:20]:
            correct += 1

    return {"rate": correct / len(probes), "n_probes": len(probes), "n_correct": correct}


def drift_score(
    atomic: dict[str, float],
    cbr: dict[str, float],
    entailment: dict[str, float],
    structural: dict | None = None,
    *,
    w_atomic: float = 0.25,
    w_cbr: float = 0.50,
    w_entailment: float = 0.25,
    w_structural: float = 0.20,
) -> dict[str, float]:
    """Weighted composite drift score. 0 = perfect preservation, 1 = complete loss.

    structural is optional and only included when applicable (procedure pages only).
    When present it slightly dilutes the other weights via renormalization, so the
    absolute values of w_* do not need to sum to 1.
    """
    losses: dict[str, tuple[float, float]] = {}

    f1 = atomic.get("f1")
    if f1 is not None and not math.isnan(f1):
        losses["atomic_loss"] = (1.0 - f1, w_atomic)

    cbr_val = cbr.get("cbr")
    if cbr_val is not None and not math.isnan(cbr_val):
        losses["cbr_loss"] = (1.0 - cbr_val, w_cbr)

    ent = entailment.get("rate")
    if ent is not None and not math.isnan(ent):
        losses["entailment_loss"] = (1.0 - ent, w_entailment)

    if structural and structural.get("applicable"):
        s_val = structural.get("structural_loss")
        if s_val is not None and not math.isnan(float(s_val)):
            losses["structural_loss"] = (float(s_val), w_structural)

    if not losses:
        return {"drift": float("nan")}

    total_w = sum(w for _, w in losses.values())
    drift = sum(v * w for v, w in losses.values()) / total_w
    return {"drift": drift, **{k: v for k, (v, _) in losses.items()}}


# --- structural layer (prerequisite ordering, detour fidelity) ---------------


def _wrap_as_html(text: str) -> str:
    """Minimal HTML wrapper so ordering.parse() can consume plain text."""
    paras = "\n".join(
        f"<p>{_html_lib.escape(p.strip())}</p>"
        for p in re.split(r"\n{2,}", text)
        if p.strip()
    )
    return f"<main>\n{paras}\n</main>"


async def analyse_structural(llm, body: str, url: str = "regen://local", is_html: bool = False, title: str | None = None):
    """Run ordering.analyse() on either HTML or plain text. Returns PageReport or None on error."""
    from .ordering import analyse
    try:
        return await analyse(llm, body if is_html else _wrap_as_html(body), url, title=title)
    except Exception:
        return None


def structural_drift(orig, regen) -> dict:
    """Compare prerequisite structure between original and regenerated PageReports.

    Only meaningful when orig.is_procedure is True. The key signal is
    detour_rate_delta: how much more detour-prone the regen doc became (positive = worse).
    Prerequisites are matched across docs by word-overlap (Jaccard >= 0.4) since the
    regen LLM may paraphrase prerequisite names.

    Returns structural_loss in [0, 1] for use as an optional term in drift_score().
    """
    if not orig.is_procedure:
        return {"applicable": False, "structural_loss": float("nan")}

    orig_det = _detour_rate(orig)
    regen_det = _detour_rate(regen)

    if not regen.is_procedure:
        return {
            "applicable": True,
            "procedure_lost": True,
            "n_orig_prereqs": len(orig.prerequisites),
            "n_regen_prereqs": len(regen.prerequisites),
            "prereq_recall": 0.0,
            "orig_detour_rate": orig_det,
            "regen_detour_rate": float("nan"),
            "detour_rate_delta": float("nan"),
            "intro_position_mae": float("nan"),
            "structural_loss": 1.0,
        }

    # Greedy best-match orig → regen by Jaccard on word tokens
    regen_avail = list(range(len(regen.prerequisites)))
    matched: list[tuple[dict, dict]] = []
    for oq in orig.prerequisites:
        if not regen_avail:
            break
        best_score, best_i = max(
            (_jaccard(oq["what"], regen.prerequisites[i]["what"]), i) for i in regen_avail
        )
        if best_score >= 0.4:
            matched.append((oq, regen.prerequisites[best_i]))
            regen_avail.remove(best_i)

    prereq_recall = len(matched) / len(orig.prerequisites) if orig.prerequisites else float("nan")
    detour_delta = (regen_det - orig_det) if not (math.isnan(regen_det) or math.isnan(orig_det)) else float("nan")

    n_blocks = max(orig.n_blocks, 1)
    pos_errors = [
        abs((oq.get("introduced_at_block") or 0) - (rq.get("introduced_at_block") or 0)) / n_blocks
        for oq, rq in matched
        if oq.get("introduced_at_block") is not None and rq.get("introduced_at_block") is not None
    ]
    intro_mae = float(np.mean(pos_errors)) if pos_errors else float("nan")

    pr_loss = 1.0 - prereq_recall if not math.isnan(prereq_recall) else 1.0
    dr_loss = min(1.0, max(0.0, detour_delta)) if not math.isnan(detour_delta) else 0.0
    structural_loss = 0.6 * pr_loss + 0.4 * dr_loss

    return {
        "applicable": True,
        "procedure_lost": False,
        "n_orig_prereqs": len(orig.prerequisites),
        "n_regen_prereqs": len(regen.prerequisites),
        "prereq_recall": prereq_recall,
        "orig_detour_rate": orig_det,
        "regen_detour_rate": regen_det,
        "detour_rate_delta": detour_delta,
        "intro_position_mae": intro_mae,
        "structural_loss": structural_loss,
    }


def _detour_rate(report) -> float:
    if not report.prerequisites:
        return 0.0
    return len(report.detours) / len(report.prerequisites)


def _jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
