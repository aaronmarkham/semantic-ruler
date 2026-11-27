"""Atomic fact extraction and conditional relationship graph for drift evaluation.

A FactGraph pairs a list of atomic FactAtoms with typed FactEdges that capture
the logical connectives (conditionals, sequences, contradictions) that prose
documentation carries but a flat fact store drops. The graph is the unit of
comparison in the drift pipeline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from spiritwriter.llm import LLMProvider


class EdgeKind(str, Enum):
    IMPLIES = "implies"         # A alone is sufficient for B
    REQUIRES = "requires"       # B is true ONLY when ALL antecedents are true
    CONTRADICTS = "contradicts" # A and B cannot both hold
    SEQUENCE = "sequence"       # B presupposes A (ordering dependency)
    NEGATES = "negates"         # A is the logical negation of B


@dataclass
class FactAtom:
    text: str
    fact_id: str = field(default="")

    def __post_init__(self):
        if not self.fact_id:
            self.fact_id = hashlib.sha256(self.text.encode()).hexdigest()[:16]


@dataclass
class FactEdge:
    kind: EdgeKind
    antecedents: list[str]  # fact_ids
    consequent: str          # fact_id
    confidence: float = 1.0


@dataclass
class FactGraph:
    facts: list[FactAtom] = field(default_factory=list)
    edges: list[FactEdge] = field(default_factory=list)

    def fact_by_id(self) -> dict[str, FactAtom]:
        return {f.fact_id: f for f in self.facts}


EXTRACT_PROMPT = """\
Extract a structured fact graph from the documentation below.

STEP 1 — Atomic facts:
List every distinct factual claim. Each fact must be:
- One complete, self-contained sentence
- Assigned a short stable ID (f01, f02, ...)
- Exact: preserve negations, conditions, quantifiers, and scope word-for-word

STEP 2 — Logical relationships between facts:
For each relationship name the kind:
- implies: antecedent alone is sufficient for the consequent
- requires: consequent is true ONLY when ALL listed antecedents hold (use for 2+ conditions)
- contradicts: antecedent and consequent cannot both hold simultaneously
- sequence: consequent presupposes the antecedent (must happen first)
- negates: antecedent is the direct logical negation of the consequent

Add a confidence score (0.0–1.0). Omit low-confidence (<0.5) edges.

Documentation:
{text}

Respond with JSON only:
{{"facts": [{{"id": "f01", "text": "..."}}], "edges": [{{"kind": "requires", "antecedents": ["f01", "f02"], "consequent": "f03", "confidence": 0.9}}]}}
"""

REGEN_PROMPT = """\
Reconstruct documentation prose from the following facts.

Rules:
- Include every fact — omit nothing
- Do not add facts not in the list
- Preserve all conditional relationships: if a fact only holds when other facts hold, \
make that explicit in the prose rather than flattening it to an unconditional statement
- Write coherent paragraphs, not a bulleted list

Facts:
{facts}
"""


EXTRACT_PROMPT_STRICT = """\
Extract a structured fact graph from the documentation below.

STEP 1 — Atomic facts:
List every distinct factual claim. Each fact must be:
- One short, self-contained sentence that makes ONE unconditional claim
- Assigned a short stable ID (f01, f02, ...)
- Free of conditions: no "only when", "if", "unless", "before", "after", "requires", \
"must ... in order to", "except", "otherwise". When the source says A holds only when B and C, \
write A, B and C as three separate facts and put the dependence in STEP 2
- Exact in what it does claim: preserve negations, quantifiers and scope

STEP 2 — Logical relationships between facts:
Every condition you removed in STEP 1 must appear here as an edge. For each relationship name the kind:
- implies: antecedent alone is sufficient for the consequent
- requires: consequent is true ONLY when ALL listed antecedents hold (use for 2+ conditions)
- contradicts: antecedent and consequent cannot both hold simultaneously
- sequence: consequent presupposes the antecedent (must happen first)
- negates: antecedent is the direct logical negation of the consequent

Add a confidence score (0.0–1.0). Omit low-confidence (<0.5) edges.

Documentation:
{text}

Respond with JSON only:
{{"facts": [{{"id": "f01", "text": "..."}}], "edges": [{{"kind": "requires", "antecedents": ["f01", "f02"], "consequent": "f03", "confidence": 0.9}}]}}
"""

MIN_EDGE_CONFIDENCE = 0.5

ATOMIZERS = {"sentence": EXTRACT_PROMPT, "strict": EXTRACT_PROMPT_STRICT}


async def extract_facts(llm: LLMProvider, text: str, model: str | None = None, atomizer: str = "sentence") -> FactGraph:
    """Extract atomic facts and conditional edges from a text passage.

    The "sentence" atomizer keeps each claim as one self-contained sentence,
    conditions included, and records the conditions it notices as edges as
    well. The "strict" atomizer forbids conditions inside atoms: every
    dependence has to be an edge, which is what a fact store under test
    actually drops.
    """
    prompt = ATOMIZERS[atomizer].format(text=text)
    kwargs = {"model": model} if model else {}
    for _ in range(3):
        reply = await llm.query(prompt, **kwargs)
        try:
            data = _parse_json(reply)
            facts = [
                FactAtom(text=f["text"], fact_id=f["id"])
                for f in data.get("facts", [])
                if isinstance(f, dict) and f.get("id") and f.get("text")
            ]
            edges = []
            for e in data.get("edges", []):
                try:
                    edge = FactEdge(
                        kind=EdgeKind(e["kind"]),
                        antecedents=list(e["antecedents"]),
                        consequent=e["consequent"],
                        confidence=float(e.get("confidence", 1.0)),
                    )
                except (KeyError, ValueError):
                    continue
                # The prompt asks the model to omit edges under 0.5; it does not always. An edge it
                # did not believe must not be scored, and must not be fed back by the repair pass.
                if edge.confidence >= MIN_EDGE_CONFIDENCE:
                    edges.append(edge)
            if facts:
                return FactGraph(facts=facts, edges=edges)
        except (ValueError, KeyError, TypeError):
            continue
    raise RuntimeError("LLM did not return a parseable fact graph after 3 attempts")


REPAIR_PROMPT = """\
Reconstruct documentation prose from the following facts. A first draft written
from the facts alone lost what is listed below the facts; this draft must carry it.

Rules:
- Include every fact — omit nothing
- Do not add facts not in the list, and do not comment on what the list lacks
- Write coherent paragraphs, not a bulleted list{list_rule}
{section_rules}
Facts:
{facts}
{sections}"""

REPAIR_EDGE_RULE = """\
- State each relationship below in the prose, in the place where it applies: \
a "requires" relationship means the consequent holds ONLY when ALL its antecedents hold, \
say so; a "sequence" relationship means the antecedent must happen first, keep that order
"""
REPAIR_PREREQ_RULE = """\
- Open with a short "Before you begin" section that names each prerequisite listed below, \
with its steps or where to find them, before any step that needs it
"""


def _edge_words(e: FactEdge) -> str:
    ants = " and ".join(f"[{a}]" for a in e.antecedents)
    return {
        EdgeKind.REQUIRES: f"[{e.consequent}] holds only when {ants} all hold",
        EdgeKind.IMPLIES: f"{ants} is enough for [{e.consequent}]",
        EdgeKind.CONTRADICTS: f"{ants} and [{e.consequent}] cannot both hold",
        EdgeKind.SEQUENCE: f"{ants} must happen before [{e.consequent}]",
        EdgeKind.NEGATES: f"{ants} is the negation of [{e.consequent}]",
    }.get(e.kind, f"{e.kind.value}: {ants} -> [{e.consequent}]")


async def regenerate_doc(
    llm: LLMProvider,
    graph: FactGraph,
    model: str | None = None,
    *,
    repair_edges: list[FactEdge] | None = None,
    prerequisites: list[str] | None = None,
) -> str:
    """Reconstruct documentation prose from a fact graph.

    From the atoms alone by default: the edges in `graph` are not shown to the
    model, which is the round trip under test. With `repair_edges` or
    `prerequisites`, the draft is asked to carry those back: the relationships
    the evaluation found missing and the prerequisites it found sprung on the
    reader or dropped.
    """
    facts_text = "\n".join(f"[{f.fact_id}] {f.text}" for f in graph.facts)
    kwargs = {"model": model} if model else {}
    if not repair_edges and not prerequisites:
        return await llm.query(REGEN_PROMPT.format(facts=facts_text), **kwargs)
    rules, sections = "", ""
    if repair_edges:
        rules += REPAIR_EDGE_RULE
        sections += "\nRelationships to state:\n" + "\n".join(f"- {_edge_words(e)}" for e in repair_edges) + "\n"
    if prerequisites:
        rules += REPAIR_PREREQ_RULE
        sections += "\nPrerequisites to state up front:\n" + "\n".join(f"- {p}" for p in prerequisites) + "\n"
    prompt = REPAIR_PROMPT.format(facts=facts_text, sections=sections, section_rules=rules,
                                  list_rule=", except for the prerequisites section" if prerequisites else "")
    return await llm.query(prompt, **kwargs)


def _parse_json(reply: str) -> Any:
    start = reply.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    return json.JSONDecoder().raw_decode(reply, start)[0]
