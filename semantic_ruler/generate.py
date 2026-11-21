"""Counterfactual data generation through a spiritwriter LLMProvider (Section 3.1, 3.5).

The pair prompt follows the paper's Appendix A.1 template. Two deliberate changes:
the provider has no seed parameter, so one call asks for all N pairs of a block
(which also keeps the pairs distinct from each other), and the JSON shape is
requested in the prompt rather than through an API response schema.
"""

from __future__ import annotations

import json
import random
from typing import Any

from spiritwriter.llm import LLMProvider
from spiritwriter.llm.anthropic import JSONExtractor

from .concepts import Block, Concept

PAIR_PROMPT = """\
Task: Provide {n} different {subject}s for the {block} domain ({focus}) and, for each one, \
create two PAIRED counterfactual variants:
(1) {neg_label} (NEG) and (2) {pos_label} (POS).

Requirements:
- Maintain the same functional purpose and product requirements in both the NEG and POS variant of a pair.
- The two variants of a pair should be of similar length (within 10-15%), in a professional and neutral tone.
- They should differ strictly in {dimensions}.
- Avoid superficial marketing terminology; all statements must be concrete, technical, and operational.
- Each pair must cover a different scenario from the other pairs.

Constraints:
- Each description must be 7-10 sentences long.
- Context: {context}
- Use realistic but generic details appropriate for the domain.

Respond with JSON only, in exactly this shape:
{{"pairs": [{{"neg": "...", "pos": "..."}}]}}
"""

LEVEL_PROMPT = """\
Context: {context} Domain: {block} ({focus}).
Task: Write {k} independent {subject}s in which {scale_name} is exactly lambda={lam:.1f} \
on a scale where 0.0 = {neg_label} and 1.0 = {pos_label}.

Requirements:
- Each description must be 7-10 sentences long, in a professional and neutral tone.
- Concrete, technical, and operational statements only; no marketing terminology.
- Each description must cover a different scenario.
- Do not mention lambda, the scale, or any numeric rating in the text.

Respond with JSON only, in exactly this shape:
{{"texts": ["..."]}}
"""


JUDGE_PROMPT = """\
You are rating {subject}s for {scale_name}.
Scale: 0.0 = {neg_label}; 1.0 = {pos_label}. Use the full continuous range and two decimals.
Rate each numbered description independently, on its own merits.

{items}

Respond with JSON only, in exactly this shape, one entry per description id:
{{"scores": [{{"id": 1, "score": 0.00}}]}}
"""


def pair_prompt(concept: Concept, block: Block, n: int) -> str:
    return PAIR_PROMPT.format(
        n=n,
        subject=concept.subject,
        block=block.name,
        focus=block.focus,
        neg_label=concept.neg_label,
        pos_label=concept.pos_label,
        dimensions=concept.dimensions,
        context=concept.context,
    )


def level_prompt(concept: Concept, block: Block, lam: float, k: int) -> str:
    return LEVEL_PROMPT.format(
        k=k,
        subject=concept.subject,
        block=block.name,
        focus=block.focus,
        scale_name=concept.scale_name,
        lam=lam,
        neg_label=concept.neg_label,
        pos_label=concept.pos_label,
        context=concept.context,
    )


def _parse_json(reply: str) -> Any:
    """Parse the first JSON object in a reply, ignoring anything after it.

    AnthropicProvider's Agent SDK path returns the reply text twice
    (`{...}{...}`), which strict parsers reject as trailing data.
    """
    start = reply.find("{")
    if start >= 0:
        try:
            return json.JSONDecoder().raw_decode(reply, start)[0]
        except json.JSONDecodeError:
            pass
    return JSONExtractor.extract(reply)


async def _query_json(llm: LLMProvider, prompt: str, model: str | None, attempts: int = 3) -> Any:
    kwargs = {"model": model} if model else {}
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return _parse_json(await llm.query(prompt, **kwargs))
        except ValueError as e:  # unparseable reply: ask again
            last = e
    raise RuntimeError(f"LLM did not return parseable JSON after {attempts} attempts: {last}")


async def generate_pairs(
    llm: LLMProvider, concept: Concept, block: Block, n: int, model: str | None = None
) -> list[tuple[str, str]]:
    """N (neg, pos) counterfactual pairs for one block."""
    pairs: list[tuple[str, str]] = []
    for _ in range(3):  # smaller models sometimes return fewer items than asked
        data = await _query_json(llm, pair_prompt(concept, block, n), model)
        pairs = [
            (p["neg"].strip(), p["pos"].strip())
            for p in data.get("pairs", [])
            if isinstance(p, dict) and p.get("neg") and p.get("pos")
        ]
        if len(pairs) >= n:
            return pairs[:n]
    raise RuntimeError(f"{block.name}: asked for {n} pairs, got {len(pairs)}")


async def judge_texts(
    llm: LLMProvider, concept: Concept, texts: list[str], model: str | None = None
) -> list[float]:
    """LLM-as-judge baseline (Section 4.1): one direct 0-1 rating per text.

    Texts are shuffled before rating so the judge cannot read the level off their
    order; scores come back in the caller's order.
    """
    order = list(range(len(texts)))
    random.Random(0).shuffle(order)
    items = "\n\n".join(f"[{n + 1}] {texts[i]}" for n, i in enumerate(order))
    prompt = JUDGE_PROMPT.format(
        subject=concept.subject,
        scale_name=concept.scale_name,
        neg_label=concept.neg_label,
        pos_label=concept.pos_label,
        items=items,
    )
    for _ in range(3):  # judges occasionally skip or repeat an id
        data = await _query_json(llm, prompt, model)
        try:
            by_id = {int(d["id"]): float(d["score"]) for d in data.get("scores", [])}
        except (KeyError, TypeError, ValueError):
            continue
        if sorted(by_id) == list(range(1, len(texts) + 1)):
            scores = [0.0] * len(texts)
            for n, i in enumerate(order):
                scores[i] = min(1.0, max(0.0, by_id[n + 1]))
            return scores
    raise RuntimeError(f"judge did not return one score per text ({len(texts)} texts)")


async def generate_level_texts(
    llm: LLMProvider, concept: Concept, block: Block, lam: float, k: int, model: str | None = None
) -> list[str]:
    """k texts targeting intensity lambda in one block (level-guided generation)."""
    texts: list[str] = []
    for _ in range(3):  # smaller models sometimes return fewer items than asked
        data = await _query_json(llm, level_prompt(concept, block, lam, k), model)
        texts = [t.strip() for t in data.get("texts", []) if isinstance(t, str) and t.strip()]
        if len(texts) >= k:
            return texts[:k]
    raise RuntimeError(f"{block.name} lambda={lam}: asked for {k} texts, got {len(texts)}")
