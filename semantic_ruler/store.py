"""Persist anchors, level texts, and rulers as spiritwriter memory shards.

A shard's ID hashes its atoms, scope, and origin, so everything that defines a
ruler (vector, calibration bounds, embedding model, anchor shard IDs) is stored
in atoms: the ruler's shard ID changes if any of them change.
"""

from __future__ import annotations

import base64
import hashlib
import json

import numpy as np
from spiritwriter.fabric.shard import AtomKind, DecayClass, MemoryShard, ShardAtom, ShardRef
from spiritwriter.fabric.store import ShardStore

from .concepts import Block, Concept
from .ruler import Ruler

ORIGIN = "semantic-ruler"


def _scope(concept: Concept) -> str:
    return f"ruler:{concept.slug}"


def anchors_ref(concept: Concept, block: Block) -> str:
    return f"{concept.slug}/anchors/{block.slug}"


def levels_ref(concept: Concept, block: Block) -> str:
    return f"{concept.slug}/levels/{block.slug}"


def ruler_ref(concept: Concept, embedding_model: str) -> str:
    return f"{concept.slug}/ruler/{embedding_model}"


# --- anchors -----------------------------------------------------------------


def save_anchors(
    store: ShardStore,
    concept: Concept,
    block: Block,
    pairs: list[tuple[str, str]],
    *,
    generator: str,
    trace_ref: str | None = None,
) -> ShardRef:
    atoms = []
    for i, (neg, pos) in enumerate(pairs):
        entity = f"anchor:{block.slug}:{i:02d}"
        atoms.append(ShardAtom(text=neg, kind=AtomKind.CONTEXT, entity=entity, key="neg"))
        atoms.append(ShardAtom(text=pos, kind=AtomKind.CONTEXT, entity=entity, key="pos"))
    shard = MemoryShard(
        atoms=atoms,
        scope=_scope(concept),
        origin=ORIGIN,
        decay_class=DecayClass.PERMANENT,
        tags=[f"{concept.name} anchors: {block.name}"],
        meta={"generator": generator, "block": block.name, "n_pairs": len(pairs)},
        trace_ref=trace_ref,
    )
    ref = store.put(shard)
    store.set_ref(anchors_ref(concept, block), ref.shard_id)
    return ref


def load_anchors(
    store: ShardStore, concept: Concept, block: Block
) -> tuple[str, list[tuple[str, str]]] | None:
    """Return (shard_id, pairs) or None if this block has no stored anchors."""
    shard = store.resolve_ref(anchors_ref(concept, block))
    if shard is None:
        return None
    by_entity: dict[str, dict[str, str]] = {}
    for a in shard.atoms:
        by_entity.setdefault(a.entity, {})[a.key] = a.text
    pairs = [(v["neg"], v["pos"]) for _, v in sorted(by_entity.items())]
    return shard.shard_id, pairs


# --- level-guided test texts -------------------------------------------------


def save_levels(
    store: ShardStore,
    concept: Concept,
    block: Block,
    texts_by_level: dict[float, list[str]],
    *,
    generator: str,
    trace_ref: str | None = None,
) -> ShardRef:
    atoms = [
        ShardAtom(
            text=text,
            kind=AtomKind.CONTEXT,
            entity=f"level:{block.slug}:{lam:.1f}:{j:02d}",
            key="lambda",
            value=f"{lam:.1f}",
        )
        for lam, texts in sorted(texts_by_level.items())
        for j, text in enumerate(texts)
    ]
    shard = MemoryShard(
        atoms=atoms,
        scope=_scope(concept),
        origin=ORIGIN,
        decay_class=DecayClass.PERMANENT,
        tags=[f"{concept.name} level texts: {block.name}"],
        meta={"generator": generator, "block": block.name},
        trace_ref=trace_ref,
    )
    ref = store.put(shard)
    store.set_ref(levels_ref(concept, block), ref.shard_id)
    return ref


def load_levels(
    store: ShardStore, concept: Concept, block: Block
) -> tuple[str, dict[float, list[str]]] | None:
    shard = store.resolve_ref(levels_ref(concept, block))
    if shard is None:
        return None
    out: dict[float, list[str]] = {}
    for a in sorted(shard.atoms, key=lambda a: a.entity):
        out.setdefault(float(a.value), []).append(a.text)
    return shard.shard_id, out


# --- LLM-judge scores for the test texts -------------------------------------


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def save_judgements(
    store: ShardStore, concept: Concept, block: Block, scores: dict[str, float], *, generator: str
) -> ShardRef:
    """scores maps level text -> judge score; texts are referenced by hash."""
    atoms = [
        ShardAtom(
            text=f"Judge score {score:.2f}",
            kind=AtomKind.FACT,
            entity=f"judged:{text_key(text)}",
            key="score",
            value=repr(float(score)),
        )
        for text, score in sorted(scores.items(), key=lambda kv: text_key(kv[0]))
    ]
    shard = MemoryShard(
        atoms=atoms,
        scope=_scope(concept),
        origin=ORIGIN,
        decay_class=DecayClass.PERMANENT,
        tags=[f"{concept.name} judge scores: {block.name}"],
        meta={"generator": generator, "block": block.name},
    )
    ref = store.put(shard)
    store.set_ref(f"{concept.slug}/judge/{block.slug}", ref.shard_id)
    return ref


def load_judgements(store: ShardStore, concept: Concept, block: Block) -> dict[str, float]:
    """Return text_key -> judge score (empty if none stored)."""
    shard = store.resolve_ref(f"{concept.slug}/judge/{block.slug}")
    if shard is None:
        return {}
    return {a.entity.split(":", 1)[1]: float(a.value) for a in shard.atoms}


# --- the ruler ---------------------------------------------------------------


def _encode_vector(v: np.ndarray) -> str:
    return base64.b64encode(np.asarray(v, dtype="<f8").tobytes()).decode("ascii")


def _decode_vector(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f8").astype(np.float64)


def save_ruler(
    store: ShardStore,
    concept: Concept,
    ruler: Ruler,
    anchor_shard_ids: list[str],
    *,
    trace_ref: str | None = None,
) -> ShardRef:
    entity = f"ruler:{concept.slug}"

    def fact(key: str, value: str, text: str) -> ShardAtom:
        return ShardAtom(text=text, kind=AtomKind.FACT, entity=entity, key=key, value=value)

    atoms = [
        fact("concept", ruler.concept, f"Semantic ruler measuring: {ruler.concept}"),
        fact("embedding_model", ruler.embedding_model, f"Embedding model: {ruler.embedding_model}"),
        fact("blocks", json.dumps(list(ruler.blocks)), f"Blocks: {', '.join(ruler.blocks)}"),
        fact("q_low", repr(ruler.q_low), f"Raw score mapped to p=0: {ruler.q_low:.6f}"),
        fact("q_high", repr(ruler.q_high), f"Raw score mapped to p=1: {ruler.q_high:.6f}"),
        fact(
            "anchor_shards",
            json.dumps(sorted(anchor_shard_ids)),
            f"Built from {len(anchor_shard_ids)} anchor shards",
        ),
        fact(
            "vector",
            _encode_vector(ruler.vector),
            f"Unit concept vector, {ruler.vector.shape[0]} dims, base64 little-endian float64",
        ),
    ]
    shard = MemoryShard(
        atoms=atoms,
        scope=_scope(concept),
        origin=ORIGIN,
        decay_class=DecayClass.PERMANENT,
        tags=[f"{concept.name} ruler ({ruler.embedding_model})"],
        trace_ref=trace_ref,
    )
    ref = store.put(shard)
    store.set_ref(ruler_ref(concept, ruler.embedding_model), ref.shard_id)
    return ref


def ruler_from_shard(shard: MemoryShard) -> Ruler:
    facts = {a.key: a.value for a in shard.atoms}
    return Ruler(
        concept=facts["concept"],
        vector=_decode_vector(facts["vector"]),
        q_low=float(facts["q_low"]),
        q_high=float(facts["q_high"]),
        embedding_model=facts["embedding_model"],
        blocks=tuple(json.loads(facts["blocks"])),
    )


def load_ruler(store: ShardStore, concept: Concept, embedding_model: str) -> Ruler | None:
    shard = store.resolve_ref(ruler_ref(concept, embedding_model))
    return ruler_from_shard(shard) if shard else None
