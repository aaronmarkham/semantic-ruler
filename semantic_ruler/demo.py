"""Case Study I end to end: generate anchors, build the ruler, validate it.

    semantic-ruler                 # live: Claude generates, OpenAI embeds
    semantic-ruler --mock          # offline: synthetic text, hashed bag-of-words
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from spiritwriter.fabric.emitter import TraceEmitter, verify_chain
from spiritwriter.fabric.store import ShardStore
from spiritwriter.llm import LLMProvider

from . import store as shards
from .concepts import CONCEPTS, Concept
from .embed import KEYCHAIN_SERVICE, CachedEmbedder, Embedder, HashEmbedder, OpenAIEmbedder
from .generate import generate_level_texts, generate_pairs, judge_texts
from .ruler import (
    Ruler,
    anchor_cohesion,
    block_coherence,
    centroid_direction,
    evaluate,
    fit_direction,
    fit_ruler,
    paired_deltas,
    power_utility,
    random_direction,
)

LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass
class DemoResult:
    ruler: Ruler
    ruler_shard_id: str
    coherence: dict[str, float]
    metrics: dict[str, dict[str, float]]  # method -> metrics
    per_block: dict[str, dict[str, float]]  # block -> metrics for the ruler
    lam: np.ndarray
    p: np.ndarray
    block_of: list[str]
    cohesion: dict[str, float] = field(default_factory=dict)
    judge: np.ndarray | None = None  # LLM-judge scores for the same texts, if requested
    anchors: dict[str, tuple[np.ndarray, np.ndarray]] = field(repr=False, default_factory=dict)
    trace_path: Path | None = None
    plot_path: Path | None = None


async def _gather_limited(coros, limit: int):
    sem = asyncio.Semaphore(limit)

    async def run(c):
        async with sem:
            return await c

    return await asyncio.gather(*(run(c) for c in coros))


async def run_demo(
    concept: Concept,
    llm: LLMProvider,
    embedder: Embedder,
    store: ShardStore,
    *,
    n_pairs: int = 8,
    n_variants: int = 3,
    levels: tuple[float, ...] = LEVELS,
    generator: str = "unknown",
    model: str | None = None,
    regenerate: bool = False,
    judge: bool = False,
    concurrency: int = 4,
    trace_path: Path | None = None,
    log=print,
) -> DemoResult:
    trace = None
    if trace_path is not None:
        trace = TraceEmitter(uuid.uuid4().hex[:12], shards.ORIGIN, str(trace_path))
        trace.emit("run_started", concept=concept.name, generator=generator,
                   embedding_model=embedder.model, n_pairs=n_pairs, n_variants=n_variants)  # fmt: skip

    def trace_ref() -> str | None:
        return trace.current_trace_ref() if trace else None

    # 1. Blocked counterfactual anchors (Section 3.1) ---------------------------
    anchor_ids: dict[str, str] = {}
    pairs_by_block: dict[str, list[tuple[str, str]]] = {}
    missing = []
    for block in concept.blocks:
        found = None if regenerate else shards.load_anchors(store, concept, block)
        if found and len(found[1]) >= n_pairs:
            anchor_ids[block.name], pairs_by_block[block.name] = found[0], found[1][:n_pairs]
        else:
            missing.append(block)
    if missing:
        log(f"Generating {n_pairs} anchor pairs for {len(missing)} block(s) with {generator}...")
        fresh = await _gather_limited(
            (generate_pairs(llm, concept, b, n_pairs, model) for b in missing), concurrency
        )
        for block, pairs in zip(missing, fresh):
            ref = shards.save_anchors(
                store, concept, block, pairs, generator=generator, trace_ref=trace_ref()
            )
            anchor_ids[block.name], pairs_by_block[block.name] = ref.shard_id, pairs
            if trace:
                trace.emit("anchors_generated", block=block.name, shard_id=ref.shard_id)
    reused = len(concept.blocks) - len(missing)
    if reused:
        log(f"Reused stored anchors for {reused} block(s).")

    # 2. Concept vector and calibration (Sections 3.2, 3.3) ---------------------
    neg, pos = {}, {}
    for block in concept.blocks:
        pairs = pairs_by_block[block.name]
        emb = embedder.embed([t for pair in pairs for t in pair])
        neg[block.name], pos[block.name] = emb[0::2], emb[1::2]
    ruler = fit_ruler(concept.name, neg, pos, embedder.model)
    coherence = block_coherence(paired_deltas(neg, pos))
    ruler_ref = shards.save_ruler(
        store, concept, ruler, list(anchor_ids.values()), trace_ref=trace_ref()
    )
    if trace:
        trace.emit("ruler_fitted", shard_id=ruler_ref.shard_id, q_low=ruler.q_low,
                   q_high=ruler.q_high, anchor_shards=sorted(anchor_ids.values()))  # fmt: skip

    # 3. Level-guided validation texts (Section 3.5) ----------------------------
    levels_by_block: dict[str, dict[float, list[str]]] = {}
    jobs = []
    for block in concept.blocks:
        found = None if regenerate else shards.load_levels(store, concept, block)
        have = found[1] if found else {}
        if all(len(have.get(lam, [])) >= n_variants for lam in levels):
            levels_by_block[block.name] = {lam: have[lam][:n_variants] for lam in levels}
        else:
            jobs.extend((block, lam) for lam in levels)
    if jobs:
        log(f"Generating level-guided texts: {len(jobs)} calls x {n_variants} variants...")
        fresh = await _gather_limited(
            (generate_level_texts(llm, concept, b, lam, n_variants, model) for b, lam in jobs),
            concurrency,
        )
        staged: dict[str, dict[float, list[str]]] = {}
        for (block, lam), texts in zip(jobs, fresh):
            staged.setdefault(block.name, {})[lam] = texts
        for block in concept.blocks:
            if block.name in staged:
                ref = shards.save_levels(store, concept, block, staged[block.name],
                                         generator=generator, trace_ref=trace_ref())  # fmt: skip
                levels_by_block[block.name] = staged[block.name]
                if trace:
                    trace.emit("levels_generated", block=block.name, shard_id=ref.shard_id)

    texts, lam_list, block_of = [], [], []
    for block in concept.blocks:
        for lam in levels:
            for t in levels_by_block[block.name][lam]:
                texts.append(t)
                lam_list.append(lam)
                block_of.append(block.name)
    lam = np.asarray(lam_list)
    test = embedder.embed(texts)

    # 4. Score and compare against baselines (Section 4.1) ----------------------
    anchors = np.vstack([*neg.values(), *pos.values()])
    concept_emb = embedder.embed([concept.name])[0]
    candidates = {
        "semantic ruler (paired, blocked)": ruler,
        "centroid difference": fit_direction(
            concept.name, centroid_direction(neg, pos), anchors, embedder.model
        ),
        "cosine to concept name": fit_direction(concept.name, concept_emb, anchors, embedder.model),
        "random direction": fit_direction(
            concept.name, random_direction(anchors.shape[1]), anchors, embedder.model
        ),
    }
    metrics = {name: evaluate(lam, r.score(test)) for name, r in candidates.items()}
    p = ruler.score(test)
    blocks_arr = np.asarray(block_of)
    per_block = {
        b.name: evaluate(lam[blocks_arr == b.name], p[blocks_arr == b.name]) for b in concept.blocks
    }
    # 5. Optional LLM-as-judge baseline on the same texts ----------------------
    judge_scores = None
    if judge:
        known: dict[str, float] = {}
        for block in concept.blocks:
            known.update(shards.load_judgements(store, concept, block))
        by_block = {
            b.name: [t for t, bn in zip(texts, block_of) if bn == b.name] for b in concept.blocks
        }
        todo = [
            b
            for b in concept.blocks
            if regenerate or any(shards.text_key(t) not in known for t in by_block[b.name])
        ]
        if todo:
            log(f"Judging test texts with {generator}: {len(todo)} call(s)...")
            fresh = await _gather_limited(
                (judge_texts(llm, concept, by_block[b.name], model) for b in todo), concurrency
            )
            for block, scores in zip(todo, fresh):
                mapping = dict(zip(by_block[block.name], scores))
                shards.save_judgements(store, concept, block, mapping, generator=generator)
                known.update({shards.text_key(t): s for t, s in mapping.items()})
        judge_scores = np.asarray([known[shards.text_key(t)] for t in texts])
        metrics["LLM judge (direct rating)"] = evaluate(lam, judge_scores)

    if trace:
        trace.emit("ruler_evaluated", shard_id=ruler_ref.shard_id,
                   metrics=metrics["semantic ruler (paired, blocked)"])  # fmt: skip

    return DemoResult(
        ruler=ruler,
        ruler_shard_id=ruler_ref.shard_id,
        cohesion=anchor_cohesion(neg, pos),
        judge=judge_scores,
        coherence=coherence,
        metrics=metrics,
        per_block=per_block,
        lam=lam,
        p=p,
        block_of=block_of,
        anchors={b: (neg[b], pos[b]) for b in neg},
        trace_path=trace_path,
    )


# --- reporting ---------------------------------------------------------------


def report(result: DemoResult, log=print) -> None:
    log(f"\nRuler shard: {result.ruler_shard_id}")
    log(
        f"  model {result.ruler.embedding_model}, {result.ruler.vector.shape[0]} dims, "
        f"calibration [{result.ruler.q_low:.4f}, {result.ruler.q_high:.4f}]"
    )
    log("\nAnchor diagnostics (median cosine):")
    log(f"  {'block':<26} {'to block centroid':>18} {'delta to mean delta':>20}")
    for block, c in result.coherence.items():
        log(f"  {block:<26} {result.cohesion.get(block, float('nan')):>18.3f} {c:>20.3f}")
    log("  (the paper's 0.80-0.85 for this embedding model is the first column)")

    log(f"\nLevel-guided validation, n = {len(result.lam)} texts:")
    log(f"  {'method':<34} {'r':>7} {'R2':>7} {'rho':>7} {'RMSE':>7} {'MAE':>7}")
    for name, m in result.metrics.items():
        log(
            f"  {name:<34} {m['pearson_r']:>7.3f} {m['r2']:>7.3f} {m['spearman_rho']:>7.3f} "
            f"{m['rmse']:>7.3f} {m['mae']:>7.3f}"
        )
    if result.judge is not None:
        agree = evaluate(result.judge, result.p)
        log(f"  ruler vs judge agreement: r = {agree['pearson_r']:.3f}, MAE = {agree['mae']:.3f}")
    log("\nRuler by block:")
    blocks = np.asarray(result.block_of)
    for block, m in result.per_block.items():
        line = f"  {block:<26} r = {m['pearson_r']:.3f}  RMSE = {m['rmse']:.3f}"
        if result.judge is not None:
            bias = float((result.p - result.judge)[blocks == block].mean())
            line += f"  mean(ruler - judge) = {bias:+.3f}"
        log(line)

    log("\nMean score by requested level, with the illustrative utility U(p) = p^2:")
    log(f"  {'lambda':>6} {'mean p':>8} {'mean U':>8}")
    for lam in np.unique(result.lam):
        sel = result.p[result.lam == lam]
        log(f"  {lam:>6.1f} {sel.mean():>8.3f} {power_utility(sel, 2.0).mean():>8.3f}")


def plot(result: DemoResult, path: Path) -> Path:
    """Fig. 2 (anchor trajectories in PCA space) and Fig. 3 (lambda vs p)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    allv = np.vstack([np.vstack(pair) for pair in result.anchors.values()])
    mean = allv.mean(axis=0)
    _, _, vt = np.linalg.svd(allv - mean, full_matrices=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for color, (block, (neg, pos)) in zip(colors, result.anchors.items()):
        n2, p2 = (neg - mean) @ vt[:2].T, (pos - mean) @ vt[:2].T
        ax1.scatter(n2[:, 0], n2[:, 1], marker="o", facecolors="none", edgecolors=color)
        ax1.scatter(p2[:, 0], p2[:, 1], marker="o", color=color, label=block)
        for a, b in zip(n2, p2):
            ax1.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.5})
    ax1.set(title="Anchor pairs: NEG (hollow) to POS (filled)", xlabel="PC1", ylabel="PC2")
    ax1.legend(fontsize=8)

    blocks = np.asarray(result.block_of)
    rng = np.random.default_rng(0)
    for color, block in zip(colors, result.anchors):
        sel = blocks == block
        jitter = rng.uniform(-0.02, 0.02, sel.sum())
        ax2.scatter(result.lam[sel] + jitter, result.p[sel], color=color, alpha=0.7, label=block)
    ax2.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    m = result.metrics["semantic ruler (paired, blocked)"]
    ax2.set(
        title=f"Level-guided validation: r = {m['pearson_r']:.3f}, RMSE = {m['rmse']:.3f}",
        xlabel="requested intensity (lambda)",
        ylabel="calibrated score p",
    )
    ax2.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="semantic-ruler", description=__doc__.splitlines()[0])
    ap.add_argument("--concept", default="sustainability", choices=sorted(CONCEPTS))
    ap.add_argument("--pairs", type=int, default=8, help="anchor pairs per block (paper: 8)")
    ap.add_argument("--variants", type=int, default=3, help="test texts per block per level")
    ap.add_argument("--model", default="claude-sonnet-5-5", help="generator model")
    ap.add_argument("--embedding-model", default="text-embedding-3-large")
    ap.add_argument("--store", type=Path, default=None, help="shard store directory")
    ap.add_argument("--out", type=Path, default=Path("out"), help="plots and traces")
    ap.add_argument("--regenerate", action="store_true", help="ignore stored anchors and texts")
    ap.add_argument("--judge", action="store_true", help="add the LLM-as-judge baseline")
    ap.add_argument("--mock", action="store_true", help="offline: synthetic text, hash embedder")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    concept = CONCEPTS[args.concept]
    if args.mock:
        from .mock import mock_generator

        llm, generator, model = mock_generator(), "mock", None
        embedder: Embedder = HashEmbedder()
        store_dir = args.store or args.out / "mock-store"
    else:
        from spiritwriter.llm import AnthropicProvider
        from spiritwriter.secrets import configure

        configure(service_name=KEYCHAIN_SERVICE)
        llm, generator, model = AnthropicProvider(model=args.model), args.model, args.model
        embedder = CachedEmbedder(OpenAIEmbedder(args.embedding_model), ".cache/embeddings")
        store_dir = args.store or Path("data/store")

    trace_path = args.out / f"trace-{uuid.uuid4().hex[:8]}.jsonl"
    result = asyncio.run(
        run_demo(
            concept,
            llm,
            embedder,
            ShardStore(store_dir),
            n_pairs=args.pairs,
            n_variants=args.variants,
            generator=generator,
            model=model,
            regenerate=args.regenerate,
            judge=args.judge,
            trace_path=trace_path,
        )
    )
    report(result)

    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    print(f"\nTrace: {trace_path} ({len(events)} events, chain valid: {verify_chain(events)})")
    print(f"Shard store: {store_dir}")
    if not args.no_plot:
        try:
            print(f"Plot: {plot(result, args.out / f'{concept.slug}-{embedder.model}.png')}")
        except ImportError:
            print("Plot skipped: install matplotlib (pip install -e .[plot])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
