"""Generate several independent anchor draws per ruler concept.

Draw 0 lives in data/aws-intents/store (the draw used by the first runs);
draws 1..K-1 in data/aws-intents/draws/<k>. Existing anchors are reused.

    python experiments/aws_intents/make_draws.py . [K]
"""

import asyncio
import sys
from pathlib import Path

from spiritwriter.fabric.store import ShardStore
from spiritwriter.llm import AnthropicProvider
from spiritwriter.secrets import configure

sys.path.insert(0, str(Path(__file__).parent))
from concepts_aws import CONTRASTS  # noqa: E402

from semantic_ruler import store as shards  # noqa: E402
from semantic_ruler.embed import KEYCHAIN_SERVICE  # noqa: E402
from semantic_ruler.generate import generate_pairs  # noqa: E402

MODEL = "claude-sonnet-5-5"
N_PAIRS = 8


def draw_store(root: Path, k: int) -> ShardStore:
    return ShardStore(root / "data" / "aws-intents" / ("store" if k == 0 else f"draws/{k}"))


async def main(root: Path, n_draws: int) -> None:
    configure(service_name=KEYCHAIN_SERVICE)
    llm = AnthropicProvider(model=MODEL)
    sem = asyncio.Semaphore(4)
    failures = []

    async def job(k, concept, block):
        store = draw_store(root, k)
        if shards.load_anchors(store, concept, block):
            return
        async with sem:
            try:
                pairs = await generate_pairs(llm, concept, block, N_PAIRS, MODEL)
            except Exception as e:  # report and keep going; rerun to fill gaps
                failures.append(f"draw {k} {concept.slug} {block.name}: {e}")
                return
        shards.save_anchors(store, concept, block, pairs, generator=MODEL)

    await asyncio.gather(*[
        job(k, concept, block)
        for k in range(n_draws) for _, _, concept in CONTRASTS.values() for block in concept.blocks
    ])
    for name, (_, _, concept) in CONTRASTS.items():
        have = sum(all(shards.load_anchors(draw_store(root, k), concept, b) for b in concept.blocks) for k in range(n_draws))
        print(f"  {name:<34} {have}/{n_draws} complete draws")
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve(), int(sys.argv[2]) if len(sys.argv) > 2 else 5))
