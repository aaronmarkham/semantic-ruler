"""End-to-end run with the mock generator and hash embedder: no network, no keys."""

import asyncio
import json

import numpy as np
from spiritwriter.fabric.emitter import verify_chain
from spiritwriter.fabric.store import ShardStore
from spiritwriter.llm import MockLLMProvider

from semantic_ruler import store as shards
from semantic_ruler.concepts import SUSTAINABILITY
from semantic_ruler.demo import main, run_demo
from semantic_ruler.embed import CachedEmbedder, HashEmbedder
from semantic_ruler.mock import mock_generator

RULER = "semantic ruler (paired, blocked)"


def run(tmp_path, llm=None, **kw):
    return asyncio.run(
        run_demo(
            SUSTAINABILITY,
            llm or mock_generator(),
            HashEmbedder(),
            ShardStore(tmp_path / "store"),
            n_pairs=4,
            n_variants=2,
            generator="mock",
            log=lambda *_: None,
            **kw,
        )
    )


def test_pipeline_tracks_requested_level(tmp_path):
    result = run(tmp_path)
    assert len(result.lam) == 4 * 6 * 2
    assert result.metrics[RULER]["pearson_r"] > 0.9
    assert result.metrics[RULER]["pearson_r"] > result.metrics["random direction"]["pearson_r"]
    assert set(result.per_block) == {b.name for b in SUSTAINABILITY.blocks}


def test_ruler_round_trips_through_shard_store(tmp_path):
    result = run(tmp_path)
    store = ShardStore(tmp_path / "store")
    loaded = shards.load_ruler(store, SUSTAINABILITY, HashEmbedder().model)
    assert np.array_equal(loaded.vector, result.ruler.vector)
    assert (loaded.q_low, loaded.q_high) == (result.ruler.q_low, result.ruler.q_high)
    assert loaded.blocks == result.ruler.blocks
    assert store.get_ref(shards.ruler_ref(SUSTAINABILITY, loaded.embedding_model)) == (
        result.ruler_shard_id
    )


def test_second_run_reuses_stored_text_and_reproduces_ruler_id(tmp_path):
    first = run(tmp_path)
    llm = mock_generator()
    second = run(tmp_path, llm=llm)
    assert llm.calls == []
    assert second.ruler_shard_id == first.ruler_shard_id


def test_trace_chain_is_valid_and_linked(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    result = run(tmp_path, trace_path=trace_path)
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert verify_chain(events)
    assert [e["type"] for e in events][0] == "run_started"
    assert events[-1]["type"] == "ruler_evaluated"
    shard = ShardStore(tmp_path / "store").get(result.ruler_shard_id)
    assert shard.trace_ref.startswith("chain:")


def test_judge_baseline_is_scored_and_cached(tmp_path):
    result = run(tmp_path, judge=True)
    assert result.judge.shape == result.lam.shape
    assert result.metrics["LLM judge (direct rating)"]["pearson_r"] > 0.95
    llm = mock_generator()
    again = run(tmp_path, llm=llm, judge=True)
    assert llm.calls == []  # judge scores reused from the shard store
    assert np.array_equal(again.judge, result.judge)


def test_doubled_llm_reply_is_parsed(tmp_path):
    # AnthropicProvider's Agent SDK path returns the reply text twice.
    inner = mock_generator()
    doubled = MockLLMProvider(lambda prompt: inner._resolve(prompt) * 2)
    assert run(tmp_path, llm=doubled).metrics[RULER]["pearson_r"] > 0.9


def test_cached_embedder_hits_on_repeat(tmp_path):
    emb = CachedEmbedder(HashEmbedder(), tmp_path / "cache")
    a = emb.embed(["recycled coolant", "landfilled chips"])
    b = emb.embed(["recycled coolant", "landfilled chips"])
    assert np.array_equal(a, b)
    assert (emb.hits, emb.misses) == (2, 2)


def test_cli_mock_run(tmp_path, capsys):
    assert main(["--mock", "--pairs", "3", "--variants", "1", "--out", str(tmp_path), "--no-plot"]) == 0
    out = capsys.readouterr().out
    assert "Ruler shard:" in out and "chain valid: True" in out
