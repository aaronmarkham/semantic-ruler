"""Offline stand-in for the generator LLM, for tests and `--mock` runs.

Replies are synthetic word salad whose share of "sustainable" vocabulary tracks
the requested level. Paired with HashEmbedder this exercises the whole pipeline
with no network and no keys; the numbers it yields say nothing about real models.
"""

from __future__ import annotations

import hashlib
import json
import random
import re

from spiritwriter.llm import MockLLMProvider

NEG_WORDS = [
    "landfilled", "discarded", "virgin", "unmonitored", "wasteful",
    "discharged", "idling", "scrapped", "vented", "single-use",
]  # fmt: skip
POS_WORDS = [
    "recycled", "recovered", "renewable", "metered", "efficient",
    "closed-loop", "reclaimed", "optimized", "captured", "reusable",
]  # fmt: skip
FILLER = [
    "fixture", "tolerance", "batch", "operator", "spindle", "cycle", "inspection",
    "housing", "bracket", "manifold", "flange", "gasket", "shift", "schedule",
]  # fmt: skip
SLOTS = 10


def _text(rng: random.Random, block: str, lam: float) -> str:
    n_pos = round(lam * SLOTS)
    signal = rng.sample(POS_WORDS, n_pos) + rng.sample(NEG_WORDS, SLOTS - n_pos)
    jargon = [w.lower() for w in re.findall(r"[A-Za-z]+", block)] * 3
    words = signal + jargon + rng.sample(FILLER, 5)
    rng.shuffle(words)
    return " ".join(words) + "."


def _drift_reply(prompt: str) -> str | None:
    """Offline answers for the drift pipeline: a sentence per fact, no edges, and cautious probes."""
    if "Extract a structured fact graph" in prompt:
        text = prompt.split("Documentation:", 1)[1].rsplit("Respond with JSON only", 1)[0]
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20][:40]
        return json.dumps({"facts": [{"id": f"f{i + 1:02d}", "text": s} for i, s in enumerate(sents)], "edges": []})
    if "Reconstruct documentation prose" in prompt:
        facts = re.findall(r"^\[f\d+\] (.*)$", prompt, flags=re.MULTILINE)
        return " ".join(facts)
    if "Generate" in prompt and "yes/no questions" in prompt:
        return json.dumps({"questions": [{"q": "Is the document consistent with itself?", "answer": "yes"}]})
    if "Answer yes, no, or unknown" in prompt:
        return "unknown"
    if "Relationships:" in prompt and "STATES or clearly ENTAILS" in prompt:
        n = len(re.findall(r"^\d+\. ", prompt.split("Relationships:", 1)[1], flags=re.MULTILINE))
        return json.dumps({"answers": [{"i": k + 1, "verdict": "unknown", "quote": ""} for k in range(n)]})
    return None


def _reply(prompt: str) -> str:
    drift = _drift_reply(prompt)
    if drift is not None:
        return drift
    if "Rate each numbered" in prompt:  # judge: share of "sustainable" vocabulary
        scores = [
            {"id": int(i), "score": sum(w in text for w in POS_WORDS) / SLOTS}
            for i, text in re.findall(r"^\[(\d+)\] (.*)$", prompt, flags=re.MULTILINE)
        ]
        return json.dumps({"scores": scores})
    rng = random.Random(int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:8], "big"))
    block = re.search(r"Domain: (.+?) \(|for the (.+?) domain", prompt)
    block_name = next(g for g in block.groups() if g)
    level = re.search(r"lambda=([\d.]+)", prompt)
    if level:
        k = int(re.search(r"Write (\d+) independent", prompt).group(1))
        lam = float(level.group(1))
        return json.dumps({"texts": [_text(rng, block_name, lam) for _ in range(k)]})
    n = int(re.search(r"Provide (\d+) different", prompt).group(1))
    return json.dumps(
        {
            "pairs": [
                {"neg": _text(rng, block_name, 0.0), "pos": _text(rng, block_name, 1.0)}
                for _ in range(n)
            ]
        }
    )


def mock_generator() -> MockLLMProvider:
    return MockLLMProvider(_reply)
