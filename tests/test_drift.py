"""Drift metrics: atom matching, bridge recall, prerequisite matching, repeat summaries.

The embedder here is a lookup table of hand-built vectors, so each test states
the similarity it relies on instead of depending on a model.
"""

import math

import numpy as np

from semantic_ruler.drift import (
    MATCH_THRESHOLD,
    atomic_f1,
    cond_bridge_recall,
    structural_drift,
    summarize_repeats,
)
from semantic_ruler.factgraph import EdgeKind, FactAtom, FactEdge, FactGraph
from semantic_ruler.ordering import PageReport


class TableEmbedder:
    """Embeds known strings to fixed vectors; unknown strings get their own axis."""

    def __init__(self, table: dict[str, np.ndarray]):
        self.table = dict(table)
        self.model = "table"

    def embed(self, texts):
        out = []
        for t in texts:
            if t not in self.table:
                v = np.zeros(32)
                v[16 + len(self.table) % 16] = 1.0
                self.table[t] = v
            out.append(self.table[t])
        return np.array(out)


def _axis(i, dim=32):
    v = np.zeros(dim)
    v[i] = 1.0
    return v


def _near(base, cos, noise_axis):
    """A unit vector at the given cosine to `base`, tilted along an unrelated axis."""
    t = math.sqrt(1 / cos**2 - 1)
    v = base + t * _axis(noise_axis)
    return v / np.linalg.norm(v)


def _graph(texts, edges=()):
    facts = [FactAtom(text=t, fact_id=f"f{i + 1:02d}") for i, t in enumerate(texts)]
    return FactGraph(facts=facts, edges=[FactEdge(kind=k, antecedents=a, consequent=c, confidence=w) for k, a, c, w in edges])


ORIG = ["log group only when all three hold", "role has CreateLogGroup", "role has PutLogEvents", "one successful invocation"]
PARA = ["created automatically only when all three conditions hold", "the role has the CreateLogGroup permission",
        "the role has the PutLogEvents permission", "the function completes a successful invocation"]
TABLE = {o: _axis(i) for i, o in enumerate(ORIG)}
# Paraphrases sit at cosine 0.75 to their original: inside the 0.70 cut, outside the old 0.85 one.
TABLE.update({p: _near(_axis(i), 0.75, 8 + i) for i, p in enumerate(PARA)})
TABLE["the missing logs can go unnoticed"] = _axis(20)          # content the regeneration added
TABLE["the log group is created automatically"] = _near(_axis(0), 0.72, 12)  # a shorter restatement of ORIG[0]


def test_paraphrased_atoms_match_at_the_calibrated_threshold():
    emb = TableEmbedder(TABLE)
    r = atomic_f1(_graph(ORIG), _graph(PARA), emb)
    assert r["recall"] == 1.0 and r["precision"] == 1.0 and r["f1"] == 1.0
    strict = atomic_f1(_graph(ORIG), _graph(PARA), emb, threshold=0.85)
    assert strict["recall"] == 0.0


def test_added_content_counts_against_precision_only():
    emb = TableEmbedder(TABLE)
    r = atomic_f1(_graph(ORIG), _graph(PARA + ["the missing logs can go unnoticed"]), emb)
    assert r["recall"] == 1.0
    assert r["precision"] == 4 / 5
    assert r["unsupported"] == ["f05"]


def test_recall_is_one_to_one_but_precision_is_not():
    emb = TableEmbedder(TABLE)
    # Two regenerated atoms both restate ORIG[0]; ORIG[1..3] have no counterpart.
    regen = _graph([PARA[0], "the log group is created automatically"])
    r = atomic_f1(_graph(ORIG), regen, emb)
    assert r["recall"] == 1 / 4          # one original claimed, not two
    assert r["precision"] == 1.0         # both regenerated atoms are supported
    assert r["unsupported"] == []


def test_bridge_survives_when_the_consequent_is_restated():
    emb = TableEmbedder(TABLE)
    orig = _graph(ORIG, [(EdgeKind.REQUIRES, ["f02", "f03", "f04"], "f01", 0.9)])
    # The regeneration keeps the long restatement as an atom (f01) and hangs the edge on the short one (f05).
    regen = _graph(PARA + ["the log group is created automatically"], [(EdgeKind.REQUIRES, ["f02", "f03", "f04"], "f05", 1.0)])
    assert cond_bridge_recall(orig, regen, emb)["cbr"] == 1.0


def test_bridge_is_lost_when_an_antecedent_is_missing():
    emb = TableEmbedder(TABLE)
    orig = _graph(ORIG, [(EdgeKind.REQUIRES, ["f02", "f03", "f04"], "f01", 0.9), (EdgeKind.SEQUENCE, ["f04"], "f01", 0.5)])
    regen = _graph(PARA, [(EdgeKind.REQUIRES, ["f02", "f03"], "f01", 1.0)])
    r = cond_bridge_recall(orig, regen, emb)
    assert r["cbr"] == 0.0
    assert r["by_kind"] == {"requires": 0.0, "sequence": 0.0}


def test_bridge_weights_by_confidence():
    emb = TableEmbedder(TABLE)
    orig = _graph(ORIG, [(EdgeKind.REQUIRES, ["f02", "f03", "f04"], "f01", 0.9), (EdgeKind.SEQUENCE, ["f04"], "f01", 0.3)])
    regen = _graph(PARA, [(EdgeKind.REQUIRES, ["f02", "f03", "f04"], "f01", 1.0)])
    assert math.isclose(cond_bridge_recall(orig, regen, emb)["cbr"], 0.9 / 1.2)


def _report(prereqs, procedure=True):
    return PageReport(url="u", title="t", goal="g", is_procedure=procedure, main_procedure_block=0, n_blocks=10, n_shown=10,
                      prerequisites=[{"what": w, "detour": d, "introduced_at_block": b} for w, d, b in prereqs])


def test_prerequisites_match_by_embedding_when_an_embedder_is_given():
    table = {"S3 logging bucket in the same account": _axis(0), "KMS key policy for the customer managed key": _axis(1),
             "an Amazon S3 bucket in the account that manages the protection pack": _near(_axis(0), 0.8, 8),
             "key policy letting AWS WAF use the KMS key": _near(_axis(1), 0.8, 9)}
    orig = _report([("S3 logging bucket in the same account", "", 1), ("KMS key policy for the customer managed key", "", 2)])
    regen = _report([("key policy letting AWS WAF use the KMS key", "alone", 7), ("an Amazon S3 bucket in the account that manages the protection pack", "alone", 5)])
    with_words = structural_drift(orig, regen)
    with_vectors = structural_drift(orig, regen, TableEmbedder(table))
    assert with_words["prereq_recall"] == 0.0            # no word overlap to speak of
    assert with_vectors["prereq_recall"] == 1.0
    assert with_vectors["detour_rate_delta"] == 1.0
    assert math.isclose(with_vectors["structural_loss"], 0.4)   # recall kept, every prerequisite became a detour


def test_losing_the_procedure_is_total_loss():
    orig = _report([("a bucket", "", 1)])
    r = structural_drift(orig, _report([("a bucket", "alone", 1)], procedure=False))
    assert r["procedure_lost"] and r["structural_loss"] == 1.0 and math.isnan(r["detour_rate_delta"])


def test_summarize_repeats_reports_spread_and_median_run():
    runs = [{"atomic_f1": {"f1": f}, "cond_bridge_recall": {"cbr": c}, "entailment": {"rate": 1.0}, "structural": {"applicable": False},
             "drift": {"drift": d}} for f, c, d in ((0.3, 0.0, 0.7), (0.7, 0.6, 0.4), (0.4, 0.0, 0.68))]
    s = summarize_repeats(runs)
    assert s["n"] == 3 and s["cbr"] == {"median": 0.0, "min": 0.0, "max": 0.6, "n": 3}
    assert s["drift"]["median"] == 0.68 and s["shown"] == 2
    assert "structural_loss" not in s
    assert MATCH_THRESHOLD == 0.70


def test_bridge_recall_names_the_lost_edges():
    emb = TableEmbedder(TABLE)
    orig = _graph(ORIG, [(EdgeKind.REQUIRES, ["f02", "f03", "f04"], "f01", 0.9), (EdgeKind.SEQUENCE, ["f04"], "f01", 0.5)])
    regen = _graph(PARA, [(EdgeKind.REQUIRES, ["f02", "f03", "f04"], "f01", 1.0)])
    assert cond_bridge_recall(orig, regen, emb)["lost"] == [1]


def test_structural_drift_names_unmatched_and_sprung_prerequisites():
    table = {"a bucket": _axis(0), "a key policy": _axis(1), "the bucket for the logs": _near(_axis(0), 0.8, 8)}
    orig = _report([("a bucket", "", 1), ("a key policy", "", 2)])
    regen = _report([("the bucket for the logs", "alone", 9)])
    r = structural_drift(orig, regen, TableEmbedder(table))
    assert r["unmatched"] == ["a key policy"]
    assert r["sprung"] == ["the bucket for the logs"]


def test_repair_prompt_states_relationships_in_words():
    from semantic_ruler.factgraph import _edge_words
    e = FactEdge(kind=EdgeKind.REQUIRES, antecedents=["f02", "f03"], consequent="f01", confidence=0.9)
    assert _edge_words(e) == "[f01] holds only when [f02] and [f03] all hold"
    s = FactEdge(kind=EdgeKind.SEQUENCE, antecedents=["f04"], consequent="f01", confidence=0.5)
    assert _edge_words(s) == "[f04] must happen before [f01]"


def test_granularity_counts_folded_conditions_and_atoms_per_sentence():
    from semantic_ruler.drift import granularity
    text = "A is created only when B holds. C is enabled by default. D must happen before E."
    g = _graph(["A is created only when B holds.", "C is enabled by default.", "D must happen before E."])
    r = granularity(g, text)
    assert r["n_sentences"] == 3 and r["atoms_per_sentence"] == 1.0
    assert r["folded"] == 2 / 3


def test_conditional_load_reads_the_graph_and_the_text():
    from semantic_ruler.drift import conditional_load
    flat = _graph(["A.", "B.", "C."])
    assert conditional_load(flat, "A. B. C.")["load"] == 0.0
    nested = _graph(["A.", "B.", "C.", "D."], [(EdgeKind.REQUIRES, ["f01", "f02", "f03"], "f04", 0.9), (EdgeKind.SEQUENCE, ["f04"], "f01", 0.5)])
    r = conditional_load(nested, "A only when B and C. If D, then A.")
    assert r["widest_requires"] == 3 and r["longest_chain"] >= 2 and 0 < r["load"] < 1
    assert r["markers_per_sentence"] == 1.0


def test_extraction_drops_edges_the_model_did_not_believe():
    import asyncio
    from semantic_ruler.factgraph import extract_facts

    class Reply:
        async def query(self, prompt, **kw):
            return ('{"facts": [{"id": "f01", "text": "A."}, {"id": "f02", "text": "B."}], '
                    '"edges": [{"kind": "requires", "antecedents": ["f01"], "consequent": "f02", "confidence": 0.9}, '
                    '{"kind": "implies", "antecedents": ["f02"], "consequent": "f01", "confidence": 0.0}]}')

    g = asyncio.run(extract_facts(Reply(), "A. B."))
    assert [e.kind.value for e in g.edges] == ["requires"]


def test_edge_statement_uses_the_atoms_words():
    from semantic_ruler.drift import edge_statement
    g = _graph(["The log group is created automatically.", "The role has CreateLogGroup.", "The role has PutLogEvents."],
               [(EdgeKind.REQUIRES, ["f02", "f03"], "f01", 0.9), (EdgeKind.SEQUENCE, ["f03"], "f01", 0.5)])
    by = g.fact_by_id()
    assert edge_statement(g.edges[0], by) == '"The log group is created automatically" holds only when all of the following hold: "The role has CreateLogGroup"; "The role has PutLogEvents".'
    assert edge_statement(g.edges[1], by) == '"The role has PutLogEvents" must happen before "The log group is created automatically".'


def test_bridge_entailment_scores_the_readers_verdicts():
    import asyncio
    from semantic_ruler.drift import bridge_entailment

    class Reader:
        def __init__(self):
            self.prompts = []

        async def query(self, prompt, **kw):
            self.prompts.append(prompt)
            return '{"answers": [{"i": 1, "verdict": "yes", "quote": "only when"}, {"i": 2, "verdict": "no", "quote": ""}]}'

    g = _graph(["A.", "B.", "C."], [(EdgeKind.REQUIRES, ["f02", "f03"], "f01", 0.8), (EdgeKind.SEQUENCE, ["f03"], "f01", 0.2)])
    reader = Reader()
    r = asyncio.run(bridge_entailment(g, "A only when B and C.", reader))
    assert r["rate"] == 0.8 and r["n_yes"] == 1 and r["lost"] == [1] and r["verdicts"] == ["yes", "no"]
    assert "A only when B and C." in reader.prompts[0] and '1. "A" holds only when' in reader.prompts[0]


def test_probe_grading_reads_the_first_word():
    from semantic_ruler.drift import grade_probe_reply
    assert grade_probe_reply("No.") == "no"
    assert grade_probe_reply("Not enough information to say.") == "unknown"
    assert grade_probe_reply("yes, because the role has both permissions") == "yes"
    assert grade_probe_reply("I do not know.") == "unknown"
    assert grade_probe_reply("Unknown") == "unknown"


def test_entailment_rate_counts_valid_probes_only_and_keeps_replies():
    import asyncio
    from semantic_ruler.drift import entailment_rate

    class Model:
        async def query(self, prompt, **kw):
            if "yes/no questions" in prompt:
                return '{"questions": [{"q": "A?", "answer": "yes"}, {"q": "B?", "answer": "no"}, {"q": "C?", "answer": "maybe"}]}'
            return "Not enough information." if "B?" in prompt else "Yes."

    g = _graph(["A.", "B.", "C."], [(EdgeKind.REQUIRES, ["f02", "f03"], "f01", 0.9)])
    r = asyncio.run(entailment_rate(g, "doc", Model()))
    assert r["n_probes"] == 2 and r["n_generated"] == 3 and r["n_correct"] == 1 and r["rate"] == 0.5
    assert r["replies"] == ["Yes.", "Not enough information.", None]


def test_extraction_tolerates_sloppy_edges_and_duplicate_ids():
    import asyncio
    from semantic_ruler.factgraph import extract_facts

    class Reply:
        async def query(self, prompt, **kw):
            return ('{"facts": [{"id": "f01", "text": "A."}, {"id": "f02", "text": "B."}, {"id": "f01", "text": "A again."}], '
                    '"edges": [{"kind": "implies", "antecedents": "f01", "consequent": "f02", "confidence": 0.9}, '
                    '{"kind": "implies", "antecedents": ["f09"], "consequent": "f02", "confidence": 0.9}]}')

    g = asyncio.run(extract_facts(Reply(), "A. B."))
    assert [f.fact_id for f in g.facts] == ["f01", "f02"]
    assert len(g.edges) == 1 and g.edges[0].antecedents == ["f01"]


def test_contradiction_matches_either_way_round():
    emb = TableEmbedder(TABLE)
    orig = _graph(ORIG, [(EdgeKind.CONTRADICTS, ["f02"], "f03", 0.8)])
    regen = _graph(PARA, [(EdgeKind.CONTRADICTS, ["f03"], "f02", 1.0)])
    assert cond_bridge_recall(orig, regen, emb)["cbr"] == 1.0


def test_procedure_without_prerequisites_loses_nothing():
    orig = _report([])
    r = structural_drift(orig, _report([]))
    assert r["structural_loss"] == 0.0


def test_edge_statement_has_no_backslash_in_fstring_expressions():
    import ast, inspect
    from semantic_ruler import drift
    src = inspect.getsource(drift.edge_statement)
    ast.parse(src, feature_version=(3, 10))   # a backslash inside an f-string expression fails here
