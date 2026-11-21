# semantic-ruler

A demo of *Semantic rulers: Continuous measurement of abstract concepts in LLM
embedding spaces* (Abonyi & Tombor, Array 32 (2026) 101249,
[doi:10.1016/j.array.2026.101249](https://doi.org/10.1016/j.array.2026.101249)),
built on [spiritwriter](../spiritwriter-core) primitives. The paper is in this
folder as `semantic-rulers.pdf`.

A semantic ruler is a calibrated direction in an embedding space. Projecting any
text onto it gives a score `p` in [0, 1] for how strongly the text expresses one
abstract concept, with one embedding call and one dot product.

## How it works

1. **Anchors.** An LLM writes counterfactual pairs: the same scenario at the low
   and high extreme of the concept, in several unrelated domains ("blocks").
2. **Concept vector.** Embed each pair and subtract. Average the differences
   within each block, normalise, then average across blocks. Shared topic cancels
   inside each pair; block-specific jargon cancels across blocks.
3. **Calibration.** The 5th and 95th percentile of the anchors' raw projections
   become 0 and 1.
4. **Utility (optional).** A separate, explicit function such as `U(p) = p^2`
   turns the score into a preference. The ruler measures; the utility judges.

## Run it

```bash
pip install -e ".[dev,plot]"
python -m pytest tests/ -v      # offline, no keys
semantic-ruler --mock           # offline end-to-end run on synthetic text
semantic-ruler                  # live: Claude generates, OpenAI embeds
```

The live run reproduces the paper's Case Study I (sustainability in
manufacturing, 4 blocks x 8 pairs) and validates the ruler on texts generated at
requested intensity levels 0.0 to 1.0. It reads `ANTHROPIC_API_KEY` and
`OPENAI_API_KEY` from the OS keychain service `claude-studio`; set
`SEMANTIC_RULER_KEYCHAIN_SERVICE` to use another. Generated text is kept in
`data/store`, so later runs reuse it unless you pass `--regenerate`.

## Results so far (2026-11-21)

One live run of Case Study I: `claude-sonnet-5-5` as generator and judge,
`text-embedding-3-large` as embedder, 72 test texts.

| Method | Pearson r | RMSE | Paper's r |
|---|---|---|---|
| Semantic ruler | 0.853 | 0.187 | 0.956 |
| Centroid difference | 0.855 | 0.186 | 0.953 |
| Cosine to concept name | 0.216 | 0.446 | 0.713 |
| Random direction | -0.207 | 0.467 | -0.161 |
| LLM judge | 0.966 | 0.093 | 0.952 |

The ordering of the embedding methods matches the paper, but the ruler does not
reach parity with the LLM judge here. What the follow-up checks showed:

- The gap is in the ruler, not the test texts: the judge rates the same texts
  at r = 0.966 with no difference between blocks.
- The ruler's error is a per-block offset. Against the judge it reads injection
  molding +0.14 high and surface treatment -0.16 low; within a block r is
  0.88-0.93. The anchors themselves show no such offset.
- The ruler generalises to an unseen block: built from three blocks and scored
  on the fourth, pooled r is 0.856.
- More anchors do not help (r plateaus by 6 pairs per block), and text length
  is not a confound.
- No alternative direction tried on this data did much better: per-block
  calibration 0.888, a shrinkage-whitened mean delta 0.882, projecting out the
  block-centroid subspace 0.877. A ridge regression trained directly on three
  blocks' test texts scored the fourth at only 0.80, and a ruler built from
  graded six-level anchors scored 0.77.

These variants were all compared on the same 72 texts, so the small gains are
not reliable evidence.

A second run with `claude-haiku-5-5` as generator and judge (store in
`data/store-haiku`) did not close the gap: ruler r = 0.808, judge r = 0.938.
Crossing the two runs, with the two rulers at cosine 0.82 to each other:

| Ruler built from | Sonnet test texts | Haiku test texts |
|---|---|---|
| Sonnet anchors | 0.853 | 0.736 |
| Haiku anchors | 0.898 | 0.808 |

So a smaller generator does not explain the paper's higher numbers, the result
depends on which anchor set and which test set were drawn, and each cell is a
single run with no confidence interval. Why the paper's ruler matched its judge
is still not established.

## What spiritwriter provides

| Need | Primitive |
|---|---|
| Generate pairs and level texts | `LLMProvider`, `JSONExtractor`, `MockLLMProvider` |
| Keep anchors, test texts, and the ruler | `MemoryShard` in a `ShardStore` |
| Record what produced what | `TraceEmitter` hash chain, linked from each shard's `trace_ref` |
| API keys | `spiritwriter.secrets` |

The ruler shard holds the vector, calibration bounds, embedding model, and the
IDs of its anchor shards, all in atoms, so its content address identifies the
instrument exactly. Embedding is the one piece spiritwriter lacks; the small
`Embedder` interface lives in `semantic_ruler/embed.py`.

## Layout

| File | Contents |
|---|---|
| `semantic_ruler/ruler.py` | The paper's equations in numpy, baselines, metrics |
| `semantic_ruler/embed.py` | OpenAI, offline hash, and disk-cached embedders |
| `semantic_ruler/concepts.py` | Concept and block definitions |
| `semantic_ruler/generate.py` | Prompt templates and generation |
| `semantic_ruler/store.py` | Shard persistence |
| `semantic_ruler/demo.py` | The Case Study I pipeline and CLI |

## Differences from the paper

- Claude generates the text and `text-embedding-3-large` embeds it; the paper's
  primary stack was Gemini for both, with OpenAI embeddings as a cross-check.
- One LLM call returns all pairs for a block, because the provider has no seed
  parameter.
- No outlier removal, no unblocked ablation, and no images (Case Studies II and
  III) yet. The LLM-as-judge baseline is behind `--judge`.
- The paper does not give its level-guided prompt for Case Study I; the one here
  is adapted from its Case Study II template.
