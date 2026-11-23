# semantic-ruler

A demo of *Semantic rulers: Continuous measurement of abstract concepts in LLM
embedding spaces* (Abonyi & Tombor, Array 32 (2026) 101249,
[doi:10.1016/j.array.2026.101249](https://doi.org/10.1016/j.array.2026.101249)),
built on [spiritwriter](https://pypi.org/project/spiritwriter/) primitives. The
paper is open access under CC BY 4.0 and is in this folder as
`semantic-rulers.pdf`.

Two interactive pages walk through the results:
[Rulers and Detours](https://aaronmarkham.github.io/semantic-ruler/) (the whole
project) and
[The Detour Map](https://aaronmarkham.github.io/semantic-ruler/detour-map.html)
(the documentation findings, as a map you walk).

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

So a smaller generator does not explain the paper's higher numbers.

### Run-to-run variance

Six anchor draws per generator (the original plus five fresh ones from the same
prompt, stored under `data/variance/`), each scored on three test sets. Mean
Pearson r, with the standard deviation across the six draws:

| Anchors from | Sonnet test A | Sonnet test B | Haiku test |
|---|---|---|---|
| Sonnet | 0.854 (0.023) | 0.853 (0.021) | 0.765 (0.028) |
| Haiku | 0.878 (0.022) | 0.844 (0.018) | 0.795 (0.021) |

- The first run was typical. Across 36 ruler/test combinations r ranges from
  0.735 to 0.898; none approaches the paper's 0.956.
- Redrawing the anchors moves r by about 0.02 (one standard deviation).
- The test set is the larger uncertainty: a bootstrap over the 72 texts gives a
  95% interval of about 0.78 to 0.91 for one ruler on one Sonnet test set.
- The apparent advantage of Haiku anchors in the cross table above does not
  hold up: it appears on test A and reverses on test B.
- Haiku's test texts are harder for every ruler.
- Pooling all six draws (48 pairs per block) gives 0.86 on the Sonnet tests, no
  better than the best single draw, so anchor count is not the limit.
- Rulers from the same generator agree at cosine 0.88-0.90; across generators
  0.80. The generator leaves a mark on the direction.

Why the paper's ruler matched its judge is still not established, but sampling
luck in this setup does not account for it. The scripts behind these checks are
in `experiments/`; each takes the repo root as its argument, for example
`python experiments/variance.py .`, and reuses stored text and cached embeddings.

## Detours in how-to docs

A second line of work in this repo measures how often a how-to page makes the
reader stop and go do something else first. A **detour** is a prerequisite the
page mentions only at the point where it is needed (or later, or never) and
gives no steps for. The experiments and their numbers are in
[experiments/doc_ordering](experiments/doc_ordering/README.md); the reusable
part is `semantic_ruler.ordering`, which reads AWS documentation pages.

```bash
semantic-ruler-docs https://docs.aws.amazon.com/waf/latest/developerguide/logging-s3.html
semantic-ruler-docs <url> --chain --max-depth 3   # follow off-page prerequisites, print one checklist
semantic-ruler-docs <url> --json
```

```python
from semantic_ruler.llm_plain import PlainTextProvider
from semantic_ruler.ordering import Fetcher, analyse, checklist, trace

llm, fetch = PlainTextProvider("claude-sonnet-5-5"), Fetcher(".cache/docs")
report = await analyse(llm, fetch(url), url)   # report.prerequisites, report.detours
tree = await trace(llm, url, fetch)            # pages opened for off-page prerequisites
items = checklist(tree)                        # one front-loaded list, in do-order
```

An interactive account of the whole project, with the data behind each chart,
is built from `experiments/explainer/` (`build_data.py` collects the numbers,
`explainer.html` is the page). A shorter, walkable version of the detour
findings is `adventure.html` in the same folder, built by `build_trail.py`; its
multi-level trees come from `experiments/doc_ordering/chains.py`.
`python -I experiments/explainer/build_site.py .` wraps both pages into the
static site in `docs/`, which GitHub Pages serves.

The prerequisites come from an LLM reading the page, so the output is a
reading, not ground truth; each detour carries a `doubts` list of scripted
reasons to distrust it. Pages are fetched only from `docs.aws.amazon.com`, and
the default provider runs the model with no tools because the prompt contains
fetched text.

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
| `semantic_ruler/ordering.py` | Doc page parsing, prerequisite extraction, detours, chains, checklists, and the `semantic-ruler-docs` CLI |
| `semantic_ruler/llm_plain.py` | An LLM provider with no tools, for prompts that contain fetched text |

## Differences from the paper

- Claude generates the text and `text-embedding-3-large` embeds it; the paper's
  primary stack was Gemini for both, with OpenAI embeddings as a cross-check.
- One LLM call returns all pairs for a block, because the provider has no seed
  parameter.
- No outlier removal, no unblocked ablation, and no images (Case Studies II and
  III) yet. The LLM-as-judge baseline is behind `--judge`.
- The paper does not give its level-guided prompt for Case Study I; the one here
  is adapted from its Case Study II template.

## Licence

The code and result files are under the [Apache License 2.0](LICENSE).

- `semantic-rulers.pdf` is the paper by J. Abonyi and R. Tombor, redistributed
  unchanged under its own licence,
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- The documentation experiments read pages from `docs.aws.amazon.com`. The raw
  pages are not in this repository; the result files hold URLs and short
  model-written descriptions of what each page asks for. This project is not
  affiliated with or endorsed by Amazon Web Services.
- Account IDs in the result files are the placeholder IDs AWS uses in its
  documentation examples.
