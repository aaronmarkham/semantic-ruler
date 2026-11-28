# Semantic drift through a fact store

What does a document lose when it is distilled into atomic facts and written back
from them? The hypothesis is that the words survive and the connective tissue goes:
the conjunctions, the conditions, and the order that says what to have ready first.
This folder holds the pipeline that measures it, a repair pass that feeds what was
lost back to the regenerator, and the recorded runs behind the
[Drift page](https://aaronmarkham.github.io/semantic-ruler/drift.html).

## The round trip

`run.py` takes a URL or a file and does five things: extract a fact graph (atoms and
conditional edges) from the source, regenerate prose from the atoms alone with the
edges withheld, extract a graph from the regeneration, score the two against each
other, and, for procedure pages, compare their prerequisite structure.

```
python -I experiments/doc_drift/run.py <url-or-file>
python -I experiments/doc_drift/run.py <url> --atomizer strict   # no conditions inside atoms
python -I experiments/doc_drift/run.py <url> --repair            # feed back what was lost, score again
python -I experiments/doc_drift/run.py <url> --repeats 3         # regenerate three times, report the range
python -I experiments/doc_drift/run.py <url> --json
```

It needs a Claude model for extraction and regeneration (through the Claude Agent SDK,
with every tool removed because the prompts carry fetched text) and an OpenAI key for
embeddings. `--mock` runs offline with a hash embedder and says nothing about meaning.

## Two extractors

The page calls them extractors; the flag calls them atomizers.

- **sentence** (default): one self-contained sentence per fact, conditions kept inside.
  Across the recorded sources 48% of its atoms carry a conditional word, so it stores
  half the connective tissue inside the facts and the round trip measures regeneration
  more than it measures the store.
- **strict** (`--atomizer strict`): no conditions inside atoms; every dependence has to be
  an edge. This is the extractor the hypothesis needs. On pages it still folds between a
  tenth and a half of the atoms.

## What is measured

The metrics are pure functions in `semantic_ruler/drift.py`.

| Layer | Function | What it asks |
|---|---|---|
| Atomic F1 | `atomic_f1` | Did the facts survive? Embedding match at 0.70 cosine, one-to-one for recall, many-to-one for precision. |
| Conditional bridge recall | `cond_bridge_recall` | Did the edges survive in a re-extraction of the regeneration? Graph against graph; noisy, because the re-extraction varies. |
| Text bridge recall | `bridge_entailment` | Does the regenerated text state or entail each original edge? One model call, one verdict and quote per edge. The measure the repair comparison uses. |
| Entailment probes | `entailment_rate` | Does a reader of the regeneration answer multi-fact yes/no questions right? |
| Structural drift | `structural_drift` | For procedures: did the prerequisites survive, and did they move from up front to the step that needs them? Uses the detour extractor from `semantic_ruler.ordering`. |
| Granularity | `granularity` | How much did the extractor keep inside the atoms: atoms per sentence, share carrying a condition. |
| Conditional load | `conditional_load` | How many ifs, ands and buts the source asks a reader to hold. |
| Composite | `drift_score` | Weighted losses: atomic 0.25, graph bridges 0.50, probes 0.25, structural 0.20 when applicable. |

Every metric also returns what it could not find: lost edge indices, unmatched and
sprung prerequisites, unsupported atoms. The repair pass (`--repair`) hands those back
to the regenerator, as relationships stated in words and prerequisites to put first,
and scores the new draft. `--repair-scope` chooses what is fed back: the edges the
graph matcher lost (default), the ones the text reader lost (`text`), or all of them.

## The recorded runs

`results/` is what the page is built from. Each `*-runs.json` holds repeated runs on one
source: the original graph, the regenerated text, the regenerated graph, the probes,
every score, the structural reports, the repair pass with its own scores, and the text
bridge verdicts.

| Folder | Extractor | Contents |
|---|---|---|
| `results/` | sentence | `lambda` (the Lambda logging passage, 5 runs), `s3public` (the Block Public Access passage, 5), `cwlogs` (CloudWatch Logs export page, 3), `waf` (WAF logging page, 3) |
| `results/cohort/` | sentence | ten procedure pages from the detour set, 2 runs each with repair |
| `results/strict/`, `results/strict/cohort/` | strict | the same sources |
| `results/before-fix/` | sentence | the Lambda and WAF runs made before the matchers were calibrated |
| `results/variants/` | sentence | runs that fed back every edge instead of the lost ones |

The two passages are in `results/lambda-logging.txt` and `results/s3-public-access.txt`.
The pages are fetched from docs.aws.amazon.com and not stored; the prompts that carried
their text are stripped from the records.

To make more: `record.py` runs one source N times and writes a runs file; `cohort.sh`
runs a list of pages two times each with repair; `rescore_bridge.py` adds the text bridge
measure to runs recorded before it existed; `regrade_probes.py` re-grades a run's probes
by the reply's first word and keeps the replies (the recorded runs have all been through it;
each run's stored question set was put to both drafts, so `entailment_before_regrade`
holds the earlier figure). The probes' expected answers lean yes by about seven to three,
so a pass rate sits above a floor near 70%. `python -I experiments/explainer/build_drift.py .`
turns the results into the page's data file, and
`python -I experiments/explainer/build_site.py .` rebuilds the page.

## What the runs showed

The Drift page has the full account. In short: the atoms come back at 88 to 100% from
every source under either extractor. Under the sentence extractor the two passages come
through almost untouched and the ten pages lose about half their bridges and displace
their prerequisites on six. Under the strict extractor the passages lose half their
probes, the pages spring half their prerequisites on the reader, and the repair pass,
read from the text, puts the bridges back on eight pages of ten and the order back on
every page where it had moved. Conditional load does not predict any of it at thirteen
sources. Ten pages is a rate, not a law.
