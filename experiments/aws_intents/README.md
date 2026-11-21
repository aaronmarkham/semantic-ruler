# AWS bucket intents

Question: can semantic rulers pull the *intent* of a job apart from its shared
steps? "Create an S3 bucket" appears in many AWS jobs, but how access to the
bucket is granted differs. This experiment checks whether rulers built from
generic counterfactual pairs can read that difference off real documentation.

## Method

1. `fetch_docs.py` fetches 36 public AWS doc pages about jobs that involve a
   bucket. From each page it keeps the paragraphs that mention a bucket (code
   removed) and, separately, the policy JSON in code blocks. Raw HTML and the
   extracted prose are gitignored; rerun the script to regenerate them.
2. `facts.py` derives permission facts from the policy JSON alone, with no LLM
   and without reading the prose. 23 pages have S3 grants in their policies:
   11 grant an AWS service principal through the bucket policy, 10 grant an
   IAM role or user through an identity policy, 2 allow public read.
3. `analyze.py` embeds the prose (`text-embedding-3-large`), builds two rulers
   from pairs that Claude writes about *fictional* services in four workload
   blocks, scores the real prose, and compares against the facts.

```bash
python -I experiments/aws_intents/fetch_docs.py data/aws-docs-cache experiments/aws_intents/corpus.json
python -I experiments/aws_intents/facts.py experiments/aws_intents/corpus.json experiments/aws_intents/facts.json
python experiments/aws_intents/analyze.py .
```

## Results (2026-11-21, one run, n = 23)

| Method | Mechanism AUC (11 of 23) | Exposure AUC (2 of 23) |
|---|---|---|
| Ruler from generated pairs | 0.708 | 1.000 |
| Cosine to the positive-label text | 0.326 | 0.952 |
| Random direction (mean of 200) | 0.495 | 0.515 |
| Supervised centroid difference on the real docs, leave-one-out | 0.879 | n/a |
| LLM judge reading the prose | 1.000 | 1.000 |

Raw embeddings, leave-one-out nearest neighbour over the three classes: 0.87
(majority baseline 0.48).

- Raw embeddings already group these jobs by permission pattern fairly well.
  The expectation that shared topic would swamp the difference was wrong here:
  mean cosine is 0.61 within a class and 0.57 between classes.
- Subtracting the corpus centroid changes little (1-NN 0.91, one more page of
  23). Removing a constant vector cannot change Euclidean neighbours.
- The mechanism ruler transfers weakly from synthetic pairs to real prose. Its
  clearest error is ranking the two static-website pages at the top: it picked
  up "the bucket policy carries the grant", not "the principal is a service".
  Post hoc, excluding those two pages gives AUC 0.836, and scoring against
  "any bucket-policy grant" gives 0.862. Those are readings chosen after
  seeing the result.
- The exposure ruler separates the two public pages perfectly, but with two
  positives and a cosine baseline at 0.95 that is weak evidence.
- An LLM judge reading the same prose is perfect on both labels.
- The two ruler scores used as coordinates classify the three classes at 0.65
  leave-one-out, below raw nearest neighbour.

## Limits

- 23 pages, one run, one embedding model. In this corpus the mechanism label
  is confounded with workload: most service-principal pages are log delivery.
- The labels come from which policies a page happens to show, which is a proxy
  for what the job requires.
- The prose is truncated to the first 6,000 characters of bucket paragraphs.
