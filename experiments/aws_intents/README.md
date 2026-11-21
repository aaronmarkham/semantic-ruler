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

## Second test: same service, different facts

`analyze_variants.py` works on doc *sections* (heading plus prose, code
removed). 26 pages were added for same-service variants, giving 62 pages and
85 sections whose own policy JSON grants S3 access to exactly one kind of
principal, across 21 services:

| Label | Meaning | Sections |
|---|---|---|
| S | bucket policy names a service principal | 22 |
| A | bucket policy names an account, role or user | 20 |
| I | identity-based policy on a role or user | 41 |
| P | bucket policy allows anyone to read | 2 |

```bash
python experiments/aws_intents/analyze_variants.py .
```

### Raw similarity follows the service more than the facts

- For 46 of 61 sections (75%), the nearest section from the same service with
  *different* facts is closer than the nearest section from another service
  with the *same* facts.
- Nearest-neighbour label accuracy is 0.65 (majority baseline 0.48), down from
  0.87 at page level, and 0.62 when the neighbour must come from the same
  service.

### Rulers order same-service variants

Share of same-service section pairs put in the right order. "Mean" gives each
service equal weight; S3 holds most of the pairs, so it is shown separately.

| Contrast | Method | Mean over services | Services other than S3 | S3 |
|---|---|---|---|---|
| S vs I | Ruler from generated pairs | 1.00 | 6 of 6 pairs | 0.99 (78) |
| S vs I | Ruler from other services' real sections | 0.96 | | |
| S vs I | LLM judge | 1.00 | | |
| A vs I | Ruler from generated pairs | 0.91 | 11 of 11 pairs | 0.55 (182) |
| A vs I | Ruler from other services' real sections | 0.94 | | |
| A vs I | LLM judge | 0.96 | | |
| S vs A | Ruler from generated pairs | 0.88 | 3 of 3 pairs | 0.75 (84) |
| S vs A | Ruler from other services' real sections | 0.82 | | |
| S vs A | LLM judge | 1.00 | | |

A random direction scores about 0.50.

- Outside S3 the generated rulers ordered all 20 same-service pairs correctly.
  That is few pairs, spread over CloudTrail, CloudWatch Logs, SES, Athena,
  Firehose, DynamoDB, RDS and CloudFront.
- Within S3 the cross-account ruler is at chance (0.55). There the "A" label
  covers bucket policies that name a user or account for reasons other than
  crossing accounts (folder access, tagging, MFA), so the label and the
  ruler's concept do not match; the judge also drops to 0.80 pooled.
- A ruler built from other services' real section pairs, using the paper's
  paired-difference method, transfers to a held-out service (0.82 to 0.96).
  The services' fact-change vectors are only weakly aligned with each other
  (cosine 0.07 to 0.24), which is still far from orthogonal in 3,072 dimensions.
- Removing headings from the embedded text changes little (S vs I and A vs I
  unchanged outside S3; S vs A drops from 3 of 3 to 2 of 3).

Read together with the first test: raw embeddings are adequate for telling
whole jobs apart, and lose track of facts when the service is held constant.
That is where a ruler adds something.

## Third test: larger corpus, several draws, intervals

The second test rested on 20 same-service pairs outside S3 and one anchor draw.
This test replaces the hand-picked page list with a systematic crawl and adds
spread and intervals.

- `crawl_guides.py` walks the tables of contents of 100 AWS service guides,
  fetches the 40 pages per guide whose titles best match S3/permission
  keywords (2,853 pages), and keeps the 242 with S3 grants in their policy JSON.
- Merged with the hand-picked pages and de-duplicated: 375 single-principal
  sections in 78 services (S 57, A 50, I 259, P 9). The 57 services that were
  not in the second test are the confirmation set.
- `make_draws.py` generates five independent anchor draws per ruler.
- `analyze_confirm.py` scores everything. Intervals are 95%, from resampling
  services; the figure is the share of same-service pairs ordered correctly,
  averaged over services.

```bash
python -I experiments/aws_intents/crawl_guides.py data/aws-docs-cache/crawl experiments/aws_intents/corpus_crawl.json 40
python experiments/aws_intents/make_draws.py . 5
python experiments/aws_intents/analyze_confirm.py . 5
```

| Contrast (services) | Ruler from generated pairs | Ruler from other services' real sections | LLM judge |
|---|---|---|---|
| S vs I (19) | 0.84 [0.71, 0.95] | 0.99 [0.96, 1.00] | 1.00 |
| A vs I, cross-account concept (21) | 0.85 [0.74, 0.94] | 0.93 [0.82, 1.00] | 0.97 |
| A vs I, redefined concept (21) | 0.77 [0.64, 0.89] | 0.93 [0.82, 1.00] | 1.00 |
| S vs A (5) | 0.74 [0.34, 1.00] | 0.72 [0.36, 0.96] | 1.00 |

New services only:

| Contrast (new services) | Generated | From real sections | LLM judge |
|---|---|---|---|
| S vs I (10) | 0.78 [0.57, 0.97] | 1.00 (10 of 10 above chance) | 1.00 |
| A vs I, cross-account concept (11) | 0.82 [0.66, 0.97] | 0.91 [0.73, 1.00] | 0.95 |
| A vs I, redefined concept (11) | 0.76 [0.55, 0.94] | 0.91 [0.73, 1.00] | 1.00 |

A random direction averages 0.50, with a 95% range of about 0.21 to 0.79 at
this number of services.

What changed from the second test:

- **The generated rulers work, less well than first reported.** "All 20 pairs
  correct" became 0.84 and 0.85 averaged over about 20 services, with 16 or 17
  services above chance and 3 below. On new services alone the intervals reach
  down to 0.57 and 0.66. The draw used in the second test happened to be the
  best of five for S vs I (0.91; the others 0.80 to 0.86).
- **A ruler taken from the docs themselves is clearly better.** Built from
  other services' real section pairs and tested on a held-out service, it
  orders 0.99 and 0.93 of pairs correctly, and 10 of 10 and 10 of 11 new
  services are above chance. It needs labelled real pairs, which here came
  from the policy JSON.
- **The raw-similarity premise did not hold up.** On the larger corpus the
  same-service/different-facts neighbour is the closer one for 45% of sections
  (interval 39% to 51%), down from 75%. The comparison is sensitive to label
  balance: identity-policy sections are now 69% of the corpus, which enlarges
  the pool of same-facts sections in other services.
- **Redefining the A-vs-I concept to match the label made the generated ruler
  worse** (0.77 against 0.85) while the judge improved to 1.00.
- **S vs A is still undecided:** five services, intervals from 0.34 to 1.00.
- **The LLM judge is at or near 1.00 throughout.**

## Limits

- The crawl ranks pages by title keywords, so it favours pages that announce
  S3 or permissions in their titles.
- "Mean over services" gives a service with one pair the same weight as one
  with fifty; most new services have fewer than five pairs.
- The first test's numbers came from an earlier extraction; `fetch_docs.py`
  now keeps headings out of the prose, so rerunning `analyze.py` will differ
  slightly.
- The second test has 20 same-service pairs outside S3. Each contrast was run
  once, with one anchor draw.
- 23 pages, one run, one embedding model. In this corpus the mechanism label
  is confounded with workload: most service-principal pages are log delivery.
- The labels come from which policies a page happens to show, which is a proxy
  for what the job requires.
- The prose is truncated to the first 6,000 characters of bucket paragraphs.
