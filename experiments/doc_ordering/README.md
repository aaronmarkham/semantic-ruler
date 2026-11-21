# Doc ordering and backpedal

Question: how often does a how-to page make the reader stop and go do something
else first? The motivating case was enabling AWS WAF logging: start the
procedure, find it needs a bucket, go make one, find the bucket needs a policy,
go learn that, and end up with several lines of investigation open.

## Method

- `stream.py` turns a cached docs page into ordered, numbered blocks: headings,
  paragraphs with their links to other doc pages, and code blocks (policy JSON
  is summarised by principal kind).
- `extract.py` holds the prompt. For each page Claude lists the prerequisites
  of the page's job, the block where the reader first needs each one, the block
  where the page first says so, and whether the steps for it are on this page,
  on another page, or missing.
- `backpedal.py` runs this over a sample of procedural pages (up to two per
  service) from the `aws_intents` corpus and computes the measures below.
- `chain.py` starts at one page and follows every prerequisite whose steps are
  on another page, printing the tree of pages a reader ends up with open.
- `reorder.py` turns the saved extractions into front-loaded checklists.
- `stability.py` repeats the extraction and measures agreement between runs.
- `detours.py` runs the extraction on every cached page and reports detours.
- `join_intents.py` joins the result to the permission facts and rulers of
  `aws_intents`.

Prompts contain fetched doc text, so they go through
`semantic_ruler.llm_plain.PlainTextProvider`, which runs the model with no
tools; both scripts assert that the session reported none.

```bash
python experiments/doc_ordering/backpedal.py . 60 2
python experiments/doc_ordering/chain.py . https://docs.aws.amazon.com/waf/latest/developerguide/logging-management-configure.html 4 14
```

Definitions: a prerequisite is a **surprise** when the page mentions it only at
the block where it is needed, later, or never. It is **off-page** when the page
sends the reader elsewhere for the steps.

## Results (2026-11-21, extraction run 1; see Stability for runs 2 and 3)

60 pages from 42 services were analysed. 52 were step-by-step procedures with
216 prerequisites between them, 4.2 per page.

| Measure, procedure pages | Value |
|---|---|
| Prerequisites that are surprises | 31% |
| Prerequisites with steps on another page | 24% |
| Prerequisites with no steps anywhere on the page | 40% |
| Pages that send the reader to at least one other page | 34 of 52 |
| Pages with three or more open lines (other pages plus on-page surprises) | 14 of 52 |

By type, on procedure pages:

| Prerequisite | Count | Surprise | Off-page | No steps |
|---|---|---|---|---|
| S3 bucket | 41 | 29% | 15% | 63% |
| Bucket policy | 20 | 20% | 5% | 0% |
| IAM role | 27 | 26% | 37% | 22% |
| IAM identity policy | 32 | 34% | 22% | 16% |
| KMS key | 12 | 42% | 42% | 50% |
| KMS key policy | 15 | 47% | 20% | 13% |

Bucket policies are the best-handled prerequisite: the steps are always on the
page. Buckets themselves are usually assumed to exist. KMS keys and key
policies are the most likely to appear without warning.

### The WAF logging job across pages

```
Configuring logging for a protection pack (web ACL)        procedure
  needs: a configured logging destination                  -> another page
  AWS WAF logging destinations                             hub page
    Sending traffic logs to an Amazon S3 bucket
      needs: S3 bucket named aws-waf-logs-...              -> S3 guide (not traced, see limits)
      needs: identity permissions for the WAF admin        -> IAM guide
      needs: bucket policy for the log delivery service    -> S3 guide
      needs: KMS key policy, if the bucket uses a KMS key     on this page
      Access management for AWS resources                  reference page
      Adding a bucket policy by using the S3 console       procedure
        needs: console access, an existing bucket, and optionally an Access Analyzer
        Getting started with IAM Access Analyzer           4 further prerequisites
```

Six pages across three guides and four levels, starting from a procedure whose
own page names one prerequisite. The bucket-creation page would be a seventh.

## Reordering into a checklist

`reorder.py` uses no LLM. It reads the saved extractions and does two things.

```bash
python experiments/doc_ordering/reorder.py .
```

**Per page.** A checklist before step 1 removes a surprise but not the work.
Of the 68 surprises, 19 have their steps on the page, so the page is walking
the reader through them in order and nothing needs moving. The other 49 are
detours: 17 send the reader to another page and 32 give no steps. That is 23%
of all prerequisites. 32 of the 52 pages have no detours, so a checklist would
change 20 pages.

**Across pages.** The WAF tree above flattens to
[checklist_waf.md](checklist_waf.md): 12 prerequisites met over six pages
become 10 items, 4 required and 6 that apply only in some setups.

1. Identity permissions for the WAF administrator
2. A bucket named `aws-waf-logs-...` in the same account
3. Console access to edit the bucket's policy
4. The bucket policy for the log delivery service (may be created for you)

Items are ordered so each comes after what it depends on: a followed page's
own prerequisites go before the thing that page creates, and within a page a
fixed type order applies (access, then resources, then the policies that refer
to them). A followed page that asks for "an existing bucket" when its parent
already lists one is merged into the parent's item. A prerequisite that led to
a hub page is replaced by the option taken there. Everything under a
conditional item is conditional.

Limits of this step: "conditional" and "may be created for you" come from
keyword matches on the extracted description, the type order is a convention
rather than something read from the docs, and the checklist inherits every
error in the extraction. Only the WAF checklist was read against the pages.
An earlier measure, prerequisites first mentioned after the main procedure
begins, was dropped because the extracted start block is often the top of the
page.

## Stability across three extraction runs

`stability.py` ran the extraction twice more on the same 60 pages and compared
every pair of runs. Prerequisites are matched between runs by type and word
overlap, so item-level agreement is measured on the items both runs found.

| Field | Agreement between two runs |
|---|---|
| Page is a procedure | 0.94 |
| Share of prerequisites found by both runs | 0.88 |
| Steps on page / other page / none | 0.94 |
| Same link, when both say other page | 1.00 |
| Introduced-at block, within 3 | 0.92 |
| Needed-at block, within 3 | 0.86 |
| Surprise or not | 0.92 |

| Measure, procedure pages | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Pages / prerequisites | 52 / 216 | 54 / 237 | 52 / 220 |
| Surprises | 31% | 32% | 26% |
| Steps on another page | 24% | 26% | 25% |
| No steps | 40% | 41% | 38% |
| Detours | 23% | 24% | 19% |
| Pages that send the reader elsewhere | 34 | 37 | 32 |

223 prerequisites were found by all three runs. Of the 206 on procedure pages,
49 are surprises in every run, 132 in none and 25 are split, so the surprise
rate among agreed items is between 24% and 36%.

The headline rates hold to within about six points. Where steps live is the
most repeatable field and "needed at" the least. The per-type surprise rates
do not hold: identity policies came out at 34%, 32% and 15%, and the KMS rows
have 12 to 15 items each. Read the by-type table above for its off-page and
no-steps columns only.

## Joining ordering to permission mechanisms

Question: do surprises concentrate on pages whose permission mechanism differs
from that of similar-looking pages in other services? `join_intents.py` takes
each page's mechanism from its policy JSON (S: bucket policy names a service
principal, A: bucket policy names an account or role, I: identity policy, P:
public read), finds its 5 most similar pages from other services by raw
embedding, and scores mismatch two ways: by the facts (Jaccard distance
between mechanism sets) and by a ruler (distance along the S-vs-I direction
built from other services' real sections). The outcome is the page's count of
permission prerequisites that are detours, averaged over the three runs.

```bash
python experiments/doc_ordering/join_intents.py . 5
```

48 procedure pages could be joined. Spearman correlations:

| Predictor | vs permission detours | 95% interval |
|---|---|---|
| Mechanism differs from neighbours (facts) | -0.13 | -0.41 to +0.19 |
| Mechanism differs from neighbours (ruler) | -0.20 | -0.48 to +0.11 |
| Similarity to neighbours (control) | +0.08 | -0.21 to +0.34 |

**The hypothesis is not supported.** No mismatch measure predicts detours,
surprises or off-page steps, at 5 or 10 neighbours, and the point estimates
lean the other way. With 48 pages the intervals still allow a moderate effect
in either direction.

Two things did show up, both exploratory (16 tests were run, and these were
not the ones planned):

- Pages that use more than one mechanism send the reader off-page for more of
  their permission prerequisites: 28% of them on the 10 multi-mechanism pages
  against 15% on the 28 single-mechanism pages (Spearman 0.41 on the rate,
  permutation p = 0.01).
- The mechanism itself seems to matter more than how it compares to
  neighbours. None of the 9 service-principal pages sends the reader away for
  a permission prerequisite; 4 of 22 identity-policy pages and 2 of 4
  account-principal pages do. These counts are too small to test.

One check passed: the page-level ruler, built without the page's own service,
orders S-only pages above I-only pages with AUC 0.94 across 192 pages, so the
null result is not because the ruler fails to read the mechanism.

## Detours on every cached page

A **detour** is a prerequisite the page springs on the reader (a surprise) and
gives no steps for. It is either *sent elsewhere* (the page links to another
page) or *left alone* (no steps, no link). `detours.py` extends the extraction
from the first 60 pages to all 281 cached pages and reports the metric.

```bash
python experiments/doc_ordering/detours.py .
```

219 of the 281 pages are procedures, from 75 services, with 842
prerequisites. Intervals resample services.

| Measure | All 219 pages | First sample (52) | New pages (167) |
|---|---|---|---|
| Detours per prerequisite | 21% [16, 27] | 23% [14, 32] | 21% [16, 27] |
| of which sent elsewhere | 9% [6, 12] | 8% | 9% |
| of which left alone | 12% [9, 17] | 15% | 12% |
| Pages with at least one detour | 46% [37, 55] | 38% | 49% |
| Pages with two or more | 20% [14, 26] | 23% | 19% |

The rate found on the first sample holds on the pages outside it.

How the 181 detours arise: 128 are mentioned at the very block where they are
needed, 46 after it, and 7 never. Two scripted checks flag 19 as doubtful (10
sit in a prerequisites section, so they are announced up front; 9 have the
need placed at the top of the page, which makes anything later look late).
Without those the rate is 19%.

| Prerequisite | Count | Detour | Sent elsewhere | Left alone |
|---|---|---|---|---|
| Tool or access | 60 | 35% | 8% | 27% |
| KMS key | 47 | 32% | 9% | 23% |
| IAM role | 110 | 25% | 11% | 14% |
| Other AWS resource | 196 | 25% | 12% | 13% |
| S3 bucket | 153 | 22% | 3% | 18% |
| Account or service setting | 37 | 22% | 19% | 3% |
| KMS key policy | 47 | 21% | 17% | 4% |
| IAM identity policy | 134 | 13% | 9% | 4% |
| Bucket policy | 58 | 0% | 0% | 0% |

No bucket policy in 58 is a detour: pages that need one give it. Buckets, KMS
keys and tools are the ones readers are left alone with.

### Mechanism, tested on the new pages

The two exploratory findings from the join were tested on the 167 procedure
pages that were not in the first sample.

| Finding | Result on new pages | Confirmed |
|---|---|---|
| Multi-mechanism pages send more permission prerequisites off-page | 39% on 22 pages vs 29% on 92, p = 0.27 | No |
| Service-principal pages send the reader away for a permission prerequisite less often than identity-policy pages | 2 of 21 vs 34 of 98, p = 0.03 | Yes |
| Same comparison, any detour | 5 of 21 vs 51 of 98, p = 0.03 | Yes |
| Same comparison, permission detours | 3 of 21 vs 22 of 98, p = 0.57 | No |

Over all 219 pages, 23% of service-principal pages have a detour against 52%
of identity-policy pages. The permutation tests treat pages as independent,
which they are not within a service, so read the p-values as loose.

By service, among those with three or more procedure pages, the detour rate
runs from 0% (Elastic Load Balancing, RDS, DataSync, MWAA) to 90% (Glue, 10
prerequisites). These are small counts per service.

Limits of the detour numbers: the 221 added pages have one extraction run
each; six "left alone" detours were read against their pages and two of them
were artefacts of the need being placed too early, which is what the second
scripted check now flags; and pages the model classes as procedures include
some that describe a setup in prose and have no numbered steps.

## Checks on the extraction

- No block number was out of range.
- 60 of 61 cited off-page links are really on the page.
- 36 of 44 policy prerequisites point to within 25 blocks before the matching
  policy JSON, which is located without an LLM.
- One page (WAF logs to S3) was read block by block against its extraction;
  the four prerequisites and their positions matched.

## Limits

- The prerequisite lists are an LLM's reading with no independent ground
  truth beyond the checks above. One page checked by reading is not a
  validation sample.
- Three runs agree on most fields (see Stability), but single pages still
  flip: the WAF S3 page was classed once as a reference page and once as a
  procedure. The chain trace and the checklist come from one run each.
- "Needed at" is the hardest field. On reference pages it coincides with
  "introduced at" almost by construction, so the surprise rate is reported for
  procedure pages only.
- The sample is drawn from pages that have S3 grants in their policy JSON, so
  it over-represents jobs with permission prerequisites.
- The tracer follows optional prerequisites as readily as required ones (the
  Access Analyzer branch above is optional), takes only the S3 branch at hub
  pages, and failed on the S3 "Creating a bucket" page, whose content did not
  parse.
