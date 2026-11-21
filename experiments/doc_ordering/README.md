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

## Results (2026-11-21, one extraction run)

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
- The extraction is not stable on judgement calls. Run twice, the WAF S3 page
  was classed once as a reference page and once as a procedure, and its bucket
  policy once as "steps on this page" and once as "steps on another page".
- "Needed at" is the hardest field. On reference pages it coincides with
  "introduced at" almost by construction, so the surprise rate is reported for
  procedure pages only.
- The sample is drawn from pages that have S3 grants in their policy JSON, so
  it over-represents jobs with permission prerequisites.
- The tracer follows optional prerequisites as readily as required ones (the
  Access Analyzer branch above is optional), takes only the S3 branch at hub
  pages, and failed on the S3 "Creating a bucket" page, whose content did not
  parse.
