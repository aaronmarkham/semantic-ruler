"""Fetch public AWS documentation pages for jobs that involve an S3 bucket.

Stdlib only. Raw HTML goes to a cache directory (gitignored); the derived corpus
(bucket-focused prose with code removed, plus the policy JSON found in code
blocks) goes to corpus.json next to this script.

    python -I experiments/aws_intents/fetch_docs.py <cache_dir> <corpus.json>
"""

import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

D = "https://docs.aws.amazon.com"
# (job id, service, url)
JOBS = [
    ("cloudtrail-trail-logs", "CloudTrail", f"{D}/awscloudtrail/latest/userguide/create-s3-bucket-policy-for-cloudtrail.html"),
    ("alb-access-logs", "Elastic Load Balancing", f"{D}/elasticloadbalancing/latest/application/enable-access-logging.html"),
    ("nlb-access-logs", "Elastic Load Balancing", f"{D}/elasticloadbalancing/latest/network/load-balancer-access-logs.html"),
    ("vpc-flow-logs", "VPC", f"{D}/vpc/latest/userguide/flow-logs-s3.html"),
    ("config-delivery", "Config", f"{D}/config/latest/developerguide/s3-bucket-policy.html"),
    ("s3-server-access-logs", "S3", f"{D}/AmazonS3/latest/userguide/enable-server-access-logging.html"),
    ("s3-inventory-destination", "S3", f"{D}/AmazonS3/latest/userguide/configure-inventory.html"),
    ("waf-logs", "WAF", f"{D}/waf/latest/developerguide/logging-s3.html"),
    ("guardduty-export", "GuardDuty", f"{D}/guardduty/latest/ug/guardduty_exportfindings.html"),
    ("redshift-audit-logs", "Redshift", f"{D}/redshift/latest/mgmt/db-auditing.html"),
    ("cloudfront-standard-logs", "CloudFront", f"{D}/AmazonCloudFront/latest/DeveloperGuide/standard-logging-legacy-s3.html"),
    ("ses-receive-to-s3", "SES", f"{D}/ses/latest/dg/receiving-email-permissions.html"),
    ("static-website", "S3", f"{D}/AmazonS3/latest/userguide/WebsiteAccessPermissionsReqd.html"),
    ("static-website-tutorial", "S3", f"{D}/AmazonS3/latest/userguide/HostingWebsiteOnS3Setup.html"),
    ("cloudfront-oac-origin", "CloudFront", f"{D}/AmazonCloudFront/latest/DeveloperGuide/private-content-restricting-access-to-s3.html"),
    ("firehose-s3-destination", "Data Firehose", f"{D}/firehose/latest/dev/controlling-access.html"),
    ("s3-replication", "S3", f"{D}/AmazonS3/latest/userguide/setting-repl-config-perm-overview.html"),
    ("athena-query-results", "Athena", f"{D}/athena/latest/ug/querying.html"),
    ("athena-s3-access", "Athena", f"{D}/athena/latest/ug/s3-permissions.html"),
    ("lambda-s3-trigger", "Lambda", f"{D}/lambda/latest/dg/with-s3-example.html"),
    ("glue-crawler-role", "Glue", f"{D}/glue/latest/dg/create-an-iam-role.html"),
    ("sagemaker-execution-role", "SageMaker", f"{D}/sagemaker/latest/dg/sagemaker-roles.html"),
    ("codepipeline-artifacts", "CodePipeline", f"{D}/codepipeline/latest/userguide/S3-artifact-encryption.html"),
    ("transfer-family-sftp", "Transfer Family", f"{D}/transfer/latest/userguide/requirements-roles.html"),
    ("datasync-s3-location", "DataSync", f"{D}/datasync/latest/userguide/create-s3-location.html"),
    ("bedrock-kb-s3", "Bedrock", f"{D}/bedrock/latest/userguide/kb-permissions.html"),
    ("ssm-session-logs", "Systems Manager", f"{D}/systems-manager/latest/userguide/session-manager-logging.html"),
    ("emr-logs", "EMR", f"{D}/emr/latest/ManagementGuide/emr-plan-debugging.html"),
    ("backup-s3", "Backup", f"{D}/aws-backup/latest/devguide/s3-backups.html"),
    ("macie-results", "Macie", f"{D}/macie/latest/user/discovery-results-repository-s3.html"),
    ("cross-account-bucket", "S3", f"{D}/AmazonS3/latest/userguide/example-walkthroughs-managing-access-example2.html"),
    ("presigned-urls", "S3", f"{D}/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html"),
    ("mediaconvert-role", "MediaConvert", f"{D}/mediaconvert/latest/ug/creating-the-iam-role-in-iam.html"),
    ("transcribe-s3", "Transcribe", f"{D}/transcribe/latest/dg/security_iam_id-based-policy-examples.html"),
    ("cloudformation-stacksets-template", "CloudFormation", f"{D}/AWSCloudFormation/latest/UserGuide/cfn-console-create-stack.html"),
    ("kinesis-video-or-iot", "IoT", f"{D}/iot/latest/developerguide/s3-rule-action.html"),
]

BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "dt", "dd", "td", "th", "div", "br", "tr"}


class DocParser(HTMLParser):
    """Split a docs page into prose blocks and code blocks."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_main = 0
        self.skip = 0
        self.in_pre = 0
        self.prose: list[str] = []
        self.code: list[str] = []
        self._buf: list[str] = []
        self._code: list[str] = []
        self.title = ""
        self._in_title = False

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        if text:
            self.prose.append(text)
        self._buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "title":
            self._in_title = True
        if a.get("id") == "main-col-body" or (tag == "main" and not self.in_main):
            self.in_main = 1
        if not self.in_main:
            return
        if tag in ("script", "style", "nav", "awsdocs-language-banner", "awsdocs-page-header"):
            self.skip += 1
        if tag == "pre":
            self._flush()
            self.in_pre += 1
        elif tag in BLOCK and not self.in_pre:
            self._flush()

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if not self.in_main:
            return
        if tag in ("script", "style", "nav", "awsdocs-language-banner", "awsdocs-page-header"):
            self.skip = max(0, self.skip - 1)
        if tag == "pre" and self.in_pre:
            self.in_pre -= 1
            if not self.in_pre:
                self.code.append("".join(self._code).strip())
                self._code = []
        elif tag in BLOCK and not self.in_pre:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if not self.in_main or self.skip:
            return
        (self._code if self.in_pre else self._buf).append(data)


def fetch(url: str, dest: Path) -> str | None:
    if dest.exists():
        return dest.read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; docs-reader)"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode("utf-8", "replace")
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"  FAILED {url}: {e}")
        return None
    dest.write_text(body, encoding="utf-8")
    time.sleep(0.5)
    return body


def policies(code_blocks: list[str]) -> list[dict]:
    """Parse code blocks that are IAM-style policy documents."""
    out = []
    for block in code_blocks:
        if '"Statement"' not in block:
            continue
        text = html.unescape(block)
        start = text.find("{")
        if start < 0:
            continue
        try:
            doc = json.JSONDecoder().raw_decode(text, start)[0]
        except json.JSONDecodeError:
            # docs often use placeholders or trailing commas; keep the raw text
            out.append({"unparsed": text[:4000]})
            continue
        if isinstance(doc, dict) and "Statement" in doc:
            out.append(doc)
    return out


def main(cache_dir: str, corpus_path: str) -> None:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    corpus = []
    for job, service, url in JOBS:
        body = fetch(url, cache / f"{job}.html")
        if body is None:
            continue
        p = DocParser()
        p.feed(body)
        p._flush()
        bucket = [t for t in p.prose if re.search(r"\bbucket", t, re.I)]
        pol = policies(p.code)
        print(f"  {job:<36} prose {len(p.prose):>4} blocks, bucket {len(bucket):>3}, code {len(p.code):>3}, policies {len(pol):>2}  | {p.title.strip()[:60]}")
        corpus.append({
            "job": job, "service": service, "url": url, "title": p.title.strip(),
            "n_prose_blocks": len(p.prose), "bucket_prose": bucket, "policies": pol,
        })
    Path(corpus_path).write_text(json.dumps(corpus, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(corpus)}/{len(JOBS)} pages -> {corpus_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
