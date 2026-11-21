"""Systematic corpus: walk AWS doc guides' tables of contents instead of hand-picking pages.

For each guide, pages are ranked by title keywords (S3, bucket, permissions,
cross-account, ...) and the top N are fetched. Only pages with at least one
section whose policy JSON grants S3 access are kept. Stdlib only.

    python -I experiments/aws_intents/crawl_guides.py <cache_dir> <corpus_crawl.json> [pages_per_guide]
"""

import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from facts import facts  # noqa: E402
from fetch_docs import DocParser, policies  # noqa: E402

D = "https://docs.aws.amazon.com"
UA = {"User-Agent": "Mozilla/5.0 (compatible; docs-reader)"}
# (service, guide path)
GUIDES = [
    ("S3", "AmazonS3/latest/userguide"), ("CloudTrail", "awscloudtrail/latest/userguide"),
    ("Elastic Load Balancing", "elasticloadbalancing/latest/application"),
    ("Elastic Load Balancing", "elasticloadbalancing/latest/network"),
    ("Elastic Load Balancing", "elasticloadbalancing/latest/classic"),
    ("VPC", "vpc/latest/userguide"), ("Config", "config/latest/developerguide"),
    ("WAF", "waf/latest/developerguide"), ("GuardDuty", "guardduty/latest/ug"),
    ("Redshift", "redshift/latest/mgmt"), ("Redshift", "redshift/latest/dg"),
    ("CloudFront", "AmazonCloudFront/latest/DeveloperGuide"), ("SES", "ses/latest/dg"),
    ("Data Firehose", "firehose/latest/dev"), ("Athena", "athena/latest/ug"),
    ("Lambda", "lambda/latest/dg"), ("Glue", "glue/latest/dg"), ("SageMaker", "sagemaker/latest/dg"),
    ("CodePipeline", "codepipeline/latest/userguide"), ("CodeBuild", "codebuild/latest/userguide"),
    ("Transfer Family", "transfer/latest/userguide"), ("DataSync", "datasync/latest/userguide"),
    ("Bedrock", "bedrock/latest/userguide"), ("Systems Manager", "systems-manager/latest/userguide"),
    ("EMR", "emr/latest/ManagementGuide"), ("Backup", "aws-backup/latest/devguide"),
    ("Macie", "macie/latest/user"), ("MediaConvert", "mediaconvert/latest/ug"),
    ("Transcribe", "transcribe/latest/dg"), ("CloudFormation", "AWSCloudFormation/latest/UserGuide"),
    ("IoT", "iot/latest/developerguide"), ("CloudWatch Logs", "AmazonCloudWatch/latest/logs"),
    ("CloudWatch", "AmazonCloudWatch/latest/monitoring"),
    ("DynamoDB", "amazondynamodb/latest/developerguide"), ("RDS", "AmazonRDS/latest/UserGuide"),
    ("Aurora", "AmazonRDS/latest/AuroraUserGuide"), ("MSK", "msk/latest/developerguide"),
    ("OpenSearch Service", "opensearch-service/latest/developerguide"),
    ("Step Functions", "step-functions/latest/dg"), ("Inspector", "inspector/latest/user"),
    ("Security Hub", "securityhub/latest/userguide"), ("Security Lake", "security-lake/latest/userguide"),
    ("QuickSight", "quicksight/latest/user"), ("Comprehend", "comprehend/latest/dg"),
    ("Textract", "textract/latest/dg"), ("Rekognition", "rekognition/latest/dg"),
    ("Personalize", "personalize/latest/dg"), ("Forecast", "forecast/latest/dg"),
    ("Kendra", "kendra/latest/dg"), ("DMS", "dms/latest/userguide"),
    ("Storage Gateway", "filegateway/latest/files3"), ("FSx", "fsx/latest/LustreGuide"),
    ("ECS", "AmazonECS/latest/developerguide"), ("EKS", "eks/latest/userguide"),
    ("Batch", "batch/latest/userguide"), ("AppFlow", "appflow/latest/userguide"),
    ("Connect", "connect/latest/adminguide"), ("API Gateway", "apigateway/latest/developerguide"),
    ("EventBridge", "eventbridge/latest/userguide"), ("SNS", "sns/latest/dg"),
    ("Kinesis Video Streams", "kinesisvideostreams/latest/dg"), ("IoT SiteWise", "iot-sitewise/latest/userguide"),
    ("HealthLake", "healthlake/latest/devguide"), ("Neptune", "neptune/latest/userguide"),
    ("DocumentDB", "documentdb/latest/developerguide"), ("Timestream", "timestream/latest/developerguide"),
    ("ElastiCache", "AmazonElastiCache/latest/dg"), ("EC2", "AWSEC2/latest/UserGuide"),
    ("VM Import/Export", "vm-import/latest/userguide"), ("Image Builder", "imagebuilder/latest/userguide"),
    ("Cost and Usage Reports", "cur/latest/userguide"), ("Billing", "awsaccountbilling/latest/aboutv2"),
    ("Lake Formation", "lake-formation/latest/dg"), ("EMR Serverless", "emr/latest/EMR-Serverless-UserGuide"),
    ("MWAA", "mwaa/latest/userguide"), ("Managed Flink", "managed-flink/latest/java"),
    ("Kinesis Data Streams", "streams/latest/dev"), ("Elastic Beanstalk", "elasticbeanstalk/latest/dg"),
    ("Amplify", "amplify/latest/userguide"), ("CodeDeploy", "codedeploy/latest/userguide"),
    ("CodeArtifact", "codeartifact/latest/ug"), ("Audit Manager", "audit-manager/latest/userguide"),
    ("Detective", "detective/latest/userguide"), ("Network Firewall", "network-firewall/latest/developerguide"),
    ("Route 53", "Route53/latest/DeveloperGuide"), ("Global Accelerator", "global-accelerator/latest/dg"),
    ("Application Migration", "mgn/latest/ug"), ("WorkSpaces", "workspaces/latest/adminguide"),
    ("Clean Rooms", "clean-rooms/latest/userguide"), ("Entity Resolution", "entityresolution/latest/userguide"),
    ("HealthOmics", "omics/latest/dev"), ("Ground Station", "ground-station/latest/ug"),
    ("Location", "location/latest/developerguide"), ("Pinpoint", "pinpoint/latest/userguide"),
    ("IAM", "IAM/latest/UserGuide"), ("KMS", "kms/latest/developerguide"),
    ("Organizations", "organizations/latest/userguide"), ("Control Tower", "controltower/latest/userguide"),
]
WEIGHTS = [(r"\bs3\b|amazon s3|bucket", 5), (r"cross.account", 4), (r"permission|polic", 3),
           (r"\brole\b|\biam\b|access", 2), (r"log|export|import|deliver|destination|encrypt", 2)]


def get(url: str, dest: Path | None = None) -> str | None:
    if dest is not None and dest.exists():
        return dest.read_text(encoding="utf-8")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                body = r.read().decode("utf-8", "replace")
            if dest is not None:
                dest.write_text(body, encoding="utf-8")
            time.sleep(0.2)
            return body
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2 * (attempt + 1))
    return None


def flatten(node, out):
    for item in node.get("contents", []):
        if item.get("href") and "#" not in item["href"] and item["href"].endswith(".html"):
            out.append((item.get("title", ""), item["href"]))
        flatten(item, out)


def rank(title: str) -> int:
    t = title.lower()
    return sum(w for pat, w in WEIGHTS if re.search(pat, t))


def page(service, guide, title, href, cache: Path):
    slug = re.sub(r"[^A-Za-z0-9]+", "-", f"{guide}/{href}")[:150]
    body = get(f"{D}/{guide}/{href}", cache / f"{slug}.html")
    if body is None:
        return None
    p = DocParser()
    p.feed(body)
    p._flush()
    sections = [{"heading": s["heading"], "level": s["level"], "prose": s["prose"], "policies": policies(s["code"])}
                for s in p.sections if s["prose"] or s["code"]]
    if not any(facts(s)["n_s3_allow"] for s in sections):
        return None
    return {"job": f"{guide}/{href}", "service": service, "url": f"{D}/{guide}/{href}", "title": title, "sections": sections}


def main(cache_dir: str, out_path: str, per_guide: int) -> None:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    corpus, total = [], 0
    for service, guide in GUIDES:
        toc = get(f"{D}/{guide}/toc-contents.json", cache / (re.sub(r"[^A-Za-z0-9]+", "-", guide) + "-toc.json"))
        if toc is None:
            print(f"  {service:<26} no table of contents at {guide}")
            continue
        pages: list[tuple[str, str]] = []
        flatten(json.loads(toc), pages)
        pages = list(dict.fromkeys(pages))
        chosen = [p for p in sorted(pages, key=lambda p: -rank(p[0])) if rank(p[0]) > 0][:per_guide]
        with ThreadPoolExecutor(4) as pool:
            found = [r for r in pool.map(lambda p: page(service, guide, p[0], p[1], cache), chosen) if r]
        total += len(chosen)
        n_sec = sum(1 for j in found for s in j["sections"] if facts(s)["n_s3_allow"])
        print(f"  {service:<26} {len(pages):>5} pages in guide, fetched {len(chosen):>3}, kept {len(found):>3} with {n_sec:>3} S3-grant sections")
        corpus.extend(found)
    Path(out_path).write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    print(f"{len(corpus)} pages kept of {total} fetched -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 40)
