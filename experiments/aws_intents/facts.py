"""Derive permission facts for each job from the policy JSON in its doc page.

Deterministic: no LLM, and it never reads the prose that gets embedded.

    python -I experiments/aws_intents/facts.py experiments/aws_intents/corpus.json experiments/aws_intents/facts.json
"""

import json
import re
import sys

WRITE = re.compile(r"s3:(Put|Delete|AbortMultipart|Replicate|\*)", re.I)
READ = re.compile(r"s3:(GetObject|List|\*)", re.I)


def as_list(x):
    return x if isinstance(x, list) else [x]


def statements(policy):
    """Yield statement dicts; unparsed policies are split on Sid/Effect heuristically."""
    if "unparsed" in policy:
        # crude split: each '"Effect"' starts a statement-like chunk
        chunks = re.split(r'(?="Effect")', policy["unparsed"])
        for c in chunks[1:]:
            yield {"_raw": c}
    else:
        for s in as_list(policy.get("Statement", [])):
            if isinstance(s, dict):
                yield s


def describe(stmt):
    """Return (effect, actions_text, principal_kind, service_principals, has_condition)."""
    if "_raw" in stmt:
        raw = stmt["_raw"]
        effect = "Allow" if re.match(r'"Effect"\s*:\s*"Allow"', raw) else "Deny"
        actions = " ".join(re.findall(r'"(s3:[A-Za-z*]+)"', raw))
        services = re.findall(r'"([a-z0-9.\-]+\.amazonaws\.com)"', raw) if '"Service"' in raw else []
        if services:
            kind = "service"
        elif re.search(r'"Principal"\s*:\s*("\*"|\{\s*"AWS"\s*:\s*"\*")', raw):
            kind = "anyone"
        elif '"Principal"' in raw:
            kind = "aws"
        else:
            kind = None
        return effect, actions, kind, services, '"Condition"' in raw
    effect = stmt.get("Effect", "Allow")
    actions = " ".join(a for a in as_list(stmt.get("Action", [])) if isinstance(a, str))
    pr = stmt.get("Principal")
    kind, services = None, []
    if pr == "*" or (isinstance(pr, dict) and pr.get("AWS") == "*"):
        kind = "anyone"
    elif isinstance(pr, dict) and "Service" in pr:
        kind, services = "service", [s for s in as_list(pr["Service"]) if isinstance(s, str)]
    elif pr is not None:
        kind = "aws"
    return effect, actions, kind, services, "Condition" in stmt


def facts(job):
    f = {
        "n_policies": len(job["policies"]),
        "n_unparsed": sum("unparsed" in p for p in job["policies"]),
        "n_s3_allow": 0,
        "bucket_policy_service": False,  # bucket policy grants an AWS service principal S3 access
        "bucket_policy_service_write": False,
        "bucket_policy_service_read": False,
        "public_read": False,  # anyone may GetObject, unconditionally
        "bucket_policy_aws_principal": False,  # bucket policy names an account/role/user
        "identity_policy_s3": False,  # S3 actions granted with no Principal (role/user policy)
        "identity_write": False,
        "identity_read": False,
        "service_principals": set(),
    }
    for policy in job["policies"]:
        for stmt in statements(policy):
            effect, actions, kind, services, cond = describe(stmt)
            if effect != "Allow" or "s3:" not in actions:
                continue
            f["n_s3_allow"] += 1
            w, r = bool(WRITE.search(actions)), bool(READ.search(actions))
            if kind == "service":
                f["bucket_policy_service"] = True
                f["bucket_policy_service_write"] |= w
                f["bucket_policy_service_read"] |= bool(re.search(r"s3:GetObject", actions))
                f["service_principals"].update(services)
            elif kind == "anyone":
                if "s3:GetObject" in actions and not cond:
                    f["public_read"] = True
                elif cond:
                    f["bucket_policy_aws_principal"] = True
            elif kind == "aws":
                f["bucket_policy_aws_principal"] = True
            else:
                f["identity_policy_s3"] = True
                f["identity_write"] |= w
                f["identity_read"] |= r
    f["service_principals"] = sorted(f["service_principals"])
    return f


if __name__ == "__main__":
    corpus = json.load(open(sys.argv[1], encoding="utf-8"))
    out = {}
    hdr = f"{'job':<34} pol unp s3 | svcBP w r | pub | awsBP | idPol w r | principals"
    print(hdr)
    for job in corpus:
        f = facts(job)
        out[job["job"]] = f
        b = lambda k: "x" if f[k] else "."
        print(f"{job['job']:<34} {f['n_policies']:>3} {f['n_unparsed']:>3} {f['n_s3_allow']:>2} |   {b('bucket_policy_service')}   {b('bucket_policy_service_write')} {b('bucket_policy_service_read')} |  {b('public_read')}  |   {b('bucket_policy_aws_principal')}   |   {b('identity_policy_s3')}   {b('identity_write')} {b('identity_read')} | {', '.join(f['service_principals'])[:60]}")
    json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), indent=1)
