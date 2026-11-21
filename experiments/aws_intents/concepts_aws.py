"""Ruler concepts for the AWS bucket-intents experiments.

Each contrast is (positive label, negative label, concept); a high ruler score
means the positive label. Labels: S = bucket policy names a service principal,
A = bucket policy names an account, role or user, I = identity-based policy.
"""

from semantic_ruler.concepts import Block, Concept

CTX = (
    "Documentation for a cloud provider's object storage. Use a fictional service name; "
    "do not name any real cloud service other than the storage service and IAM."
)
BLOCKS = (
    Block("Log delivery", "a service delivers log files into the bucket"),
    Block("Analytics ingestion", "a service reads input data and writes query or job results"),
    Block("Media and document processing", "a service reads source files and writes processed output"),
    Block("Backup and replication", "a service copies objects for durability or disaster recovery"),
)
SAME_FLOW = "; the data flow, i.e. what is written to or read from the bucket, must be identical"
# Each contrast: (positive label, negative label, concept). Positive = high ruler score.
CONTRASTS = {
    "S vs I": ("S", "I", Concept(
        name="Bucket access granted by service-principal bucket policy",  # ruler from the first experiment
        scale_name="reliance on a bucket policy that names a service principal",
        subject="setup guide passage about giving a service access to a storage bucket",
        neg_label="access is granted only through an IAM role that the service assumes; the role's identity-based policy lists the bucket permissions and the bucket policy is not changed",
        pos_label="access is granted only through the bucket's resource-based policy, which names the service's service principal; no IAM role is created",
        dimensions="how access is granted (which principal is named and which policy carries the permissions)" + SAME_FLOW,
        context=CTX, blocks=BLOCKS)),
    "A vs I": ("A", "I", Concept(
        name="Cross-account bucket policy versus same-account identity policy",
        scale_name="reliance on a bucket policy that names a principal from another account",
        subject="setup guide passage about giving a role or user access to a storage bucket",
        neg_label="the bucket and the role are in the same account; access is granted only by the role's identity-based policy and the bucket policy is not changed",
        pos_label="the bucket is owned by a different account; the bucket owner adds a bucket policy statement naming the other account's role or account ID as the principal",
        dimensions="whether access crosses an account boundary and which policy carries the grant" + SAME_FLOW,
        context=CTX, blocks=BLOCKS)),
    "S vs A": ("S", "A", Concept(
        name="Service principal versus account principal in a bucket policy",
        scale_name="naming a service principal rather than an account principal",
        subject="setup guide passage about a bucket policy statement that grants access to a storage bucket",
        neg_label="the bucket policy's Principal is an AWS account ID, IAM role or IAM user",
        pos_label="the bucket policy's Principal is a service principal (a service's own identity, such as service-name.amazonaws.com), usually restricted with source-account or source-ARN conditions",
        dimensions="which kind of principal the bucket policy names" + SAME_FLOW,
        context=CTX, blocks=BLOCKS)),
    # Declared 2026-11-21 before the crawled corpus was scored. The first "A vs I"
    # concept was about crossing accounts, which the label does not capture: "A"
    # only says the bucket policy names an IAM principal.
    "A vs I (resource vs identity)": ("A", "I", Concept(
        name="IAM principal granted by bucket policy versus by identity policy",
        scale_name="reliance on a bucket policy that names an IAM principal",
        subject="setup guide passage about giving a role or user access to a storage bucket",
        neg_label="access is granted only by an identity-based policy attached to the role or user; the bucket policy is not changed",
        pos_label="access is granted by a statement in the bucket's resource-based policy whose Principal is an AWS account ID, IAM role or IAM user",
        dimensions="which policy carries the grant (the bucket's resource-based policy or the principal's identity-based policy)" + SAME_FLOW,
        context=CTX, blocks=BLOCKS)),
}
