"""Embedding providers. spiritwriter-core has no embedder, so the interface lives here."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

# Keychain service holding the API keys (see `spiritwriter secrets -s <service> list`).
KEYCHAIN_SERVICE = os.environ.get("SEMANTIC_RULER_KEYCHAIN_SERVICE", "claude-studio")


class Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) float64 matrix, one row per text."""


class OpenAIEmbedder:
    """OpenAI embeddings over plain HTTPS; the key comes from the OS keychain."""

    URL = "https://api.openai.com/v1/embeddings"

    def __init__(self, model: str = "text-embedding-3-large", batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size
        self._key: str | None = None

    def _api_key(self) -> str:
        if self._key is None:
            from spiritwriter.secrets import configure, get_api_key

            configure(service_name=KEYCHAIN_SERVICE)
            key = get_api_key("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    f"OPENAI_API_KEY not found in keychain service {KEYCHAIN_SERVICE!r} or the "
                    f"environment. Set it with: spiritwriter secrets -s {KEYCHAIN_SERVICE} "
                    "set OPENAI_API_KEY"
                )
            self._key = key
        return self._key

    def _post(self, batch: Sequence[str]) -> list[list[float]]:
        req = urllib.request.Request(
            self.URL,
            data=json.dumps({"model": self.model, "input": list(batch)}).encode(),
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.load(resp)["data"]
                return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                    detail = e.read().decode("utf-8", "replace")[:300]
                    raise RuntimeError(f"OpenAI embeddings HTTP {e.code}: {detail}") from None
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            rows.extend(self._post(texts[i : i + self.batch_size]))
        return np.asarray(rows, dtype=np.float64)


def bedrock_runtime_client(region: str | None = None):
    """A bedrock-runtime client from the standard AWS credential chain."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise RuntimeError("Bedrock needs boto3: pip install -e '.[bedrock]'") from None
    # botocore reads AWS_DEFAULT_REGION and the profile's region, but not
    # AWS_REGION (which the AWS CLI does honour), so check it too.
    region = region or boto3.Session().region_name or os.environ.get("AWS_REGION")
    if not region:
        raise RuntimeError("No AWS region for Bedrock: pass --aws-region or set AWS_DEFAULT_REGION")
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(retries={"max_attempts": 8, "mode": "adaptive"}, read_timeout=300),
    )


class BedrockEmbedder:
    """Amazon Bedrock embeddings (Cohere Embed or Titan Text) through boto3.

    Defaults to Cohere Embed English v3, which tracked the requested level best
    of the Bedrock models on Case Study I (r = 0.82, against 0.65 for Titan v2).

    Credentials and region come from the standard AWS chain (AWS_PROFILE,
    AWS_REGION, ~/.aws/config, SSO, instance roles); nothing is read from the
    keychain. Vectors are L2-normalised so they match the OpenAI embedder.
    """

    COHERE_BATCH = 96  # Cohere Embed's per-request text limit on Bedrock

    def __init__(
        self,
        model: str = "cohere.embed-english-v3",
        region: str | None = None,
        dimensions: int = 1024,
        client=None,
    ):
        if "titan-embed" not in model and "cohere.embed" not in model:
            raise ValueError(
                f"unsupported Bedrock embedding model {model!r}: "
                "use an amazon.titan-embed-* or cohere.embed-* model ID"
            )
        self.model = model
        self.dimensions = dimensions
        self._region = region
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = bedrock_runtime_client(self._region)
        return self._client

    def _invoke(self, body: dict) -> dict:
        resp = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(resp["body"].read())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[list[float]] = []
        if "cohere.embed" in self.model:
            for i in range(0, len(texts), self.COHERE_BATCH):
                batch = list(texts[i : i + self.COHERE_BATCH])
                out = self._invoke({"texts": batch, "input_type": "search_document"})
                emb = out["embeddings"]
                rows.extend(emb["float"] if isinstance(emb, dict) else emb)
        else:  # Titan takes one text per request
            for text in texts:
                out = self._invoke(
                    {"inputText": text, "dimensions": self.dimensions, "normalize": True}
                )
                rows.append(out["embedding"])
        mat = np.asarray(rows, dtype=np.float64).reshape(len(texts), -1)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.where(norms == 0, 1.0, norms)


class HashEmbedder:
    """Deterministic bag-of-words embedder for offline tests and --mock runs.

    Each word hashes to a signed slot; no semantics beyond shared vocabulary.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.model = f"hash-bow-{dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z][a-z\-]+", text.lower()):
                h = hashlib.sha256(word.encode()).digest()
                out[row, int.from_bytes(h[:4], "big") % self.dim] += 1.0 if h[4] & 1 else -1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


class CachedEmbedder:
    """Content-addressed disk cache: sha256(model, text) -> .npy vector."""

    def __init__(self, inner: Embedder, cache_dir: str | Path):
        self.inner = inner
        self.model = inner.model
        self.dir = Path(cache_dir) / re.sub(r"[^A-Za-z0-9._-]", "_", inner.model)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self.model}\0{text}".encode("utf-8")).hexdigest()
        return self.dir / f"{digest}.npy"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        paths = [self._path(t) for t in texts]
        vecs: list[np.ndarray | None] = [np.load(p) if p.exists() else None for p in paths]
        todo = [i for i, v in enumerate(vecs) if v is None]
        self.hits += len(texts) - len(todo)
        self.misses += len(todo)
        if todo:
            fresh = self.inner.embed([texts[i] for i in todo])
            for i, vec in zip(todo, fresh):
                np.save(paths[i], vec)
                vecs[i] = vec
        return np.vstack(vecs) if vecs else np.zeros((0, 0))
