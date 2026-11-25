"""BedrockEmbedder request/response handling against a fake client: no network, no AWS."""

import io
import json

import numpy as np
import pytest

from semantic_ruler.embed import BedrockEmbedder


class FakeClient:
    def __init__(self):
        self.calls = []

    def invoke_model(self, modelId, body, contentType, accept):
        req = json.loads(body)
        self.calls.append((modelId, req))
        if "cohere" in modelId:
            out = {"embeddings": [[float(len(t)), 1.0, 0.0] for t in req["texts"]]}
        else:
            out = {"embedding": [float(len(req["inputText"])), 0.0, 1.0]}
        return {"body": io.BytesIO(json.dumps(out).encode())}


def test_titan_one_request_per_text_and_normalised():
    fake = FakeClient()
    emb = BedrockEmbedder("amazon.titan-embed-text-v2:0", client=fake, dimensions=512)
    out = emb.embed(["ab", "abcd"])
    assert out.shape == (2, 3)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert len(fake.calls) == 2
    model, req = fake.calls[0]
    assert model == "amazon.titan-embed-text-v2:0"
    assert req == {"inputText": "ab", "dimensions": 512, "normalize": True}


def test_cohere_batches_at_96():
    fake = FakeClient()
    emb = BedrockEmbedder("cohere.embed-english-v3", client=fake)
    out = emb.embed([f"t{i}" for i in range(100)])
    assert out.shape == (100, 3)
    assert [len(req["texts"]) for _, req in fake.calls] == [96, 4]
    assert fake.calls[0][1]["input_type"] == "search_document"


def test_default_is_cohere_english_v3():
    assert BedrockEmbedder(client=FakeClient()).model == "cohere.embed-english-v3"


def test_rejects_non_embedding_model():
    with pytest.raises(ValueError):
        BedrockEmbedder("openai.gpt-oss-120b-1:0")
