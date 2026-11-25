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
        if "cohere.embed-v4" in modelId:  # v4 answers embeddings_by_type
            rows = [[3.0, 4.0, 0.0] for _ in req["texts"]]
            out = {"embeddings": {"float": rows}, "response_type": "embeddings_by_type"}
        elif "cohere" in modelId:
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


def test_cohere_v4_dict_response():
    out = BedrockEmbedder("cohere.embed-v4:0", client=FakeClient()).embed(["a", "b"])
    assert np.allclose(out, [[0.6, 0.8, 0.0], [0.6, 0.8, 0.0]])


def test_empty_input_makes_no_request():
    fake = FakeClient()
    out = BedrockEmbedder(client=fake).embed([])
    assert out.shape == (0, 1024)
    assert fake.calls == []


def test_cohere_v3_rejects_long_text_before_calling_bedrock():
    fake = FakeClient()
    emb = BedrockEmbedder("cohere.embed-english-v3", client=fake)
    with pytest.raises(ValueError, match="index 1 .2049 chars"):
        emb.embed(["ok", "x" * 2049])
    assert fake.calls == []
    assert emb.embed(["x" * 2048]).shape == (1, 3)  # the limit itself is allowed


def test_cohere_v4_has_no_length_cap():
    assert BedrockEmbedder("cohere.embed-v4:0", client=FakeClient()).embed(["x" * 13200]).shape == (1, 3)
