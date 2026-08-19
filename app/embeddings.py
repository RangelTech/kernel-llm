"""Text embeddings via provider APIs (litellm), plus a deterministic stub.

The stub maps tokens to hashed one-hot positions — no semantics, but shared
words yield high cosine similarity, which is exactly what the test suite
needs to exercise retrieval without spending tokens.
"""

import hashlib
import math

DIMENSIONS = 768


def _stub_vector(text: str) -> list[float]:
    vector = [0.0] * DIMENSIONS
    for token in text.lower().split():
        index = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIMENSIONS
        vector[index] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


async def embed_texts(spec: dict, texts: list[str]) -> list[list[float]]:
    """spec: {provider, model, api_key}. provider 'stub' is deterministic."""
    if spec.get("provider") == "stub":
        return [_stub_vector(t) for t in texts]

    import litellm

    # "gemini/text-embedding-004" foi descontinuado pela Google (confirmado em
    # produção 2026-08-19: toda ingestão de RAG falhava com
    # litellm.NotFoundError 404 "models/text-embedding-004 is not found").
    # gemini-embedding-001 é o substituto ativo; ele é 3072-dim por padrão,
    # então pedimos explicitamente os 768 que o resto da plataforma assume
    # (coluna pgvector, DIMENSIONS acima).
    model = spec.get("model") or "gemini/gemini-embedding-001"
    kwargs: dict = {"model": model, "input": texts, "api_key": spec.get("api_key")}
    if "text-embedding-3" in model or "gemini-embedding" in model:
        kwargs["dimensions"] = DIMENSIONS
    response = await litellm.aembedding(**kwargs)
    return [item["embedding"] for item in response.data]
