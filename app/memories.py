"""Long-term memory: fact extraction after a conversation turn and semantic
retrieval at the start of the next one. Namespaced (tenant_id, user_id);
deduplicated by embedding similarity so repeated facts don't accumulate."""

import json
import logging

from app.embeddings import embed_texts
from app.providers import ModelConfig, complete
from app.trace import _get_pool

logger = logging.getLogger(__name__)

_DEDUPE_SIMILARITY = 0.92
_EXTRACTION_PROMPT = (
    "Você extrai fatos duradouros sobre o usuário e a empresa dele a partir "
    "de uma conversa. Retorne APENAS um array JSON de strings curtas, cada "
    "uma um fato útil para personalizar conversas futuras (preferências, "
    "contexto do negócio, decisões). Ignore small talk e detalhes efêmeros. "
    "Sem fatos novos: retorne []."
)


async def _mensagens_ja_lidas(thread_id: str) -> int:
    pool = await _get_pool()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT messages_read FROM memory_extraction_state WHERE thread_id = %s",
            (thread_id,),
        )
        linha = await cursor.fetchone()
    return int(linha[0]) if linha else 0


async def _marcar_lidas(thread_id: str, total: int) -> None:
    pool = await _get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """INSERT INTO memory_extraction_state (thread_id, messages_read)
               VALUES (%s, %s)
               ON CONFLICT (thread_id) DO UPDATE
                   SET messages_read = GREATEST(
                           memory_extraction_state.messages_read, EXCLUDED.messages_read
                       ),
                       updated_at = now()""",
            (thread_id, total),
        )


async def extract_memories(
    *, thread_id: str, tenant_id, user_id, model: dict, embedding: dict
) -> int:
    """Read the recent exchange from the checkpointed thread, extract facts,
    store the genuinely new ones. Returns how many were saved."""
    from app.graph import get_graph

    graph = await get_graph()
    state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", []) if state.values else []
    if not messages:
        return 0

    # Só o que ainda não foi lido. A extração roda ao fim de cada turno, e a
    # janela é maior que um turno: sem esta marca, o mesmo trecho voltava a ser
    # extraído a cada turno seguinte e virava fato novo com outra redação — que
    # a deduplicação por similaridade não pega. Medido numa conversa real: 215
    # memórias para ~60 turnos.
    lidas = await _mensagens_ja_lidas(thread_id)
    novas = messages[lidas:]
    if not novas:
        return 0

    transcript = []
    for m in novas[-8:]:
        role = "assistente" if m.type == "ai" else "usuário"
        content = m.content if isinstance(m.content, str) else json.dumps(m.content)
        transcript.append(f"{role}: {content[:1500]}")

    async def swallow(_):
        return None

    result = await complete(
        ModelConfig(**model),
        [
            {"role": "system", "content": _EXTRACTION_PROMPT},
            {"role": "user", "content": "\n".join(transcript)},
        ],
        swallow,
    )
    raw = result.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    try:
        facts = [f for f in json.loads(raw) if isinstance(f, str) and f.strip()]
    except json.JSONDecodeError:
        logger.warning("memory extraction returned non-JSON: %.200s", raw)
        return 0
    if not facts:
        await _marcar_lidas(thread_id, len(messages))
        return 0

    vectors = await embed_texts(embedding, facts)
    pool = await _get_pool()
    saved = 0
    async with pool.connection() as conn:
        for fact, vector in zip(facts, vectors, strict=True):
            cursor = await conn.execute(
                """SELECT 1 FROM memories
                    WHERE tenant_id = %s AND user_id = %s
                      AND 1 - (embedding <=> %s::vector) > %s
                    LIMIT 1""",
                (tenant_id, user_id, str(vector), _DEDUPE_SIMILARITY),
            )
            if await cursor.fetchone():
                continue  # near-duplicate of an existing memory
            await conn.execute(
                """INSERT INTO memories (tenant_id, user_id, content, embedding, source_chat)
                   VALUES (%s, %s, %s, %s, %s)""",
                (tenant_id, user_id, fact.strip(), str(vector), thread_id),
            )
            saved += 1
    await _marcar_lidas(thread_id, len(messages))
    return saved


async def recall(*, tenant_id, user_id, query: str, embedding: dict, top_k: int = 3) -> list[str]:
    """Most relevant memories for this turn — capped so context stays cheap."""
    if not tenant_id or not user_id:
        return []
    try:
        [vector] = await embed_texts(embedding, [query])
        pool = await _get_pool()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                """SELECT content FROM memories
                    WHERE tenant_id = %s AND user_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s""",
                (tenant_id, user_id, str(vector), top_k),
            )
            rows = await cursor.fetchall()
        return [r[0] for r in rows]
    except Exception:  # noqa: BLE001 — memory must never break a turn
        logger.exception("memory recall failed")
        return []
