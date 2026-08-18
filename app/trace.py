"""Tool-call trace persistence. Uses its own tiny pool so trace writes never
compete with the checkpointer, and failures never break a run."""

import json
import logging

from psycopg_pool import AsyncConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)

_pool: AsyncConnectionPool | None = None


async def _get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            settings.database_url,
            min_size=0,
            max_size=3,
            kwargs={"autocommit": True},
            # Conexão ociosa pode ter morrido com o banco atrás do Traefik.
            check=AsyncConnectionPool.check_connection,
            open=False,
        )
        await _pool.open()
    return _pool


async def close_trace_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def insert_usage(
    *, tenant_id, user_id, chat_id, agent_name, provider, model,
    prompt_tokens, completion_tokens, cost_usd, latency_ms,
) -> None:
    try:
        pool = await _get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """INSERT INTO usage_records
                       (tenant_id, user_id, chat_id, agent_name, provider, model,
                        prompt_tokens, completion_tokens, cost_usd, latency_ms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    tenant_id, user_id, chat_id, agent_name, provider, model,
                    prompt_tokens, completion_tokens, cost_usd, latency_ms,
                ),
            )
    except Exception:  # noqa: BLE001 — accounting must never break a run
        logger.exception("failed to record usage for %s/%s", agent_name, model)


async def insert_tool_call(
    *, tenant_id, chat_id, agent_name, tool_name, input, output, status, duration_ms
) -> None:
    try:
        pool = await _get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """INSERT INTO tool_calls
                       (tenant_id, chat_id, agent_name, tool_name, input, output,
                        status, duration_ms)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    tenant_id,
                    chat_id,
                    agent_name,
                    tool_name,
                    json.dumps(input, ensure_ascii=False),
                    output,
                    status,
                    duration_ms,
                ),
            )
    except Exception:  # noqa: BLE001 — tracing must never break a run
        logger.exception("failed to record tool call %s/%s", agent_name, tool_name)
