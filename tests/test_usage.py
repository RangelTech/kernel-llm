"""Usage accounting: every model call in a run leaves a usage_records row."""

import uuid

import psycopg
import pytest

from app.config import settings
from tests.conftest import sse_events as _events

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

STUB = {"provider": "stub", "model": "stub-1"}


async def test_supervisor_and_specialist_calls_are_recorded(client):
    thread = f"t-{uuid.uuid4().hex[:8]}"
    tenant, user = str(uuid.uuid4()), str(uuid.uuid4())
    await client.post(
        "/stub/script",
        json={
            "rules": [
                ["some os numeros", 'TOOL:contas_agent:{"task": "faca a soma"}'],
                ["faca a soma", "o resultado é 42"],
                ["resultado é 42", "a soma deu 42"],
            ],
            "default": "ok",
        },
    )
    r = await client.post(
        "/v1/runs",
        json={
            "thread_id": thread,
            "message": "some os numeros",
            "supervisor": {"prompt": "Coordene.", "model": STUB},
            "agents": [
                {
                    "name": "contas_agent",
                    "description": "faz contas",
                    "prompt": "Você soma.",
                    "model": STUB,
                }
            ],
            "max_steps": 4,
            "tenant_id": tenant,
            "user_id": user,
        },
    )
    assert any(e == "done" for e, _ in _events(r.text))

    with psycopg.connect(settings.database_url) as conn:
        rows = conn.execute(
            """SELECT agent_name, prompt_tokens, completion_tokens
                 FROM usage_records WHERE chat_id = %s""",
            (thread,),
        ).fetchall()
    agents = {r[0] for r in rows}
    assert "supervisor" in agents
    assert "contas_agent" in agents
    # Stub estimates are word counts — must be nonzero for real exchanges.
    assert all(r[1] > 0 for r in rows)
    assert any(r[2] > 0 for r in rows)
