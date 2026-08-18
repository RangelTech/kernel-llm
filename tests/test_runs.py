"""Black-box tests of /v1/runs over the HTTP seam with the stub provider.

Everything here needs Postgres (the checkpointer) — marked integration.
"""

import uuid

import pytest

from tests.conftest import sse_events as _events

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


STUB = {"provider": "stub", "model": "stub-1"}


def _run_payload(message: str, thread_id: str | None = None, agents: list | None = None) -> dict:
    return {
        "thread_id": thread_id or f"t-{uuid.uuid4().hex[:8]}",
        "message": message,
        "supervisor": {"prompt": "Você é um assistente de testes.", "model": STUB},
        "agents": agents or [],
        "max_steps": 4,
    }


def _specialist(name: str, description: str = "especialista de teste") -> dict:
    return {
        "name": name,
        "description": description,
        "prompt": f"Você é o especialista {name}.",
        "model": STUB,
    }


async def _script(client, rules, default="ok"):
    await client.post("/stub/script", json={"rules": rules, "default": default})


async def test_run_streams_tokens_and_finishes_with_done(client):
    await _script(client, [], default="uma resposta com cinco palavras")
    r = await client.post("/v1/runs", json=_run_payload("olá"))
    assert r.status_code == 200
    events = _events(r.text)
    tokens = [d["text"] for e, d in events if e == "token"]
    dones = [d for e, d in events if e == "done"]
    assert len(tokens) == 5
    assert "".join(tokens) == "uma resposta com cinco palavras"
    assert dones == [{"text": "uma resposta com cinco palavras"}]


async def test_conversation_history_persists_across_turns(client):
    thread = f"t-{uuid.uuid4().hex[:8]}"
    await _script(client, [], default="entendido")
    await client.post("/v1/runs", json=_run_payload("primeira mensagem", thread))
    await client.post("/v1/runs", json=_run_payload("segunda mensagem", thread))

    from app.graph import get_graph

    graph = await get_graph()
    state = await graph.aget_state({"configurable": {"thread_id": thread}})
    contents = [m.content for m in state.values["messages"]]
    assert "primeira mensagem" in contents
    assert "segunda mensagem" in contents
    assert len(state.values["messages"]) == 4


async def test_supervisor_routes_to_the_right_specialist(client):
    # Script: financial questions trigger the tool call; the specialist's own
    # prompt then matches the second rule and answers deterministically.
    await _script(
        client,
        [
            ["fluxo de caixa", 'TOOL:financeiro_agent:{"task": "analisar fluxo de caixa"}'],
            ["especialista financeiro_agent", "análise: caixa saudável"],
            ["analisar fluxo de caixa", "análise: caixa saudável"],
        ],
        default="não sei",
    )
    r = await client.post(
        "/v1/runs",
        json=_run_payload(
            "como está meu fluxo de caixa?",
            agents=[
                _specialist("financeiro_agent", "questões financeiras"),
                _specialist("rh_agent", "questões de RH"),
            ],
        ),
    )
    events = _events(r.text)
    agent_events = [d for e, d in events if e == "agent"]
    assert {"name": "financeiro_agent", "status": "start"} in agent_events
    assert {"name": "financeiro_agent", "status": "done"} in agent_events
    assert all(a["name"] != "rh_agent" for a in agent_events)
    done = next(d for e, d in events if e == "done")
    assert done["text"]  # supervisor produced a final answer after the tool round


async def test_unknown_specialist_is_reported_not_fatal(client):
    await _script(
        client,
        [["chame o fantasma", 'TOOL:fantasma_agent:{"task": "boo"}']],
        default="segui sem o fantasma",
    )
    r = await client.post(
        "/v1/runs",
        json=_run_payload("chame o fantasma", agents=[_specialist("real_agent")]),
    )
    events = _events(r.text)
    assert any(e == "done" for e, _ in events)
    assert not any(e == "error" for e, _ in events)


async def test_step_limit_ends_the_turn(client):
    # Every supervisor round keeps requesting the tool -> the budget must trip
    # and the turn must still end with a final answer.
    await _script(
        client,
        [
            ["girar", 'TOOL:loop_agent:{"task": "girar"}'],
            ["especialista loop_agent", 'TOOL:loop_agent:{"task": "girar"}'],
            ["limite de passos atingido", "parei pelo limite"],
        ],
        default='TOOL:loop_agent:{"task": "girar"}',
    )
    r = await client.post(
        "/v1/runs",
        json=_run_payload("girar", agents=[_specialist("loop_agent")]),
    )
    events = _events(r.text)
    assert any(e == "limit" for e, _ in events)
    done = next(d for e, d in events if e == "done")
    assert done["text"] == "parei pelo limite"


async def test_provider_failure_becomes_an_error_event(client):
    await _script(client, [["erro proposital", "__RAISE__"]], default="ok")
    r = await client.post("/v1/runs", json=_run_payload("erro proposital"))
    events = _events(r.text)
    assert events[-1][0] == "error"
    assert not any(e == "done" for e, _ in events)


async def test_distinct_threads_do_not_share_history(client):
    a, b = f"t-{uuid.uuid4().hex[:6]}", f"t-{uuid.uuid4().hex[:6]}"
    await _script(client, [], default="ok")
    await client.post("/v1/runs", json=_run_payload("mensagem só do A", a))
    await client.post("/v1/runs", json=_run_payload("mensagem só do B", b))

    from app.graph import get_graph

    graph = await get_graph()
    state_b = await graph.aget_state({"configurable": {"thread_id": b}})
    contents = " ".join(m.content for m in state_b.values["messages"])
    assert "mensagem só do A" not in contents
