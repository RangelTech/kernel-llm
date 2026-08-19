"""MongoDB contra um banco de verdade — sobe como `services:` no workflow de
CI (imagem oficial `mongo`). Mongo não é SQL: cobre tanto o nível de driver
(`app/datasources.py`) quanto a tool própria `query_mongo` (`app/tools.py`,
não reaproveita `run_sql_query`) fim-a-fim pelo grafo real.

Pulado fora do CI (sem MONGO_HOST) — mesmo raciocínio dos outros engines
novos: sem um Mongo de verdade, "funciona" não prova nada sobre
list_collection_names/find reais.
"""

import os
import time
import uuid

import pytest

from app import datasources
from tests.conftest import sse_events as _events

MONGO_HOST = os.environ.get("MONGO_HOST")
MONGO_PORT = int(os.environ.get("MONGO_PORT", "27017"))
MONGO_DATABASE = os.environ.get("MONGO_DATABASE", "kernel_test")

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        not MONGO_HOST, reason="MONGO_HOST não configurado (sem container de teste)"
    ),
]

STUB = {"provider": "stub", "model": "stub-1"}


def _wait_and_seed():
    import pymongo

    last_exc = None
    client = None
    for _ in range(60):
        try:
            client = pymongo.MongoClient(
                MONGO_HOST, MONGO_PORT, serverSelectionTimeoutMS=3_000
            )
            client.admin.command("ping")
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2)
    if client is None:
        raise RuntimeError(f"MongoDB não respondeu a tempo: {last_exc}")

    db = client[MONGO_DATABASE]
    db.drop_collection("test_clientes")
    db.test_clientes.insert_many(
        [
            {"_id": 1, "nome": "ACME", "ativo": True},
            {"_id": 2, "nome": "Contoso", "ativo": True},
            {"_id": 3, "nome": "Fabrikam", "ativo": False},
        ]
    )
    client.close()


@pytest.fixture(scope="module", autouse=True)
def _seeded():
    _wait_and_seed()


@pytest.fixture
def datasource():
    return {
        "kind": "mongodb",
        "config": {"host": MONGO_HOST, "port": MONGO_PORT, "database": MONGO_DATABASE},
        "secret": None,
    }


# ---------- app/datasources.py (nível de driver) ----------

async def test_test_connection_ok(datasource):
    ok, detail = await datasources.test_connection(datasource)
    assert ok, detail


async def test_list_tables_lists_collections_with_sampled_fields(datasource):
    catalogo = await datasources.list_tables(datasource)
    colecao = next(t for t in catalogo if t["table"] == "test_clientes")
    nomes_campos = {c["name"] for c in colecao["columns"]}
    assert {"nome", "ativo"} <= nomes_campos


async def test_query_mongo_applies_filter_and_projection(datasource):
    columns, rows = await datasources.query_mongo(
        datasource, "test_clientes", {"ativo": True}, {"_id": 0, "nome": 1}, 10
    )
    assert [c["name"] for c in columns] == ["nome"]
    assert sorted(r[0] for r in rows) == ["ACME", "Contoso"]


async def test_query_mongo_has_no_write_path():
    # Mongo nunca entra em _WRITE_ENGINES: execute_write/execute_transaction
    # devem recusar, não silenciosamente virar um find.
    assert "mongodb" not in datasources._WRITE_ENGINES
    with pytest.raises(ValueError, match="não suportado"):
        await datasources.execute_write(
            {"kind": "mongodb", "config": {}, "secret": None},
            "UPDATE test_clientes SET ativo = 0 WHERE _id = 1",
            ["test_clientes"],
        )


# ---------- app/tools.py::query_mongo fim-a-fim pelo grafo real ----------

def _payload(message: str) -> dict:
    return {
        "thread_id": f"t-{uuid.uuid4().hex[:8]}",
        "message": message,
        "supervisor": {"prompt": "Coordene.", "model": STUB},
        "agents": [
            {
                "name": "dados_agent",
                "description": "consultas a dados",
                "prompt": "Você consulta dados.",
                "model": STUB,
                "tools": ["query_mongo", "describe_datasources"],
            }
        ],
        "max_steps": 4,
        "tenant_id": None,
        "datasources": [
            {
                "name": "crm",
                "kind": "mongodb",
                "config": {"host": MONGO_HOST, "port": MONGO_PORT, "database": MONGO_DATABASE},
            }
        ],
    }


async def _script(client, rules, default="ok"):
    await client.post("/stub/script", json={"rules": rules, "default": default})


async def test_query_mongo_tool_materializes_dataset_artifact(client):
    await _script(
        client,
        [
            ["clientes ativos", 'TOOL:dados_agent:{"task": "liste clientes ativos"}'],
            [
                "liste clientes ativos",
                'TOOL:query_mongo:{"datasource": "crm", "collection": "test_clientes",'
                ' "filter_json": "{\\"ativo\\": true}"}',
            ],
            ["artifact_id", "encontrei os clientes ativos"],
        ],
        default="feito",
    )
    r = await client.post("/v1/runs", json=_payload("quais clientes estão ativos?"))
    events = _events(r.text)

    artifact_events = [d for e, d in events if e == "artifact"]
    assert len(artifact_events) == 1
    assert artifact_events[0]["kind"] == "dataset"

    tool_events = [d for e, d in events if e == "tool"]
    assert any(t["tool"] == "query_mongo" and t["status"] == "ok" for t in tool_events)


async def test_query_mongo_tool_rejects_non_mongo_datasource(client):
    payload = _payload("teste")
    payload["datasources"] = [
        {"name": "crm", "kind": "sqlite", "config": {"path": datasources.make_temp_sqlite("")}}
    ]
    await _script(
        client,
        [
            ["teste", 'TOOL:dados_agent:{"task": "consulte"}'],
            [
                "consulte",
                'TOOL:query_mongo:{"datasource": "crm", "collection": "x"}',
            ],
            ["não é uma fonte MongoDB", "confirmado, não é Mongo"],
        ],
        default="feito",
    )
    r = await client.post("/v1/runs", json=payload)
    events = _events(r.text)
    assert any(e == "done" for e, _ in events)
