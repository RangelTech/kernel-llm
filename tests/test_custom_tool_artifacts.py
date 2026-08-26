"""produto-06 (25-26/08/2026): fecha o gap real de teste do endpoint que o
Custom Tool Runner usa pra publicar artifacts sem carregar bytes pelo corpo
HTTP do kernel (register-init -> upload direto -> register-complete, ver
`app/main.py`). Backend real (Cloud Run) usa S3-compatible via HMAC contra o
GCS -- aqui, com `storage_backend=local` (fixture de sessão em
`conftest.py`), `upload_url` vem `None` e o "upload direto" vira escrita no
disco local mesmo, o que já é suficiente pra provar o contrato HTTP e o
registro em banco sem precisar de credencial de nuvem real."""

import uuid

import psycopg
import pytest

from app.config import settings

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

TENANT_ID = str(uuid.uuid4())  # coluna `artifacts.tenant_id` eh uuid real


async def test_register_init_then_complete_publishes_artifact(client):
    init = await client.post(
        "/v1/artifacts/register-init",
        json={
            "tenant_id": TENANT_ID,
            "kind": "file",
            "extension": "csv",
            "content_type": "text/csv",
        },
    )
    assert init.status_code == 200
    target = init.json()
    assert target["artifact_id"]
    assert target["upload_url"] is None  # backend local (fixture de teste), nao S3/GCS real

    # Local backend: "upload direto" (que em producao seria um PUT numa URL
    # assinada) vira escrita de fato no storage_path que register-init
    # devolveu -- mesmo caminho que register-complete vai conferir o tamanho.
    from pathlib import Path

    payload = b"mes,total\njan,1000\nfev,1500\n"
    caminho = Path(target["storage_path"])
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_bytes(payload)

    complete = await client.post(
        "/v1/artifacts/register-complete",
        json={
            "artifact_id": target["artifact_id"],
            "tenant_id": TENANT_ID,
            "kind": "file",
            "extension": "csv",
            "content_type": "text/csv",
            "title": "relatorio.csv",
            "agent_name": "custom_tool",
        },
    )
    assert complete.status_code == 200
    descriptor = complete.json()
    assert descriptor["artifact_id"] == target["artifact_id"]
    assert descriptor["title"] == "relatorio.csv"

    with psycopg.connect(settings.database_url) as conn:
        row = conn.execute(
            "SELECT tenant_id, kind, agent_name, storage_path, content_type"
            " FROM artifacts WHERE id = %s",
            (target["artifact_id"],),
        ).fetchone()
    assert str(row[0]) == TENANT_ID
    assert row[1] == "file"
    assert row[2] == "custom_tool"
    assert row[3] == target["storage_path"]
    assert row[4] == "text/csv"

    from app.storage import load_payload

    assert load_payload(row[3]) == payload


async def test_register_complete_rejects_upload_never_verified(client):
    # artifact_id inventado -- nunca passou por register-init, nao existe
    # arquivo no storage_path esperado. Simula tentativa de pular a etapa
    # de upload (cliente malicioso ou distraido chamando complete direto).
    complete = await client.post(
        "/v1/artifacts/register-complete",
        json={
            "artifact_id": "00000000-0000-0000-0000-000000000000",
            "tenant_id": TENANT_ID,
            "kind": "file",
            "extension": "csv",
            "content_type": "text/csv",
            "title": "nao-deveria-existir.csv",
        },
    )
    assert complete.status_code == 404


async def test_register_complete_rejects_oversized_upload(client, monkeypatch):
    init = await client.post(
        "/v1/artifacts/register-init",
        json={"tenant_id": TENANT_ID, "kind": "file", "extension": "bin"},
    )
    target = init.json()

    import app.main as main_module

    monkeypatch.setattr(main_module, "_MAX_CUSTOM_ARTIFACT_BYTES", 10)

    from pathlib import Path

    caminho = Path(target["storage_path"])
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_bytes(b"x" * 100)  # acima do limite forcado (10 bytes)

    complete = await client.post(
        "/v1/artifacts/register-complete",
        json={
            "artifact_id": target["artifact_id"],
            "tenant_id": TENANT_ID,
            "kind": "file",
            "extension": "bin",
        },
    )
    assert complete.status_code == 413
    assert not caminho.exists()  # payload rejeitado eh apagado, nao fica orfao no disco
