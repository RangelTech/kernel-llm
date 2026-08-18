"""Multimodal attachments: documents inlined, audio transcribed (stub),
images becoming content blocks — all through the /v1/runs seam."""

import uuid

import pytest
from app.storage import save_payload
from tests.conftest import sse_events as _events

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

STUB = {"provider": "stub", "model": "stub-1"}


def _payload(message: str, attachments: list[dict]) -> dict:
    return {
        "thread_id": f"t-{uuid.uuid4().hex[:8]}",
        "message": message,
        "supervisor": {"prompt": "Responda.", "model": STUB},
        "agents": [],
        "max_steps": 3,
        "attachments": attachments,
        "transcription": {"provider": "stub"},
    }


def _store(name: str, data: bytes, content_type: str) -> str:
    return save_payload(f"tests/attachments/{uuid.uuid4()}/{name}", data, content_type)


async def _script(client, rules, default="ok"):
    await client.post("/stub/script", json={"rules": rules, "default": default})


async def test_document_attachment_content_reaches_the_model(client):
    path = _store("contrato.txt", b"O valor do contrato e R$ 55.000 anuais.", "text/plain")
    await _script(client, [["55.000", "o contrato vale R$ 55.000"]], default="nao li")
    r = await client.post(
        "/v1/runs",
        json=_payload(
            "qual o valor do contrato?",
            [
                {
                    "kind": "file",
                    "name": "contrato.txt",
                    "content_type": "text/plain",
                    "storage_path": path,
                }
            ],
        ),
    )
    done = next(d for e, d in _events(r.text) if e == "done")
    assert "55.000" in done["text"]


async def test_audio_attachment_is_transcribed_into_the_message(client):
    path = _store("nota.webm", b"lembrete: pagar fornecedor sexta-feira", "audio/webm")
    await _script(client, [["pagar fornecedor", "anotado: pagar fornecedor sexta"]], default="nada")
    r = await client.post(
        "/v1/runs",
        json=_payload(
            "",
            [
                {
                    "kind": "audio",
                    "name": "nota.webm",
                    "content_type": "audio/webm",
                    "storage_path": path,
                }
            ],
        )
        | {"message": "transcreva e anote"},
    )
    done = next(d for e, d in _events(r.text) if e == "done")
    assert "pagar fornecedor" in done["text"]


async def test_image_attachment_becomes_a_content_block(client):
    # 1x1 PNG.
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c626001000000ffff03000006000557bfabd40000000049454e44ae426082"
    )
    path = _store("print.png", png, "image/png")
    await _script(client, [["imagem anexada", "recebi sua imagem print.png"]], default="sem imagem")
    r = await client.post(
        "/v1/runs",
        json=_payload(
            "o que tem nessa imagem?",
            [
                {
                    "kind": "image",
                    "name": "print.png",
                    "content_type": "image/png",
                    "storage_path": path,
                }
            ],
        ),
    )
    done = next(d for e, d in _events(r.text) if e == "done")
    assert "print.png" in done["text"]


async def test_broken_attachment_does_not_kill_the_turn(client):
    await _script(client, [["falha ao carregar", "não consegui ler o anexo"]], default="ok")
    r = await client.post(
        "/v1/runs",
        json=_payload(
            "leia o anexo",
            [
                {
                    "kind": "file",
                    "name": "sumiu.txt",
                    "content_type": "text/plain",
                    "storage_path": "/caminho/que/nao/existe.txt",
                }
            ],
        ),
    )
    events = _events(r.text)
    assert any(e == "done" for e, _ in events)
    assert not any(e == "error" for e, _ in events)
