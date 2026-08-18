"""Um turno que nunca fica em silêncio precisa acabar mesmo assim.

O único limite era o silêncio entre dois eventos. Como cada ferramenta emite um
evento, um turno que consulta o banco a cada poucos segundos reiniciava a espera
indefinidamente. Medido numa conversa real: oito minutos e 66 chamadas num turno
só. Quem o cortava era o Cloud Run aos 600 s — a conexão caía sem o evento de
erro que o cliente espera.
"""

import asyncio

import pytest
from app import runs
from tests.conftest import sse_events as _events

pytestmark = pytest.mark.anyio


class _FilaTagarela:
    """Entrega um evento a cada 10 ms, para sempre: nunca fica em silêncio."""

    def __init__(self):
        self.entregues = 0

    async def get(self):
        await asyncio.sleep(0.01)
        self.entregues += 1
        return ("token", {"text": "."})

    async def put(self, item):
        """O produtor real também escreve na fila; sem isto ele morre e o teste
        passa por engano, medindo a queda em vez do teto."""


async def _coletar(gerador, maximo=400):
    texto = ""
    async for pedaco in gerador:
        texto += pedaco
        if len(_events(texto)) >= maximo:
            break
    return _events(texto)


async def test_turno_tagarela_termina_no_teto_total(monkeypatch):
    monkeypatch.setattr(runs.settings, "turn_timeout_seconds", 5.0)
    monkeypatch.setattr(runs.settings, "turn_total_timeout_seconds", 0.2)

    fila = _FilaTagarela()
    monkeypatch.setattr(asyncio, "Queue", lambda: fila)

    resposta = await runs.create_run(_pedido())
    eventos = await _coletar(resposta.body_iterator)

    assert eventos[-1][0] == "error"
    assert eventos[-1][1]["detail"] == "timeout"
    assert fila.entregues > 1, "o teto cortou antes de o turno sequer andar"


async def test_silencio_ainda_corta_antes_do_teto_total(monkeypatch):
    """O limite de silêncio continua valendo — e é o que corta primeiro."""
    monkeypatch.setattr(runs.settings, "turn_timeout_seconds", 0.05)
    monkeypatch.setattr(runs.settings, "turn_total_timeout_seconds", 30.0)

    class _FilaMuda:
        async def get(self):
            await asyncio.sleep(30)

        async def put(self, item):
            pass

    monkeypatch.setattr(asyncio, "Queue", lambda: _FilaMuda())

    resposta = await runs.create_run(_pedido())
    eventos = await _coletar(resposta.body_iterator)

    assert eventos[-1] == ("error", {"detail": "timeout"})


def _pedido() -> runs.RunRequest:
    return runs.RunRequest(
        thread_id="t-teto",
        message="oi",
        supervisor={"prompt": "p", "model": {"provider": "stub", "model": "stub-1"}},
        agents=[],
        max_steps=2,
    )
