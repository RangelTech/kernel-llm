"""Duas falhas que passavam como sucesso.

**Turno sem texto.** Observado em campo (C1 turno 11, "Compare essa despesa com
a de Campinas"): o turno terminou sem resposta e sem erro. `done` com texto
vazio é indistinguível de resposta entregue — o cliente fica olhando para a
tela e nenhum alarme dispara. É o pior modo de falha que um chat tem.

**Chamadas de ferramenta sem teto.** `specialist_max_tool_rounds` limita as
rodadas de UM especialista; o supervisor chama vários no mesmo turno. Medido:
30 chamadas e 26 datasets publicados para responder "faça um gráfico de linha".
Nada errava — só custava caro e demorava.
"""

import json

import pytest

from app.config import settings
from app.graph import _estourou_o_teto

pytestmark = pytest.mark.anyio


def test_o_teto_conta_o_turno_inteiro_e_nao_um_especialista():
    """O contador vive no run_config porque ele é o único objeto que atravessa
    todos os especialistas do mesmo turno."""
    run_config = {"max_tool_calls_per_turn": 3}
    assert [_estourou_o_teto(run_config) for _ in range(5)] == [
        False,
        False,
        False,
        True,
        True,
    ]


def test_sem_configuracao_o_teto_vem_do_padrao_da_plataforma():
    run_config: dict = {}
    for _ in range(settings.max_tool_calls_per_turn):
        assert _estourou_o_teto(run_config) is False
    assert _estourou_o_teto(run_config) is True


def test_turnos_diferentes_nao_herdam_a_contagem():
    """Cada turno monta o seu run_config; se a contagem vazasse entre turnos, a
    segunda pergunta de uma conversa longa já nasceria sem orçamento."""
    primeiro = {"max_tool_calls_per_turn": 2}
    _estourou_o_teto(primeiro)
    _estourou_o_teto(primeiro)
    assert _estourou_o_teto(primeiro) is True

    segundo = {"max_tool_calls_per_turn": 2}
    assert _estourou_o_teto(segundo) is False


async def test_turno_sem_texto_vira_erro_e_nao_done_vazio(monkeypatch, caplog):
    """`done` com texto vazio é o que fez a falha passar despercebida."""
    from app import runs

    class _GrafoMudo:
        async def astream(self, *_args, **_kwargs):
            # Termina normalmente, sem nunca produzir mensagem do assistente.
            for item in []:
                yield item

    async def falso_grafo():
        return _GrafoMudo()

    async def sem_anexos(mensagem, *_args, **_kwargs):
        return mensagem

    monkeypatch.setattr(runs, "get_graph", falso_grafo)
    monkeypatch.setattr("app.attachments.build_user_content", sem_anexos)

    corpo = {
        "thread_id": "t1",
        "message": "Compare essa despesa com a de Campinas.",
        "supervisor": {"prompt": "x", "model": {"provider": "stub", "model": "stub"}},
    }
    resposta = await runs.create_run(runs.RunRequest(**corpo))

    eventos = []
    async for pedaco in resposta.body_iterator:
        eventos.append(pedaco)
    fluxo = "".join(eventos)

    assert "event: error" in fluxo
    assert json.dumps({"detail": "empty_answer"}, ensure_ascii=False) in fluxo
    assert "event: done" not in fluxo


async def test_turno_com_texto_continua_entregando_done(monkeypatch):
    """A guarda não pode transformar resposta boa em erro."""
    from app import runs

    class _Mensagem:
        content = "A despesa de Campinas foi de R$ 10."

    class _GrafoFalante:
        async def astream(self, *_args, **_kwargs):
            yield "values", {"messages": [_Mensagem()]}

    async def falso_grafo():
        return _GrafoFalante()

    async def sem_anexos(mensagem, *_args, **_kwargs):
        return mensagem

    monkeypatch.setattr(runs, "get_graph", falso_grafo)
    monkeypatch.setattr("app.attachments.build_user_content", sem_anexos)

    resposta = await runs.create_run(
        runs.RunRequest(
            thread_id="t2",
            message="E Campinas?",
            supervisor={"prompt": "x", "model": {"provider": "stub", "model": "stub"}},
        )
    )
    fluxo = "".join([p async for p in resposta.body_iterator])

    assert "event: done" in fluxo
    assert "R$ 10" in fluxo
    assert "empty_answer" not in fluxo
