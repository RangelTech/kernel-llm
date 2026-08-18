"""O especialista não pode pedir confirmação ao usuário.

Observado no QA de PIX: o usuário pediu a cobrança, o supervisor pediu
confirmação, o usuário confirmou ("Sim, confirmo... Pode gerar") — e o
especialista `financeiro` pediu confirmação de novo. E de novo, por quatro
turnos seguidos, sem emitir nada.

Não era o modelo teimando: o prompt do especialista, escrito por quem montou o
template, dizia "confirme o valor com o usuário antes de emitir". Só que o
especialista recebe apenas a `task` — nunca a conversa. A confirmação existia
no histórico e ele não tinha como vê-la, então a regra era insatisfazível por
construção e a conversa girava para sempre.

A correção é de plataforma, não de template: quem monta um agente não deveria
precisar saber que especialista não enxerga histórico. O kernel diz isso a ele.
"""

import pytest
from app.graph import (
    CLAUSULA_DO_ESPECIALISTA,
    WRITE_CONFIRMATION_CLAUSE,
    _nota_dos_artefatos,
    _run_specialist,
    build_supervisor_prompt,
)
from app.providers import Completion

pytestmark = pytest.mark.anyio


async def test_o_especialista_e_avisado_de_que_nao_fala_com_o_usuario(monkeypatch):
    vistos = []

    async def falso_complete(config, messages, on_delta, tools=None):
        vistos.append(messages)
        return Completion(content="cobrança emitida")

    async def sem_usage(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.graph.complete", falso_complete)
    monkeypatch.setattr("app.graph._record_usage", sem_usage)

    saida = await _run_specialist(
        {
            "name": "financeiro",
            "prompt": "Confirme o valor com o usuário antes de emitir.",
            "model": {"provider": "stub", "model": "stub"},
            "tools": [],
        },
        "Emita uma cobrança PIX de R$ 0,01.",
        lambda _evento: None,
        {},
        {},
        None,
        None,
    )

    assert saida == "cobrança emitida"
    sistema = vistos[0][0]["content"]
    # O prompt de quem montou o template continua valendo; a regra da plataforma
    # entra depois, para desfazer justamente a parte insatisfazível dele.
    assert "Confirme o valor com o usuário antes de emitir." in sistema
    assert "NUNCA peça confirmação" in sistema
    assert "NÃO vê a conversa" in CLAUSULA_DO_ESPECIALISTA


def test_o_supervisor_e_mandado_repassar_a_confirmacao_na_tarefa():
    """O outro lado do mesmo defeito: só calar o especialista faria a cobrança
    sair sem que ninguém tivesse confirmado. Quem colhe a confirmação é o
    supervisor, e ele precisa dizer isso na tarefa que delega."""
    prompt = build_supervisor_prompt("Você coordena.", [], True)
    assert WRITE_CONFIRMATION_CLAUSE in prompt
    assert "JÁ confirmou" in prompt
    assert "não pergunte de novo" in prompt


async def test_o_especialista_e_avisado_do_artefato_antes_de_redigir(monkeypatch):
    """A negativa nascia no especialista, não no supervisor.

    Ele chamava a tool, o QR ia para a tela por evento de stream — que ele não
    vê — e então escrevia "não consigo exibir imagem". Avisar só o supervisor
    depois deixava a resposta errada ser escrita para só então ser desmentida.
    """
    from app.providers import ToolCall

    chamadas = []

    async def falso_complete(config, messages, on_delta, tools=None):
        chamadas.append([dict(m) for m in messages])
        if len(chamadas) == 1:
            return Completion(
                tool_calls=[ToolCall(id="1", name="gera", arguments="{}")]
            )
        return Completion(content="pronto")

    async def sem_usage(*_args, **_kwargs):
        return None

    async def falsa_tool(*_args, **_kwargs):
        return '{"artifact_id": "a1", "kind": "image", "title": "QR Code PIX"}'

    monkeypatch.setattr("app.graph.complete", falso_complete)
    monkeypatch.setattr("app.graph._record_usage", sem_usage)
    monkeypatch.setattr("app.graph._record_tool_call", sem_usage)
    monkeypatch.setattr("app.graph._execute_tool", falsa_tool)

    saida = await _run_specialist(
        {
            "name": "financeiro",
            "prompt": "Cuide de cobrança.",
            "model": {"provider": "stub", "model": "stub"},
            "tools": ["gera"],
        },
        "Gere a cobrança.",
        lambda _evento: None,
        {},
        {"gera": {"type": "function", "function": {"name": "gera"}}},
        None,
        None,
    )

    # Segunda chamada ao modelo: é nela que o especialista redige a resposta.
    conversa = "\n".join(str(m.get("content")) for m in chamadas[1])
    assert "QR Code PIX" in conversa
    assert "está vendo isso na tela" in conversa
    # E o supervisor também precisa saber, porque é ele quem fala com o usuário.
    assert "QR Code PIX" in saida


def test_o_supervisor_sabe_o_que_ja_apareceu_na_tela():
    """Observado no QA: o QR Code do PIX foi publicado e apareceu para o usuário,
    e no mesmo turno o agente respondeu "não tenho como exibir uma imagem aqui".
    O artefato vai por evento de stream direto ao cliente; o supervisor, que
    escreve a resposta, só vê o texto do especialista."""
    nota = _nota_dos_artefatos(
        [{"artifact_id": "a1", "kind": "image", "title": "QR Code PIX — R$ 0,01"}]
    )
    assert "QR Code PIX" in nota
    assert "está vendo isso na tela" in nota
    assert "NUNCA diga que não consegue mostrar" in nota


def test_sem_artefato_nenhuma_nota_e_grudada_na_resposta():
    assert _nota_dos_artefatos([]) == ""


def test_sem_a_exigencia_de_confirmacao_o_prompt_fica_intacto():
    assert build_supervisor_prompt("Você coordena.", [], False) == "Você coordena."
