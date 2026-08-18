"""O resultado de uma ferramenta não pode entrar inteiro no prompt.

Medido numa conversa de 30 turnos: o especialista `despesas` acumulou 889 mil
tokens de prompt, contra 10 mil do supervisor. O corte de histórico da conversa
existe desde antes e ataca a parte menor — o gasto está no retorno das
ferramentas, que volta inteiro e reentra no contexto a cada rodada seguinte.

Cortar não perde dado: o payload continua materializado no artefato e o
artifact_id encadeia para gráfico, planilha e sandbox sem passar pelo modelo.
"""

import json

import pytest

from app.config import settings
from app.graph import _limitar_saida, _run_specialist
from app.providers import Completion, ToolCall

pytestmark = pytest.mark.anyio


def test_saida_curta_passa_intacta():
    assert _limitar_saida("dois mil reais", [], 1000) == "dois mil reais"


def test_saida_longa_e_cortada_e_o_corte_e_anunciado():
    """Truncar em silêncio é pior que não truncar: o modelo soma uma coluna pela
    metade e apresenta o número como total."""
    saida = _limitar_saida("x" * 5000, [], 1000)
    assert len(saida) < 2000
    assert saida.startswith("x" * 1000)
    assert "CORTADO PELA PLATAFORMA" in saida
    assert "5000" in saida and "1000" in saida
    assert "NÃO some nem conte" in saida


def test_o_corte_diz_onde_esta_o_resultado_completo():
    """Sem o artifact_id, cortar viraria beco sem saída: o modelo não teria como
    chegar ao conjunto todo."""
    saida = _limitar_saida("x" * 5000, [{"artifact_id": "art-9"}], 1000)
    assert "art-9" in saida


def test_limite_zero_desliga_o_corte():
    assert _limitar_saida("x" * 5000, [], 0) == "x" * 5000


async def test_o_especialista_recebe_a_saida_ja_cortada(monkeypatch):
    """O corte é do prompt, não do artefato nem do registro: o que a ferramenta
    devolveu continua inteiro para publicar e para o trace."""
    vistos = []
    registrados = []

    async def falso_complete(config, messages, on_delta, tools=None):
        vistos.append([dict(m) for m in messages])
        if len(vistos) == 1:
            return Completion(tool_calls=[ToolCall(id="1", name="consulta", arguments="{}")])
        return Completion(content="pronto")

    async def sem_usage(*_args, **_kwargs):
        return None

    async def registrar_chamada(run_config, agent, tool, arguments, output, *_a, **_k):
        registrados.append(output)

    grande = json.dumps({"artifact_id": "art-1", "kind": "dataset", "linhas": "y" * 9000})

    async def falsa_tool(*_args, **_kwargs):
        return grande

    monkeypatch.setattr("app.graph.complete", falso_complete)
    monkeypatch.setattr("app.graph._record_usage", sem_usage)
    monkeypatch.setattr("app.graph._record_tool_call", registrar_chamada)
    monkeypatch.setattr("app.graph._execute_tool", falsa_tool)

    await _run_specialist(
        {
            "name": "despesas",
            "prompt": "Você analisa despesas.",
            "model": {"provider": "stub", "model": "stub"},
            "tools": ["consulta"],
        },
        "Some as despesas.",
        lambda _evento: None,
        {"tool_output_limit": 2000},
        {"consulta": {"type": "function", "function": {"name": "consulta"}}},
        None,
        None,
    )

    mensagem_da_tool = next(m for m in vistos[1] if m.get("role") == "tool")
    assert len(mensagem_da_tool["content"]) < len(grande)
    assert "CORTADO PELA PLATAFORMA" in mensagem_da_tool["content"]
    assert "art-1" in mensagem_da_tool["content"]
    # O trace e a publicação do artefato veem o resultado inteiro.
    assert registrados == [grande]


async def test_sem_configuracao_o_teto_vem_do_padrao_da_plataforma(monkeypatch):
    """Template antigo, sem o campo, não pode rodar sem teto nenhum."""
    vistos = []

    async def falso_complete(config, messages, on_delta, tools=None):
        vistos.append([dict(m) for m in messages])
        if len(vistos) == 1:
            return Completion(tool_calls=[ToolCall(id="1", name="consulta", arguments="{}")])
        return Completion(content="pronto")

    async def sem_usage(*_args, **_kwargs):
        return None

    async def falsa_tool(*_args, **_kwargs):
        return "z" * (settings.tool_output_limit_default + 5000)

    monkeypatch.setattr("app.graph.complete", falso_complete)
    monkeypatch.setattr("app.graph._record_usage", sem_usage)
    monkeypatch.setattr("app.graph._record_tool_call", sem_usage)
    monkeypatch.setattr("app.graph._execute_tool", falsa_tool)

    await _run_specialist(
        {
            "name": "despesas",
            "prompt": "Você analisa despesas.",
            "model": {"provider": "stub", "model": "stub"},
            "tools": ["consulta"],
        },
        "Some as despesas.",
        lambda _evento: None,
        {},  # run_config sem tool_output_limit
        {"consulta": {"type": "function", "function": {"name": "consulta"}}},
        None,
        None,
    )

    mensagem_da_tool = next(m for m in vistos[1] if m.get("role") == "tool")
    assert "CORTADO PELA PLATAFORMA" in mensagem_da_tool["content"]
