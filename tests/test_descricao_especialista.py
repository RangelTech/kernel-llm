"""O supervisor precisa saber quais especialistas têm documentos.

Observado no QA: um .txt e um .docx foram enviados, indexados e ligados a um
especialista com `query_agent_rag`. Perguntado sobre o conteúdo, o agente
respondeu "não tenho acesso a documentos internos da sua empresa" — porque a
descrição que o supervisor lê continuava falando só de bases públicas.

Nada estava quebrado: o vínculo existia no banco. O supervisor simplesmente não
tinha como saber.
"""

from app.graph import _agent_tool_defs, _descricao_do_especialista


def test_especialista_com_documentos_anuncia_isso():
    descricao = _descricao_do_especialista(
        {"name": "gerais", "description": "Dados do município.", "file_ids": ["a", "b"]}
    )
    assert "Dados do município." in descricao
    assert "2 documento" in descricao
    assert "interna" in descricao


def test_especialista_sem_documentos_fica_igual():
    agente = {"name": "editais", "description": "Lê editais do PNCP.", "file_ids": []}
    assert _descricao_do_especialista(agente) == "Lê editais do PNCP."


def test_a_descricao_chega_na_ferramenta_do_supervisor():
    defs = _agent_tool_defs(
        [
            {"name": "gerais", "description": "Contexto.", "file_ids": ["x"]},
            {"name": "itens", "description": "Itens.", "file_ids": []},
        ]
    )
    por_nome = {d["function"]["name"]: d["function"]["description"] for d in defs}
    assert "documento" in por_nome["gerais"]
    assert por_nome["itens"] == "Itens."
