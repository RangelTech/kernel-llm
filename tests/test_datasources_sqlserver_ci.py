"""SQL Server contra um banco de verdade — sobe como `services:` no workflow
de CI (`.github/workflows/ci.yml`, imagem `mcr.microsoft.com/mssql/server`).

Sem o container (dev local sem docker), a suíte inteira aqui é pulada: exigir
MSSQL_HOST evita falso-negativo (driver "funciona" contra nada) e falso-
positivo de quebrar o dev que não tem o serviço rodando.
"""

import os
import time

import pytest

from app import datasources

MSSQL_HOST = os.environ.get("MSSQL_HOST")
MSSQL_PORT = int(os.environ.get("MSSQL_PORT", "1433"))
MSSQL_PASSWORD = os.environ.get("MSSQL_PASSWORD", "TestPassw0rd!")
MSSQL_DATABASE = os.environ.get("MSSQL_DATABASE", "kernel_test")

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        not MSSQL_HOST, reason="MSSQL_HOST não configurado (sem container de teste)"
    ),
]


def _wait_and_seed():
    """Bloqueia até o SQL Server aceitar conexão (imagem demora a subir, sem
    health-cmd portável no workflow), cria o banco e uma tabela pequena."""
    import pymssql

    last_exc = None
    for _ in range(60):
        try:
            conn = pymssql.connect(
                server=MSSQL_HOST, port=str(MSSQL_PORT), user="sa",
                password=MSSQL_PASSWORD, login_timeout=5, timeout=10,
                autocommit=True,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2)
    else:
        raise RuntimeError(f"SQL Server não respondeu a tempo: {last_exc}")

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"IF DB_ID('{MSSQL_DATABASE}') IS NULL CREATE DATABASE {MSSQL_DATABASE}"
            )
    finally:
        conn.close()

    conn = pymssql.connect(
        server=MSSQL_HOST, port=str(MSSQL_PORT), database=MSSQL_DATABASE,
        user="sa", password=MSSQL_PASSWORD, autocommit=True,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "IF OBJECT_ID('dbo.test_clientes', 'U') IS NOT NULL "
                "DROP TABLE dbo.test_clientes"
            )
            cur.execute(
                "CREATE TABLE dbo.test_clientes ("
                "id INT PRIMARY KEY, nome VARCHAR(100), ativo BIT)"
            )
            cur.execute(
                "INSERT INTO dbo.test_clientes (id, nome, ativo) VALUES "
                "(1, 'ACME', 1), (2, 'Contoso', 1), (3, 'Fabrikam', 0)"
            )
    finally:
        conn.close()


@pytest.fixture(scope="module", autouse=True)
def _seeded():
    _wait_and_seed()


@pytest.fixture
def datasource():
    return {
        "kind": "sqlserver",
        "config": {
            "host": MSSQL_HOST, "port": MSSQL_PORT, "database": MSSQL_DATABASE, "user": "sa",
        },
        "secret": MSSQL_PASSWORD,
    }


async def test_test_connection_ok(datasource):
    ok, detail = await datasources.test_connection(datasource)
    assert ok, detail


async def test_list_tables_finds_seeded_table(datasource):
    catalogo = await datasources.list_tables(datasource)
    tabela = next(t for t in catalogo if t["table"] == "dbo.test_clientes")
    nomes_colunas = {c["name"] for c in tabela["columns"]}
    assert {"id", "nome", "ativo"} <= nomes_colunas


async def test_query_returns_seeded_rows(datasource):
    columns, rows = await datasources.execute_query(
        datasource, "SELECT id, nome FROM dbo.test_clientes ORDER BY id", 10
    )
    assert [c["name"] for c in columns] == ["id", "nome"]
    assert rows == [[1, "ACME"], [2, "Contoso"], [3, "Fabrikam"]]


async def test_write_respects_allowlist_and_where(datasource):
    with pytest.raises(ValueError, match="não está na lista"):
        await datasources.execute_write(
            datasource, "UPDATE dbo.outra_tabela SET x=1 WHERE id=1", ["test_clientes"]
        )
    with pytest.raises(ValueError, match="sem WHERE"):
        await datasources.execute_write(
            datasource, "UPDATE dbo.test_clientes SET ativo=0", ["test_clientes"]
        )

    table, affected, _ = await datasources.execute_write(
        datasource,
        "UPDATE dbo.test_clientes SET ativo = 0 WHERE id = 3",
        ["test_clientes"],
    )
    assert table == "test_clientes"
    assert affected == 1

    _, rows = await datasources.execute_query(
        datasource, "SELECT ativo FROM dbo.test_clientes WHERE id = 3", 1
    )
    assert rows == [[False]]


async def test_write_blocks_ddl(datasource):
    with pytest.raises(ValueError, match="INSERT, UPDATE ou DELETE"):
        await datasources.execute_write(
            datasource, "DROP TABLE dbo.test_clientes", ["test_clientes"]
        )
