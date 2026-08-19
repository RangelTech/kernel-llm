"""Firebird contra um banco de verdade — sobe como `services:` no workflow de
CI (`jacobalberty/firebird`, imagem comunitária de referência citada na spec;
Firebird Foundation não publica imagem oficial própria).

Pulado fora do CI (sem FIREBIRD_HOST) pelo mesmo motivo do SQL Server: driver
"funcionando" contra um mock não prova nada sobre o catálogo RDB$ real.
"""

import os
import time

import pytest

from app import datasources

FIREBIRD_HOST = os.environ.get("FIREBIRD_HOST")
FIREBIRD_PORT = int(os.environ.get("FIREBIRD_PORT", "3050"))
FIREBIRD_DATABASE = os.environ.get("FIREBIRD_DATABASE", "/firebird/data/test.fdb")
FIREBIRD_PASSWORD = os.environ.get("FIREBIRD_PASSWORD", "TestPassw0rd!")

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        not FIREBIRD_HOST, reason="FIREBIRD_HOST não configurado (sem container de teste)"
    ),
]


def _dsn():
    return f"{FIREBIRD_HOST}/{FIREBIRD_PORT}:{FIREBIRD_DATABASE}"


def _wait_and_seed():
    import firebird.driver as fbd

    last_exc = None
    conn = None
    for _ in range(60):
        try:
            conn = fbd.connect(database=_dsn(), user="SYSDBA", password=FIREBIRD_PASSWORD)
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2)
    if conn is None:
        raise RuntimeError(f"Firebird não respondeu a tempo: {last_exc}")

    try:
        with conn.cursor() as cur:
            try:
                cur.execute("DROP TABLE TEST_CLIENTES")
                conn.commit()
            except Exception:  # noqa: BLE001 — tabela ainda não existe na 1a run
                conn.rollback()
            cur.execute(
                "CREATE TABLE TEST_CLIENTES ("
                "ID INTEGER NOT NULL PRIMARY KEY, NOME VARCHAR(100), ATIVO SMALLINT)"
            )
            conn.commit()
            cur.execute(
                "INSERT INTO TEST_CLIENTES (ID, NOME, ATIVO) VALUES (1, 'ACME', 1)"
            )
            cur.execute(
                "INSERT INTO TEST_CLIENTES (ID, NOME, ATIVO) VALUES (2, 'Contoso', 1)"
            )
            cur.execute(
                "INSERT INTO TEST_CLIENTES (ID, NOME, ATIVO) VALUES (3, 'Fabrikam', 0)"
            )
            conn.commit()
    finally:
        conn.close()


@pytest.fixture(scope="module", autouse=True)
def _seeded():
    _wait_and_seed()


@pytest.fixture
def datasource():
    return {
        "kind": "firebird",
        "config": {
            "host": FIREBIRD_HOST, "port": FIREBIRD_PORT, "database": FIREBIRD_DATABASE,
            "user": "SYSDBA",
        },
        "secret": FIREBIRD_PASSWORD,
    }


async def test_test_connection_ok(datasource):
    ok, detail = await datasources.test_connection(datasource)
    assert ok, detail


async def test_list_tables_finds_seeded_table(datasource):
    catalogo = await datasources.list_tables(datasource)
    tabela = next(t for t in catalogo if t["table"] == "TEST_CLIENTES")
    nomes_colunas = {c["name"] for c in tabela["columns"]}
    assert {"ID", "NOME", "ATIVO"} <= nomes_colunas


async def test_query_returns_seeded_rows(datasource):
    columns, rows = await datasources.execute_query(
        datasource, "SELECT ID, NOME FROM TEST_CLIENTES ORDER BY ID", 10
    )
    assert [c["name"] for c in columns] == ["ID", "NOME"]
    assert rows == [[1, "ACME"], [2, "Contoso"], [3, "Fabrikam"]]


async def test_write_respects_allowlist_and_where(datasource):
    with pytest.raises(ValueError, match="não está na lista"):
        await datasources.execute_write(
            datasource, "UPDATE OUTRA_TABELA SET X=1 WHERE ID=1", ["test_clientes"]
        )
    with pytest.raises(ValueError, match="sem WHERE"):
        await datasources.execute_write(
            datasource, "UPDATE TEST_CLIENTES SET ATIVO=0", ["test_clientes"]
        )

    table, affected, _ = await datasources.execute_write(
        datasource, "UPDATE TEST_CLIENTES SET ATIVO = 0 WHERE ID = 3", ["test_clientes"]
    )
    assert table == "test_clientes"
    assert affected == 1


async def test_write_blocks_ddl(datasource):
    with pytest.raises(ValueError, match="INSERT, UPDATE ou DELETE"):
        await datasources.execute_write(
            datasource, "DROP TABLE TEST_CLIENTES", ["test_clientes"]
        )
