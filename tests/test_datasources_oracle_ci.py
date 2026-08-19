"""Oracle contra um banco de verdade — sobe como `services:` no workflow de
CI.

Desvio de spec deliberado: a spec cita `container-registry.oracle.com/database/free`
(imagem oficial da Oracle), mas esse registry exige login autenticado
(conta Oracle SSO) mesmo pra imagem gratuita — não dá pra fazer pull anônimo
num runner do GitHub Actions sem guardar uma credencial Oracle como secret,
o que este repo não tem e a spec não previu. Trocado por `gvenzl/oracle-free`
(imagem comunitária, mantida especificamente para uso em CI/testes, pull
público sem login, healthcheck embutido) — mesmo motor Oracle Database Free,
mesma superfície de catálogo `ALL_TABLES`/`ALL_TAB_COLUMNS`.

Pulado fora do CI (sem ORACLE_HOST) pelo mesmo motivo dos outros engines
novos.
"""

import os
import time

import pytest

from app import datasources

ORACLE_HOST = os.environ.get("ORACLE_HOST")
ORACLE_PORT = int(os.environ.get("ORACLE_PORT", "1521"))
ORACLE_SERVICE = os.environ.get("ORACLE_SERVICE", "FREEPDB1")
ORACLE_USER = os.environ.get("ORACLE_USER", "testuser")
ORACLE_PASSWORD = os.environ.get("ORACLE_PASSWORD", "TestPassw0rd1")

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        not ORACLE_HOST, reason="ORACLE_HOST não configurado (sem container de teste)"
    ),
]


def _wait_and_seed():
    import oracledb

    dsn = f"{ORACLE_HOST}:{ORACLE_PORT}/{ORACLE_SERVICE}"
    last_exc = None
    conn = None
    for _ in range(60):
        try:
            conn = oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=dsn)
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(5)
    if conn is None:
        raise RuntimeError(f"Oracle não respondeu a tempo: {last_exc}")

    try:
        with conn.cursor() as cur:
            try:
                cur.execute("DROP TABLE test_clientes")
            except oracledb.DatabaseError:
                pass  # ORA-00942: tabela ainda não existe na 1a run
            cur.execute(
                "CREATE TABLE test_clientes ("
                "id NUMBER PRIMARY KEY, nome VARCHAR2(100), ativo NUMBER(1))"
            )
            cur.execute("INSERT INTO test_clientes VALUES (1, 'ACME', 1)")
            cur.execute("INSERT INTO test_clientes VALUES (2, 'Contoso', 1)")
            cur.execute("INSERT INTO test_clientes VALUES (3, 'Fabrikam', 0)")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(scope="module", autouse=True)
def _seeded():
    _wait_and_seed()


@pytest.fixture
def datasource():
    return {
        "kind": "oracle",
        "config": {
            "host": ORACLE_HOST, "port": ORACLE_PORT,
            "database": ORACLE_SERVICE, "user": ORACLE_USER,
        },
        "secret": ORACLE_PASSWORD,
    }


async def test_test_connection_ok(datasource):
    ok, detail = await datasources.test_connection(datasource)
    assert ok, detail


async def test_list_tables_finds_seeded_table_via_all_tables_catalog(datasource):
    catalogo = await datasources.list_tables(datasource)
    tabela = next(
        t for t in catalogo if t["table"].upper() == f"{ORACLE_USER.upper()}.TEST_CLIENTES"
    )
    nomes_colunas = {c["name"] for c in tabela["columns"]}
    assert {"ID", "NOME", "ATIVO"} <= nomes_colunas


async def test_query_returns_seeded_rows(datasource):
    columns, rows = await datasources.execute_query(
        datasource, "SELECT id, nome FROM test_clientes ORDER BY id", 10
    )
    assert [c["name"] for c in columns] == ["ID", "NOME"]
    assert rows == [[1, "ACME"], [2, "Contoso"], [3, "Fabrikam"]]


async def test_write_respects_allowlist_and_where(datasource):
    with pytest.raises(ValueError, match="não está na lista"):
        await datasources.execute_write(
            datasource, "UPDATE outra_tabela SET x=1 WHERE id=1", ["test_clientes"]
        )
    with pytest.raises(ValueError, match="sem WHERE"):
        await datasources.execute_write(
            datasource, "UPDATE test_clientes SET ativo=0", ["test_clientes"]
        )

    table, affected, _ = await datasources.execute_write(
        datasource, "UPDATE test_clientes SET ativo = 0 WHERE id = 3", ["test_clientes"]
    )
    assert table == "test_clientes"
    assert affected == 1


async def test_write_blocks_ddl(datasource):
    with pytest.raises(ValueError, match="INSERT, UPDATE ou DELETE"):
        await datasources.execute_write(
            datasource, "DROP TABLE test_clientes", ["test_clientes"]
        )
