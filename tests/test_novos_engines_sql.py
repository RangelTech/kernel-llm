"""SQL Server / Oracle / Firebird: mesma forma dos engines existentes, mas sem
banco real — o driver de conexão é mockado (nenhum destes 3 tem instância
disponível no ambiente de CI/local deste repo)."""

import sys
import types

import pytest

from app import datasources

pytestmark = pytest.mark.anyio


class _FakeCursor:
    def __init__(self, description, rows):
        self.description = description
        self._rows = rows
        self.rowcount = len(rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql):
        self.last_sql = sql

    def fetchmany(self, n):
        return self._rows[:n]

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, description, rows):
        self._cursor = _FakeCursor(description, rows)
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def close(self):
        pass


# ---------- SQL Server (pymssql) ----------

def test_query_sqlserver_returns_columns_and_rows(monkeypatch):
    fake_conn = _FakeConn([("id", 3), ("nome", 129)], [[1, "a"], [2, "b"]])
    fake_module = types.SimpleNamespace(connect=lambda **kw: fake_conn)
    monkeypatch.setitem(sys.modules, "pymssql", fake_module)

    columns, rows = datasources._query_sqlserver(
        {"host": "h", "port": 1433, "database": "db", "user": "u"},
        "secret",
        "SELECT * FROM t",
        10,
    )
    assert columns == [{"name": "id", "type": "3"}, {"name": "nome", "type": "129"}]
    assert rows == [[1, "a"], [2, "b"]]


def test_write_sqlserver_commits_and_reports_rowcount(monkeypatch):
    fake_conn = _FakeConn(None, [])
    fake_module = types.SimpleNamespace(connect=lambda **kw: fake_conn)
    monkeypatch.setitem(sys.modules, "pymssql", fake_module)

    count, returned = datasources._write_sqlserver(
        {"host": "h", "database": "db", "user": "u"}, "secret",
        "UPDATE t SET a = 1 WHERE id = 1",
    )
    assert count == 0
    assert returned is None
    assert fake_conn.committed


# ---------- Oracle (oracledb) ----------

def test_oracle_dsn_uses_host_port_service():
    dsn = datasources._oracle_dsn({"host": "orahost", "port": 1521, "database": "ORCL"})
    assert dsn == "orahost:1521/ORCL"


def test_query_oracle_returns_columns_and_rows(monkeypatch):
    fake_conn = _FakeConn([("ID", "DB_TYPE_NUMBER"), ("NOME", "DB_TYPE_VARCHAR")], [[1, "x"]])
    fake_module = types.SimpleNamespace(connect=lambda **kw: fake_conn)
    monkeypatch.setitem(sys.modules, "oracledb", fake_module)

    columns, rows = datasources._query_oracle(
        {"host": "h", "port": 1521, "database": "ORCL", "user": "u"},
        "secret",
        "SELECT * FROM t",
        10,
    )
    assert columns[0]["name"] == "ID"
    assert rows == [[1, "x"]]


def test_write_oracle_commits(monkeypatch):
    fake_conn = _FakeConn(None, [])
    fake_module = types.SimpleNamespace(connect=lambda **kw: fake_conn)
    monkeypatch.setitem(sys.modules, "oracledb", fake_module)

    count, returned = datasources._write_oracle(
        {"host": "h", "database": "ORCL", "user": "u"}, "secret",
        "INSERT INTO t (id) VALUES (1)",
    )
    assert returned is None
    assert fake_conn.committed


# ---------- Firebird (firebird-driver) ----------

def test_firebird_dsn_with_host_and_without():
    assert datasources._firebird_dsn({"host": "fbhost", "port": 3050, "database": "/data/db.fdb"}) == (
        "fbhost/3050:/data/db.fdb"
    )
    assert datasources._firebird_dsn({"database": "/data/db.fdb"}) == "/data/db.fdb"


def test_query_firebird_returns_columns_and_rows(monkeypatch):
    fake_conn = _FakeConn([("ID", 500), ("NOME", 14)], [[1, "y"]])
    fake_module = types.SimpleNamespace(connect=lambda **kw: fake_conn)
    fake_package = types.SimpleNamespace(driver=fake_module)
    monkeypatch.setitem(sys.modules, "firebird", fake_package)
    monkeypatch.setitem(sys.modules, "firebird.driver", fake_module)

    columns, rows = datasources._query_firebird(
        {"host": "h", "port": 3050, "database": "/data/db.fdb", "user": "SYSDBA"},
        "secret",
        "SELECT * FROM T",
        10,
    )
    assert columns[0]["name"] == "ID"
    assert rows == [[1, "y"]]


def test_write_firebird_commits(monkeypatch):
    fake_conn = _FakeConn(None, [])
    fake_module = types.SimpleNamespace(connect=lambda **kw: fake_conn)
    fake_package = types.SimpleNamespace(driver=fake_module)
    monkeypatch.setitem(sys.modules, "firebird", fake_package)
    monkeypatch.setitem(sys.modules, "firebird.driver", fake_module)

    count, returned = datasources._write_firebird(
        {"host": "h", "database": "/data/db.fdb", "user": "SYSDBA"}, "secret",
        "DELETE FROM T WHERE ID = 1",
    )
    assert returned is None
    assert fake_conn.committed


# ---------- registration ----------

def test_engines_registered_for_query_and_write():
    for kind in ("sqlserver", "oracle", "firebird"):
        assert kind in datasources._ENGINES
        assert kind in datasources._WRITE_ENGINES


# ---------- catalog: sqlserver reuses the information_schema-shaped path ----------

async def test_list_tables_sqlserver_uses_information_schema_pattern(monkeypatch):
    chamadas = []

    async def _execute_query(datasource, query, max_rows=None):
        chamadas.append(query)
        if "INFORMATION_SCHEMA.TABLES" in query:
            return ["t"], [["dbo.pedidos"]]
        return ["t", "c", "d"], [["dbo.pedidos", "id", "int"]]

    monkeypatch.setattr(datasources, "execute_query", _execute_query)

    catalogo = await datasources.list_tables({"kind": "sqlserver"})
    assert catalogo == [{"table": "dbo.pedidos", "columns": [{"name": "id", "type": "int"}]}]


# ---------- catalog: oracle / firebird need their own dialect ----------

async def test_list_tables_oracle_uses_all_tables_catalog(monkeypatch):
    async def _execute_query(datasource, query, max_rows=None):
        if "all_tables" in query:
            return ["n"], [["HR.EMPLOYEES"]]
        assert "all_tab_columns" in query
        return ["c", "t"], [["ID", "NUMBER"], ["NOME", "VARCHAR2"]]

    monkeypatch.setattr(datasources, "execute_query", _execute_query)

    catalogo = await datasources.list_tables({"kind": "oracle"})
    assert catalogo == [
        {
            "table": "HR.EMPLOYEES",
            "columns": [{"name": "ID", "type": "NUMBER"}, {"name": "NOME", "type": "VARCHAR2"}],
        }
    ]


async def test_list_tables_firebird_uses_rdb_catalog(monkeypatch):
    async def _execute_query(datasource, query, max_rows=None):
        if "rdb$relations" in query:
            return ["n"], [["PEDIDOS"]]
        assert "rdb$relation_fields" in query
        return ["c", "t"], [["ID", "LONG"]]

    monkeypatch.setattr(datasources, "execute_query", _execute_query)

    catalogo = await datasources.list_tables({"kind": "firebird"})
    assert catalogo == [{"table": "PEDIDOS", "columns": [{"name": "ID", "type": "LONG"}]}]
