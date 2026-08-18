"""Datasource query execution — one engine per kind.

All drivers are synchronous; calls run in a worker thread so the event loop
never blocks. Queries are read-only by contract: anything that is not a
SELECT/WITH is refused before touching the database.
"""

import json
import re
import sqlite3
import tempfile

import anyio

_READONLY = re.compile(r"^\s*(--[^\n]*\n|\s)*(select|with)\b", re.IGNORECASE)


def ensure_readonly(sql: str) -> None:
    if not _READONLY.match(sql or ""):
        raise ValueError("apenas consultas SELECT/WITH são permitidas nesta tool")


def _rows_to_python(rows) -> list[list]:
    return [[value for value in row] for row in rows]


def _query_postgresql(config: dict, secret: str | None, sql: str, max_rows: int):
    import psycopg

    conninfo = psycopg.conninfo.make_conninfo(
        host=config.get("host", "localhost"),
        port=int(config.get("port", 5432)),
        dbname=config.get("database", ""),
        user=config.get("user", ""),
        password=secret or "",
        connect_timeout=10,
    )
    with psycopg.connect(conninfo) as conn:
        conn.read_only = True
        with conn.cursor() as cursor:
            cursor.execute(sql)
            columns = [
                {"name": d.name, "type": str(d.type_code)} for d in cursor.description
            ]
            rows = cursor.fetchmany(max_rows)
    return columns, _rows_to_python(rows)


def _query_mysql(config: dict, secret: str | None, sql: str, max_rows: int):
    import pymysql

    conn = pymysql.connect(
        host=config.get("host", "localhost"),
        port=int(config.get("port", 3306)),
        database=config.get("database", ""),
        user=config.get("user", ""),
        password=secret or "",
        connect_timeout=10,
        read_timeout=60,
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            columns = [{"name": d[0], "type": str(d[1])} for d in cursor.description]
            rows = cursor.fetchmany(max_rows)
    finally:
        conn.close()
    return columns, _rows_to_python(rows)


def _query_sqlite(config: dict, _secret: str | None, sql: str, max_rows: int):
    conn = sqlite3.connect(config.get("path", ":memory:"), timeout=10)
    try:
        cursor = conn.execute(sql)
        columns = [{"name": d[0], "type": ""} for d in cursor.description or []]
        rows = cursor.fetchmany(max_rows)
    finally:
        conn.close()
    return columns, _rows_to_python(rows)


def _bigquery_client(config: dict, secret: str | None):
    from google.cloud import bigquery
    from google.oauth2 import service_account

    project = config.get("project", "")
    if secret:
        info = json.loads(secret)
        credentials = service_account.Credentials.from_service_account_info(info)
        return bigquery.Client(project=project or info.get("project_id"), credentials=credentials)
    return bigquery.Client(project=project or None)


def _query_bigquery(config: dict, secret: str | None, sql: str, max_rows: int):
    client = _bigquery_client(config, secret)
    job = client.query(sql)
    result = job.result(max_results=max_rows)
    columns = [{"name": f.name, "type": f.field_type} for f in result.schema]
    rows = [[value for value in row.values()] for row in result]
    return columns, rows


_ENGINES = {
    "postgresql": _query_postgresql,
    "mysql": _query_mysql,
    "sqlite": _query_sqlite,
    "bigquery": _query_bigquery,
}


async def execute_query(datasource: dict, sql: str, max_rows: int):
    """(columns, rows) for a read-only query against the datasource."""
    ensure_readonly(sql)
    engine = _ENGINES.get(datasource["kind"])
    if engine is None:
        raise ValueError(f"tipo de datasource não suportado: {datasource['kind']}")
    return await anyio.to_thread.run_sync(
        lambda: engine(datasource.get("config", {}), datasource.get("secret"), sql, max_rows)
    )


# Só os nomes. Sai barato e é o que garante que nenhuma tabela suma da lista
# quando as colunas não couberem.
_LIST_TABLE_NAMES_SQL = {
    "postgresql": """
        SELECT table_schema || '.' || table_name
          FROM information_schema.tables
         WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
         ORDER BY 1 LIMIT 500""",
    "mysql": """
        SELECT CONCAT(table_schema, '.', table_name)
          FROM information_schema.tables
         WHERE table_schema = DATABASE()
         ORDER BY 1 LIMIT 500""",
}

_LIST_TABLES_SQL = {
    "postgresql": """
        SELECT table_schema || '.' || table_name, column_name, data_type
          FROM information_schema.columns
         WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
         ORDER BY 1, ordinal_position LIMIT 500""",
    "mysql": """
        SELECT CONCAT(table_schema, '.', table_name), column_name, data_type
          FROM information_schema.columns
         WHERE table_schema = DATABASE()
         ORDER BY 1, ordinal_position LIMIT 500""",
}


# Quantas tabelas vêm com as colunas detalhadas. Listar coluna de tudo estoura o
# contexto num dataset grande; listar só as primeiras N e **calar sobre o
# resto** é pior ainda. Medido no QA: o dataset tinha 180 tabelas, o corte era
# 50, e as tabelas do SIOPE — o assunto inteiro do template — estavam nas
# posições 75 a 80. O modelo nunca soube que existiam e passou o turno
# garimpando INFORMATION_SCHEMA: 54 consultas, mais de dez minutos, sem
# resposta. Nome de tabela é barato; é o que faz o modelo pedir a coluna certa
# de uma vez.
TABELAS_COM_COLUNAS = 50

_SEM_COLUNAS = (
    "colunas não listadas aqui — consulte INFORMATION_SCHEMA.COLUMNS desta tabela"
)


def _catalogo(nomes: list[str], colunas_de) -> list[dict]:
    """Catálogo do dataset: todas as tabelas, colunas nas primeiras."""
    catalogo = []
    for posicao, nome in enumerate(nomes):
        if posicao < TABELAS_COM_COLUNAS:
            catalogo.append({"table": nome, "columns": colunas_de(nome)})
        else:
            catalogo.append({"table": nome, "columns": _SEM_COLUNAS})
    return catalogo


def _escolhidas(datasource: dict, nomes: list[str]) -> list[str]:
    """Filtra pelas tabelas que o dono da fonte declarou em `config.tables`.

    Num lake de 180 tabelas onde 7 interessam, ordem alfabética decide o que o
    modelo enxerga — e as 7 do caso real caíam fora. Declarar quais valem
    resolve isso melhor do que qualquer limite: a lista fica curta e todas vêm
    com coluna. Sem declaração, nada muda.

    Aceita o nome com ou sem o prefixo do dataset, porque quem configura pensa
    na tabela, não no caminho.
    """
    escolhidas = datasource.get("config", {}).get("tables") or []
    if not escolhidas:
        return nomes
    querido = {str(t).strip() for t in escolhidas}
    return [n for n in nomes if n in querido or n.split(".")[-1] in querido]


async def list_tables(datasource: dict) -> list[dict]:
    """[{table, columns:[{name,type}]}] — capped, for the model's orientation."""
    kind = datasource["kind"]
    if kind in _LIST_TABLES_SQL:
        _, linhas_nomes = await execute_query(
            datasource, _LIST_TABLE_NAMES_SQL[kind], max_rows=500
        )
        nomes = _escolhidas(datasource, [linha[0] for linha in linhas_nomes])

        _, rows = await execute_query(
            datasource, _LIST_TABLES_SQL[kind], max_rows=500
        )
        # As colunas vêm num só resultado limitado: uma tabela larga consome a
        # cota e as seguintes ficam sem coluna. Ficar sem coluna é aceitável —
        # sumir da lista, não.
        tables: dict[str, list] = {}
        for table, column, data_type in rows:
            tables.setdefault(table, []).append({"name": column, "type": data_type})
        return [{"table": n, "columns": tables.get(n) or _SEM_COLUNAS} for n in nomes]

    if kind == "sqlite":
        columns, rows = await execute_query(
            datasource,
            "SELECT name FROM sqlite_master WHERE type = 'table' LIMIT 100",
            max_rows=100,
        )
        out = []
        for (table,) in rows:
            _, info = await execute_query(
                datasource, f"SELECT * FROM pragma_table_info('{table}')", max_rows=100
            )
            out.append(
                {"table": table, "columns": [{"name": r[1], "type": r[2]} for r in info]}
            )
        return out

    if kind == "bigquery":
        def _bq():
            client = _bigquery_client(
                datasource.get("config", {}), datasource.get("secret")
            )
            dataset_id = datasource.get("config", {}).get("dataset", "")
            out = []
            datasets = (
                [client.get_dataset(dataset_id)] if dataset_id else list(client.list_datasets())[:5]
            )
            for ds in datasets:
                ref = ds.reference if hasattr(ds, "reference") else ds
                tabelas = list(client.list_tables(ref))
                por_nome = {
                    f"{t.dataset_id}.{t.table_id}": t for t in tabelas
                }

                def colunas_de(nome, _por_nome=por_nome):
                    schema = client.get_table(_por_nome[nome].reference).schema
                    return [{"name": f.name, "type": f.field_type} for f in schema]

                out.extend(_catalogo(_escolhidas(datasource, list(por_nome)), colunas_de))
            return out

        return await anyio.to_thread.run_sync(_bq)

    raise ValueError(f"tipo de datasource não suportado: {kind}")


_WRITE_FIRST = re.compile(r"^\s*(insert|update|delete)\b", re.IGNORECASE)
_FORBIDDEN_WRITE = re.compile(
    r"\b(drop|alter|create|truncate|grant|revoke|vacuum|attach|pragma)\b", re.IGNORECASE
)
_TARGET_TABLE = re.compile(
    r"^\s*(?:insert\s+into|update|delete\s+from)\s+([\"'`]?[\w.]+[\"'`]?)",
    re.IGNORECASE,
)


def validate_write(statement: str, allowed_tables: list[str]) -> str:
    """Guardrails for write statements. Returns the target table or raises."""
    body = statement.strip().rstrip(";")
    if ";" in body:
        raise ValueError("apenas um statement por chamada")
    if not _WRITE_FIRST.match(body):
        raise ValueError("apenas INSERT, UPDATE ou DELETE são permitidos")
    if _FORBIDDEN_WRITE.search(body):
        raise ValueError("DDL/estruturais são proibidos nesta tool")
    first_word = body.split(None, 1)[0].lower()
    if first_word in ("update", "delete") and not re.search(r"\bwhere\b", body, re.IGNORECASE):
        raise ValueError(f"{first_word.upper()} sem WHERE é proibido")
    match = _TARGET_TABLE.match(body)
    if not match:
        raise ValueError("não foi possível identificar a tabela alvo")
    table = match.group(1).strip("\"'`").lower()
    normalized = {t.lower() for t in allowed_tables}
    # Accept both bare and schema-qualified declarations.
    bare = table.split(".")[-1]
    if table not in normalized and bare not in normalized:
        raise ValueError(
            f"tabela '{table}' não está na lista de escrita permitida deste template"
        )
    return bare


def _write_postgresql(config: dict, secret: str | None, sql: str):
    import psycopg

    conninfo = psycopg.conninfo.make_conninfo(
        host=config.get("host", "localhost"),
        port=int(config.get("port", 5432)),
        dbname=config.get("database", ""),
        user=config.get("user", ""),
        password=secret or "",
        connect_timeout=10,
    )
    with psycopg.connect(conninfo) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            count = cursor.rowcount
            returned = None
            if cursor.description is not None:  # RETURNING clause present
                returned = [list(r) for r in cursor.fetchall()]
        conn.commit()
    return count, returned


def _write_mysql(config: dict, secret: str | None, sql: str) -> int:
    import pymysql

    conn = pymysql.connect(
        host=config.get("host", "localhost"),
        port=int(config.get("port", 3306)),
        database=config.get("database", ""),
        user=config.get("user", ""),
        password=secret or "",
        connect_timeout=10,
    )
    try:
        with conn.cursor() as cursor:
            count = cursor.execute(sql)
            returned = [[cursor.lastrowid]] if cursor.lastrowid else None
        conn.commit()
    finally:
        conn.close()
    return count, returned


def _write_sqlite(config: dict, _secret: str | None, sql: str):
    conn = sqlite3.connect(config.get("path", ":memory:"), timeout=10)
    try:
        cursor = conn.execute(sql)
        conn.commit()
        returned = None
        if cursor.description is not None:
            returned = [list(r) for r in cursor.fetchall()]
        elif cursor.lastrowid:
            returned = [[cursor.lastrowid]]
        return cursor.rowcount, returned
    finally:
        conn.close()


def _write_bigquery(config: dict, secret: str | None, sql: str):
    client = _bigquery_client(config, secret)
    job = client.query(sql)
    job.result()
    return job.num_dml_affected_rows or 0, None


_WRITE_ENGINES = {
    "postgresql": _write_postgresql,
    "mysql": _write_mysql,
    "sqlite": _write_sqlite,
    "bigquery": _write_bigquery,
}


async def execute_write(datasource: dict, sql: str, allowed_tables: list[str]):
    """(target_table, affected_rows, returned_rows) after all guardrails pass.
    returned_rows carries RETURNING output (or lastrowid) so agents can chain
    parent->child inserts without guessing ids."""
    table = validate_write(sql, allowed_tables)
    engine = _WRITE_ENGINES.get(datasource["kind"])
    if engine is None:
        raise ValueError(f"tipo de datasource não suportado: {datasource['kind']}")
    rows, returned = await anyio.to_thread.run_sync(
        lambda: engine(datasource.get("config", {}), datasource.get("secret"), sql)
    )
    return table, rows, returned


_PLACEHOLDER = re.compile(r"\{\{\s*returned:(\d+)(?::(\d+))?\s*\}\}")


def _substitute_placeholders(statement: str, results: list[dict]) -> str:
    """Replace {{returned:N}} / {{returned:N:C}} with the value from an earlier
    statement's RETURNING (row 0, column N-th index / C), so a child insert can
    reference the parent's generated id inside the same transaction."""

    def repl(match: re.Match) -> str:
        idx = int(match.group(1))
        col = int(match.group(2)) if match.group(2) else 0
        if idx >= len(results) or not results[idx].get("returned"):
            raise ValueError(f"{{{{returned:{idx}}}}} não disponível (statement sem RETURNING)")
        value = results[idx]["returned"][0][col]
        # Numeric ids inline as-is; anything else is single-quoted safely.
        if isinstance(value, (int, float)):
            return str(value)
        return "'" + str(value).replace("'", "''") + "'"

    return _PLACEHOLDER.sub(repl, statement)


def _run_transaction(kind: str, config: dict, secret: str | None, statements: list[str]):
    """Execute all statements in ONE transaction. Placeholders are resolved
    left to right. Any failure rolls back the whole batch."""
    if kind == "postgresql":
        import psycopg

        conninfo = psycopg.conninfo.make_conninfo(
            host=config.get("host", "localhost"), port=int(config.get("port", 5432)),
            dbname=config.get("database", ""), user=config.get("user", ""),
            password=secret or "", connect_timeout=10,
        )
        results: list[dict] = []
        with psycopg.connect(conninfo) as conn:
            try:
                with conn.cursor() as cur:
                    for stmt in statements:
                        cur.execute(_substitute_placeholders(stmt, results))
                        returned = (
                            [list(r) for r in cur.fetchall()] if cur.description else None
                        )
                        results.append({"affected_rows": cur.rowcount, "returned": returned})
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return results

    if kind == "sqlite":
        conn = sqlite3.connect(config.get("path", ":memory:"), timeout=10)
        results = []
        try:
            for stmt in statements:
                cur = conn.execute(_substitute_placeholders(stmt, results))
                if cur.description is not None:
                    returned = [list(r) for r in cur.fetchall()]
                elif cur.lastrowid:
                    returned = [[cur.lastrowid]]
                else:
                    returned = None
                results.append({"affected_rows": cur.rowcount, "returned": returned})
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return results

    if kind == "mysql":
        import pymysql

        conn = pymysql.connect(
            host=config.get("host", "localhost"), port=int(config.get("port", 3306)),
            database=config.get("database", ""), user=config.get("user", ""),
            password=secret or "", connect_timeout=10,
        )
        results = []
        try:
            with conn.cursor() as cur:
                for stmt in statements:
                    count = cur.execute(_substitute_placeholders(stmt, results))
                    returned = [[cur.lastrowid]] if cur.lastrowid else None
                    results.append({"affected_rows": count, "returned": returned})
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return results

    raise ValueError(f"transações não suportadas para datasource '{kind}'")


async def execute_transaction(
    datasource: dict, statements: list[str], allowed_tables: list[str]
) -> list[dict]:
    """Guarded, atomic multi-statement write. Every statement passes the same
    guardrails as a single write; all run in one transaction (all-or-nothing).
    Returns per-statement {table, affected_rows, returned}."""
    if not statements:
        raise ValueError("nenhum statement informado")
    if len(statements) > 50:
        raise ValueError("máximo de 50 statements por transação")
    tables = [validate_write(s, allowed_tables) for s in statements]
    rows = await anyio.to_thread.run_sync(
        lambda: _run_transaction(
            datasource["kind"], datasource.get("config", {}),
            datasource.get("secret"), statements,
        )
    )
    return [{"table": t, **r} for t, r in zip(tables, rows, strict=True)]


async def test_connection(datasource: dict) -> tuple[bool, str]:
    try:
        await execute_query(datasource, "SELECT 1", max_rows=1)
        return True, ""
    except Exception as exc:  # noqa: BLE001 — the point is reporting it
        return False, str(exc)[:500]


def make_temp_sqlite(rows_sql: str) -> str:
    """Test helper: temp sqlite database seeded with the given SQL script."""
    path = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(path)
    conn.executescript(rows_sql)
    conn.commit()
    conn.close()
    return path
