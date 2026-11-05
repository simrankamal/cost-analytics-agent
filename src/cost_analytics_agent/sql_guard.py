import sqlglot
from sqlglot import exp

BLOCKED_FUNCTIONS = {
    "glob",
    "getenv",
    "current_setting",
    "query",
    "query_table",
    "sniff_csv",
    "parquet_scan",
    "parquet_metadata",
    "parquet_schema",
    "duckdb_settings",
    "duckdb_secrets",
    "duckdb_extensions",
}
FORBIDDEN_NODES = tuple(
    getattr(exp, name)
    for name in [
        "Insert",
        "Update",
        "Delete",
        "Merge",
        "Create",
        "Drop",
        "Alter",
        "Command",
        "Copy",
        "Set",
        "Use",
        "Transaction",
        "Commit",
        "Rollback",
        "Attach",
        "Detach",
        "Pragma",
        "LoadData",
    ]
    if hasattr(exp, name)
)


class UnsafeSQLError(ValueError):
    pass


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def guard(sql: str, allowed_tables: set[str], max_rows: int = 1000) -> str:
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQLError(f"could not parse SQL: {e}") from e
    if len(statements) != 1:
        raise UnsafeSQLError("exactly one statement is allowed")
    root = statements[0]
    if not isinstance(root, exp.Query):
        raise UnsafeSQLError("only SELECT queries are allowed")

    for node in root.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise UnsafeSQLError(f"{type(node).__name__} is not allowed")

    ctes = {cte.alias_or_name.lower() for cte in root.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables} | ctes
    for table in root.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise UnsafeSQLError("table functions are not allowed")
        if table.args.get("db") or table.args.get("catalog"):
            raise UnsafeSQLError(f"qualified table names are not allowed: {table.sql()}")
        if table.name.lower() not in allowed:
            raise UnsafeSQLError(f"unknown table: {table.name}")

    for func in root.find_all(exp.Func):
        name = _function_name(func)
        if name.startswith("read_") or name in BLOCKED_FUNCTIONS:
            raise UnsafeSQLError(f"function not allowed: {name}")

    limit = root.args.get("limit")
    current = None
    if limit is not None and isinstance(limit.expression, exp.Literal):
        current = int(limit.expression.name)
    if current is None or current > max_rows:
        root = root.limit(max_rows)
    return root.sql(dialect="duckdb")
