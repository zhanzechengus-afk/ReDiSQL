import math
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope


def parse_query(sql):
    statements = sqlglot.parse(sql, read="sqlite")
    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Expected one SQL query")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise ValueError("Expected a read-only query")
    forbidden = tuple(
        cls for name in ("Into", "Lock", "Command", "Pragma", "Attach", "Detach", "DML", "DDL")
        if (cls := getattr(exp, name, None)) is not None
    )
    if any(isinstance(node, forbidden) for node in tree.walk()):
        raise ValueError("Expected a read-only query")
    return tree


def normalize_sql(sql):
    return parse_query(sql).sql(dialect="sqlite", pretty=False, comments=False, normalize=False)


def schema_mapping(schema):
    return {
        table["name"]: {column["name"]: column.get("type") or "TEXT" for column in table["columns"]}
        for table in schema["tables"]
    }


def sqlite_string_columns(tree, schema):
    catalog = {table["name"].casefold(): {column["name"].casefold() for column in table["columns"]} for table in schema["tables"]}

    def sources(scope):
        return {alias: source for alias, (node, source) in scope.selected_sources.items()}

    def source_columns(source, seen):
        if isinstance(source, exp.Table):
            return catalog.get(source.name.casefold(), set())
        return outputs(source, seen)

    def outputs(scope, seen):
        if id(scope) in seen:
            return set()
        seen = seen | {id(scope)}
        if scope.outer_columns:
            return {name.casefold() for name in scope.outer_columns}
        if scope.union_scopes:
            return outputs(scope.union_scopes[0], seen)
        available = sources(scope)
        names = set()
        for projection in scope.expression.selects:
            if isinstance(projection, exp.Star):
                for source in available.values():
                    names.update(source_columns(source, seen))
            elif isinstance(projection, exp.Column) and projection.is_star:
                for alias, source in available.items():
                    if alias.casefold() == projection.table.casefold():
                        names.update(source_columns(source, seen))
            elif projection.alias_or_name:
                names.add(projection.alias_or_name.casefold())
        return names

    def visible(scope):
        names = set()
        for source in sources(scope).values():
            names.update(source_columns(source, {id(scope)}))
        if scope.parent is not None and scope.can_be_correlated:
            names.update(visible(scope.parent))
        return names

    string_columns = []
    for scope in traverse_scope(tree) or []:
        known = visible(scope)
        aliases = {projection.alias.casefold() for projection in scope.expression.selects if projection.alias}
        for column in list(scope.columns):
            if column.table or not isinstance(column.this, exp.Identifier) or not column.this.args.get("quoted"):
                continue
            ancestor = column
            while ancestor.parent is not None and ancestor.parent is not scope.expression:
                ancestor = ancestor.parent
            in_projection = ancestor.parent is scope.expression and ancestor.arg_key == "expressions"
            if column.name.casefold() not in known and (in_projection or column.name.casefold() not in aliases):
                string_columns.append(column)
    return string_columns


def _sqlite_literals(tree, schema):
    tree = tree.copy()
    for column in sqlite_string_columns(tree, schema):
        column.replace(exp.Literal.string(column.name))
    return tree


def qualify_query(tree, schema):
    return qualify(
        _sqlite_literals(tree, schema),
        dialect="sqlite",
        schema=schema_mapping(schema),
        expand_alias_refs=True,
        expand_stars=False,
        infer_schema=False,
        validate_qualify_columns=True,
        quote_identifiers=False,
        identify=False,
    )


def _identifier_position(sql, name):
    try:
        dialect = sqlglot.Dialect.get_or_raise("sqlite")
        tokens = dialect.tokenize(sql)
        needle = dialect.tokenize(name)
        for index in range(len(tokens) - len(needle) + 1):
            if needle and all(left.text.casefold() == right.text.casefold() and left.token_type.name != "STRING" for left, right in zip(tokens[index:index + len(needle)], needle)):
                return tokens[index].start
    except Exception:
        pass
    return len(sql)


def _unknown_identifier(tree, schema, sql):
    known_tables = {table["name"].casefold() for table in schema["tables"]}
    missing = []
    scopes = list(traverse_scope(tree) or [])
    for scope in scopes:
        for source in scope.sources.values():
            if isinstance(source, exp.Table):
                if source.name.casefold() not in known_tables or source.catalog or source.db.casefold() not in ("", "main"):
                    missing.append(source.name)
    if missing:
        return min(missing, key=lambda name: _identifier_position(sql, name))
    try:
        qualify_query(tree, schema)
        return None
    except Exception as error:
        names = []
        message = str(error)
        for value in re.findall(r"['\"]([^'\"]+)['\"]", message):
            for column in tree.find_all(exp.Column):
                if value.casefold() in {column.name.casefold(), column.sql().casefold()}:
                    names.append(column.sql())
        if names:
            return min(names, key=lambda name: _identifier_position(sql, name))
        columns = list(tree.find_all(exp.Column))
        return min((column.sql() for column in columns), key=lambda name: _identifier_position(sql, name), default="query")


def _syntax_position(sql, error):
    positions = []
    if isinstance(error, ParseError):
        lines = sql.splitlines(keepends=True)
        for detail in error.errors:
            line = max(1, int(detail.get("line", 1)))
            column = max(1, int(detail.get("col", 1)) - len(detail.get("highlight", "")) + 1)
            positions.append(sum(len(value) for value in lines[:line - 1]) + column)
    return min(positions, default=1)


def _safe_identifier(name):
    return re.sub(r"[\r\n\t\x00-\x1f]", " ", name)[:160]


def _hashable(value):
    if value is None:
        return ("null",)
    if isinstance(value, (bool, int, float)):
        if isinstance(value, float) and math.isnan(value):
            return ("nan",)
        return ("number", value)
    if isinstance(value, bytes):
        return ("bytes", value)
    if isinstance(value, dict):
        return ("mapping", tuple(sorted((key, _hashable(item)) for key, item in value.items())))
    return ("text", value)


def denotation_equal(reference_rows, pred_rows, ordered=False):
    if len(reference_rows) != len(pred_rows):
        return False
    if not reference_rows:
        return True
    width = len(reference_rows[0])
    if any(len(row) != width for row in reference_rows) or any(len(row) != width for row in pred_rows):
        return False
    left = [tuple(_hashable(value) for value in row) for row in reference_rows]
    right = [tuple(_hashable(value) for value in row) for row in pred_rows]
    if left == right or (not ordered and Counter(left) == Counter(right)):
        return True
    row_signatures = lambda rows: [tuple(sorted(row)) for row in rows]
    if ordered:
        if row_signatures(left) != row_signatures(right):
            return False
    elif Counter(row_signatures(left)) != Counter(row_signatures(right)):
        return False
    signature = lambda rows, index: tuple(row[index] for row in rows) if ordered else Counter(row[index] for row in rows)
    left_signatures = [signature(left, index) for index in range(width)]
    right_signatures = [signature(right, index) for index in range(width)]
    candidates = [
        [j for j in range(width) if left_signatures[i] == right_signatures[j]]
        for i in range(width)
    ]
    if any(not values for values in candidates):
        return False
    order = sorted(range(width), key=lambda index: len(candidates[index]))
    permutation = [-1] * width
    left_projections = [Counter(tuple(row[index] for index in order[:depth + 1]) for row in left) for depth in range(width)]

    def search(depth, used):
        if depth == width:
            rows = [tuple(row[index] for index in permutation) for row in right]
            return left == rows if ordered else Counter(left) == Counter(rows)
        index = order[depth]
        seen = set()
        for candidate in candidates[index]:
            vector = tuple(row[candidate] for row in right)
            if candidate in used or vector in seen:
                continue
            seen.add(vector)
            permutation[index] = candidate
            projection = Counter(tuple(row[permutation[item]] for item in order[:depth + 1]) for row in right)
            if projection == left_projections[depth] and search(depth + 1, used | {candidate}):
                return True
        return False

    return search(0, set())


def execute_readonly(db_path, sql, timeout=30):
    started = time.monotonic()
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        return {"ok": False, "rows": [], "error": "timeout", "elapsed": time.monotonic() - started}
    deadline = started + timeout
    connection = None
    rows = []
    try:
        if timeout <= 0:
            raise TimeoutError
        parse_query(sql)
        path = Path(db_path).expanduser().resolve(strict=True)
        connection = sqlite3.connect(f"file:{quote(str(path), safe='/')}?mode=ro", uri=True, timeout=float(timeout))
        connection.text_factory = lambda value: value.decode(errors="ignore")
        connection.enable_load_extension(False)
        read_actions = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE}
        forbidden_functions = {"load_extension", "readfile", "writefile", "fts3_tokenizer", "eval", "shell"}

        def authorize(action, first, second, database, source):
            if action in read_actions:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (second or first or "").casefold() not in forbidden_functions:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        connection.set_authorizer(authorize)
        connection.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        cursor = connection.execute(sql)
        while batch := cursor.fetchmany(1000):
            if time.monotonic() >= deadline:
                raise TimeoutError
            rows.extend([{"__bytes__": value.hex()} if isinstance(value, bytes) else value for value in row] for row in batch)
        return {"ok": True, "rows": rows, "error": None, "elapsed": time.monotonic() - started}
    except Exception:
        timed_out = time.monotonic() >= deadline
        return {"ok": False, "rows": [], "error": "timeout" if timed_out else "execution error", "elapsed": time.monotonic() - started}
    finally:
        if connection is not None:
            connection.close()


def validate_static(sql, schema):
    try:
        tree = parse_query(sql)
        normalized = tree.sql(dialect="sqlite", pretty=False, comments=False, normalize=False)
    except Exception as error:
        return {"ok": False, "category": "syntax", "message": f"Syntax error at {_syntax_position(sql, error)}", "sql": sql}
    unknown = _unknown_identifier(tree, schema, sql)
    if unknown is not None:
        return {"ok": False, "category": "schema", "message": f"Unresolved identifier: {_safe_identifier(unknown)}", "sql": normalized}
    return {"ok": True, "category": "ok", "message": "ok", "sql": normalized}


def validate_sql(sql, schema, db_path, reference_sql=None, timeout=30, require_denotation=True):
    static = validate_static(sql, schema)
    if not static["ok"]:
        return static
    normalized = static["sql"]
    result = execute_readonly(db_path, sql, timeout)
    if not result["ok"]:
        return {"ok": False, "category": "runtime", "message": f"Runtime error: {result['error']}", "sql": normalized}
    if require_denotation:
        if reference_sql is None:
            raise ValueError("reference_sql is required when require_denotation is True")
        reference = execute_readonly(db_path, reference_sql, timeout)
        if not reference["ok"]:
            raise ValueError("Reference query could not be executed")
        reference_tree = parse_query(reference_sql)
        ordered = reference_tree.args.get("order") is not None
        if not denotation_equal(reference["rows"], result["rows"], ordered=ordered):
            return {"ok": False, "category": "denotation", "message": "Executable query; result mismatch", "sql": normalized}
    return {"ok": True, "category": "ok", "message": "ok", "sql": normalized}
