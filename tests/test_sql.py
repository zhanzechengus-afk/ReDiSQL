import sqlite3
import time

import pytest

from redisql.sql import denotation_equal, execute_readonly, validate_sql, validate_static


@pytest.fixture
def schema():
    return {
        "tables": [
            {"name": "employee", "columns": [{"name": "id", "type": "INTEGER"}, {"name": "name", "type": "TEXT"}, {"name": "age", "type": "INTEGER"}, {"name": "department_id", "type": "INTEGER"}]},
            {"name": "department", "columns": [{"name": "id", "type": "INTEGER"}, {"name": "title", "type": "TEXT"}]},
        ]
    }


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "database.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript("CREATE TABLE employee(id INTEGER, name TEXT, age INTEGER, department_id INTEGER); CREATE TABLE department(id INTEGER, title TEXT); INSERT INTO employee VALUES(1, 'Alice', 30, 1), (2, 'Bob', 40, 1), (3, 'Bob', NULL, 2); INSERT INTO department VALUES(1, 'Research'), (2, 'Sales');")
    connection.close()
    return path


@pytest.mark.parametrize("sql", [
    "SELECT e.name FROM employee e",
    "SELECT e.name,d.title FROM employee e JOIN department d ON e.department_id=d.id",
    "WITH x AS (SELECT id,name FROM employee) SELECT x.name FROM x",
    "SELECT x.name FROM (SELECT name FROM employee) x",
    "SELECT name FROM employee e WHERE EXISTS(SELECT 1 FROM employee f WHERE f.id=e.id)",
    "SELECT MAX(age) AS maximum_age FROM employee HAVING maximum_age>10",
    "SELECT name AS employee_name FROM employee ORDER BY employee_name",
    "SELECT name FROM MAIN.employee",
    "SELECT \"NAME\" FROM \"EMPLOYEE\"",
    "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x WHERE n<5) SELECT n FROM x",
])
def test_scope_resolution(database, schema, sql):
    assert validate_sql(sql, schema, database, require_denotation=False)["ok"]


def test_static_validation_never_executes(schema, monkeypatch):
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: pytest.fail("database accessed"))
    assert validate_static("SELECT name FROM employee", schema)["ok"]
    assert validate_static("SELECT absent FROM employee", schema)["category"] == "schema"


@pytest.mark.parametrize("sql,name", [
    ("SELECT absent FROM employee", "absent"),
    ("SELECT e.absent FROM employee e", "absent"),
    ("SELECT name FROM absent", "absent"),
    ("SELECT x.name FROM employee e", "x.name"),
])
def test_unknown_identifiers_are_sanitized(database, schema, sql, name):
    result = validate_sql(sql, schema, database, require_denotation=False)
    assert result["category"] == "schema"
    assert result["message"].startswith("Unresolved identifier: ")
    assert name in result["message"]
    assert "rows" not in result


def test_ambiguous_column_is_rejected(schema):
    assert validate_static("SELECT id FROM employee JOIN department ON employee.department_id=department.id", schema)["category"] == "schema"


def test_reference_result_comparison(database, schema):
    correct = validate_sql("SELECT name FROM employee", schema, database, "SELECT name FROM employee ORDER BY name")
    assert correct["ok"]
    wrong = validate_sql("SELECT DISTINCT name FROM employee", schema, database, "SELECT name FROM employee")
    assert wrong["category"] == "denotation"
    assert wrong["message"] == "Executable query; result mismatch"
    assert "Alice" not in wrong["message"]


def test_order_is_checked_when_reference_orders(database, schema):
    result = validate_sql("SELECT age FROM employee ORDER BY age DESC", schema, database, "SELECT age FROM employee ORDER BY age ASC")
    assert result["category"] == "denotation"


def test_denotation_preserves_multiplicity_and_allows_column_permutation():
    assert denotation_equal([[1, "a"], [2, None]], [[None, 2], ["a", 1]])
    assert not denotation_equal([[1], [1], [2]], [[1], [2], [2]])
    assert denotation_equal([[1]], [[1.0]])
    assert not denotation_equal([[1], [2]], [[2], [1]], ordered=True)
    assert denotation_equal([], [])


def test_denotation_prunes_symmetric_column_candidates():
    width = 12
    cycle = [[int(column in {row, (row + 1) % width}) for column in range(width)] for row in range(width)]
    disconnected = [[int(column in {row, row // 4 * 4 + (row + 1) % 4}) for column in range(width)] for row in range(width)]
    started = time.monotonic()
    assert not denotation_equal(cycle, disconnected)
    assert time.monotonic() - started < 2
    permutation = [5, 2, 7, 0, 1, 11, 9, 4, 3, 8, 10, 6]
    reordered = [[row[index] for index in permutation] for row in reversed(cycle)]
    assert denotation_equal(cycle, reordered)


@pytest.mark.parametrize("sql", [
    "DROP TABLE employee",
    "DELETE FROM employee RETURNING id",
    "SELECT 1; DROP TABLE employee",
    "ATTACH DATABASE '/tmp/unwanted.sqlite' AS x",
    "PRAGMA writable_schema=1",
    "SELECT load_extension('/tmp/unwanted.so')",
])
def test_readonly_guard(database, sql):
    assert not execute_readonly(database, sql)["ok"]
    assert execute_readonly(database, "SELECT COUNT(*) FROM employee")["rows"] == [[3]]


def test_timeout_is_enforced(database):
    query = "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT SUM(n) FROM x"
    result = execute_readonly(database, query, timeout=0.01)
    assert not result["ok"]
    assert result["error"] == "timeout"
    assert result["elapsed"] < 1


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "absent.sqlite"
    assert not execute_readonly(path, "SELECT 1")["ok"]
    assert not path.exists()


def test_bad_reference_is_a_dataset_error(database, schema):
    with pytest.raises(ValueError, match="Reference"):
        validate_sql("SELECT name FROM employee", schema, database, "SELECT absent FROM employee")


def test_syntax_feedback_exposes_only_position(database, schema):
    result = validate_sql("SELECT ( FROM employee", schema, database, require_denotation=False)
    assert result["category"] == "syntax"
    assert result["message"].startswith("Syntax error at ")
    assert "employee" not in result["message"]


@pytest.mark.parametrize("sql", [
    'SELECT name FROM employee WHERE name="Alice"',
    'SELECT CASE WHEN name="Alice" THEN "first" ELSE "other" END FROM employee',
    'SELECT AVG(age) AS "mean_age" FROM employee HAVING "mean_age">20',
    'SELECT name AS "label" FROM employee ORDER BY "label"',
    'SELECT name AS label, "label" FROM employee',
    'SELECT name FROM employee e WHERE EXISTS (SELECT 1 FROM department d WHERE "name"="Alice")',
    'WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT "n"+1 FROM x WHERE "n"<5) SELECT "n" FROM x',
])
def test_sqlite_double_quoted_literals_and_real_names(database, schema, sql):
    result = validate_sql(sql, schema, database, sql)
    assert result["ok"]


def test_double_quoted_literals_preserve_original_text(database, schema):
    sql = 'SELECT name FROM employee WHERE name="Alice"'
    result = validate_sql(sql, schema, database, "SELECT name FROM employee WHERE name='Alice'")
    assert result["ok"]
    assert '"Alice"' in result["sql"]
    assert validate_static('SELECT e."missing" FROM employee e', schema)["category"] == "schema"
