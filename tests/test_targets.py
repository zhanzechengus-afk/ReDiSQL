import re

import pytest

from redisql.targets import build_targets


class OffsetTokenizer:
    eos_token_id = 2
    is_fast = True

    def __call__(self, text, **kwargs):
        matches = list(re.finditer(r"\w+|[^\w\s]", text))
        return {"input_ids": list(range(3, 3 + len(matches))), "offset_mapping": [(match.start(), match.end()) for match in matches]}


@pytest.fixture
def tokenizer():
    return OffsetTokenizer()


@pytest.fixture
def schema():
    return {"tables": [{"name": "employee", "columns": [{"name": "id", "type": "INTEGER"}, {"name": "name", "type": "TEXT"}, {"name": "age", "type": "INTEGER"}]}]}


def selected(target, key):
    return [target["repair_sql"][start:end] if end > start else "EOS" for (start, end), flag in zip(target["token_offsets"], target[key]) if flag]


def test_distinct_deletion_supervises_next_token_once(tokenizer, schema):
    target = build_targets("SELECT DISTINCT name FROM employee", "SELECT name FROM employee", tokenizer, schema)
    assert selected(target, "edit_mask") == ["name"]
    assert selected(target, "deletion_mask") == ["name"]
    assert not any(target["identifier_mask"])


def test_trailing_deletion_supervises_eos(tokenizer, schema):
    target = build_targets("SELECT name FROM employee LIMIT 1", "SELECT name FROM employee", tokenizer, schema)
    assert selected(target, "edit_mask") == ["EOS"]
    assert target["input_ids"][-1] == tokenizer.eos_token_id
    assert not any(target["identifier_mask"])


def test_changed_identifier_receives_identifier_weight(tokenizer, schema):
    target = build_targets("SELECT age FROM employee", "SELECT name FROM employee", tokenizer, schema)
    assert selected(target, "changed_mask") == ["name"]
    assert selected(target, "identifier_mask") == ["name"]
    assert not any(target["deletion_mask"])


def test_no_edit_has_zero_mask(tokenizer, schema):
    target = build_targets("SELECT name FROM employee", "select name from employee", tokenizer, schema)
    assert not any(target["edit_mask"])
    assert target["alignment_mode"] == "ast"


def test_changed_operator_and_literal_are_not_identifier_tokens(tokenizer, schema):
    target = build_targets("SELECT name FROM employee WHERE age > 30", "SELECT name FROM employee WHERE age >= 40", tokenizer, schema)
    assert set(selected(target, "changed_mask")) == {">", "=", "40"}
    assert not any(target["identifier_mask"])


def test_all_subwords_of_changed_identifier_are_marked(schema):
    class CharacterTokenizer(OffsetTokenizer):
        def __call__(self, text, **kwargs):
            positions = [index for index, character in enumerate(text) if not character.isspace()]
            return {"input_ids": [index + 3 for index in positions], "offset_mapping": [(index, index + 1) for index in positions]}

    target = build_targets("SELECT age FROM employee", "SELECT name FROM employee", CharacterTokenizer(), schema)
    assert selected(target, "identifier_mask") == list("name")


def test_lexical_fallback_preserves_valid_repair_ast(tokenizer, schema):
    target = build_targets("SELECT name FROM employee WHERE", "SELECT name FROM employee", tokenizer, schema)
    assert target["alignment_mode"] == "token"
    assert selected(target, "edit_mask") == ["EOS"]
    assert target["ast_nodes"]


def test_character_fallback_handles_broken_quote(tokenizer, schema):
    target = build_targets("SELECT 'unterminated", "SELECT name FROM employee", tokenizer, schema)
    assert target["alignment_mode"] == "character"
    assert any(target["edit_mask"])


def test_character_fallback_trailing_deletion_marks_eos(tokenizer, schema):
    target = build_targets("SELECT name FROM employee '", "SELECT name FROM employee", tokenizer, schema)
    assert target["alignment_mode"] == "character"
    assert selected(target, "deletion_mask") == ["EOS"]


def test_repeated_deletion_positions_are_merged(tokenizer, schema):
    target = build_targets("SELECT DISTINCT DISTINCT name FROM employee", "SELECT name FROM employee", tokenizer, schema)
    assert selected(target, "edit_mask") == ["name"]
    assert sum(target["deletion_mask"]) == 1


def test_ast_positions_exclude_eos_and_edges_reference_retained_nodes(tokenizer, schema):
    target = build_targets("SELECT name FROM employee", "SELECT COUNT(*) FROM employee WHERE age >= 30", tokenizer, schema)
    assert target["ast_nodes"][0]["positions"] == list(range(len(target["input_ids"]) - 1))
    assert all(node["positions"] and max(node["positions"]) < len(target["input_ids"]) - 1 for node in target["ast_nodes"])
    assert all(0 <= edge["parent"] < len(target["ast_nodes"]) and 0 <= edge["child"] < len(target["ast_nodes"]) for edge in target["ast_edges"])
    assert {node["label"] for node in target["ast_nodes"]} <= set(range(9))
    assert {edge["label"] for edge in target["ast_edges"]} <= set(range(6))
    assert {0, 1, 2, 3, 4} <= {edge["label"] for edge in target["ast_edges"]}


@pytest.mark.parametrize("repair", [
    "SELECT e.name FROM employee e",
    "WITH x AS (SELECT name FROM employee) SELECT x.name FROM x",
    "WITH x AS (SELECT * FROM employee) SELECT x.name FROM x",
    "SELECT x.name FROM (SELECT name FROM employee) x",
])
def test_schema_positives_use_physical_names(tokenizer, schema, repair):
    target = build_targets("SELECT age FROM employee", repair, tokenizer, schema)
    assert target["schema_positives"] == ["employee", "employee.name"]


def test_schema_inference_resolves_aliases_without_supplied_schema(tokenizer):
    target = build_targets("SELECT e.age FROM employee e", "SELECT e.name FROM employee e", tokenizer)
    assert target["schema_positives"] == ["employee", "employee.name"]


def test_duplicate_identifier_occurrences_are_deterministic(tokenizer, schema):
    arguments = ("SELECT age, name FROM employee WHERE name='a'", "SELECT name, name FROM employee WHERE name='a'", tokenizer, schema)
    first = build_targets(*arguments)
    second = build_targets(*arguments)
    assert first == second
    assert sum(first["edit_mask"]) >= 1


def test_null_and_boolean_labels(tokenizer, schema):
    target = build_targets("SELECT age FROM employee", "SELECT NULL, TRUE FROM employee", tokenizer, schema)
    assert sum(node["label"] == 7 for node in target["ast_nodes"]) == 2


def test_quoted_identifier_case_is_preserved(tokenizer, schema):
    target = build_targets('SELECT "age" FROM employee', 'SELECT "name" FROM employee', tokenizer, schema)
    assert '"name"' in target["repair_sql"]
    assert any(target["identifier_mask"])


def test_target_truncation_is_rejected(tokenizer, schema):
    with pytest.raises(ValueError, match="max_length"):
        build_targets("SELECT name FROM employee", "SELECT name FROM employee", tokenizer, schema, max_length=3)


def test_invalid_repair_is_rejected(tokenizer, schema):
    with pytest.raises(ValueError, match="Repair SQL"):
        build_targets("SELECT name FROM employee", "SELECT (", tokenizer, schema)


def test_sqlite_double_quoted_values_are_literals_not_schema_identifiers(tokenizer, schema):
    target = build_targets('SELECT name FROM employee WHERE name="Alice"', 'SELECT name FROM employee WHERE name="Bob"', tokenizer, schema)
    assert '"Bob"' in target["repair_sql"]
    assert target["schema_positives"] == ["employee", "employee.name"]
    assert not any(target["identifier_mask"])
    assert selected(target, "changed_mask") == ['"', 'Bob', '"']
    assert sum(node["label"] == 7 for node in target["ast_nodes"]) == 1


def test_case_expression_keeps_quoted_values_out_of_schema_targets(tokenizer, schema):
    query = 'SELECT CASE WHEN name="Alice" THEN "first" ELSE "other" END FROM employee'
    target = build_targets(query, query, tokenizer, schema)
    assert not any(target["edit_mask"])
    assert target["schema_positives"] == ["employee", "employee.name"]
    assert sum(node["label"] == 7 for node in target["ast_nodes"]) == 3
