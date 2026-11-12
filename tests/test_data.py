import json
import re
import sqlite3

import pytest

from redisql.data import collate_records, encode_record, encode_sql, load_records, normalized_type, retrieve_schema, select_schema_negatives, split_name


class OffsetTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, truncation=False):
        tokens = list(re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE))
        ids = [3 + sum(ord(char) for char in token.group()) % 101 for token in tokens]
        offsets = [token.span() for token in tokens]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
            offsets = [(0, 0)] + offsets
        value = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            value["offset_mapping"] = offsets
        return value

    def encode(self, text, add_special_tokens=False):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]


def column(name, kind="number", primary=False, references=None):
    return {"name": name, "type": kind, "primary_key": primary, "references": references}


def schema_for_retrieval():
    return {
        "tables": [
            {"name": "account", "columns": [column("id", primary=True), column("name", "text")]},
            {"name": "invoice", "columns": [column("id", primary=True), column("accountId", references={"table": "account", "column": "id"}), column("totalCost")]},
            {"name": "line", "columns": [column("id", primary=True), column("invoiceId", references={"table": "invoice", "column": "id"})]},
            {"name": "contact", "columns": [column("id", primary=True), column("accountId", references={"table": "account", "column": "id"})]},
            {"name": "isolated", "columns": [column("id", primary=True), column("description", "text")]},
        ]
    }


def test_split_names_handles_camel_case_acronyms_and_punctuation():
    assert split_name("HTTPServer customer_ID-orderDate") == {"http", "server", "customer", "id", "order", "date"}


def test_type_compatibility_does_not_match_substrings_in_unrelated_types():
    assert normalized_type("INTEGER") == normalized_type("DECIMAL(10, 2)") == "number"
    assert normalized_type("VARCHAR(20)") == "text"
    assert normalized_type("TIMESTAMP") == "time"
    assert normalized_type("POINT") == "point"


def test_retrieval_expands_seeds_before_capping_tables():
    schema = schema_for_retrieval()
    selected = retrieve_schema("account total cost", schema)
    assert [table["name"] for table in selected["tables"]] == ["invoice", "account", "contact", "line"]
    assert len(schema["tables"]) == 5
    assert "score" not in schema["tables"][0]


def test_column_budget_prefers_keys_and_drops_dangling_relations():
    selected = retrieve_schema("account total cost", schema_for_retrieval(), max_columns=5)
    kept = {(table["name"], col["name"]) for table in selected["tables"] for col in table["columns"]}
    assert {("invoice", "id"), ("account", "id"), ("line", "id"), ("contact", "id")} <= kept
    assert ("invoice", "totalCost") not in kept
    for table in selected["tables"]:
        for col in table["columns"]:
            if col["references"]:
                assert (col["references"]["table"], col["references"]["column"]) in kept


def test_token_budget_preserves_question_and_complete_entries():
    record = {"id": "x", "dataset": "test", "question": "account total cost", "query": "SELECT id FROM account", "db_path": "/unused.sqlite", "schema": schema_for_retrieval()}
    tokenizer = OffsetTokenizer()
    encoded = encode_record(record, tokenizer, max_input_length=46)
    assert len(encoded["input_ids"]) <= 46
    assert encoded["source_text"].startswith("[QUESTION] account total cost\n[SCHEMA]\n")
    assert "[QUESTION]" not in encoded["schema_text"]
    assert encoded["source_text"].endswith("\n")
    assert encoded["labels"][-1] == tokenizer.eos_token_id
    assert encoded["schema_items"]
    offsets = tokenizer(encoded["source_text"], return_offsets_mapping=True)["offset_mapping"]
    q_start = encoded["source_text"].index(record["question"])
    q_end = q_start + len(record["question"])
    assert all(offsets[index][0] < q_end and offsets[index][1] > q_start for index in encoded["question_positions"])
    assert all(item["positions"] for item in encoded["schema_items"])
    retained_ids = {f"{table['name']}.{col['name']}" for table in encoded["retained_schema"]["tables"] for col in table["columns"]}
    for table in encoded["retained_schema"]["tables"]:
        for col in table["columns"]:
            if col["references"]:
                assert f"{col['references']['table']}.{col['references']['column']}" in retained_ids


def test_budget_errors_do_not_silently_truncate_targets_or_questions():
    tokenizer = OffsetTokenizer()
    with pytest.raises(ValueError, match="SQL target uses"):
        encode_sql("SELECT id FROM account", tokenizer, max_output_length=3)
    record = {"question": " ".join(["long"] * 20), "query": "SELECT 1", "schema": {"tables": []}}
    with pytest.raises(ValueError, match="question and delimiters exceed"):
        encode_record(record, tokenizer, max_input_length=10)


def test_retrieval_and_serialization_do_not_depend_on_reference_sql():
    record = {"id": "x", "dataset": "test", "question": "account total cost", "query": "SELECT id FROM account", "db_path": "/unused.sqlite", "schema": schema_for_retrieval()}
    changed = dict(record, query="SELECT totalCost FROM invoice")
    first = encode_record(record, OffsetTokenizer())
    second = encode_record(changed, OffsetTokenizer())
    assert first["source_text"] == second["source_text"]
    assert first["input_ids"] == second["input_ids"]
    assert first["schema_items"] == second["schema_items"]
    assert first["labels"] != second["labels"]


def test_negatives_filter_before_top_k_and_respect_types_and_distance():
    schema = {"tables": [
        {"name": "a", "columns": [column("id", primary=True), column("text", "text")]},
        {"name": "b", "columns": [column("value", "INTEGER", references={"table": "a", "column": "id"})]},
        {"name": "c", "columns": [column("value", "REAL")]},
        {"name": "d", "columns": [column("value", "text")]},
    ]}
    result = select_schema_negatives("value", schema, ["a.id"], max_negatives=2)
    assert result["a.id"] == ["b.value", "c.value"]
    result = select_schema_negatives("value", schema, ["a.id"], max_negatives=1, available_ids={"a.id", "c.value"})
    assert result["a.id"] == ["c.value"]
    result = select_schema_negatives("value", schema, ["a.id", "b.value"], max_negatives=4)
    assert "b.value" not in result["a.id"]
    assert select_schema_negatives("value", schema, ["a.id"], available_ids={"a.id"}) == {"a.id": []}


def test_random_negatives_are_reproducible():
    schema = schema_for_retrieval()
    first = select_schema_negatives("account", schema, ["account.id"], max_negatives=2, random=True, seed=41)
    assert first == select_schema_negatives("account", schema, ["account.id"], max_negatives=2, random=True, seed=41)
    assert len(first["account.id"]) == 2


def test_load_spider_preserves_sql_names_and_foreign_keys(tmp_path):
    records = [{"db_id": "bank", "question": "Which account?", "query": "SELECT account.id FROM account"}]
    tables = [{"db_id": "bank", "table_names_original": ["account", "payment"], "column_names_original": [[-1, "*"], [0, "id"], [1, "account_id"]], "column_types": ["text", "number", "number"], "primary_keys": [1], "foreign_keys": [[2, 1]]}]
    source = tmp_path / "train.json"
    catalog = tmp_path / "tables.json"
    source.write_text(json.dumps(records))
    catalog.write_text(json.dumps(tables))
    loaded = load_records(source, db_root=tmp_path / "database", tables_path=catalog)
    assert loaded[0]["schema"]["tables"][0]["columns"][0]["primary_key"]
    assert loaded[0]["schema"]["tables"][1]["columns"][0]["references"] == {"table": "account", "column": "id"}
    assert loaded[0]["db_path"] == str(tmp_path / "database" / "bank" / "bank.sqlite")


def test_load_canonical_jsonl_resolves_paths_and_rejects_duplicate_ids(tmp_path):
    record = {"id": "one", "question": "How many?", "query": "SELECT 1", "db_path": "example.sqlite", "schema": {"tables": []}}
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps(record) + "\n")
    loaded = load_records(source)
    assert loaded[0]["db_path"] == str(tmp_path / "example.sqlite")
    source.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="IDs unique"):
        load_records(source)


def test_queryless_canonical_inference_is_explicit_and_training_stays_strict(tmp_path):
    record = {"id": "one", "question": "How many?", "db_path": "example.sqlite", "schema": {"tables": []}}
    source = tmp_path / "predict.jsonl"
    source.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="query must be"):
        load_records(source)
    loaded = load_records(source, require_query=False)[0]
    assert loaded["query"] == ""
    tokenizer = OffsetTokenizer()
    encoded = encode_record(loaded, tokenizer, include_labels=False)
    assert encoded["labels"] == [tokenizer.eos_token_id]
    assert encoded["label_text"] == ""
    with pytest.raises(ValueError, match="SQL target"):
        encode_record(loaded, tokenizer)


def test_one_pass_inference_needs_no_database_or_external_schema_catalog(tmp_path):
    record = {"id": "one", "question": "How many?", "schema": {"tables": []}}
    source = tmp_path / "predict.jsonl"
    source.write_text(json.dumps(record) + "\n")
    loaded = load_records(source, tables_path=tmp_path / "missing-tables.json", require_query=False)[0]
    assert loaded["db_path"] == ""
    assert loaded["query"] == ""
    encoded = encode_record(loaded, OffsetTokenizer(), include_labels=False)
    assert encoded["schema_text"] == ""
    with pytest.raises(ValueError, match="require db_path"):
        load_records(source)


def test_wikisql_materialization_uses_bound_values_and_human_headers(tmp_path):
    table = {"id": "1-1", "header": ["Customer Name", "Balance"], "types": ["text", "real"], "rows": [["O'Brien", 12.0], ["Anne", 4.0]]}
    example = {"table_id": "1-1", "question": "What is the balance of O'Brien?", "sql": {"sel": 1, "agg": 0, "conds": [[0, 0, "O'Brien"]]}}
    source = tmp_path / "train.jsonl"
    tables = tmp_path / "tables.jsonl"
    source.write_text(json.dumps(example) + "\n")
    tables.write_text(json.dumps(table) + "\n")
    record = load_records(source, db_root=tmp_path / "database", tables_path=tables, dataset="wikisql")[0]
    with sqlite3.connect(record["db_path"]) as connection:
        assert connection.execute(record["query"]).fetchall() == [(12.0,)]
    encoded = encode_record(record, OffsetTokenizer())
    assert "Customer Name" in encoded["schema_text"]
    assert "Balance" in encoded["schema_text"]
    assert record["query"].count("o''brien") == 1
    offsets = OffsetTokenizer()(encoded["source_text"], return_offsets_mapping=True)["offset_mapping"]
    for item in encoded["schema_items"]:
        for index in item["positions"]:
            start, end = offsets[index]
            assert encoded["source_text"][start:end] not in {"[", "]", "(", ")", ";"}


def test_wikisql_text_and_formatted_numeric_conditions_follow_execution_normalization(tmp_path):
    table = {"id": "2-1", "header": ["Customer Name", "Balance"], "types": ["text", "real"], "rows": [["ALICE", 3000.0], ["Bob", 17.5]]}
    examples = [
        {"id": "formatted", "table_id": "2-1", "question": "Who has 3,000?", "sql": {"sel": 0, "agg": 0, "conds": [[1, 0, "3,000"]]}},
        {"id": "text", "table_id": "2-1", "question": "What is Alice's balance?", "sql": {"sel": 1, "agg": 0, "conds": [[0, 0, "Alice"]]}},
        {"id": "unit", "table_id": "2-1", "question": "Who has 17.5 units?", "sql": {"sel": 0, "agg": 0, "conds": [[1, 0, "17.5 units"]]}},
    ]
    source = tmp_path / "train.jsonl"
    tables = tmp_path / "tables.jsonl"
    source.write_text("\n".join(json.dumps(example) for example in examples) + "\n")
    tables.write_text(json.dumps(table) + "\n")
    values = load_records(source, db_root=tmp_path / "db", tables_path=tables, dataset="wikisql")
    with sqlite3.connect(values[0]["db_path"]) as connection:
        assert connection.execute(values[0]["query"]).fetchall() == [("alice",)]
        assert connection.execute(values[1]["query"]).fetchall() == [(3000.0,)]
        assert connection.execute(values[2]["query"]).fetchall() == [("bob",)]


def test_wikisql_catalog_directory_supports_official_split_table_files(tmp_path):
    catalog = tmp_path / "official"
    catalog.mkdir()
    train_table = {"id": "train-1", "header": ["Name"], "types": ["text"], "rows": [["ALICE"]]}
    dev_table = {"id": "dev-1", "header": ["Name"], "types": ["text"], "rows": [["BOB"]]}
    (catalog / "train.tables.jsonl").write_text(json.dumps(train_table) + "\n")
    (catalog / "dev.tables.jsonl").write_text(json.dumps(dev_table) + "\n")
    query = {"sel": 0, "agg": 0, "conds": []}
    source = tmp_path / "dev.jsonl"
    source.write_text(json.dumps({"question": "Which name?", "table_id": "dev-1", "sql": query}) + "\n")
    value = load_records(source, db_root=tmp_path / "db", tables_path=catalog, dataset="wikisql")[0]
    with sqlite3.connect(value["db_path"]) as connection:
        assert connection.execute(value["query"]).fetchall() == [("bob",)]


def test_duplicate_catalog_ids_allow_identical_tables_but_reject_conflicts(tmp_path):
    catalog = tmp_path / "official"
    catalog.mkdir()
    table = {"id": "same-1", "header": ["Name"], "types": ["text"], "rows": [["Alice"]]}
    train = catalog / "train.tables.jsonl"
    dev = catalog / "dev.tables.jsonl"
    train.write_text(json.dumps(table) + "\n")
    dev.write_text(json.dumps(table) + "\n")
    source = tmp_path / "data.jsonl"
    source.write_text(json.dumps({"question": "Which name?", "table_id": "same-1", "sql": {"sel": 0, "agg": 0, "conds": []}}) + "\n")
    assert len(load_records(source, db_root=tmp_path / "db", tables_path=catalog, dataset="wikisql")) == 1
    dev.write_text(json.dumps(dict(table, rows=[["Bob"]])) + "\n")
    with pytest.raises(ValueError, match="conflicting table definitions"):
        load_records(source, db_root=tmp_path / "db", tables_path=catalog, dataset="wikisql")


def test_collation_masks_padding_without_changing_metadata():
    items = [{"input_ids": [1, 3, 4], "labels": [5, 2]}, {"input_ids": [1], "labels": [6, 7, 2]}]
    batch = collate_records(items, pad_token_id=0)
    assert batch["input_ids"].tolist() == [[1, 3, 4], [1, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[5, 2, -100], [6, 7, 2]]
    assert batch["metadata"] is items
