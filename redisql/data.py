import copy
import hashlib
import json
import math
import os
import random as pyrandom
import re
import sqlite3
import tempfile
from collections import deque
from decimal import Decimal, InvalidOperation
from pathlib import Path


def split_name(value):
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value))
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
    return set(re.findall(r"[^\W_]+", value.lower(), flags=re.UNICODE))


def normalized_type(value):
    value = str(value or "others").lower()
    if re.search(r"\b(?:bool|boolean)\b", value):
        return "boolean"
    if re.search(r"\b(?:date|datetime|timestamp|time|year)\b", value):
        return "time"
    if re.search(r"\b(?:tinyint|smallint|mediumint|bigint|int|integer|float|double|real|decimal|numeric|number)\b", value):
        return "number"
    if re.search(r"\b(?:text|char|varchar|nchar|nvarchar|character|string|clob)\b", value):
        return "text"
    return value


def normalize_schema(schema):
    if not isinstance(schema, dict) or not isinstance(schema.get("tables"), list):
        raise ValueError("schema must contain a tables list")
    result = copy.deepcopy(schema)
    table_names = set()
    for table_index, table in enumerate(result["tables"]):
        name = str(table.get("name", ""))
        if not name or name.casefold() in table_names:
            raise ValueError("schema table names must be nonempty and unique")
        table_names.add(name.casefold())
        table["name"] = name
        table["index"] = int(table.get("index", table_index))
        columns = table.get("columns", [])
        if not isinstance(columns, list):
            raise ValueError("table columns must be a list")
        column_names = set()
        for column_index, column in enumerate(columns):
            column_name = str(column.get("name", ""))
            if not column_name or column_name.casefold() in column_names:
                raise ValueError("schema column names must be nonempty and unique within each table")
            column_names.add(column_name.casefold())
            column["name"] = column_name
            column["index"] = int(column.get("index", column_index))
            column["type"] = str(column.get("type", "others"))
            column["primary_key"] = bool(column.get("primary_key", False))
            references = column.get("references")
            if references is not None:
                if not isinstance(references, dict) or not references.get("table") or not references.get("column"):
                    raise ValueError("column references must contain table and column")
                column["references"] = {"table": str(references["table"]), "column": str(references["column"])}
            else:
                column["references"] = None
    return result


def _read_json_records(path, wanted_ids=None):
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        first = ""
        while not first:
            char = stream.read(1)
            if not char:
                return []
            if not char.isspace():
                first = char
        stream.seek(0)
        if first == "[":
            result = json.load(stream)
            if not isinstance(result, list):
                raise ValueError("dataset JSON must be an array")
            if wanted_ids is not None:
                result = [value for value in result if str(value.get("db_id", value.get("id", ""))) in wanted_ids]
            return result
        if path.suffix.lower() == ".json":
            try:
                value = json.load(stream)
            except json.JSONDecodeError:
                stream.seek(0)
            else:
                if isinstance(value, dict) and isinstance(value.get("data"), list):
                    result = value["data"]
                    if wanted_ids is not None:
                        result = [entry for entry in result if str(entry.get("db_id", entry.get("id", ""))) in wanted_ids]
                    return result
                if isinstance(value, dict):
                    return [value] if wanted_ids is None or str(value.get("db_id", value.get("id", ""))) in wanted_ids else []
                raise ValueError("dataset JSON must contain objects")
        result = []
        for line_index, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL line {line_index} is not an object")
                if wanted_ids is None or str(value.get("db_id", value.get("id", ""))) in wanted_ids:
                    result.append(value)
        return result


def _flatten_indices(values):
    result = set()
    for value in values:
        if isinstance(value, list):
            result.update(int(item) for item in value)
        else:
            result.add(int(value))
    return result


def _spider_schema(value):
    names = value.get("table_names_original", value.get("table_names"))
    columns = value.get("column_names_original", value.get("column_names"))
    if not names or columns is None:
        raise ValueError("Spider tables file is missing table or column names")
    types = value.get("column_types", ["others"] * len(columns))
    primary_keys = _flatten_indices(value.get("primary_keys", []))
    foreign_keys = dict(value.get("foreign_keys", []))
    tables = [{"name": str(name), "index": index, "columns": []} for index, name in enumerate(names)]
    for column_index, (table_index, column_name) in enumerate(columns):
        if int(table_index) < 0:
            continue
        item = {
            "name": str(column_name),
            "type": str(types[column_index]),
            "primary_key": column_index in primary_keys,
            "index": column_index,
            "references": None,
        }
        if column_index in foreign_keys:
            target_table, target_column = columns[int(foreign_keys[column_index])]
            if int(target_table) >= 0:
                item["references"] = {"table": str(names[int(target_table)]), "column": str(target_column)}
        tables[int(table_index)]["columns"].append(item)
    return normalize_schema({"tables": tables})


def _quote_identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def _sql_literal(value):
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("SQL values must be finite")
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _wikisql_condition(value, kind):
    if isinstance(value, str):
        value = value.lower()
    if normalized_type(kind) == "number" and not isinstance(value, (int, float)):
        try:
            value = float(Decimal(str(value).replace(",", "")))
        except (InvalidOperation, ValueError):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", str(value))
            if not matches:
                raise ValueError("WikiSQL numeric condition has no numeric value")
            value = float(matches[0])
    return value


def _wikisql_query(sql, table_name, column_count, column_types=None):
    selected = int(sql["sel"])
    aggregation = int(sql.get("agg", 0))
    aggregations = ("", "MAX", "MIN", "COUNT", "SUM", "AVG")
    operators = ("=", ">", "<")
    if not 0 <= selected < column_count or not 0 <= aggregation < len(aggregations):
        raise ValueError("invalid WikiSQL selection or aggregation")
    column = _quote_identifier(f"col{selected}")
    expression = column if aggregation == 0 else f"{aggregations[aggregation]}({column})"
    conditions = []
    for index, operator, value in sql.get("conds", []):
        if not 0 <= int(index) < column_count or not 0 <= int(operator) < len(operators):
            raise ValueError("unsupported WikiSQL condition")
        kind = column_types[int(index)] if column_types is not None else "text"
        value = _wikisql_condition(value, kind)
        conditions.append(f"{_quote_identifier(f'col{int(index)}')} {operators[int(operator)]} {_sql_literal(value)}")
    query = f"SELECT {expression} FROM {_quote_identifier(table_name)}"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    return query


def _materialize_wikisql(table, db_root):
    table_id = str(table["id"])
    safe_id = re.sub(r"[^A-Za-z0-9_]", "_", table_id)
    table_name = "table_" + safe_id
    if not table.get("header") or len(table.get("types", table["header"])) != len(table["header"]):
        raise ValueError("WikiSQL tables require nonempty headers and one type per column")
    destination = Path(db_root).expanduser().resolve() / "wikisql"
    destination.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps({"format": 2, "table": table}, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
    db_path = destination / f"{safe_id}_{digest[:12]}.sqlite"
    if not db_path.exists():
        descriptor, temporary = tempfile.mkstemp(prefix=".wikisql_", suffix=".sqlite", dir=destination)
        os.close(descriptor)
        try:
            with sqlite3.connect(temporary) as connection:
                definitions = []
                for index, kind in enumerate(table.get("types", ["text"] * len(table["header"]))):
                    sqlite_type = "REAL" if normalized_type(kind) == "number" else "TEXT"
                    definitions.append(f"{_quote_identifier(f'col{index}')} {sqlite_type}")
                connection.execute(f"CREATE TABLE {_quote_identifier(table_name)} ({', '.join(definitions)})")
                rows = table.get("rows", [])
                if any(len(row) != len(definitions) for row in rows):
                    raise ValueError("WikiSQL row width does not match table header")
                placeholders = ",".join("?" for _ in definitions)
                values = [[value.lower() if isinstance(value, str) else value for value in row] for row in rows]
                connection.executemany(f"INSERT INTO {_quote_identifier(table_name)} VALUES ({placeholders})", values)
            os.replace(temporary, db_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    columns = [
        {
            "name": f"col{index}",
            "display_name": str(header),
            "type": str(table.get("types", ["text"] * len(table["header"]))[index]),
            "primary_key": False,
            "references": None,
            "index": index,
        }
        for index, header in enumerate(table["header"])
    ]
    return table_name, str(db_path), normalize_schema({"tables": [{"name": table_name, "columns": columns}]})


def _resolve_database(db_root, db_id, base):
    root = Path(db_root).expanduser() if db_root else base
    paths = [root / str(db_id) / f"{db_id}.sqlite", root / f"{db_id}.sqlite", root / f"{db_id}.db"]
    for path in paths:
        if path.is_file():
            return str(path.resolve())
    return str(paths[0].resolve())


def load_records(path, db_root=None, tables_path=None, dataset="spider", require_query=True):
    path = Path(path).expanduser().resolve()
    raw_records = _read_json_records(path)
    catalog = {}
    if tables_path is not None:
        wanted_ids = {str(item.get("db_id", item.get("table_id", ""))) for item in raw_records if isinstance(item, dict) and "schema" not in item}
        if wanted_ids:
            location = Path(tables_path).expanduser()
            sources = [location]
            if location.is_dir():
                sources = sorted(location.glob("*.tables.jsonl"))
                if not sources:
                    sources = [value for value in (location / "tables.jsonl", location / "tables.json") if value.is_file()]
                if not sources:
                    raise ValueError(f"no table catalog files found in {location}")
            for source in sources:
                for table in _read_json_records(source, wanted_ids=wanted_ids):
                    key = str(table.get("db_id", table.get("id", "")))
                    if not key:
                        raise ValueError("table catalog IDs must be nonempty")
                    if key in catalog:
                        if catalog[key] != table:
                            raise ValueError(f"conflicting table definitions for {key}")
                        continue
                    catalog[key] = table
    result = []
    seen = set()
    materialized = {}
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise ValueError("each dataset record must be an object")
        item = copy.deepcopy(raw)
        item["id"] = str(raw.get("id", raw.get("question_id", f"{dataset}:{index}")))
        item["dataset"] = str(raw.get("dataset", dataset))
        item["question"] = str(raw.get("question", ""))
        if not item["question"].strip() or item["id"] in seen:
            raise ValueError("record questions must be nonempty and IDs unique")
        seen.add(item["id"])
        if isinstance(raw.get("schema"), dict):
            item["schema"] = normalize_schema(raw["schema"])
            item["query"] = raw.get("query", raw.get("SQL", raw.get("sql")))
            if not raw.get("db_path") and require_query:
                raise ValueError("canonical records require db_path")
            if raw.get("db_path"):
                database = Path(raw["db_path"]).expanduser()
                if not database.is_absolute():
                    database = (Path(db_root).expanduser() if db_root else path.parent) / database
                item["db_path"] = str(database.resolve())
            else:
                item["db_path"] = ""
        elif "table_id" in raw and isinstance(raw.get("sql"), dict):
            table_id = str(raw["table_id"])
            table = raw.get("table", catalog.get(table_id))
            if table is None or db_root is None:
                raise ValueError("WikiSQL requires a table catalog and explicit db_root")
            table = dict(table, id=table_id)
            if table_id not in materialized:
                materialized[table_id] = _materialize_wikisql(table, db_root)
            table_name, item["db_path"], schema = materialized[table_id]
            item["schema"] = copy.deepcopy(schema)
            item["query"] = _wikisql_query(raw["sql"], table_name, len(table["header"]), table.get("types"))
        else:
            db_id = str(raw.get("db_id", ""))
            if db_id not in catalog:
                raise ValueError(f"no schema found for database {db_id!r}")
            item["schema"] = _spider_schema(catalog[db_id])
            item["query"] = raw.get("query", raw.get("SQL"))
            item["db_path"] = _resolve_database(db_root, db_id, path.parent)
        if item["query"] is None and not require_query:
            item["query"] = ""
        if not isinstance(item["query"], str) or (require_query and not item["query"].strip()):
            raise ValueError("record query must be a nonempty SQL string")
        result.append(item)
    return result


def _graph(schema):
    table_names = {table["name"].casefold(): table["name"] for table in schema["tables"]}
    graph = {table["name"]: set() for table in schema["tables"]}
    endpoints = set()
    for table in schema["tables"]:
        for column in table["columns"]:
            reference = column.get("references")
            if reference and reference["table"].casefold() in table_names:
                target = table_names[reference["table"].casefold()]
                graph[table["name"]].add(target)
                graph[target].add(table["name"])
                endpoints.add((table["name"], column["name"]))
                endpoints.add((target, reference["column"]))
    return graph, endpoints


def _column_overlap(question_tokens, table_name, column):
    name = column.get("display_name", column["name"])
    return len(question_tokens & split_name(f"{table_name}.{name}"))


def retrieve_schema(question, schema, max_tables=4, max_columns=32, seed_tables=2):
    if max_tables <= 0 or max_columns < 0 or seed_tables <= 0:
        raise ValueError("schema budgets must be positive, with nonnegative column budget")
    schema = normalize_schema(schema)
    question_tokens = split_name(question)
    graph, _ = _graph(schema)
    scored = []
    for table in schema["tables"]:
        table["score"] = max((_column_overlap(question_tokens, table["name"], column) for column in table["columns"]), default=0)
        scored.append(table)
    scored.sort(key=lambda table: (-table["score"], table["index"]))
    seeds = scored[:min(seed_tables, max_tables)]
    seed_names = {table["name"] for table in seeds}
    neighbors = set().union(*(graph[name] for name in seed_names)) if seed_names else set()
    selected = seeds + [table for table in scored if table["name"] in neighbors and table["name"] not in seed_names]
    selected = selected[:max_tables]
    _, selected_endpoints = _graph({"tables": selected})
    eligible = []
    for table in selected:
        for column in table["columns"]:
            tier = 0 if column["primary_key"] else 1 if (table["name"], column["name"]) in selected_endpoints else 2
            column["priority"] = tier
            column["score"] = _column_overlap(question_tokens, table["name"], column)
            eligible.append((tier, -column["score"], table["index"], column["index"], table["name"], column["name"]))
    eligible.sort()
    kept = {(item[-2], item[-1]) for item in eligible[:max_columns]}
    for table in selected:
        table["columns"] = [column for column in table["columns"] if (table["name"], column["name"]) in kept]
        for column in table["columns"]:
            reference = column.get("references")
            if reference and (reference["table"], reference["column"]) not in kept:
                column["references"] = None
    return {"tables": selected}


def _input_encoding(tokenizer, text):
    try:
        value = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True, truncation=False)
    except (TypeError, NotImplementedError) as error:
        raise ValueError("a tokenizer with character offset mappings is required") from error
    ids = value["input_ids"]
    offsets = value["offset_mapping"]
    if ids and isinstance(ids[0], list):
        raise ValueError("tokenizer returned a batch for a single input")
    if len(ids) != len(offsets):
        raise ValueError("tokenizer offset mappings do not match input IDs")
    return [int(item) for item in ids], [(int(start), int(end)) for start, end in offsets]


def encode_sql(sql, tokenizer, max_output_length=256):
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL target must be a nonempty string")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(sql, add_special_tokens=False)
    else:
        ids = tokenizer(sql, add_special_tokens=False)["input_ids"]
    ids = [int(item) for item in ids] + [int(tokenizer.eos_token_id)]
    if len(ids) > max_output_length:
        raise ValueError(f"SQL target uses {len(ids)} tokens, exceeding {max_output_length}")
    return ids


def _positions(offsets, start, end):
    return [index for index, (left, right) in enumerate(offsets) if right > left and left < end and right > start]


def _retained_schema(schema, table_names, column_ids):
    result = copy.deepcopy(schema)
    result["tables"] = [table for table in result["tables"] if table["name"] in table_names]
    for table in result["tables"]:
        table["columns"] = [column for column in table["columns"] if f"{table['name']}.{column['name']}" in column_ids]
        for column in table["columns"]:
            reference = column.get("references")
            if reference and f"{reference['table']}.{reference['column']}" not in column_ids:
                column["references"] = None
    return result


def _serialize(question, schema):
    prefix = "[QUESTION] "
    text = prefix + question + "\n[SCHEMA]\n"
    question_span = (len(prefix), len(prefix) + len(question))
    spans = []
    columns = []
    for table in schema["tables"]:
        start = len(text)
        text += table["name"] + "\n"
        spans.append({"id": table["name"], "kind": "table", "table": table["name"], "name": table["name"], "type": "table", "start": start, "end": start + len(table["name"]), "table_index": table["index"]})
        for column in table["columns"]:
            columns.append((column.get("priority", 2), -column.get("score", 0), table["index"], column["index"], table, column))
    columns.sort(key=lambda item: item[:4])
    for _, _, _, _, table, column in columns:
        identifier = f"{table['name']}.{column['name']}"
        name = identifier
        if column.get("display_name") and column["display_name"] != column["name"]:
            name += f" [{column['display_name']}]"
        start = len(text)
        details = [column["type"]]
        if column["primary_key"]:
            details.append("primary key")
        if column.get("references"):
            target = column["references"]
            details.append(f"references {target['table']}.{target['column']}")
        text += name + " (" + "; ".join(details) + ")\n"
        segments = [(start, start + len(identifier))]
        if name != identifier:
            segments.append((start + len(identifier) + 2, start + len(name) - 1))
        spans.append({"id": identifier, "kind": "column", "table": table["name"], "name": column["name"], "type": column["type"], "start": start, "end": start + len(name), "segments": segments, "table_index": table["index"], "column_index": column["index"], "primary_key": column["primary_key"], "references": column.get("references")})
    return text, question_span, spans, [f"{item[4]['name']}.{item[5]['name']}" for item in columns]


def encode_record(record, tokenizer, max_input_length=512, max_output_length=256, include_labels=True):
    question = str(record["question"])
    empty_text, _, _, _ = _serialize(question, {"tables": []})
    if len(_input_encoding(tokenizer, empty_text)[0]) > max_input_length:
        raise ValueError(f"question and delimiters exceed {max_input_length} tokens")
    schema = retrieve_schema(question, record["schema"])
    _, _, _, column_order = _serialize(question, schema)
    table_order = [table["name"] for table in schema["tables"]]
    while True:
        retained = _retained_schema(schema, set(table_order), set(column_order))
        source, question_span, spans, _ = _serialize(question, retained)
        ids, offsets = _input_encoding(tokenizer, source)
        if len(ids) <= max_input_length:
            break
        if column_order:
            column_order.pop()
        elif table_order:
            table_order.pop()
        else:
            raise ValueError("input cannot fit the configured token budget")
    result = copy.deepcopy(record)
    result["source_text"] = source
    result["schema_text"] = source.partition("\n[SCHEMA]\n")[2]
    result["label_text"] = record.get("query", "") if include_labels else ""
    result["input_ids"] = ids
    result["attention_mask"] = [1] * len(ids)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    result["labels"] = encode_sql(record.get("query", ""), tokenizer, max_output_length) if include_labels else [int(tokenizer.eos_token_id)]
    result["question_positions"] = _positions(offsets, *question_span)
    result["schema_items"] = []
    for item in spans:
        span = (item.pop("start"), item.pop("end"))
        segments = item.pop("segments", [span])
        positions = sorted({index for start, end in segments for index in _positions(offsets, start, end)})
        if positions:
            item["positions"] = positions
            result["schema_items"].append(item)
    result["retained_schema"] = retained
    return result


def collate_records(items, pad_token_id):
    import torch

    if not items:
        raise ValueError("cannot collate an empty batch")
    input_length = max(len(item["input_ids"]) for item in items)
    output_length = max(len(item["labels"]) for item in items)
    input_ids = torch.full((len(items), input_length), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(items), input_length), dtype=torch.long)
    labels = torch.full((len(items), output_length), -100, dtype=torch.long)
    for index, item in enumerate(items):
        input_ids[index, :len(item["input_ids"])] = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask[index, :len(item["input_ids"])] = torch.tensor(item.get("attention_mask", [1] * len(item["input_ids"])), dtype=torch.long)
        labels[index, :len(item["labels"])] = torch.tensor(item["labels"], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "metadata": items}


def _distances(graph, source):
    result = {source: 0}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(graph[current]):
            if neighbor not in result:
                result[neighbor] = result[current] + 1
                queue.append(neighbor)
    return result


def select_schema_negatives(question, schema, positives, max_negatives=4, random=False, seed=0, available_ids=None):
    if max_negatives < 0:
        raise ValueError("max_negatives must be nonnegative")
    schema = normalize_schema(schema)
    items = {}
    for table in schema["tables"]:
        items[table["name"]] = {"kind": "table", "table": table["name"], "name": table["name"], "type": "table", "table_index": table["index"], "column_index": -1}
        for column in table["columns"]:
            items[f"{table['name']}.{column['name']}"] = {"kind": "column", "table": table["name"], "name": column["name"], "display_name": column.get("display_name", column["name"]), "type": normalized_type(column["type"]), "table_index": table["index"], "column_index": column["index"]}
    positive_ids = set(str(item) for item in positives)
    unknown = positive_ids - items.keys()
    if unknown:
        raise ValueError("schema positives are not present in the supplied schema: " + ", ".join(sorted(unknown)))
    available = set(items) if available_ids is None else set(available_ids) & items.keys()
    all_positive_ids = set(positive_ids)
    positive_ids &= available
    graph, _ = _graph(schema)
    question_tokens = split_name(question)
    generator = pyrandom.Random(seed)
    result = {}
    ordered = sorted(positive_ids, key=lambda key: (items[key]["table_index"], items[key]["column_index"]))
    for positive in ordered:
        item = items[positive]
        distance = _distances(graph, item["table"])
        candidates = [key for key, candidate in items.items() if key in available and key not in all_positive_ids and candidate["kind"] == item["kind"] and (item["kind"] == "table" or candidate["type"] == item["type"])]
        def ranking(key):
            candidate = items[key]
            text = candidate["table"] if candidate["kind"] == "table" else f"{candidate['table']}.{candidate['display_name']}"
            overlap = len(question_tokens & split_name(text))
            return (-overlap, distance.get(candidate["table"], math.inf), candidate["table_index"], candidate["column_index"])
        candidates.sort(key=ranking)
        if random:
            candidates = generator.sample(candidates, min(max_negatives, len(candidates)))
        else:
            candidates = candidates[:max_negatives]
        result[positive] = candidates
    return result
