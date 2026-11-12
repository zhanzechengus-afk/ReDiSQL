from collections import Counter
from pathlib import Path
import json
import torch
from .data import collate_records, select_schema_negatives
from .sql import validate_sql, validate_static
from .storage import RepairCache, fingerprint, read_jsonl, sqlite_fingerprint, state_fingerprint, write_json, write_jsonl
from .targets import build_targets
from .teacher import repair_prompt, tokenizer_identity


def record_key(record):
    return str(record["dataset"]) + ":" + str(record["id"])


def dataset_identity(records):
    databases = {str(Path(record["db_path"]).resolve()) for record in records}
    return fingerprint({"records": [
        {key: record[key] for key in ("id", "dataset", "question", "query", "db_path", "schema")}
        for record in records
    ], "databases": {path: sqlite_fingerprint(path) for path in sorted(databases)}})


def student_identity(model):
    return state_fingerprint(model)


def add_schema_negatives(targets, record, config):
    available = {item["id"] for item in record["schema_items"]}
    negatives = select_schema_negatives(
        record["question"], record["schema"], targets["schema_positives"],
        max_negatives=config.training.max_schema_negatives,
        random=config.training.schema_negatives == "random",
        seed=config.training.seed,
        available_ids=available,
    )
    targets["schema_negatives"] = negatives
    return targets


def collect_corpus(model, tokenizer, teacher, records, config, output_dir=None):
    output = Path(output_dir or config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for record in records:
        reference = validate_sql(record["query"], record["schema"], record["db_path"], timeout=config.evaluation.timeout, require_denotation=False)
        if not reference["ok"]:
            raise ValueError(f"invalid reference SQL for {record_key(record)}")
    identities = {
        "dataset": dataset_identity(records),
        "student": student_identity(model),
        "teacher": teacher.identity,
        "tokenizer": tokenizer_identity(tokenizer),
    }
    device = next(model.parameters()).device
    statistics = Counter(records=len(records))
    collection = []
    accepted = []
    prior_mode = model.training
    model.eval()
    try:
        with RepairCache(output / "repair_cache.sqlite") as cache:
            for record in records:
                batch = collate_records([record], config.model.pad_token_id)
                with torch.no_grad():
                    generated = model.generate(
                        batch["input_ids"].to(device), batch["attention_mask"].to(device),
                        max_new_tokens=config.model.max_output_length, num_beams=1,
                    )
                draft = tokenizer.decode(generated[0, 1:].tolist(), skip_special_tokens=True).strip()
                if config.training.repair_trigger == "static":
                    feedback = validate_static(draft, record["schema"])
                else:
                    feedback = validate_sql(draft, record["schema"], record["db_path"], record["query"], timeout=config.evaluation.timeout)
                row = {"key": record_key(record), "dataset": record["dataset"], "id": record["id"], "draft": draft, "failure": feedback["category"], "requested": False, "accepted": False}
                if feedback["ok"]:
                    statistics["draft_passes"] += 1
                    collection.append(row)
                    continue
                row["requested"] = True
                statistics["repair_requests"] += 1
                statistics["failure_" + feedback["category"]] += 1
                prompt = repair_prompt(record["question"], record["schema_text"], draft, feedback)
                context = {
                    **identities,
                    "example": record_key(record),
                    "schema": record["schema"],
                    "prompt": prompt,
                    "decoding": {"greedy": True, "num_beams": 1, "max_new_tokens": config.teacher.max_new_tokens},
                    "draft": draft,
                    "feedback": {"category": feedback["category"], "message": feedback["message"]},
                }
                proposal = cache.get(context)
                if proposal is None:
                    proposal = teacher.repair(record["question"], record["schema_text"], draft, feedback)
                    cache.put(context, proposal)
                    statistics["teacher_calls"] += 1
                else:
                    statistics["resume_cache_hits"] += 1
                row["repair"] = proposal
                row["cache_key"] = fingerprint(context)
                if config.training.execution_filter:
                    validation = validate_sql(proposal, record["schema"], record["db_path"], record["query"], timeout=config.evaluation.timeout)
                else:
                    validation = validate_static(proposal, record["schema"])
                row["acceptance_category"] = validation["category"]
                if not validation["ok"]:
                    statistics["rejected_" + validation["category"]] += 1
                    collection.append(row)
                    continue
                statistics["gate_accepts"] += 1
                target_sql = record["query"] if config.training.repair_target == "gold" else proposal
                try:
                    targets = build_targets(draft, target_sql, tokenizer, schema=record["schema"], max_length=config.model.max_output_length)
                    targets = add_schema_negatives(targets, record, config)
                except (ValueError, KeyError) as error:
                    statistics["target_rejections"] += 1
                    row["target_error"] = str(error)
                    collection.append(row)
                    continue
                row["accepted"] = True
                statistics["accepted_pairs"] += 1
                item = {"key": row["key"], "dataset": record["dataset"], "id": record["id"], "draft": draft, "teacher_repair": proposal, "repair": targets["repair_sql"], "targets": targets}
                accepted.append(item)
                collection.append(row)
    finally:
        model.train(prior_mode)
    requests = statistics["repair_requests"]
    statistics = dict(statistics)
    statistics["request_rate"] = requests / len(records) if records else 0.0
    statistics["gate_acceptance_rate"] = statistics.get("gate_accepts", 0) / requests if requests else 0.0
    statistics["usable_acceptance_rate"] = statistics.get("accepted_pairs", 0) / requests if requests else 0.0
    manifest = {"identities": identities, "training": {key: getattr(config.training, key) for key in ("repair_target", "execution_filter", "repair_trigger", "schema_negatives", "max_schema_negatives", "seed")}, "statistics": statistics}
    write_jsonl(output / "collection.jsonl", collection)
    write_jsonl(output / "repair_corpus.jsonl", accepted)
    write_json(output / "corpus_manifest.json", manifest)
    return {item["key"]: item for item in accepted}, manifest


def load_corpus(path, records=None, tokenizer=None, model=None, config=None):
    path = Path(path)
    if path.is_dir():
        path = path / "repair_corpus.jsonl"
    manifest_path = path.parent / "corpus_manifest.json"
    if not manifest_path.exists():
        raise ValueError("repair corpus manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["identities"]
    if records is not None and dataset_identity(records) != expected["dataset"]:
        raise ValueError("repair corpus belongs to a different dataset")
    if tokenizer is not None and tokenizer_identity(tokenizer) != expected["tokenizer"]:
        raise ValueError("repair corpus tokenizer differs")
    if model is not None and student_identity(model) != expected["student"]:
        raise ValueError("repair corpus does not match the frozen student")
    if config is not None:
        for key, value in manifest["training"].items():
            if key in {"execution_filter", "repair_trigger"} and value != getattr(config.training, key):
                raise ValueError(f"repair corpus differs in {key}")
    rows = read_jsonl(path)
    result = {row["key"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate repair corpus keys")
    if records is not None:
        indexed = {record_key(record): record for record in records}
        if not set(result) <= set(indexed):
            raise ValueError("repair corpus contains unknown examples")
        if config is not None:
            previous_target = manifest["training"]["repair_target"]
            if previous_target != config.training.repair_target:
                if tokenizer is None:
                    raise ValueError("changing repair targets requires the tokenizer")
                for key, row in result.items():
                    record = indexed[key]
                    sql = record["query"] if config.training.repair_target == "gold" else row["teacher_repair"]
                    row["targets"] = build_targets(row["draft"], sql, tokenizer, schema=record["schema"], max_length=config.model.max_output_length)
                    row["repair"] = row["targets"]["repair_sql"]
            for key, row in result.items():
                row["targets"] = add_schema_negatives(row["targets"], indexed[key], config)
    elif config is not None and (manifest["training"]["repair_target"] != config.training.repair_target or manifest["training"]["schema_negatives"] != config.training.schema_negatives):
        raise ValueError("changing repair_target or schema_negatives requires encoded records")
    return result, manifest

