from pathlib import Path
import importlib.util
import sys
import time
import numpy as np
import torch
from .config import ModelConfig
from .data import collate_records, encode_record
from .model import ReDiSQLModel
from .runtime import autocast_context, load_checkpoint, synchronize
from .sql import denotation_equal, execute_readonly, parse_query, validate_sql, validate_static


def load_student(path, device="cpu"):
    payload = load_checkpoint(path)
    values = payload["config"]
    values = values["model"] if "model" in values else values
    config = ModelConfig(**values)
    student = ReDiSQLModel(config, auxiliary=False)
    state = {key: value for key, value in payload["model"].items() if not key.startswith(("question_projection.", "schema_projection.", "node_classifier.", "edge_classifier."))}
    student.load_state_dict(state, strict=True)
    return student.to(device).eval()


@torch.no_grad()
def predict_encoded(model, tokenizer, records, config, mode="student", teacher=None):
    if not records:
        return []
    if mode not in {"student", "execution_rerank", "one_shot"}:
        raise ValueError("unknown prediction mode")
    if mode != "student" and any(not record.get("db_path") for record in records):
        raise ValueError("database paths are required for execution-based inference")
    device = next(model.parameters()).device
    batch = collate_records(records, model.config.pad_token_id)
    inputs = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    with autocast_context(device, config.training.dtype):
        if mode == "execution_rerank":
            candidates = model.generate_candidates(inputs, mask, num_beams=5, max_new_tokens=model.config.max_output_length)
            sqls = []
            for record, options in zip(records, candidates):
                decoded = [tokenizer.decode(option["input_ids"][1:], skip_special_tokens=True).strip() for option in options]
                chosen = decoded[0]
                for sql in decoded:
                    if validate_sql(sql, record["schema"], record["db_path"], timeout=config.evaluation.timeout, require_denotation=False)["ok"]:
                        chosen = sql
                        break
                sqls.append(chosen)
        else:
            values = model.generate(inputs, mask, max_new_tokens=model.config.max_output_length, num_beams=config.evaluation.num_beams)
            sqls = [tokenizer.decode(row[1:].tolist(), skip_special_tokens=True).strip() for row in values]
    predictions = []
    for record, sql in zip(records, sqls):
        called = False
        if mode == "one_shot":
            if teacher is None:
                raise ValueError("one_shot inference requires a teacher")
            failure = validate_sql(sql, record["schema"], record["db_path"], timeout=config.evaluation.timeout, require_denotation=False)
            if not failure["ok"]:
                called = True
                proposal = teacher.repair(record["question"], record["schema_text"], sql, failure)
                if validate_sql(proposal, record["schema"], record["db_path"], timeout=config.evaluation.timeout, require_denotation=False)["ok"]:
                    sql = proposal
        predictions.append({"id": str(record["id"]), "dataset": record["dataset"], "sql": sql, "teacher_called": called, "execution_used": mode != "student"})
    return predictions


def predict_records(model, tokenizer, records, config, mode="student", teacher=None):
    result = []
    size = config.evaluation.batch_size
    if size < 1:
        raise ValueError("evaluation batch_size must be positive")
    for start in range(0, len(records), size):
        encoded = [encode_record(record, tokenizer, model.config.max_input_length, model.config.max_output_length, include_labels=False) for record in records[start:start + size]]
        result.extend(predict_encoded(model, tokenizer, encoded, config, mode, teacher))
    return result


class ExecutionEvaluator:
    def __init__(self, records, config):
        self.records = records
        self.config = config
        self.references = {}
        self.official = None
        keys = [(str(record["dataset"]), str(record["id"])) for record in records]
        if len(set(keys)) != len(keys):
            raise ValueError("evaluation record IDs must be unique")
        if any(not isinstance(record.get("query"), str) or not record["query"].strip() for record in records):
            raise ValueError("evaluation requires reference SQL")
        if any(not record.get("db_path") for record in records):
            raise ValueError("evaluation requires database paths")
        if config.evaluation.official_evaluator:
            path = Path(config.evaluation.official_evaluator).resolve(strict=True)
            sys.path.insert(0, str(path.parent))
            specification = importlib.util.spec_from_file_location("redisql_external_evaluator", path)
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            if not hasattr(module, "eval_exec_match"):
                raise ValueError("official evaluator must expose eval_exec_match")
            if hasattr(module, "TIMEOUT"):
                module.TIMEOUT = config.evaluation.timeout
            self.official = module.eval_exec_match
        for record in records:
            key = (record["db_path"], record["query"])
            if key not in self.references:
                rows = execute_readonly(record["db_path"], record["query"], config.evaluation.timeout)
                if not rows["ok"]:
                    raise ValueError(f"reference query failed for {record['id']}")
                tree = parse_query(record["query"])
                self.references[key] = (rows["rows"], tree.args.get("order") is not None)

    def evaluate(self, predictions):
        indexed = {(str(row["dataset"]), str(row["id"])): row for row in predictions}
        if len(indexed) != len(predictions) or len(predictions) != len(self.records):
            raise ValueError("prediction IDs must be unique and cover the evaluation set")
        outcomes = []
        errors = {}
        for record in self.records:
            key = (str(record["dataset"]), str(record["id"]))
            if key not in indexed:
                raise ValueError(f"missing prediction for {key}")
            prediction = indexed[key]
            sql = prediction["sql"]
            static = validate_static(sql, record["schema"])
            category = static["category"]
            correct = False
            if static["ok"]:
                result = execute_readonly(record["db_path"], sql, self.config.evaluation.timeout)
                if result["ok"]:
                    if self.official is not None:
                        database = record["db_path"]
                        if self.config.evaluation.test_suite_database:
                            identifier = record.get("db_id", Path(database).stem)
                            candidate = Path(self.config.evaluation.test_suite_database) / identifier / Path(database).name
                            database = str(candidate.resolve(strict=True))
                        correct = bool(self.official(database, sql, record["query"], plug_value=False, keep_distinct=True, progress_bar_for_each_datapoint=False))
                    else:
                        reference, ordered = self.references[(record["db_path"], record["query"])]
                        correct = denotation_equal(reference, result["rows"], ordered)
                    category = "ok" if correct else "denotation"
                else:
                    category = "runtime"
            errors[category] = errors.get(category, 0) + 1
            outcomes.append({**prediction, "correct": correct, "category": category})
        count = len(outcomes)
        correct = sum(row["correct"] for row in outcomes)
        return {
            "protocol": "official_exec_match" if self.official else "sqlite_denotation",
            "count": count,
            "correct": correct,
            "execution_accuracy": 100.0 * correct / count if count else 0.0,
            "categories": errors,
            "outcomes": outcomes,
        }


def profile_model(model, tokenizer, records, config, mode="student", teacher=None):
    if not records:
        raise ValueError("profiling requires requests")
    if config.evaluation.timing_rounds < 1 or config.evaluation.batch_size < 1:
        raise ValueError("profiling rounds and batch size must be positive")
    device = next(model.parameters()).device
    batches = [records[index:index + config.evaluation.batch_size] for index in range(0, len(records), config.evaluation.batch_size)]
    for index in range(config.evaluation.warmup_iterations):
        predict_records(model, tokenizer, batches[index % len(batches)], config, mode, teacher)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    latencies = []
    total = 0.0
    teacher_calls = 0
    for _ in range(config.evaluation.timing_rounds):
        for batch in batches:
            synchronize(device)
            started = time.perf_counter()
            predictions = predict_records(model, tokenizer, batch, config, mode, teacher)
            synchronize(device)
            elapsed = time.perf_counter() - started
            total += elapsed
            latencies.extend([elapsed * 1000] * len(batch))
            teacher_calls += sum(row["teacher_called"] for row in predictions)
    return {
        "requests": len(latencies), "requests_per_round": len(records),
        "rounds": config.evaluation.timing_rounds, "warmup_iterations": config.evaluation.warmup_iterations,
        "batch_size": config.evaluation.batch_size, "mode": mode, "device": str(device),
        "mean_ms": float(np.mean(latencies)), "p95_ms": float(np.percentile(latencies, 95)),
        "throughput_queries_per_second": len(latencies) / total,
        "teacher_call_rate": teacher_calls / len(latencies),
        "cpu_threads": torch.get_num_threads(),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }

