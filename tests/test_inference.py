import re
import shutil
import sqlite3
from types import SimpleNamespace

import pytest
import torch

from redisql import inference
from redisql.config import Config, ModelConfig
from redisql.inference import ExecutionEvaluator, predict_records, profile_model


class InferenceTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, sqls):
        self.sqls = sqls
        self.inputs = []

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, truncation=False):
        self.inputs.append(text)
        tokens = list(re.finditer(r"\w+|[^\w\s]", text))
        ids = [3 + sum(map(ord, token.group())) % 101 for token in tokens]
        offsets = [token.span() for token in tokens]
        if add_special_tokens:
            ids.insert(0, self.bos_token_id)
            offsets.insert(0, (0, 0))
        result = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def decode(self, ids, skip_special_tokens=True):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return self.sqls[next(value for value in ids if value not in {0, 1, 2})]


class InferenceModel(torch.nn.Module):
    def __init__(self, tokens, candidates=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.config = ModelConfig(vocab_size=128, d_model=8, n_heads=2, encoder_layers=1, decoder_layers=1, dim_feedforward=16, max_input_length=128, max_output_length=64, dropout=0.0)
        self.tokens = tokens
        self.candidates = candidates
        self.generation_calls = []

    def generate(self, input_ids, attention_mask, **kwargs):
        self.generation_calls.append(kwargs)
        values = [[1, self.tokens[min(index, len(self.tokens) - 1)], 2] for index in range(len(input_ids))]
        return torch.tensor(values, device=input_ids.device)

    def generate_candidates(self, input_ids, attention_mask, **kwargs):
        self.generation_calls.append(kwargs)
        return [self.candidates for _ in range(len(input_ids))]


class InferenceTeacher:
    def __init__(self, proposal="SELECT COUNT(*) FROM t"):
        self.proposal = proposal
        self.calls = []

    def repair(self, question, schema, draft, feedback):
        self.calls.append((question, schema, draft, feedback))
        assert "SECRET_GOLD" not in str(self.calls[-1])
        return self.proposal


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "test.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        connection.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (1, "b"), (2, "c")])
    return str(path)


def record(database, identifier="one", query="SELECT id FROM t"):
    return {"id": identifier, "dataset": "unit", "question": "Which records?", "query": query, "db_path": database, "schema": {"tables": [{"name": "t", "columns": [{"name": "id", "type": "number"}, {"name": "name", "type": "text"}]}]}}


def prediction(identifier, sql, dataset="unit"):
    return {"id": identifier, "dataset": dataset, "sql": sql}


def test_native_execution_evaluation_aligns_ids_and_preserves_multisets_and_order(database):
    records = [record(database, "a"), record(database, "b", "SELECT id FROM t ORDER BY id ASC"), record(database, "c", "SELECT DISTINCT id FROM t")]
    evaluator = ExecutionEvaluator(records, Config())
    result = evaluator.evaluate([prediction("c", "SELECT id FROM t"), prediction("a", "SELECT id FROM t ORDER BY id DESC"), prediction("b", "SELECT id FROM t ORDER BY id DESC")])
    assert result["protocol"] == "sqlite_denotation"
    assert result["execution_accuracy"] == pytest.approx(100 / 3)
    assert [row["id"] for row in result["outcomes"]] == ["a", "b", "c"]
    assert [row["correct"] for row in result["outcomes"]] == [True, False, False]
    assert result["categories"] == {"ok": 1, "denotation": 2}


@pytest.mark.parametrize("values", [
    [prediction("a", "SELECT id FROM t")],
    [prediction("a", "SELECT id FROM t"), prediction("a", "SELECT id FROM t")],
    [prediction("a", "SELECT id FROM t"), prediction("unknown", "SELECT id FROM t")],
    [prediction("a", "SELECT id FROM t"), prediction("b", "SELECT id FROM t", dataset="other")],
])
def test_evaluation_rejects_incomplete_duplicate_or_unknown_prediction_ids(database, values):
    evaluator = ExecutionEvaluator([record(database, "a"), record(database, "b")], Config())
    with pytest.raises(ValueError):
        evaluator.evaluate(values)


@pytest.mark.parametrize("query", [None, "", " ", "missing"])
def test_evaluation_requires_reference_sql(database, query):
    value = record(database)
    if query == "missing":
        value.pop("query")
    else:
        value["query"] = query
    with pytest.raises(ValueError, match="reference|query"):
        ExecutionEvaluator([value], Config())


def test_evaluation_rejects_duplicate_record_ids(database):
    values = [record(database, "a"), record(database, "a")]
    with pytest.raises(ValueError, match="duplicate|unique"):
        ExecutionEvaluator(values, Config())


def test_external_evaluator_receives_unchanged_sql_protocol_flags_and_suite_path(database, tmp_path):
    adapter = tmp_path / "external.py"
    adapter.write_text("TIMEOUT = 0\nseen = []\ndef eval_exec_match(database, prediction, reference, **kwargs):\n    seen.append((database, prediction, reference, kwargs))\n    return True\n")
    suite = tmp_path / "suite" / "bank"
    suite.mkdir(parents=True)
    destination = suite / "test.sqlite"
    shutil.copyfile(database, destination)
    config = Config()
    config.evaluation.timeout = 7.5
    config.evaluation.official_evaluator = str(adapter)
    config.evaluation.test_suite_database = str(tmp_path / "suite")
    value = record(database)
    value["db_id"] = "bank"
    evaluator = ExecutionEvaluator([value], config)
    result = evaluator.evaluate([prediction("one", "SELECT DISTINCT id FROM t")])
    assert result["protocol"] == "official_exec_match"
    assert result["execution_accuracy"] == 100
    assert evaluator.official.__globals__["TIMEOUT"] == 7.5
    assert evaluator.official.__globals__["seen"] == [(str(destination), "SELECT DISTINCT id FROM t", "SELECT id FROM t", {"plug_value": False, "keep_distinct": True, "progress_bar_for_each_datapoint": False})]


@pytest.mark.parametrize(("sql", "category"), [
    ("SELECT FROM", "syntax"),
    ("SELECT missing FROM t", "schema"),
    ("SELECT ABS(-9223372036854775808)", "runtime"),
])
def test_execution_failures_have_separate_categories(database, sql, category):
    evaluator = ExecutionEvaluator([record(database)], Config())
    result = evaluator.evaluate([prediction("one", sql)])
    assert result["correct"] == 0
    assert result["outcomes"][0]["category"] == category


def test_student_prediction_uses_no_database_teacher_or_reference_sql(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected database or teacher use")
    monkeypatch.setattr(inference, "validate_sql", forbidden)
    monkeypatch.setattr(inference, "validate_static", forbidden)
    monkeypatch.setattr(inference, "execute_readonly", forbidden)
    value = record("")
    value.pop("query")
    tokenizer = InferenceTokenizer({10: "SELECT id FROM t"})
    model = InferenceModel([10])
    teacher = SimpleNamespace(repair=forbidden)
    result = predict_records(model, tokenizer, [value], Config(), teacher=teacher)
    assert result == [{"id": "one", "dataset": "unit", "sql": "SELECT id FROM t", "teacher_called": False, "execution_used": False}]
    assert all("SECRET_GOLD" not in text for text in tokenizer.inputs)


@pytest.mark.parametrize("draft", ["SELECT FROM", "SELECT missing FROM t", "SELECT ABS(-9223372036854775808)"])
def test_one_shot_repairs_only_observable_failures_without_gold(database, draft):
    value = record(database, query="SECRET_GOLD")
    teacher = InferenceTeacher()
    tokenizer = InferenceTokenizer({10: draft})
    result = predict_records(InferenceModel([10]), tokenizer, [value], Config(), mode="one_shot", teacher=teacher)
    assert result[0]["sql"] == "SELECT COUNT(*) FROM t"
    assert result[0]["teacher_called"] is True
    assert len(teacher.calls) == 1
    assert teacher.calls[0][3]["category"] in {"syntax", "schema", "runtime"}


def test_one_shot_does_not_use_reference_denotation_as_a_deployment_trigger(database):
    value = record(database, query="SELECT COUNT(*) FROM t")
    teacher = InferenceTeacher()
    tokenizer = InferenceTokenizer({10: "SELECT id FROM t"})
    result = predict_records(InferenceModel([10]), tokenizer, [value], Config(), mode="one_shot", teacher=teacher)
    assert result[0]["sql"] == "SELECT id FROM t"
    assert result[0]["teacher_called"] is False
    assert teacher.calls == []


def test_one_shot_rejects_invalid_proposal_and_keeps_the_original_draft(database):
    value = record(database)
    teacher = InferenceTeacher("DELETE FROM t")
    tokenizer = InferenceTokenizer({10: "SELECT missing FROM t"})
    result = predict_records(InferenceModel([10]), tokenizer, [value], Config(), mode="one_shot", teacher=teacher)
    assert result[0]["sql"] == "SELECT missing FROM t"
    assert result[0]["teacher_called"] is True
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM t").fetchone() == (3,)


def test_execution_reranking_selects_the_best_executable_candidate(database):
    tokenizer = InferenceTokenizer({10: "SELECT missing FROM t", 11: "SELECT name FROM t", 12: "SELECT id FROM t"})
    candidates = [{"input_ids": [1, 10, 2], "score": -1.0}, {"input_ids": [1, 11, 2], "score": -2.0}, {"input_ids": [1, 12, 2], "score": -3.0}]
    model = InferenceModel([10], candidates)
    result = predict_records(model, tokenizer, [record(database)], Config(), mode="execution_rerank")
    assert result[0]["sql"] == "SELECT name FROM t"
    assert result[0]["execution_used"] is True
    assert result[0]["teacher_called"] is False
    assert model.generation_calls[0]["num_beams"] == 5


def test_execution_reranking_falls_back_to_top_beam_when_none_executes(database):
    tokenizer = InferenceTokenizer({10: "SELECT missing FROM t", 11: "SELECT FROM"})
    candidates = [{"input_ids": [1, 10, 2], "score": -1.0}, {"input_ids": [1, 11, 2], "score": -2.0}]
    result = predict_records(InferenceModel([10], candidates), tokenizer, [record(database)], Config(), mode="execution_rerank")
    assert result[0]["sql"] == "SELECT missing FROM t"


def test_execution_modes_require_database_paths():
    value = record("")
    tokenizer = InferenceTokenizer({10: "SELECT id FROM t"})
    for mode in ("one_shot", "execution_rerank"):
        with pytest.raises(ValueError, match="database paths"):
            predict_records(InferenceModel([10]), tokenizer, [value], Config(), mode=mode)


def test_one_shot_requires_an_explicit_teacher(database):
    tokenizer = InferenceTokenizer({10: "SELECT id FROM t"})
    with pytest.raises(ValueError, match="requires a teacher"):
        predict_records(InferenceModel([10]), tokenizer, [record(database)], Config(), mode="one_shot")


def test_profile_reports_per_request_batch_latency_and_request_throughput(monkeypatch):
    config = Config()
    config.evaluation.batch_size = 2
    config.evaluation.timing_rounds = 2
    config.evaluation.warmup_iterations = 2
    calls = []
    def fake_predict(model, tokenizer, records, config, mode, teacher):
        calls.append([row["id"] for row in records])
        return [{"teacher_called": row["id"] % 2 == 0} for row in records]
    clock = iter([0.0, 0.01, 1.0, 1.02, 2.0, 2.04, 3.0, 3.01, 4.0, 4.02, 5.0, 5.04])
    monkeypatch.setattr(inference, "predict_records", fake_predict)
    monkeypatch.setattr(inference, "synchronize", lambda device: None)
    monkeypatch.setattr(inference.time, "perf_counter", lambda: next(clock))
    result = profile_model(InferenceModel([10]), None, [{"id": index} for index in range(5)], config, mode="one_shot", teacher=InferenceTeacher())
    assert len(calls) == 8
    assert calls[:2] == [[0, 1], [2, 3]]
    assert result["warmup_iterations"] == 2
    assert result["mode"] == "one_shot"
    assert result["requests"] == 10
    assert result["requests_per_round"] == 5
    assert result["mean_ms"] == pytest.approx(20)
    assert result["p95_ms"] == pytest.approx(40)
    assert result["throughput_queries_per_second"] == pytest.approx(10 / 0.14)
    assert result["teacher_call_rate"] == pytest.approx(0.6)


def test_profiling_rejects_empty_inputs_and_invalid_batch_sizes():
    config = Config()
    model = InferenceModel([10])
    with pytest.raises(ValueError, match="requires requests"):
        profile_model(model, None, [], config)
    config.evaluation.batch_size = 0
    with pytest.raises(ValueError, match="batch size"):
        profile_model(model, None, [{"id": 0}], config)
