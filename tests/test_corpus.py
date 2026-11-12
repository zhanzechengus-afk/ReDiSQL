import copy
import json
import sqlite3

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from redisql.config import Config, EvaluationConfig, ModelConfig
from redisql.corpus import collect_corpus, load_corpus
from redisql.data import encode_record
from redisql.teacher import repair_prompt


class FixedStudent(torch.nn.Module):
    def __init__(self, tokenizer, draft):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.tokenizer = tokenizer
        self.draft = draft

    def generate(self, input_ids, attention_mask, **kwargs):
        values = [self.tokenizer.bos_token_id] + self.tokenizer.encode(self.draft, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        return torch.tensor([values] * len(input_ids), dtype=torch.long, device=input_ids.device)


class FixedTeacher:
    def __init__(self, proposal):
        self.proposal = proposal
        self.calls = []

    @property
    def identity(self):
        return {"model": "unit-test-teacher", "weights": "fixed"}

    def repair(self, question, schema, draft, feedback):
        self.calls.append({"question": question, "schema": schema, "draft": draft, "feedback": feedback, "prompt": repair_prompt(question, schema, draft, feedback)})
        return self.proposal


@pytest.fixture
def tokenizer():
    words = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "SELECT", "FROM", "WHERE", "ORDER", "BY", "AS", "DISTINCT", "COUNT", "ABS", "employee", "department", "name", "age", "id", "missing", "title", "TEXT", "INTEGER", "List", "Show", "names", "ages", "(", ")", ",", ".", ":", ";", "[", "]", "QUESTION", "SCHEMA", "primary", "key", "1", "0", ">", "-", "9223372036854775808"]
    backend = Tokenizer(WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 1)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]", clean_up_tokenization_spaces=False)


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "records.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript("CREATE TABLE employee(id INTEGER, name TEXT, age INTEGER); CREATE TABLE department(id INTEGER,title TEXT); INSERT INTO employee VALUES(1,'Alice',30),(2,'Bob',40); INSERT INTO department VALUES(1,'Research');")
    connection.close()
    return path


def records_for(database, tokenizer, count=1, query="SELECT name FROM employee"):
    schema = {"tables": [
        {"name": "employee", "columns": [{"name": "id", "type": "INTEGER", "primary_key": True}, {"name": "name", "type": "TEXT"}, {"name": "age", "type": "INTEGER"}]},
        {"name": "department", "columns": [{"name": "id", "type": "INTEGER", "primary_key": True}, {"name": "title", "type": "TEXT"}]},
    ]}
    return [encode_record({"id": str(index), "dataset": "fixture", "question": "List names" if index == 0 else "Show names", "query": query, "db_path": str(database), "schema": copy.deepcopy(schema)}, tokenizer, max_input_length=512, max_output_length=64) for index in range(count)]


def config_for(tokenizer, output):
    return Config(model=ModelConfig(vocab_size=len(tokenizer), d_model=8, n_heads=2, encoder_layers=1, decoder_layers=1, dim_feedforward=16, max_input_length=512, max_output_length=64, dropout=0), evaluation=EvaluationConfig(timeout=1), output_dir=str(output))


def test_collection_roundtrip_and_cached_requests_are_not_subtracted(database, tokenizer, tmp_path):
    records = records_for(database, tokenizer, count=2)
    config = config_for(tokenizer, tmp_path / "corpus")
    model = FixedStudent(tokenizer, "SELECT age FROM employee")
    teacher = FixedTeacher("SELECT name FROM employee")
    corpus, first = collect_corpus(model, tokenizer, teacher, records, config)
    assert model.training
    assert len(corpus) == 2
    assert first["statistics"]["repair_requests"] == 2
    assert first["statistics"]["teacher_calls"] == 2
    assert first["statistics"]["accepted_pairs"] == 2
    for item in corpus.values():
        targets = item["targets"]
        assert targets["input_ids"][-1] == tokenizer.eos_token_id
        assert targets["schema_positives"] == ["employee", "employee.name"]
        assert any(targets["edit_mask"])
        assert any(targets["identifier_mask"])
    loaded, manifest = load_corpus(config.output_dir, records=records, tokenizer=tokenizer, model=model, config=config)
    assert loaded == corpus
    assert manifest == first
    cached, second = collect_corpus(model, tokenizer, teacher, records, config)
    assert cached == corpus
    assert second["statistics"]["repair_requests"] == 2
    assert second["statistics"]["resume_cache_hits"] == 2
    assert second["statistics"].get("teacher_calls", 0) == 0
    assert second["statistics"]["gate_acceptance_rate"] == 1
    assert len(teacher.calls) == 2


def test_prompt_never_contains_gold_sql_or_reference_rows(database, tokenizer, tmp_path):
    records = records_for(database, tokenizer)
    config = config_for(tokenizer, tmp_path / "corpus")
    teacher = FixedTeacher("SELECT name FROM employee")
    collect_corpus(FixedStudent(tokenizer, "SELECT missing FROM employee"), tokenizer, teacher, records, config)
    assert len(teacher.calls) == 1
    prompt = teacher.calls[0]["prompt"]
    assert records[0]["query"] not in prompt
    assert "Alice" not in prompt
    assert "Bob" not in prompt
    assert "Unresolved identifier: missing" in prompt


def test_gold_target_ablation_retains_accepted_identities(database, tokenizer, tmp_path):
    records = records_for(database, tokenizer)
    config = config_for(tokenizer, tmp_path / "corpus")
    model = FixedStudent(tokenizer, "SELECT age FROM employee")
    proposal = "SELECT name FROM employee WHERE id > 0"
    teacher = FixedTeacher(proposal)
    original, manifest = collect_corpus(model, tokenizer, teacher, records, config)
    config.training.repair_target = "gold"
    gold, replacement = collect_corpus(model, tokenizer, teacher, records, config)
    assert set(gold) == set(original)
    key = next(iter(original))
    assert original[key]["teacher_repair"] == gold[key]["teacher_repair"] == proposal
    assert original[key]["draft"] == gold[key]["draft"]
    assert gold[key]["repair"] == "SELECT name FROM employee"
    assert original[key]["repair"] != gold[key]["repair"]
    assert replacement["statistics"]["resume_cache_hits"] == 1
    assert len(teacher.calls) == 1


def test_static_trigger_ignores_denotation_errors(database, tokenizer, tmp_path):
    records = records_for(database, tokenizer)
    config = config_for(tokenizer, tmp_path / "static")
    config.training.repair_trigger = "static"
    teacher = FixedTeacher("SELECT name FROM employee")
    corpus, manifest = collect_corpus(FixedStudent(tokenizer, "SELECT age FROM employee"), tokenizer, teacher, records, config)
    assert not corpus
    assert not teacher.calls
    assert manifest["statistics"]["draft_passes"] == 1
    assert manifest["statistics"].get("repair_requests", 0) == 0


def test_no_execution_filter_accepts_static_valid_runtime_failure(database, tokenizer, tmp_path):
    records = records_for(database, tokenizer)
    model = FixedStudent(tokenizer, "SELECT missing FROM employee")
    proposal = "SELECT ABS(-9223372036854775808) FROM employee"
    config = config_for(tokenizer, tmp_path / "unfiltered")
    config.training.repair_trigger = "static"
    config.training.execution_filter = False
    accepted, manifest = collect_corpus(model, tokenizer, FixedTeacher(proposal), records, config)
    assert len(accepted) == 1
    assert manifest["statistics"]["gate_accepts"] == 1
    config.training.execution_filter = True
    config.output_dir = str(tmp_path / "filtered")
    rejected, filtered = collect_corpus(model, tokenizer, FixedTeacher(proposal), records, config)
    assert not rejected
    assert filtered["statistics"]["rejected_runtime"] == 1


@pytest.mark.parametrize("query", ["SELECT missing FROM employee", "SELECT ABS(-9223372036854775808) FROM employee"])
def test_invalid_gold_is_rejected_even_for_static_trigger(database, tokenizer, tmp_path, query):
    records = records_for(database, tokenizer, query=query)
    config = config_for(tokenizer, tmp_path / "invalid")
    config.training.repair_trigger = "static"
    with pytest.raises(ValueError):
        collect_corpus(FixedStudent(tokenizer, "SELECT name FROM employee"), tokenizer, FixedTeacher("SELECT name FROM employee"), records, config)


def test_database_wal_mutation_invalidates_repair_cache(database, tokenizer, tmp_path):
    records = records_for(database, tokenizer)
    config = config_for(tokenizer, tmp_path / "corpus")
    model = FixedStudent(tokenizer, "SELECT age FROM employee")
    teacher = FixedTeacher("SELECT name FROM employee")
    collect_corpus(model, tokenizer, teacher, records, config)
    writer = sqlite3.connect(database)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("INSERT INTO employee VALUES(3,'Carol',50)")
        writer.commit()
        assert database.with_name(database.name + "-wal").exists()
        with pytest.raises(ValueError, match="different dataset"):
            load_corpus(config.output_dir, records=records)
        corpus, manifest = collect_corpus(model, tokenizer, teacher, records, config)
        assert len(corpus) == 1
        assert manifest["statistics"]["teacher_calls"] == 1
        assert manifest["statistics"].get("resume_cache_hits", 0) == 0
        assert len(teacher.calls) == 2
    finally:
        writer.close()


def test_student_weight_mutation_invalidates_cache(database, tokenizer, tmp_path):
    records = records_for(database, tokenizer)
    config = config_for(tokenizer, tmp_path / "corpus")
    model = FixedStudent(tokenizer, "SELECT age FROM employee")
    teacher = FixedTeacher("SELECT name FROM employee")
    collect_corpus(model, tokenizer, teacher, records, config)
    with torch.no_grad():
        model.anchor.fill_(1)
    with pytest.raises(ValueError, match="frozen student"):
        load_corpus(config.output_dir, model=model)
    _, manifest = collect_corpus(model, tokenizer, teacher, records, config)
    assert manifest["statistics"]["teacher_calls"] == 1
    assert len(teacher.calls) == 2


def test_manifest_rejects_changed_dataset_and_training_contract(database, tokenizer, tmp_path):
    records = records_for(database, tokenizer)
    config = config_for(tokenizer, tmp_path / "corpus")
    model = FixedStudent(tokenizer, "SELECT age FROM employee")
    collect_corpus(model, tokenizer, FixedTeacher("SELECT name FROM employee"), records, config)
    changed = copy.deepcopy(records)
    changed[0]["question"] = "Show ages"
    with pytest.raises(ValueError, match="different dataset"):
        load_corpus(config.output_dir, records=changed)
    config.training.repair_target = "gold"
    with pytest.raises(ValueError, match="repair_target"):
        load_corpus(config.output_dir, config=config)


def test_manifest_and_duplicate_keys_are_required(database, tokenizer, tmp_path):
    records = records_for(database, tokenizer)
    config = config_for(tokenizer, tmp_path / "corpus")
    collect_corpus(FixedStudent(tokenizer, "SELECT age FROM employee"), tokenizer, FixedTeacher("SELECT name FROM employee"), records, config)
    path = tmp_path / "corpus" / "repair_corpus.jsonl"
    value = json.loads(path.read_text().splitlines()[0])
    path.write_text(json.dumps(value) + "\n" + json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(path)
    path.with_name("corpus_manifest.json").unlink()
    with pytest.raises(ValueError, match="manifest"):
        load_corpus(path)
