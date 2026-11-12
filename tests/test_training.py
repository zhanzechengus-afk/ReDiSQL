import copy
import json

import pytest
import torch

from redisql.config import Config, EvaluationConfig, ModelConfig
from redisql.inference import load_student, predict_records
from redisql.runtime import load_checkpoint
from redisql.training import train_model
from redisql import training
from test_corpus import database, records_for, tokenizer


class UniformTeacher:
    def __init__(self, vocabulary_size, interrupt_after=None):
        self.vocabulary_size = vocabulary_size
        self.interrupt_after = interrupt_after
        self.calls = 0

    @property
    def identity(self):
        return {"model": "training-fixture", "weights": "uniform-fixed"}

    def logits(self, input_ids, attention_mask, labels):
        self.calls += 1
        if self.interrupt_after is not None and self.calls > self.interrupt_after:
            raise InterruptedError("epoch-boundary fixture interruption")
        return torch.zeros(
            labels.shape[0], labels.shape[1], self.vocabulary_size,
            device=input_ids.device,
        )

    def repair(self, question, schema, draft, feedback):
        return "SELECT name FROM employee"


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def training_config(tokenizer, output, epochs=2):
    config = Config(
        model=ModelConfig(
            vocab_size=len(tokenizer), d_model=8, n_heads=2,
            encoder_layers=1, decoder_layers=1, dim_feedforward=16,
            max_input_length=128, max_output_length=12, dropout=0.2,
        ),
        evaluation=EvaluationConfig(batch_size=2, timeout=1, warmup_iterations=0),
        output_dir=str(output),
    )
    config.training.epochs = epochs
    config.training.warmup_epochs = 1
    config.training.selection_start_epoch = 2
    config.training.batch_size = 1
    config.training.effective_batch_size = 2
    config.training.learning_rate = 1e-3
    config.training.dtype = "float32"
    config.training.device = "cpu"
    config.training.deterministic = True
    return config


def raw_records(database, tokenizer):
    values = records_for(database, tokenizer, count=3)
    fields = ("id", "dataset", "question", "query", "db_path", "schema")
    return [{name: value[name] for name in fields} for value in values]


def run(config, tokenizer, records, phase="train", checkpoint=None, teacher=None):
    return train_model(
        config, phase=phase, checkpoint=checkpoint,
        tokenizer=tokenizer, teacher=teacher or UniformTeacher(len(tokenizer)),
        train_records=records, dev_records=records[:1],
    )


def assert_checkpoint_training_equal(left, right):
    assert left["epoch"] == right["epoch"]
    assert left["global_step"] == right["global_step"]
    assert left["best_score"] == right["best_score"]
    assert left["bad_epochs"] == right["bad_epochs"]
    assert left["scheduler"] == right["scheduler"]
    assert left["optimizer"]["groups"] == right["optimizer"]["groups"]
    assert left["model"].keys() == right["model"].keys()
    for name in left["model"]:
        assert torch.equal(left["model"][name], right["model"][name]), name
    for name, state in left["optimizer"]["state"].items():
        for key, value in state.items():
            other = right["optimizer"]["state"][name][key]
            assert torch.equal(value, other) if torch.is_tensor(value) else value == other
    assert torch.equal(left["rng"][0]["torch"], right["rng"][0]["torch"])


def test_tiny_real_sqlite_pipeline_and_warmup_fit_resume_parity(database, tokenizer, tmp_path):
    records = raw_records(database, tokenizer)
    complete = training_config(tokenizer, tmp_path / "complete")
    result = run(complete, tokenizer, records)
    assert result["global_step"] == 4
    output = tmp_path / "complete"
    assert all((output / name).is_file() for name in ("warmup.pt", "last.pt", "best.pt", "student.pt"))
    manifest = json.loads((output / "corpus/corpus_manifest.json").read_text())
    assert manifest["statistics"]["accepted_pairs"] == 3
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert history[1]["examples"] == 3
    assert all(history[1][name] > 0 for name in ("repair", "schema", "ast"))
    deployed = load_student(output / "student.pt")
    assert deployed.parameter_counts()["auxiliary"] == 0
    predictions = predict_records(deployed, tokenizer, records[:1], complete)
    assert len(predictions) == 1
    assert predictions[0]["execution_used"] is False
    staged = copy.deepcopy(complete)
    staged.output_dir = str(tmp_path / "staged")
    warmup = run(staged, tokenizer, records, phase="warmup")
    assert warmup["global_step"] == 2
    resumed = run(staged, tokenizer, records, phase="fit", checkpoint=tmp_path / "staged/warmup.pt")
    assert resumed["global_step"] == 4
    assert_checkpoint_training_equal(
        load_checkpoint(output / "last.pt"),
        load_checkpoint(tmp_path / "staged/last.pt"),
    )


def test_epoch_boundary_interruption_restores_optimizer_and_dropout_rng(database, tokenizer, tmp_path):
    records = raw_records(database, tokenizer)
    complete = training_config(tokenizer, tmp_path / "complete", epochs=3)
    run(complete, tokenizer, records)
    interrupted = copy.deepcopy(complete)
    interrupted.output_dir = str(tmp_path / "interrupted")
    with pytest.raises(InterruptedError, match="epoch-boundary"):
        run(interrupted, tokenizer, records, teacher=UniformTeacher(len(tokenizer), interrupt_after=6))
    checkpoint = tmp_path / "interrupted/last.pt"
    assert load_checkpoint(checkpoint)["epoch"] == 2
    result = run(interrupted, tokenizer, records, checkpoint=checkpoint)
    assert result["global_step"] == 6
    assert_checkpoint_training_equal(
        load_checkpoint(tmp_path / "complete/last.pt"),
        load_checkpoint(checkpoint),
    )


def test_committed_last_checkpoint_has_its_selected_best_artifacts(database, tokenizer, tmp_path, monkeypatch):
    records = raw_records(database, tokenizer)
    config = training_config(tokenizer, tmp_path / "interrupted")
    original = training.atomic_checkpoint

    def interrupt(path, payload):
        original(path, payload)
        if path.name == "last.pt" and payload["epoch"] == 2:
            raise InterruptedError("committed last checkpoint")

    monkeypatch.setattr(training, "atomic_checkpoint", interrupt)
    with pytest.raises(InterruptedError, match="committed last"):
        run(config, tokenizer, records)
    output = tmp_path / "interrupted"
    saved = load_checkpoint(output / "last.pt")
    best = load_checkpoint(output / "best.pt")
    deployed = load_student(output / "student.pt")
    assert best["epoch"] == saved["epoch"] == 2
    for name, value in deployed.state_dict().items():
        assert torch.equal(value, saved["model"][name])
