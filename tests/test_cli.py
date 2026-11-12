import json
from pathlib import Path

import pytest

from redisql import cli


def forbid(*args, **kwargs):
    raise AssertionError("unexpected model download or device initialization")


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_help_and_version_are_available_without_loading_models(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_tokenizer", forbid)
    monkeypatch.setattr(cli, "FrozenTeacher", forbid)
    monkeypatch.setattr(cli, "resolve_device", forbid)
    with pytest.raises(SystemExit) as result:
        cli.main(["--help"])
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert all(name in output for name in ("parameters", "train", "collect", "predict", "evaluate", "profile", "audit", "paired"))
    with pytest.raises(SystemExit) as result:
        cli.main(["--version"])
    assert result.value.code == 0
    assert "ReDiSQL 0.1.0" in capsys.readouterr().out


def test_default_parameters_use_meta_device_and_match_the_paper_without_downloads(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_tokenizer", forbid)
    monkeypatch.setattr(cli, "FrozenTeacher", forbid)
    monkeypatch.setattr(cli, "resolve_device", forbid)
    original = cli.ReDiSQLModel
    devices = []
    def create_model(*args, **kwargs):
        model = original(*args, **kwargs)
        devices.extend(parameter.device.type for parameter in model.parameters())
        return model
    monkeypatch.setattr(cli, "ReDiSQLModel", create_model)
    cli.main(["parameters"])
    result = json.loads(capsys.readouterr().out)
    assert result == {"backbone": 98_709_504, "auxiliary": 536_079, "total": 99_245_583}
    assert devices and set(devices) == {"meta"}


def test_configuration_overrides_use_yaml_types_and_preserve_spaces(tmp_path):
    arguments = cli.build_parser().parse_args(["parameters", "--set", "training.seed=17", "--set", "training.deterministic=true", "--set", "training.betas=[0.8, 0.9]", "--set", f"output_dir={tmp_path / 'space dir'}"])
    config = cli.configured(arguments)
    assert config.training.seed == 17
    assert config.training.deterministic is True
    assert list(config.training.betas) == [0.8, 0.9]
    assert config.output_dir == str(tmp_path / "space dir")


@pytest.mark.parametrize("name", ["spider", "wikisql"])
def test_shipped_dataset_configurations_load_without_network(name, monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", forbid)
    path = Path(__file__).resolve().parents[1] / "configs" / f"{name}.yaml"
    arguments = cli.build_parser().parse_args(["parameters", "--config", str(path)])
    config = cli.configured(arguments)
    assert config.data.dataset == name
    assert config.training.method == "redisql"
    assert config.model.vocab_size == 32016


@pytest.mark.parametrize(("override", "message"), [
    ("model.unknown=2", "unknown configuration key"),
    ("unknown.branch=2", "unknown configuration key"),
    ("model.d_model", "KEY=VALUE"),
    ("model.d_model=7", "divisible"),
    ("model.n_heads=0", "positive"),
    ("training.batch_size=0", "positive"),
])
def test_invalid_overrides_fail_cleanly_before_model_creation(override, message, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ReDiSQLModel", forbid)
    monkeypatch.setattr(cli, "load_tokenizer", forbid)
    with pytest.raises(SystemExit) as result:
        cli.main(["parameters", "--set", override])
    assert result.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("value", [{"unknown": 1}, {"model": {"unknown": 1}}])
def test_unknown_configuration_fields_fail_cleanly(tmp_path, value, monkeypatch, capsys):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(cli, "ReDiSQLModel", forbid)
    with pytest.raises(SystemExit) as result:
        cli.main(["parameters", "--config", str(path)])
    assert result.value.code == 2
    assert "unknown" in capsys.readouterr().err


def test_missing_dataset_is_reported_before_tokenizer_download(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "missing-train.json"
    monkeypatch.setattr(cli, "load_tokenizer", forbid)
    with pytest.raises(SystemExit) as result:
        cli.main(["prepare", "--split", "train", "--set", f"data.train_path={missing}", "--output", str(tmp_path / "prepared")])
    assert result.value.code == 2
    assert str(missing) in capsys.readouterr().err
    assert not (tmp_path / "prepared").exists()


def test_audit_command_computes_counts_rates_and_intervals_from_input(tmp_path, capsys):
    source = tmp_path / "audit.jsonl"
    target = tmp_path / "result.json"
    labels = ["correct"] * 6 + ["incorrect"] * 3 + ["undecided"]
    write_lines(source, [{"dataset": "example", "id": str(index), "label": label} for index, label in enumerate(labels)])
    cli.main(["audit", "--input", str(source), "--output", str(target)])
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(target.read_text())
    values = result["example"]
    assert values["count"] == 10
    assert values["incorrect"]["count"] == 3
    assert values["incorrect"]["rate"] == pytest.approx(0.3)
    assert values["undecided"]["rate"] == pytest.approx(0.1)
    assert values["incorrect"]["wilson_95_ci"] == pytest.approx([0.1077912674, 0.6032218525])
    assert target.read_bytes().endswith(b"\n")


@pytest.mark.parametrize("rows", [
    [{"dataset": "example", "id": "1", "label": "unknown"}],
    [{"dataset": "example", "id": "1", "label": "correct"}, {"dataset": "example", "id": "1", "label": "incorrect"}],
])
def test_audit_command_rejects_invalid_labels_and_duplicate_ids(tmp_path, rows, capsys):
    source = tmp_path / "audit.jsonl"
    target = tmp_path / "result.json"
    write_lines(source, rows)
    with pytest.raises(SystemExit) as result:
        cli.main(["audit", "--input", str(source), "--output", str(target)])
    assert result.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not target.exists()


def test_paired_command_uses_paired_differences_and_groups_datasets(tmp_path, capsys):
    source = tmp_path / "scores.jsonl"
    target = tmp_path / "paired.json"
    rows = [{"dataset": "first", "seed": index, "first": 70 + index, "second": 71 + index} for index in range(4)]
    rows += [{"dataset": "second", "seed": 0, "first": 10, "second": 10}, {"dataset": "second", "seed": 1, "first": 20, "second": 30}]
    write_lines(source, rows)
    cli.main(["paired", "--input", str(source), "--output", str(target), "--set", "training.seed=13"])
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(target.read_text())
    assert result["first"]["paired_runs"] == 4
    assert result["first"]["first_mean"] == 71.5
    assert result["first"]["second_mean"] == 72.5
    assert result["first"]["mean_delta"] == 1
    assert result["first"]["delta_sd"] == 0
    assert result["first"]["paired_bootstrap_95_ci"] == [1, 1]
    assert result["first"]["deltas"] == [1, 1, 1, 1]
    assert result["first"]["seed"] == 13
    assert result["second"]["mean_delta"] == 5
    assert result["second"]["paired_bootstrap_95_ci"] == [0, 10]


def test_paired_command_rejects_duplicate_dataset_seed_rows(tmp_path, capsys):
    source = tmp_path / "scores.jsonl"
    target = tmp_path / "paired.json"
    row = {"dataset": "example", "seed": 7, "first": 1, "second": 2}
    write_lines(source, [row, row])
    with pytest.raises(SystemExit) as result:
        cli.main(["paired", "--input", str(source), "--output", str(target)])
    assert result.value.code == 2
    assert "unique dataset/seed" in capsys.readouterr().err
    assert not target.exists()


def test_missing_analysis_input_is_a_clear_cli_error(tmp_path, capsys):
    source = tmp_path / "missing.jsonl"
    target = tmp_path / "out.json"
    with pytest.raises(SystemExit) as result:
        cli.main(["audit", "--input", str(source), "--output", str(target)])
    assert result.value.code == 2
    assert str(source) in capsys.readouterr().err
    assert not target.exists()
