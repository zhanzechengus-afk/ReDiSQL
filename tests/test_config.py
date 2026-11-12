import ast
import io
import json
import pickle
import tokenize
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from redisql.config import Config, EvaluationConfig, ModelConfig, TrainingConfig


def test_json_yaml_and_pickle_roundtrips_preserve_resume_configuration(tmp_path):
    config = Config.from_dict({"training": {"betas": [0.85, 0.97], "seed": 13}, "output_dir": "runs/space dir"})
    expected = asdict(config)
    assert config.training.betas == (0.85, 0.97)
    json_path = tmp_path / "config.json"
    config.save(json_path)
    assert asdict(Config.load(json_path)) == expected
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("training:\n  betas: [0.85, 0.97]\n  seed: 13\noutput_dir: runs/space dir\n")
    assert asdict(Config.load(yaml_path)) == expected
    assert asdict(Config.from_dict(pickle.loads(pickle.dumps(config.to_dict())))) == expected
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"config": config.to_dict()}, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert asdict(Config.from_dict(loaded["config"])) == expected
    assert isinstance(Config.load(json_path).training.betas, tuple)


def test_all_shipped_yaml_configurations_are_valid_and_keep_the_paper_architecture():
    directory = Path(__file__).resolve().parents[1] / "configs"
    paths = sorted(directory.rglob("*.yaml"))
    assert len(paths) >= 4
    for path in paths:
        config = Config.load(path)
        assert (config.model.encoder_layers, config.model.decoder_layers) == (6, 6)
        assert config.model.d_model == 512
        assert config.model.dim_feedforward == 5120
        assert config.model.vocab_size == 32016


@pytest.mark.parametrize("values", [None, [], {"model": None}, {"training": []}])
def test_configuration_objects_and_blocks_have_explicit_structure_errors(values):
    with pytest.raises(ValueError, match="object|mapping|configuration"):
        Config.from_dict(values)


@pytest.mark.parametrize("text", ["model: [unclosed", "{not valid json"])
def test_malformed_configuration_files_fail_with_configuration_errors(tmp_path, text):
    path = tmp_path / ("bad.yaml" if text.startswith("model") else "bad.json")
    path.write_text(text)
    with pytest.raises(ValueError):
        Config.load(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_optimizer_learning_rate_must_be_finite(value):
    with pytest.raises(ValueError, match="finite|optimizer|learning"):
        TrainingConfig(learning_rate=value)


def test_invalid_betas_fail_as_a_configuration_error():
    with pytest.raises(ValueError, match="betas"):
        TrainingConfig(betas=None)


@pytest.mark.parametrize(("kind", "values"), [
    (ModelConfig, {"n_heads": 0}),
    (ModelConfig, {"d_model": 7}),
    (TrainingConfig, {"warmup_epochs": 0}),
    (TrainingConfig, {"batch_size": 0}),
    (TrainingConfig, {"learning_rate": 0}),
    (TrainingConfig, {"betas": [0.9, 1.0]}),
    (EvaluationConfig, {"timeout": 0}),
    (EvaluationConfig, {"warmup_iterations": -1}),
])
def test_invalid_numeric_settings_are_rejected(kind, values):
    with pytest.raises(ValueError):
        kind(**values)


def test_authored_package_has_no_comments_docstrings_or_readme():
    directory = Path(__file__).resolve().parents[1]
    files = [*sorted((directory / "redisql").glob("*.py")), *sorted((directory / "tests").glob("*.py"))]
    for path in files:
        source = path.read_text()
        assert all(token.type != tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(source).readline)), path
        tree = ast.parse(source)
        nodes = [tree, *(node for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)))]
        assert all(ast.get_docstring(node) is None for node in nodes), path
    authored = [path for path in directory.rglob("*") if path.is_file() and not any(part in {".venv", "__pycache__", ".pytest_cache"} or part.endswith(".egg-info") for part in path.relative_to(directory).parts)]
    assert not any(path.name.lower().startswith("readme") for path in authored)
