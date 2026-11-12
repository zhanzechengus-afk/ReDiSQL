from pathlib import Path

import pytest
import yaml

from redisql.config import Config, TrainingConfig


CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PATHS = sorted(CONFIG_DIRECTORY.rglob("*.yaml"))


def test_production_schedule_keeps_training_and_optimizer_warmup_distinct():
    training = TrainingConfig()
    assert training.warmup_epochs == 10
    assert training.selection_start_epoch == training.warmup_epochs + 1 == 11
    assert training.epochs == 20
    assert training.patience == 3
    assert training.warmup_ratio == 0.06


def test_production_configuration_inventory_is_complete():
    assert len(CONFIG_PATHS) == 14


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda path: str(path.relative_to(CONFIG_DIRECTORY)))
def test_every_production_config_explicitly_pins_the_repair_schedule(path):
    raw = yaml.safe_load(path.read_text())
    assert raw["training"]["warmup_epochs"] == 10
    assert raw["training"]["selection_start_epoch"] == 11
    config = Config.load(path)
    assert config.training.warmup_epochs == 10
    assert config.training.selection_start_epoch == 11
    assert config.training.epochs == 20
    assert config.training.patience == 3
    assert config.training.warmup_ratio == 0.06


def test_small_test_schedules_can_override_production_defaults():
    training = TrainingConfig(epochs=2, warmup_epochs=1, selection_start_epoch=2)
    assert (training.warmup_epochs, training.selection_start_epoch, training.epochs) == (1, 2, 2)


def test_wikisql_paths_are_unchanged_by_the_schedule_update():
    config = Config.load(CONFIG_DIRECTORY / "wikisql.yaml")
    assert config.data.dataset == "wikisql"
    assert config.data.train_path == "data/wikisql/train.jsonl"
    assert config.data.dev_path == "data/wikisql/dev.jsonl"
    assert config.data.tables_path == "data/wikisql"
    assert config.data.db_root == "data/wikisql/databases"
    assert config.output_dir == "runs/wikisql"


@pytest.mark.parametrize(("name", "key", "value"), [
    ("full_token", "edit_mask", False),
    ("gold_targets", "repair_target", "gold"),
    ("no_ast", "ast_weight", 0.0),
    ("no_execution_filter", "execution_filter", False),
    ("no_identifier_weight", "identifier_weight", 1.0),
    ("no_schema", "schema_weight", 0.0),
    ("parser_schema_trigger", "repair_trigger", "static"),
    ("random_negatives", "schema_negatives", "random"),
    ("repair_only", "schema_weight", 0.0),
    ("repair_only", "ast_weight", 0.0),
])
def test_ablation_controls_remain_independent_of_warmup(name, key, value):
    config = Config.load(CONFIG_DIRECTORY / "ablations" / f"{name}.yaml")
    assert getattr(config.training, key) == value
    assert config.output_dir == f"runs/ablations/{name}"


@pytest.mark.parametrize(("filename", "method"), [
    ("spider.yaml", "redisql"),
    ("spider_sft.yaml", "sft"),
    ("spider_soft_kd.yaml", "soft_kd"),
    ("spider_kid.yaml", "kid"),
])
def test_baseline_methods_keep_their_training_objectives(filename, method):
    config = Config.load(CONFIG_DIRECTORY / filename)
    assert config.training.method == method
