from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import json
import math
import yaml


def _integers(instance, names):
    for name in names:
        value = getattr(instance, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")


def _floats(instance, names):
    for name in names:
        try:
            value = float(getattr(instance, name))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be finite") from error
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        setattr(instance, name, value)


@dataclass
class ModelConfig:
    vocab_size: int = 32016
    d_model: int = 512
    n_heads: int = 8
    encoder_layers: int = 6
    decoder_layers: int = 6
    dim_feedforward: int = 5120
    max_input_length: int = 512
    max_output_length: int = 256
    dropout: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self):
        _integers(self, ("vocab_size", "d_model", "n_heads", "encoder_layers", "decoder_layers", "dim_feedforward", "max_input_length", "max_output_length", "pad_token_id", "bos_token_id", "eos_token_id"))
        _floats(self, ("dropout",))
        for name in ("vocab_size", "d_model", "n_heads", "encoder_layers", "decoder_layers", "dim_feedforward", "max_input_length", "max_output_length"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if any(not 0 <= x < self.vocab_size for x in (self.pad_token_id, self.bos_token_id, self.eos_token_id)):
            raise ValueError("special token IDs must lie inside the vocabulary")


@dataclass
class DataConfig:
    train_path: str = "data/spider/train_spider.json"
    dev_path: str = "data/spider/dev.json"
    tables_path: str | None = "data/spider/tables.json"
    db_root: str | None = "data/spider/database"
    dataset: str = "spider"
    tokenizer: str = "codellama/CodeLlama-13b-Instruct-hf"
    tokenizer_revision: str | None = None
    local_files_only: bool = False
    num_workers: int = 0

    def __post_init__(self):
        _integers(self, ("num_workers",))
        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")


@dataclass
class TeacherConfig:
    model: str = "codellama/CodeLlama-13b-Instruct-hf"
    revision: str | None = None
    device: str = "auto"
    dtype: str = "bfloat16"
    local_files_only: bool = False
    max_new_tokens: int = 256

    def __post_init__(self):
        _integers(self, ("max_new_tokens",))
        if self.max_new_tokens < 1 or self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("invalid teacher token limit or dtype")


@dataclass
class TrainingConfig:
    method: str = "redisql"
    seed: int = 42
    epochs: int = 20
    warmup_epochs: int = 10
    selection_start_epoch: int = 11
    batch_size: int = 32
    effective_batch_size: int = 256
    learning_rate: float = 3e-5
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_clip: float = 1.0
    patience: int = 3
    dtype: str = "bfloat16"
    device: str = "auto"
    temperature: float = 2.0
    kd_weight: float = 1.0
    repair_weight: float = 1.0
    schema_weight: float = 0.2
    ast_weight: float = 0.1
    identifier_weight: float = 2.0
    edit_mask: bool = True
    repair_target: str = "teacher"
    execution_filter: bool = True
    repair_trigger: str = "all"
    schema_negatives: str = "hard"
    max_schema_negatives: int = 4
    kid_mask_rate: float = 0.2
    deterministic: bool = False

    def __post_init__(self):
        _integers(self, ("seed", "epochs", "warmup_epochs", "selection_start_epoch", "batch_size", "effective_batch_size", "patience", "max_schema_negatives"))
        _floats(self, ("learning_rate", "epsilon", "weight_decay", "warmup_ratio", "gradient_clip", "temperature", "kd_weight", "repair_weight", "schema_weight", "ast_weight", "identifier_weight", "kid_mask_rate"))
        try:
            self.betas = tuple(float(value) for value in self.betas)
        except (TypeError, ValueError) as error:
            raise ValueError("betas must contain two finite numbers") from error
        if self.method not in {"sft", "soft_kd", "kid", "redisql"}:
            raise ValueError("method must be sft, soft_kd, kid, or redisql")
        if self.repair_target not in {"teacher", "gold"}:
            raise ValueError("repair_target must be teacher or gold")
        if self.repair_trigger not in {"all", "static"}:
            raise ValueError("repair_trigger must be all or static")
        if self.schema_negatives not in {"hard", "random"}:
            raise ValueError("schema_negatives must be hard or random")
        if self.epochs < self.warmup_epochs or self.warmup_epochs < 1:
            raise ValueError("epochs must be at least warmup_epochs >= 1")
        if self.batch_size < 1 or self.effective_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if self.temperature <= 0 or not 0 <= self.kid_mask_rate <= 1:
            raise ValueError("invalid temperature or KID mask rate")
        if min(self.kd_weight, self.repair_weight, self.schema_weight, self.ast_weight, self.identifier_weight) < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.patience < 1 or self.selection_start_epoch < 1:
            raise ValueError("patience and selection_start_epoch must be positive")
        if len(self.betas) != 2 or any(not math.isfinite(value) or not 0 <= value < 1 for value in self.betas):
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.learning_rate <= 0 or self.epsilon <= 0 or self.gradient_clip <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if not 0 <= self.warmup_ratio <= 1 or self.max_schema_negatives < 0:
            raise ValueError("invalid warmup ratio or negative count")
        if self.dtype not in {"float32", "float16", "bfloat16"} or self.seed < 0:
            raise ValueError("invalid training dtype or seed")


@dataclass
class EvaluationConfig:
    batch_size: int = 8
    timeout: float = 30.0
    num_beams: int = 1
    warmup_iterations: int = 50
    timing_rounds: int = 5
    cpu_threads: int = 16
    official_evaluator: str | None = None
    test_suite_database: str | None = None

    def __post_init__(self):
        _integers(self, ("batch_size", "num_beams", "warmup_iterations", "timing_rounds", "cpu_threads"))
        _floats(self, ("timeout",))
        if min(self.batch_size, self.num_beams, self.timing_rounds, self.cpu_threads) < 1 or self.timeout <= 0 or self.warmup_iterations < 0:
            raise ValueError("invalid evaluation settings")


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_dir: str = "runs/spider"

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("configuration must be an object")
        allowed = {f.name for f in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        mapping = {"model": ModelConfig, "data": DataConfig, "teacher": TeacherConfig, "training": TrainingConfig, "evaluation": EvaluationConfig}
        result = {}
        for name, kind in mapping.items():
            block = value.get(name, {})
            if not isinstance(block, dict):
                raise ValueError(f"{name} must be an object")
            unknown = set(block) - {f.name for f in fields(kind)}
            if unknown:
                raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
            result[name] = kind(**block)
        return cls(**result, output_dir=value.get("output_dir", "runs/spider"))

    @classmethod
    def load(cls, path):
        try:
            with Path(path).open(encoding="utf-8") as handle:
                value = json.load(handle) if str(path).endswith(".json") else yaml.safe_load(handle)
        except yaml.YAMLError as error:
            raise ValueError("invalid YAML configuration") from error
        if not isinstance(value, dict):
            raise ValueError("configuration must be an object")
        return cls.from_dict(value)

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

