from pathlib import Path
import argparse
import json
import torch
import yaml
from .analysis import audit_summary, paired_summary
from .config import Config
from .corpus import collect_corpus, dataset_identity
from .inference import ExecutionEvaluator, load_student, predict_records, profile_model
from .model import ReDiSQLModel
from .runtime import load_checkpoint, resolve_device, seed_everything
from .storage import read_jsonl, write_json, write_jsonl
from .teacher import FrozenTeacher, tokenizer_identity
from .training import load_tokenizer, prepare_records, read_dataset, train_model


def build_parser():
    parser = argparse.ArgumentParser(prog="redisql")
    parser.add_argument("--version", action="version", version="ReDiSQL 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("parameters", "prepare", "warmup", "collect", "fit", "train", "predict", "evaluate", "profile", "audit", "paired"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        if name in {"train", "warmup", "fit"}:
            command.add_argument("--checkpoint", "--resume", type=Path, required=name == "fit")
            command.add_argument("--corpus", type=Path)
        if name == "collect":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--output", type=Path)
        if name == "prepare":
            command.add_argument("--split", choices=("train", "dev", "both"), default="both")
            command.add_argument("--output", type=Path)
        if name in {"predict", "profile"}:
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--input", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--mode", choices=("student", "execution_rerank", "one_shot"), default="student")
        if name == "evaluate":
            command.add_argument("--predictions", type=Path, required=True)
            command.add_argument("--input", type=Path)
            command.add_argument("--output", type=Path, required=True)
        if name == "audit":
            command.add_argument("--input", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
        if name == "paired":
            command.add_argument("--input", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
    return parser


def configured(arguments):
    value = Config.load(arguments.config).to_dict() if arguments.config else Config().to_dict()
    for item in arguments.set:
        if "=" not in item:
            raise ValueError("overrides must use KEY=VALUE")
        name, raw = item.split("=", 1)
        keys = name.split(".")
        cursor = value
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                raise ValueError(f"unknown configuration key: {name}")
            cursor = cursor[key]
        if keys[-1] not in cursor:
            raise ValueError(f"unknown configuration key: {name}")
        cursor[keys[-1]] = yaml.safe_load(raw)
    return Config.from_dict(value)


def run(arguments):
    config = configured(arguments)
    command = arguments.command
    if command == "parameters":
        with torch.device("meta"):
            model = ReDiSQLModel(config.model, auxiliary=True)
        return model.parameter_counts()
    if command in {"warmup", "train", "fit"}:
        return train_model(config, phase=command, checkpoint=arguments.checkpoint, corpus_path=arguments.corpus)
    if command == "prepare":
        output = arguments.output or Path(config.output_dir) / "prepared"
        splits = ("train", "dev") if arguments.split == "both" else (arguments.split,)
        datasets = {split: read_dataset(config, getattr(config.data, split + "_path")) for split in splits}
        tokenizer = load_tokenizer(config)
        sizes = {}
        for split in splits:
            raw = datasets[split]
            encoded = prepare_records(config, raw, tokenizer)
            write_jsonl(output / (split + ".jsonl"), encoded)
            sizes[split] = len(encoded)
        config.save(output / "config.json")
        tokenizer.save_pretrained(output / "tokenizer")
        return {"output_dir": str(output), "examples": sizes}
    if command == "collect":
        seed_everything(config.training.seed, config.training.deterministic)
        device = resolve_device(config.training.device)
        raw = read_dataset(config, config.data.train_path)
        payload = load_checkpoint(arguments.checkpoint)
        tokenizer = load_tokenizer(config, Path(arguments.checkpoint).parent / "tokenizer")
        if payload.get("epoch") != config.training.warmup_epochs:
            raise ValueError("collection requires the frozen warm-up checkpoint")
        saved = payload["config"]["model"]
        if saved != config.to_dict()["model"]:
            raise ValueError("checkpoint architecture differs from configuration")
        model = ReDiSQLModel(config.model, auxiliary=payload.get("auxiliary", True)).to(device)
        model.load_state_dict(payload["model"], strict=True)
        records = prepare_records(config, raw, tokenizer)
        if payload.get("dataset_identity") != dataset_identity(records) or payload.get("tokenizer_identity") != tokenizer_identity(tokenizer):
            raise ValueError("collection checkpoint data or tokenizer differs")
        teacher = FrozenTeacher(config.teacher, tokenizer)
        _, manifest = collect_corpus(model, tokenizer, teacher, records, config, arguments.output or Path(config.output_dir) / "corpus")
        return manifest["statistics"]
    if command in {"predict", "profile"}:
        device = resolve_device(config.training.device)
        if device.type == "cpu":
            torch.set_num_threads(config.evaluation.cpu_threads)
        records = read_dataset(config, arguments.input, require_query=False)
        model = load_student(arguments.checkpoint, device)
        config.model = model.config
        tokenizer = load_tokenizer(config, Path(arguments.checkpoint).parent / "tokenizer")
        teacher = FrozenTeacher(config.teacher, tokenizer) if arguments.mode == "one_shot" else None
        if command == "profile":
            result = profile_model(model, tokenizer, records, config, arguments.mode, teacher)
            write_json(arguments.output, result)
            return result
        predictions = predict_records(model, tokenizer, records, config, arguments.mode, teacher)
        write_jsonl(arguments.output, predictions)
        return {"predictions": len(predictions), "output": str(arguments.output)}
    if command == "evaluate":
        records = read_dataset(config, arguments.input or config.data.dev_path)
        predictions = read_jsonl(arguments.predictions)
        result = ExecutionEvaluator(records, config).evaluate(predictions)
        write_json(arguments.output, result)
        return {key: value for key, value in result.items() if key != "outcomes"}
    if command == "audit":
        result = audit_summary(read_jsonl(arguments.input))
        write_json(arguments.output, result)
        return result
    if command == "paired":
        rows = read_jsonl(arguments.input)
        keys = [(row["dataset"], row["seed"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("paired scores need unique dataset/seed pairs")
        groups = {}
        for row in rows:
            groups.setdefault(row["dataset"], []).append(row)
        result = {dataset: paired_summary([row["first"] for row in values], [row["second"] for row in values], seed=config.training.seed) for dataset, values in groups.items()}
        write_json(arguments.output, result)
        return result
    raise ValueError("unknown command")


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = run(arguments)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

