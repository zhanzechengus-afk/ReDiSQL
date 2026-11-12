from dataclasses import asdict
from functools import partial
from pathlib import Path
import itertools
import json
import math
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
from .config import Config
from .corpus import collect_corpus, dataset_identity, load_corpus
from .data import collate_records, encode_record, load_records
from .engine import TrainingObjective
from .inference import ExecutionEvaluator, predict_records
from .model import ReDiSQLModel
from .runtime import Runtime, atomic_checkpoint, autocast_context, load_checkpoint, seed_everything
from .storage import file_fingerprint, write_json
from .teacher import FrozenTeacher, tokenizer_identity


def load_tokenizer(config, directory=None):
    from transformers import AutoTokenizer
    source = str(directory) if directory and Path(directory).is_dir() else config.data.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(source, revision=config.data.tokenizer_revision, use_fast=True, local_files_only=config.data.local_files_only)
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer with character offsets is required")
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is None:
            raise ValueError("the shared tokenizer needs a padding token")
        tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("the tokenizer needs BOS and EOS tokens")
    if len(tokenizer) != config.model.vocab_size:
        raise ValueError(f"tokenizer vocabulary {len(tokenizer)} differs from student vocabulary {config.model.vocab_size}")
    actual = (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)
    configured = (config.model.pad_token_id, config.model.bos_token_id, config.model.eos_token_id)
    if actual != configured:
        raise ValueError(f"tokenizer special IDs {actual} differ from model special IDs {configured}")
    return tokenizer


def read_dataset(config, path, require_query=True):
    return load_records(path, db_root=config.data.db_root, tables_path=config.data.tables_path, dataset=config.data.dataset, require_query=require_query)


def prepare_records(config, records, tokenizer):
    return [encode_record(record, tokenizer, config.model.max_input_length, config.model.max_output_length) for record in records]


class IndexedDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, value):
        index, weight = value if isinstance(value, tuple) else (value, 1.0)
        return {**self.records[index], "sample_weight": float(weight)}


class ExactDistributedSampler(Sampler):
    def __init__(self, size, rank=0, world_size=1, seed=42):
        self.size = size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return math.ceil(self.size / self.world_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = [(index, 1.0) for index in torch.randperm(self.size, generator=generator).tolist()]
        total = len(self) * self.world_size
        indices.extend((0, 0.0) for _ in range(total - len(indices)))
        return iter(indices[self.rank:total:self.world_size])


def _move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if value["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in value["cuda"]])


def on_master(runtime, function):
    if runtime.world_size == 1:
        return function()
    status = None
    if runtime.master:
        try:
            status = {"ok": True, "value": function()}
        except Exception as error:
            status = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    status = runtime.broadcast(status)
    if not status["ok"]:
        raise RuntimeError(status["error"])
    return status["value"]


def _optimizer_state(optimizer, student):
    names = {parameter: name for name, parameter in student.named_parameters()}
    state = {names[parameter]: value for parameter, value in optimizer.state.items()}
    groups = [{**{key: value for key, value in group.items() if key != "params"}, "names": [names[parameter] for parameter in group["params"]]} for group in optimizer.param_groups]
    return {"state": state, "groups": groups}


def _restore_optimizer(optimizer, student, value, device):
    named = dict(student.named_parameters())
    for name, state in value["state"].items():
        if name in named:
            optimizer.state[named[name]] = {key: item.to(device) if torch.is_tensor(item) else item for key, item in state.items()}
    for group, saved in zip(optimizer.param_groups, value["groups"]):
        for key, item in saved.items():
            if key not in {"params", "names"}:
                group[key] = item


def _save(path, student, optimizer, scheduler, scaler, runtime, config, epoch, global_step, best, bad_epochs, data_identity, tokenizer_id, corpus_path, teacher_id):
    rng = _rng_state()
    states = [None] * runtime.world_size if runtime.master else None
    if runtime.world_size > 1:
        dist.gather_object(rng, states, dst=0)
    else:
        states = [rng]
    def save():
        atomic_checkpoint(path, {
            "config": config.to_dict(), "model": student.state_dict(), "auxiliary": student.auxiliary,
            "optimizer": _optimizer_state(optimizer, student), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "global_step": global_step, "best_score": best, "bad_epochs": bad_epochs,
            "dataset_identity": data_identity, "tokenizer_identity": tokenizer_id, "rng": states,
            "teacher_identity": teacher_id,
            "world_size": runtime.world_size,
            "corpus_path": str(corpus_path) if corpus_path else None,
            "corpus_fingerprint": file_fingerprint(corpus_path) if corpus_path else None,
        })
    on_master(runtime, save)


def _check_resume(payload, config, data_id, tokenizer_id):
    if payload["dataset_identity"] != data_id or payload["tokenizer_identity"] != tokenizer_id:
        raise ValueError("checkpoint data or tokenizer differs from this run")
    previous = Config.from_dict(payload["config"])
    if asdict(previous.model) != asdict(config.model):
        raise ValueError("checkpoint architecture differs")
    family = lambda method: "forward" if method in {"soft_kd", "redisql"} else method
    if family(previous.training.method) != family(config.training.method):
        raise ValueError("checkpoint warm-up objective differs")
    if payload["epoch"] > config.training.warmup_epochs:
        if asdict(previous.training) != asdict(config.training):
            raise ValueError("resumed training configuration differs")
    else:
        for field in ("warmup_epochs", "epochs", "seed", "batch_size", "effective_batch_size", "learning_rate", "betas", "epsilon", "weight_decay", "warmup_ratio", "temperature", "kd_weight"):
            a, b = getattr(previous.training, field), getattr(config.training, field)
            if list(a) != list(b) if isinstance(a, (list, tuple)) else a != b:
                raise ValueError(f"checkpoint warm-up differs in {field}")


def train_model(config, phase="train", checkpoint=None, corpus_path=None, tokenizer=None, teacher=None, train_records=None, dev_records=None):
    if phase not in {"train", "warmup", "fit"}:
        raise ValueError("unknown training phase")
    started = time.perf_counter()
    runtime = Runtime(config.training.device)
    try:
        seed_everything(config.training.seed, config.training.deterministic)
        raw_train = train_records if train_records is not None else read_dataset(config, config.data.train_path)
        raw_dev = dev_records if dev_records is not None else read_dataset(config, config.data.dev_path)
        if not raw_train or not raw_dev:
            raise ValueError("training and development sets must be nonempty")
        tokenizer = tokenizer or load_tokenizer(config)
        records = prepare_records(config, raw_train, tokenizer)
        identity = dataset_identity(records)
        tokenizer_id = tokenizer_identity(tokenizer)
        student = ReDiSQLModel(config.model, auxiliary=config.training.method == "redisql").to(runtime.device)
        payload = load_checkpoint(checkpoint) if checkpoint else None
        if phase == "fit" and payload is None:
            raise ValueError("fit requires the warm-up or resumed checkpoint")
        if phase == "fit" and payload["epoch"] < config.training.warmup_epochs:
            raise ValueError("fit requires a completed warm-up")
        if phase != "warmup" and config.training.selection_start_epoch > config.training.epochs:
            raise ValueError("selection_start_epoch exceeds the training duration")
        if payload:
            _check_resume(payload, config, identity, tokenizer_id)
            saved = {name: value for name, value in payload["model"].items() if name in student.state_dict()}
            result = student.load_state_dict(saved, strict=False)
            if any(not name.startswith(("question_projection.", "schema_projection.", "node_classifier.", "edge_classifier.")) for name in result.missing_keys):
                raise ValueError("checkpoint is missing backbone parameters")
        objective = TrainingObjective(student, config).to(runtime.device)
        wrapped = DistributedDataParallel(objective, device_ids=[runtime.device.index] if runtime.device.type == "cuda" else None, find_unused_parameters=True) if runtime.world_size > 1 else objective
        optimizer = torch.optim.AdamW(objective.parameters(), lr=config.training.learning_rate, betas=tuple(config.training.betas), eps=config.training.epsilon, weight_decay=config.training.weight_decay)
        factor = config.training.batch_size * runtime.world_size
        if config.training.effective_batch_size % factor:
            raise ValueError("effective_batch_size must be divisible by batch_size * world_size")
        accumulation = config.training.effective_batch_size // factor
        if accumulation < 1:
            raise ValueError("effective_batch_size is smaller than the distributed microbatch")
        sampler = ExactDistributedSampler(len(records), runtime.rank, runtime.world_size, config.training.seed)
        loader = DataLoader(IndexedDataset(records), batch_size=config.training.batch_size, sampler=sampler, num_workers=config.data.num_workers, collate_fn=partial(collate_records, pad_token_id=config.model.pad_token_id), pin_memory=runtime.device.type == "cuda")
        steps_per_epoch = math.ceil(len(loader) / accumulation)
        total_steps = steps_per_epoch * config.training.epochs
        warmup_steps = math.ceil(total_steps * config.training.warmup_ratio)
        def schedule(step):
            return step / max(1, warmup_steps) if step < warmup_steps else max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        scaler = torch.amp.GradScaler("cuda", enabled=runtime.device.type == "cuda" and config.training.dtype == "float16")
        start = 1
        global_step = 0
        best = None
        bad_epochs = 0
        if payload:
            _restore_optimizer(optimizer, student, payload["optimizer"], runtime.device)
            scheduler.load_state_dict(payload["scheduler"])
            scaler.load_state_dict(payload["scaler"])
            start = int(payload["epoch"]) + 1
            global_step = int(payload["global_step"])
            best = payload["best_score"]
            bad_epochs = int(payload["bad_epochs"])
        maximum_epoch = config.training.warmup_epochs if phase == "warmup" else config.training.epochs
        if start > maximum_epoch:
            raise ValueError("checkpoint has already reached the requested final epoch")
        if phase != "warmup" and bad_epochs >= config.training.patience:
            raise ValueError("checkpoint has already reached early stopping")
        output = Path(config.output_dir).resolve()
        if checkpoint is None and (output / "last.pt").exists():
            raise ValueError("output already contains a run; provide its checkpoint to resume")
        def initialize_output():
            output.mkdir(parents=True, exist_ok=True)
            config.save(output / "config.json")
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(output / "tokenizer")
            write_json(output / "runtime.json", {"torch": torch.__version__, "device": str(runtime.device), "world_size": runtime.world_size, "effective_batch_size": config.training.effective_batch_size, "parameters": student.parameter_counts()})
        on_master(runtime, initialize_output)
        if config.training.method != "sft":
            teacher = teacher or FrozenTeacher(config.teacher, tokenizer)
            if hasattr(teacher, "_load"):
                loaded = teacher._load()
                if loaded.config.vocab_size != config.model.vocab_size:
                    raise ValueError("teacher and student vocabularies differ")
        else:
            teacher = None
        teacher_id = teacher.identity if teacher is not None else None
        if payload and payload.get("teacher_identity") != teacher_id:
            raise ValueError("checkpoint teacher differs from the current teacher")
        seed_everything(config.training.seed + runtime.rank, config.training.deterministic)
        if payload:
            if payload.get("world_size", 1) != runtime.world_size:
                raise ValueError("resuming with a different world size changes the random streams")
            _restore_rng(payload["rng"][runtime.rank])
        evaluator = None
        def initialize_evaluator():
            nonlocal evaluator
            evaluator = ExecutionEvaluator(raw_dev, config)
        if phase != "warmup":
            on_master(runtime, initialize_evaluator)
        repairs = {}
        corpus_file = Path(corpus_path).resolve() if corpus_path else None
        if corpus_file and corpus_file.is_dir():
            corpus_file /= "repair_corpus.jsonl"
        if payload and payload.get("corpus_path") and corpus_file is None:
            corpus_file = Path(payload["corpus_path"])
        if payload and payload.get("corpus_fingerprint"):
            if corpus_file is None or file_fingerprint(corpus_file) != payload["corpus_fingerprint"]:
                raise ValueError("fixed repair corpus differs from the checkpoint")
        history_path = output / "history.jsonl"
        for epoch in range(start, maximum_epoch + 1):
            if config.training.method == "redisql" and epoch > config.training.warmup_epochs and not repairs:
                if corpus_file is None:
                    if epoch != config.training.warmup_epochs + 1:
                        raise ValueError("a resumed repair run must retain its fixed corpus")
                    on_master(runtime, lambda: collect_corpus(student, tokenizer, teacher, records, config, output / "corpus")[1]["statistics"])
                    corpus_file = output / "corpus" / "repair_corpus.jsonl"
                repairs, _ = load_corpus(corpus_file, records, tokenizer, student if epoch == config.training.warmup_epochs + 1 else None, config)
                if not repairs:
                    raise ValueError("no usable repairs were accepted")
            sampler.set_epoch(epoch - 1)
            wrapped.train()
            totals = torch.zeros(7, device=runtime.device, dtype=torch.float64)
            iterator = iter(loader)
            while window := list(itertools.islice(iterator, accumulation)):
                global_count = runtime.sum(sum(record["sample_weight"] for batch in window for record in batch["metadata"]))
                optimizer.zero_grad(set_to_none=True)
                for batch in window:
                    batch = _move_batch(batch, runtime.device)
                    prefixes = None
                    if config.training.method == "kid":
                        if tokenizer.unk_token_id is None:
                            raise ValueError("KID needs an unknown token")
                        with autocast_context(runtime.device, config.training.dtype):
                            prefixes = objective.kid_prefixes(batch, tokenizer.unk_token_id)
                    teacher_logits = teacher.logits(batch["input_ids"], batch["attention_mask"], prefixes if prefixes is not None else batch["labels"]) if teacher is not None else None
                    with autocast_context(runtime.device, config.training.dtype):
                        losses = wrapped(batch, teacher_logits, prefixes, repairs)
                        weights = losses["total"].new_tensor([record["sample_weight"] for record in batch["metadata"]])
                        loss = (losses["total"] * weights).sum() * runtime.world_size / global_count.clamp_min(1)
                    scaler.scale(loss).backward()
                    for index, name in enumerate(("total", "sft", "kd", "repair", "schema", "ast")):
                        totals[index] += (losses[name].detach() * weights).sum().double()
                    totals[-1] += weights.sum().double()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(objective.parameters(), config.training.gradient_clip)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale:
                    scheduler.step()
                    global_step += 1
            totals = runtime.sum(totals).cpu().tolist()
            entry = {"epoch": epoch, "global_step": global_step, "examples": int(totals[-1]), **{name: totals[index] / max(1, totals[-1]) for index, name in enumerate(("total", "sft", "kd", "repair", "schema", "ast"))}}
            improved = False
            if phase != "warmup":
                def evaluate():
                    predictions = predict_records(student, tokenizer, raw_dev, config)
                    result = evaluator.evaluate(predictions)
                    return {key: value for key, value in result.items() if key != "outcomes"}
                result = on_master(runtime, evaluate)
                entry["development"] = result
                if epoch >= config.training.selection_start_epoch:
                    score = result["execution_accuracy"]
                    improved = best is None or score > best
                    if improved:
                        best, bad_epochs = score, 0
                    else:
                        bad_epochs += 1
            def save_history():
                with history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, allow_nan=False) + "\n")
                print(json.dumps(entry, allow_nan=False), flush=True)
            on_master(runtime, save_history)
            if epoch == config.training.warmup_epochs:
                _save(output / "warmup.pt", student, optimizer, scheduler, scaler, runtime, config, epoch, global_step, best, bad_epochs, identity, tokenizer_id, None, teacher_id)
            if improved:
                _save(output / "best.pt", student, optimizer, scheduler, scaler, runtime, config, epoch, global_step, best, bad_epochs, identity, tokenizer_id, corpus_file, teacher_id)
                def export_student():
                    student.export_backbone(output / "student.pt")
                on_master(runtime, export_student)
            _save(output / "last.pt", student, optimizer, scheduler, scaler, runtime, config, epoch, global_step, best, bad_epochs, identity, tokenizer_id, corpus_file, teacher_id)
            if phase != "warmup" and bad_epochs >= config.training.patience:
                break
        def finish():
            elapsed = time.perf_counter() - started
            summary = {"output_dir": str(output), "phase": phase, "start_epoch": start, "last_epoch": epoch, "resumed_from": str(checkpoint) if checkpoint else None, "checkpoint": str(output / ("warmup.pt" if phase == "warmup" else "best.pt")), "best_execution_accuracy": best, "global_step": global_step, "wall_time_seconds": elapsed, "training_process_gpu_hours": elapsed * runtime.world_size / 3600 if runtime.device.type == "cuda" else 0.0, "gpu_hour_scope": "training_worker_devices", "teacher_service_gpu_hours": None, "database_cpu_core_hours": None}
            write_json(output / "training_summary.json", summary)
            return summary
        return on_master(runtime, finish)
    finally:
        runtime.close()

