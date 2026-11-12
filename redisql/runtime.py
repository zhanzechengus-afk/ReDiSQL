from contextlib import nullcontext
from pathlib import Path
import os
import random
import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def resolve_device(value="auto"):
    if value == "auto":
        value = f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(device)
    return device


def autocast_context(device, dtype="bfloat16"):
    if device.type == "cuda" and dtype in {"bfloat16", "float16"}:
        return torch.autocast("cuda", dtype=getattr(torch, dtype))
    return nullcontext()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class Runtime:
    def __init__(self, device="auto"):
        self.device = resolve_device(device)
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.created_group = False
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl" if self.device.type == "cuda" else "gloo")
            self.created_group = True
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

    @property
    def master(self):
        return self.rank == 0

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def sum(self, value):
        result = torch.as_tensor(value, device=self.device, dtype=torch.float64).clone()
        if self.world_size > 1:
            dist.all_reduce(result)
        return result

    def broadcast(self, value):
        if self.world_size == 1:
            return value
        values = [value]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def close(self):
        if self.created_group:
            dist.destroy_process_group()


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("invalid ReDiSQL checkpoint")
    return payload

