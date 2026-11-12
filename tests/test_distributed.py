import copy
import json
import multiprocessing
import time
import traceback
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from redisql.config import Config, ModelConfig
from redisql.data import collate_records
from redisql.engine import TrainingObjective
from redisql.model import ReDiSQLModel
from redisql.runtime import Runtime
from redisql.training import ExactDistributedSampler, on_master


def objective_fixture():
    config = Config(model=ModelConfig(vocab_size=24, d_model=8, n_heads=2, encoder_layers=1, decoder_layers=1, dim_feedforward=16, max_input_length=16, max_output_length=8, dropout=0))
    records = [
        {"id": "0", "dataset": "distributed", "input_ids": [3, 4, 5, 6], "attention_mask": [1, 1, 1, 1], "labels": [12, 13, 2], "question_positions": [0], "schema_items": [{"id": "a.x", "positions": [1]}, {"id": "a.y", "positions": [2]}, {"id": "a", "positions": [3]}]},
        {"id": "1", "dataset": "distributed", "input_ids": [7, 8, 9], "attention_mask": [1, 1, 1], "labels": [14, 2], "question_positions": [0], "schema_items": [{"id": "a.x", "positions": [1]}, {"id": "a.y", "positions": [2]}]},
    ]
    repairs = {}
    for index in range(2):
        repairs[f"distributed:{index}"] = {"targets": {
            "input_ids": [6 + index, 10 + index, 2],
            "edit_mask": [0, 1, 0],
            "identifier_mask": [0, 1, 0],
            "schema_positives": ["a.x"],
            "schema_negatives": {"a.x": ["a.y"]},
            "ast_nodes": [{"positions": [0, 1], "label": 0}, {"positions": [1], "label": 6}],
            "ast_edges": [{"parent": 0, "child": 1, "label": 0}],
        }}
    return config, records, repairs


def phases():
    return [([0, 1], [1.0, 1.0], [0]), ([0, 1], [1.0, 1.0], [1]), ([0, 1], [1.0, 1.0], []), ([0, 0], [1.0, 0.0], [0])]


def gradient_snapshot(module):
    return {name: None if parameter.grad is None else parameter.grad.detach().clone() for name, parameter in module.named_parameters()}


def distributed_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    try:
        dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
        torch.manual_seed(101)
        config, records, repairs = objective_fixture()
        objective = TrainingObjective(ReDiSQLModel(config.model, auxiliary=True), config)
        wrapped = DistributedDataParallel(objective, find_unused_parameters=True)
        optimizer = torch.optim.SGD(objective.parameters(), lr=0.05)
        snapshots = []
        for indices, weights, accepted in phases():
            optimizer.zero_grad(set_to_none=True)
            count = torch.tensor(weights[rank], dtype=torch.float64)
            dist.all_reduce(count)
            record = copy.deepcopy(records[indices[rank]])
            batch = collate_records([record], config.model.pad_token_id)
            current_repairs = {f"distributed:{index}": repairs[f"distributed:{index}"] for index in accepted}
            losses = wrapped(batch, repairs=current_repairs)
            loss = losses["total"].sum() * weights[rank] * 2 / count
            loss.backward()
            snapshots.append(gradient_snapshot(objective))
            optimizer.step()
        torch.save(snapshots, Path(output) / f"rank-{rank}.pt")
        runtime = Runtime("cpu")

        def fail():
            (Path(output) / f"callback-{rank}.called").touch()
            raise ValueError("master failure fixture")

        try:
            on_master(runtime, fail)
        except RuntimeError as error:
            message = str(error)
        else:
            raise AssertionError("Master failure was not propagated")
        value = on_master(runtime, lambda: {"source_rank": rank})
        (Path(output) / f"rank-{rank}-failure.json").write_text(json.dumps({"message": message, "value": value}))
    except BaseException:
        (Path(output) / f"rank-{rank}.error").write_text(traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("size,world_size", [(0, 2), (1, 8), (5, 2), (7, 3), (16, 4), (17, 8)])
def test_exact_sampler_counts_each_real_example_once(size, world_size):
    for epoch in range(3):
        samples = []
        lengths = []
        for rank in range(world_size):
            sampler = ExactDistributedSampler(size, rank=rank, world_size=world_size, seed=31)
            sampler.set_epoch(epoch)
            values = list(sampler)
            assert values == list(sampler)
            assert len(values) == len(sampler)
            samples.extend(values)
            lengths.append(len(values))
        assert len(set(lengths)) == 1
        assert sum(weight for index, weight in samples) == size
        assert sorted(index for index, weight in samples if weight == 1) == list(range(size))
        assert all(weight in (0.0, 1.0) for index, weight in samples)


@pytest.fixture(scope="module")
def distributed_results(tmp_path_factory):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    tmp_path = tmp_path_factory.mktemp("distributed")
    context = multiprocessing.get_context("spawn")
    rendezvous = (tmp_path / "rendezvous").resolve().as_uri()
    processes = [context.Process(target=distributed_worker, args=(rank, rendezvous, str(tmp_path))) for rank in range(2)]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 35
    try:
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        errors = [path.read_text() for path in sorted(tmp_path.glob("*.error"))]
        assert not any(process.is_alive() for process in processes), "Distributed worker timed out"
        assert all(process.exitcode == 0 for process in processes), "\n".join(errors)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    return tmp_path


def test_ddp_uneven_repairs_and_padding_match_full_batch_gradients(distributed_results):
    tmp_path = distributed_results
    actual = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]
    torch.set_num_threads(1)
    torch.manual_seed(101)
    config, records, repairs = objective_fixture()
    objective = TrainingObjective(ReDiSQLModel(config.model, auxiliary=True), config)
    optimizer = torch.optim.SGD(objective.parameters(), lr=0.05)
    for phase, (indices, weights, accepted) in enumerate(phases()):
        optimizer.zero_grad(set_to_none=True)
        batch = collate_records([copy.deepcopy(records[index]) for index in indices], config.model.pad_token_id)
        current_repairs = {f"distributed:{index}": repairs[f"distributed:{index}"] for index in accepted}
        losses = objective(batch, repairs=current_repairs)
        loss = (losses["total"] * torch.tensor(weights)).sum() / sum(weights)
        loss.backward()
        expected = gradient_snapshot(objective)
        for rank in range(2):
            for name, gradient in expected.items():
                distributed = actual[rank][phase][name]
                if gradient is None:
                    assert distributed is None, (phase, rank, name)
                else:
                    assert distributed is not None, (phase, rank, name)
                    torch.testing.assert_close(distributed, gradient, rtol=2e-4, atol=3e-5, msg=f"phase={phase} rank={rank} parameter={name}")
        optimizer.step()


def test_master_failure_is_broadcast_without_peer_hang(distributed_results):
    results = [json.loads((distributed_results / f"rank-{rank}-failure.json").read_text()) for rank in range(2)]
    assert all(value["message"] == "ValueError: master failure fixture" for value in results)
    assert all(value["value"] == {"source_rank": 0} for value in results)
    assert (distributed_results / "callback-0.called").exists()
    assert not (distributed_results / "callback-1.called").exists()
