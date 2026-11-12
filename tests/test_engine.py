import copy
from types import MethodType

import torch
from torch.nn import functional as F

from redisql.config import Config, ModelConfig
from redisql.corpus import add_schema_negatives, record_key
from redisql.data import collate_records
from redisql.engine import TrainingObjective, shift_labels
from redisql.model import ReDiSQLModel
from redisql.objectives import distillation_per_record
from redisql.targets import build_targets
from test_corpus import database, records_for, tokenizer


def objective_config(tokenizer, method="redisql"):
    config = Config(model=ModelConfig(
        vocab_size=len(tokenizer), d_model=8, n_heads=2,
        encoder_layers=1, decoder_layers=1, dim_feedforward=16,
        max_input_length=512, max_output_length=64, dropout=0,
    ))
    config.training.method = method
    return config


def repair_for(record, tokenizer, config):
    targets = build_targets(
        "SELECT age FROM employee", "SELECT name FROM employee WHERE id > 0",
        tokenizer, schema=record["schema"], max_length=64,
    )
    add_schema_negatives(targets, record, config)
    return {record_key(record): {"targets": targets}}


def test_shifted_labels_include_bos_and_exclude_ignored_padding():
    labels = torch.tensor([[4, 6, 2, -100], [8, 2, -100, -100]])
    values, mask = shift_labels(labels, 1, 0)
    assert values.tolist() == [[1, 4, 6, 2], [1, 8, 2, 0]]
    assert mask.tolist() == [[True, True, True, False], [True, True, False, False]]


def test_repairs_use_corrected_prefixes_and_one_shared_encoder(database, tokenizer):
    torch.manual_seed(8)
    config = objective_config(tokenizer)
    model = ReDiSQLModel(config.model)
    objective = TrainingObjective(model, config)
    records = records_for(database, tokenizer, count=3)
    batch = collate_records(records, 0)
    repairs = repair_for(records[0], tokenizer, config)
    prefixes = []
    memories = []
    original = model.decode

    def decode(self, decoder_input_ids, encoder_states, attention_mask,
               decoder_attention_mask=None):
        prefixes.append(decoder_input_ids.detach().clone())
        return original(decoder_input_ids, encoder_states, attention_mask, decoder_attention_mask)

    model.decode = MethodType(decode, model)

    def capture(module, arguments, output):
        output.retain_grad()
        memories.append(output)

    hook = model.encoder.register_forward_hook(capture)
    losses = objective(batch, repairs=repairs)
    hook.remove()
    assert len(memories) == 1
    assert len(prefixes) == 2
    assert torch.equal(prefixes[0], shift_labels(batch["labels"], 1, 0)[0])
    repaired = torch.tensor([next(iter(repairs.values()))["targets"]["input_ids"]])
    assert torch.equal(prefixes[1], shift_labels(repaired, 1, 0)[0])
    assert prefixes[1].shape[-1] != prefixes[0].shape[-1]
    for name in ("repair", "schema", "ast"):
        assert losses[name][0].item() > 0
        assert torch.equal(losses[name][1:], torch.zeros(2))
    auxiliary = losses["repair"] + 0.2 * losses["schema"] + 0.1 * losses["ast"]
    auxiliary.mean().backward()
    assert memories[0].grad[0].abs().sum() > 0
    assert memories[0].grad[1:].abs().sum() == 0
    assert model.question_projection.weight.grad.abs().sum() > 0
    assert model.node_classifier.weight.grad.abs().sum() > 0


def test_accepted_losses_are_averaged_over_the_full_batch(database, tokenizer):
    torch.manual_seed(13)
    config = objective_config(tokenizer)
    full = ReDiSQLModel(config.model)
    single = copy.deepcopy(full)
    records = records_for(database, tokenizer, count=3)
    repairs = repair_for(records[0], tokenizer, config)
    full_losses = TrainingObjective(full, config)(collate_records(records, 0), repairs=repairs)
    single_losses = TrainingObjective(single, config)(collate_records(records[:1], 0), repairs=repairs)
    full_aux = full_losses["repair"] + 0.2 * full_losses["schema"] + 0.1 * full_losses["ast"]
    single_aux = single_losses["repair"] + 0.2 * single_losses["schema"] + 0.1 * single_losses["ast"]
    assert torch.allclose(full_aux.mean() * 3, single_aux.mean(), atol=1e-6)
    full_aux.mean().backward()
    single_aux.mean().backward()
    assert torch.allclose(
        full.token_embeddings.weight.grad * 3,
        single.token_embeddings.weight.grad,
        atol=1e-6, rtol=1e-5,
    )


def test_kid_rewrites_only_selected_nonspecial_positions(database, tokenizer):
    torch.manual_seed(42)
    config = objective_config(tokenizer, method="kid")
    model = ReDiSQLModel(config.model, auxiliary=False)
    objective = TrainingObjective(model, config)
    records = records_for(database, tokenizer, count=2)
    batch = collate_records(records, 0)
    captured = []

    def forward(self, input_ids, attention_mask, decoder_input_ids,
                decoder_attention_mask=None):
        captured.append(decoder_input_ids.clone())
        logits = torch.zeros(*decoder_input_ids.shape, config.model.vocab_size)
        logits[..., 8] = 10
        return {"logits": logits}

    model.forward = MethodType(forward, model)
    rewritten = objective.kid_prefixes(batch, tokenizer.unk_token_id)
    for gold, replacement in zip(batch["labels"], rewritten):
        eligible = gold.ne(-100) & gold.ne(0) & gold.ne(1) & gold.ne(2)
        changed = gold.ne(replacement)
        assert changed.sum().item() == round(eligible.sum().item() * config.training.kid_mask_rate)
        assert not (changed & ~eligible).any()
        assert replacement[changed].eq(8).all()
    assert captured[0].eq(tokenizer.unk_token_id).sum() == rewritten.ne(batch["labels"]).sum()
    assert not rewritten.requires_grad


def test_kid_sft_uses_gold_while_reverse_kd_uses_rewritten_prefix(database, tokenizer):
    torch.manual_seed(23)
    config = objective_config(tokenizer, method="kid")
    model = ReDiSQLModel(config.model, auxiliary=False)
    objective = TrainingObjective(model, config)
    batch = collate_records(records_for(database, tokenizer, count=2), 0)
    prefixes = batch["labels"].clone()
    prefixes[:, 1] = 8
    teacher = torch.zeros(batch["labels"].shape + (config.model.vocab_size,))
    losses = objective(batch, teacher, prefixes)
    gold_inputs, mask = shift_labels(batch["labels"], 1, 0)
    base = model(batch["input_ids"], batch["attention_mask"], gold_inputs, mask)
    changed_inputs, changed_mask = shift_labels(prefixes, 1, 0)
    states = model.decode(changed_inputs, base["encoder_states"], batch["attention_mask"], changed_mask)
    expected = distillation_per_record(
        F.linear(states, model.token_embeddings.weight), teacher,
        batch["labels"], config.training.temperature, reverse=True,
    )
    assert torch.allclose(losses["kd"], expected, atol=1e-6)
