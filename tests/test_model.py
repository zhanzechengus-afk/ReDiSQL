from types import MethodType, SimpleNamespace

import pytest
import torch

from redisql.model import ReDiSQLModel


def config(**changes):
    values = {
        "vocab_size": 17,
        "d_model": 8,
        "n_heads": 2,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "dim_feedforward": 24,
        "max_input_length": 12,
        "max_output_length": 9,
        "dropout": 0.0,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_paper_parameter_counts():
    paper = config(
        vocab_size=32016, d_model=512, n_heads=8,
        encoder_layers=6, decoder_layers=6, dim_feedforward=5120,
        max_input_length=512, max_output_length=256,
    )
    with torch.device("meta"):
        model = ReDiSQLModel(paper)
        backbone = ReDiSQLModel(paper, auxiliary=False)
    assert model.parameter_counts() == {
        "backbone": 98709504,
        "auxiliary": 536079,
        "total": 99245583,
    }
    assert backbone.count_parameters() == 98709504
    assert not hasattr(backbone, "node_classifier")


def test_auxiliary_heads_preserve_matched_backbone_and_rng_state():
    torch.manual_seed(271)
    baseline = ReDiSQLModel(config(), auxiliary=False)
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(271)
    repair_model = ReDiSQLModel(config(), auxiliary=True)
    repair_rng = torch.get_rng_state().clone()
    assert torch.equal(baseline_rng, repair_rng)
    baseline_state = baseline.state_dict()
    repair_state = repair_model.state_dict()
    for name, tensor in baseline_state.items():
        assert torch.equal(tensor, repair_state[name]), name


def test_forward_is_causal_and_uses_shared_output_weights():
    torch.manual_seed(8)
    model = ReDiSQLModel(config()).eval()
    source = torch.tensor([[1, 4, 5, 2, 0]])
    attention = source.ne(0)
    prefix = torch.tensor([[1, 6, 7, 8]])
    changed = prefix.clone()
    changed[:, 2:] = torch.tensor([10, 11])
    first = model(source, attention, prefix)
    second = model(source, attention, changed)
    assert first["logits"].shape == (1, 4, 17)
    assert torch.allclose(first["logits"][:, :2], second["logits"][:, :2], atol=1e-6)
    expected = first["decoder_states"] @ model.token_embeddings.weight.T
    assert torch.allclose(first["logits"], expected)
    first["logits"].square().mean().backward()
    assert torch.isfinite(model.token_embeddings.weight.grad).all()
    assert model.token_embeddings.weight.grad.abs().sum() > 0


def test_encoder_ignores_padding_values():
    torch.manual_seed(11)
    model = ReDiSQLModel(config()).eval()
    source = torch.tensor([[1, 4, 5, 0, 0]])
    attention = source.ne(0)
    changed = source.clone()
    changed[:, -2:] = torch.tensor([9, 10])
    first = model.encode(source, attention)
    second = model.encode(changed, attention)
    assert torch.allclose(first[:, :3], second[:, :3], atol=1e-6)


@pytest.mark.parametrize("beams", [1, 3])
def test_generation_stops_at_eos_and_restores_training_mode(beams):
    model = ReDiSQLModel(config()).train()
    with torch.no_grad():
        model.token_embeddings.weight.zero_()
        model.token_embeddings.weight[2, 0] = 5

    def decode(self, decoder_input_ids, encoder_states, attention_mask,
               decoder_attention_mask=None):
        states = encoder_states.new_zeros(
            decoder_input_ids.shape[0], decoder_input_ids.shape[1], self.config.d_model
        )
        states[..., 0] = 1
        return states

    model.decode = MethodType(decode, model)
    source = torch.tensor([[1, 4, 2], [1, 6, 2]])
    result = model.generate(source, source.ne(0), max_new_tokens=4, num_beams=beams)
    assert result.tolist() == [[1, 2], [1, 2]]
    assert model.training


def test_export_removes_auxiliary_heads_without_changing_predictions(tmp_path):
    model = ReDiSQLModel(config()).eval()
    path = tmp_path / "student.pt"
    artifact = model.export_backbone(path)
    assert artifact["auxiliary"] is False
    assert all("classifier" not in key and "projection" not in key for key in artifact["model"])
    deployed = ReDiSQLModel(config(), auxiliary=False).eval()
    deployed.load_state_dict(torch.load(path, weights_only=True)["model"])
    source = torch.tensor([[1, 4, 2]])
    decoder = torch.tensor([[1, 5]])
    assert torch.allclose(
        model(source, source.ne(0), decoder)["logits"],
        deployed(source, source.ne(0), decoder)["logits"],
    )


def test_sequence_limits_and_empty_encoder_are_rejected():
    model = ReDiSQLModel(config())
    with pytest.raises(ValueError, match="position limit"):
        model.encode(torch.ones(1, 13, dtype=torch.long))
    with pytest.raises(ValueError, match="non-padding"):
        model.encode(torch.zeros(1, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="output limit"):
        model.generate(torch.ones(1, 2, dtype=torch.long), max_new_tokens=10)


def test_beam_candidates_are_sorted_and_suppress_input_only_tokens():
    model = ReDiSQLModel(config()).eval()
    with torch.no_grad():
        model.token_embeddings.weight.zero_()
        model.token_embeddings.weight[0, 0] = 10
        model.token_embeddings.weight[1, 0] = 9
        model.token_embeddings.weight[2, 0] = 8
        model.token_embeddings.weight[3, 0] = 7

    def decode(self, decoder_input_ids, encoder_states, attention_mask,
               decoder_attention_mask=None):
        states = encoder_states.new_zeros(
            decoder_input_ids.shape[0], decoder_input_ids.shape[1], self.config.d_model
        )
        states[..., 0] = 1
        return states

    model.decode = MethodType(decode, model)
    source = torch.tensor([[1, 4, 2]])
    candidates = model.generate_candidates(source, source.ne(0), max_new_tokens=4, num_beams=3)[0]
    assert len(candidates) == 3
    assert candidates[0]["input_ids"] == [1, 2]
    assert all(not {0, 1}.intersection(item["input_ids"][1:]) for item in candidates)
    assert [item["score"] for item in candidates] == sorted(
        [item["score"] for item in candidates], reverse=True
    )
    expected = torch.log_softmax(model.token_embeddings.weight[:, 0], 0)[2].item()
    assert candidates[0]["score"] == pytest.approx(expected)
    assert model.generate(source, source.ne(0), max_new_tokens=3).tolist() == [[1, 2]]


def test_eos_remains_available_when_it_is_also_padding():
    model = ReDiSQLModel(config(pad_token_id=2))
    scores = model._generation_log_probs(torch.zeros(1, 8))
    assert torch.isfinite(scores[0, 2])
    assert not torch.isfinite(scores[0, 1])
