import pytest
import torch
from torch import nn
from torch.nn import functional as F

from redisql.objectives import (
    ast_regularization_loss,
    distillation_per_record,
    localized_repair_per_record,
    schema_contrast_loss,
    token_cross_entropy_per_record,
)


def test_sft_averages_each_record_before_batch_reduction():
    logits = torch.tensor([
        [[2.0, 0.0], [0.0, 2.0], [3.0, -1.0]],
        [[0.0, 2.0], [7.0, -2.0], [1.0, 1.0]],
    ], requires_grad=True)
    labels = torch.tensor([[0, 1, -100], [0, -100, -100]])
    values = token_cross_entropy_per_record(logits, labels)
    expected = torch.stack([
        F.cross_entropy(logits[0, :2], labels[0, :2]),
        F.cross_entropy(logits[1, :1], labels[1, :1]),
    ])
    assert torch.allclose(values, expected)
    values.mean().backward()
    assert logits.grad[0, 2].abs().sum() == 0
    assert logits.grad[1, 1:].abs().sum() == 0


@pytest.mark.parametrize("reverse", [False, True])
def test_distillation_direction_temperature_and_teacher_detachment(reverse):
    student = torch.tensor([[[0.2, 1.0, -0.4], [9.0, 0.0, 0.0]]], requires_grad=True)
    teacher = torch.tensor([[[1.5, -0.3, 0.8], [-9.0, 0.0, 0.0]]], requires_grad=True)
    labels = torch.tensor([[1, -100]])
    temperature = 2.0
    value = distillation_per_record(student, teacher, labels, temperature, reverse)
    p = (student[0, 0] / temperature).softmax(-1)
    q = (teacher[0, 0].detach() / temperature).softmax(-1)
    expected = (
        (p * (p.log() - q.log())).sum() if reverse
        else (q * (q.log() - p.log())).sum()
    ) * temperature ** 2
    assert torch.allclose(value[0], expected, atol=1e-6)
    value.sum().backward()
    assert teacher.grad is None
    assert student.grad[0, 1].abs().sum() == 0
    assert student.grad[0, 0].abs().sum() > 0


def test_localized_identifiers_use_unweighted_mask_denominator():
    logits = torch.zeros(2, 4, 3, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, -100], [0, 1, 2, -100]])
    mask = torch.tensor([[0, 1, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
    identifiers = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)
    values = localized_repair_per_record(logits, labels, mask, identifiers)
    expected = torch.log(torch.tensor(3.0)) * 1.5
    assert torch.allclose(values, torch.tensor([expected, 0.0]))
    values.mean().backward()
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[1].abs().sum() == 0
    assert logits.grad[0, 1].abs().sum() > logits.grad[0, 2].abs().sum()


def test_full_token_control_keeps_changed_identifier_weights():
    logits = torch.zeros(1, 3, 2)
    labels = torch.tensor([[0, 1, 0]])
    mask = torch.tensor([[0, 1, 0]])
    identifiers = torch.tensor([[0, 1, 0]])
    value = localized_repair_per_record(
        logits, labels, mask, identifiers, full_token=True
    )
    assert torch.allclose(value[0], torch.log(torch.tensor(2.0)) * (4 / 3))


def test_schema_uses_cosine_and_excludes_positives_without_negatives():
    question = torch.tensor([2.0, 0.0], requires_grad=True)
    schema = torch.tensor([[3.0, 0.0], [0.0, 7.0], [-2.0, 0.0]], requires_grad=True)
    value = schema_contrast_loss(question, schema, [0, 2], [[1, 2], []])
    expected = -torch.log_softmax(torch.tensor([1.0, 0.0, -1.0]), 0)[0]
    assert torch.allclose(value, expected)
    value.backward()
    assert question.grad is not None
    assert schema.grad is not None


def test_schema_empty_negative_pools_are_differentiable_zero():
    question = torch.randn(4, requires_grad=True)
    schema = torch.randn(2, 4, requires_grad=True)
    value = schema_contrast_loss(question, schema, [0, 1], [[], []])
    assert value.item() == 0
    value.backward()
    assert question.grad.abs().sum() == 0
    assert schema.grad.abs().sum() == 0


def test_ast_averages_node_and_edge_losses_equally():
    nodes = torch.randn(3, 4, requires_grad=True)
    node_classifier = nn.Linear(4, 9)
    edge_classifier = nn.Linear(8, 6)
    node_labels = torch.tensor([0, 3, 6])
    edges = torch.tensor([[0, 1], [0, 2]])
    edge_labels = torch.tensor([2, 0])
    expected = 0.5 * (
        F.cross_entropy(node_classifier(nodes), node_labels)
        + F.cross_entropy(edge_classifier(torch.cat([nodes[edges[:, 0]], nodes[edges[:, 1]]], -1)), edge_labels)
    )
    value = ast_regularization_loss(
        nodes, node_labels, edges, edge_labels, node_classifier, edge_classifier
    )
    assert torch.allclose(value, expected)
    value.backward()
    assert nodes.grad.abs().sum() > 0
    assert node_classifier.weight.grad.abs().sum() > 0
    assert edge_classifier.weight.grad.abs().sum() > 0


def test_single_ast_node_has_zero_edge_component():
    nodes = torch.randn(1, 4, requires_grad=True)
    node_classifier = nn.Linear(4, 9)
    edge_classifier = nn.Linear(8, 6)
    labels = torch.tensor([0])
    value = ast_regularization_loss(
        nodes, labels, [], [], node_classifier, edge_classifier
    )
    expected = 0.5 * F.cross_entropy(node_classifier(nodes), labels)
    assert torch.allclose(value, expected)


def test_ignored_tokens_have_finite_zero_losses():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    labels = torch.full((2, 3), -100)
    value = token_cross_entropy_per_record(logits, labels)
    assert torch.equal(value, torch.zeros(2))
    value.sum().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("reverse", [False, True])
def test_distillation_gradient_matches_closed_form(reverse):
    temperature = 2.0
    student = torch.tensor([[[0.7, -0.1, 1.4], [0.3, 0.8, -0.2]]], requires_grad=True)
    teacher = torch.tensor([[[0.1, 0.9, -0.3], [1.0, -0.2, 0.4]]], requires_grad=True)
    labels = torch.tensor([[1, 2]])
    loss = distillation_per_record(student, teacher, labels, temperature, reverse).sum()
    loss.backward()
    p = torch.softmax(student.detach() / temperature, -1)
    q = torch.softmax(teacher.detach() / temperature, -1)
    if reverse:
        ratio = p.log() - q.log()
        divergence = (p * ratio).sum(-1, keepdim=True)
        expected = temperature * p * (ratio - divergence) / 2
    else:
        expected = temperature * (p - q) / 2
    assert torch.allclose(student.grad, expected, atol=1e-6)
    assert teacher.grad is None


def test_trailing_deletion_supervises_eos_only():
    logits = torch.zeros(1, 4, 5, requires_grad=True)
    labels = torch.tensor([[3, 4, 2, -100]])
    edit_mask = torch.tensor([[0, 0, 1, 0]])
    identifiers = torch.zeros_like(edit_mask)
    loss = localized_repair_per_record(logits, labels, edit_mask, identifiers).sum()
    assert torch.allclose(loss, torch.log(torch.tensor(5.0)))
    loss.backward()
    assert logits.grad[0, :2].abs().sum() == 0
    assert logits.grad[0, 2].abs().sum() > 0
    assert logits.grad[0, 3].abs().sum() == 0
