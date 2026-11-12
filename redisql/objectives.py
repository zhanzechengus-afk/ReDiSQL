import torch
from torch.nn import functional as F


def _validate_token_shapes(logits, labels):
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Expected logits [batch, tokens, vocabulary] and labels [batch, tokens]")


def _per_record_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def token_cross_entropy_per_record(logits, labels, pad=-100):
    _validate_token_shapes(logits, labels)
    token_losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
        reduction="none", ignore_index=pad,
    ).reshape_as(labels)
    return _per_record_mean(token_losses, labels.ne(pad))


def distillation_per_record(student_logits, teacher_logits, labels,
                            temperature=2.0, reverse=False):
    _validate_token_shapes(student_logits, labels)
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("Teacher and student logits must have equal shapes")
    if temperature <= 0:
        raise ValueError("Distillation temperature must be positive")
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(
        teacher_logits.detach().float() / temperature, dim=-1
    )
    if reverse:
        values = (
            student_log_probs.exp() * (student_log_probs - teacher_log_probs)
        ).sum(-1)
    else:
        values = F.kl_div(
            student_log_probs, teacher_log_probs, reduction="none", log_target=True
        ).sum(-1)
    return _per_record_mean(values, labels.ne(-100)) * temperature ** 2


def localized_repair_per_record(logits, labels, edit_mask, identifier_mask,
                                identifier_weight=2.0, full_token=False):
    _validate_token_shapes(logits, labels)
    if edit_mask.shape != labels.shape or identifier_mask.shape != labels.shape:
        raise ValueError("Repair masks must match labels")
    if identifier_weight < 0:
        raise ValueError("Identifier weight cannot be negative")
    valid = labels.ne(-100)
    selected = valid if full_token else valid & edit_mask.bool()
    values = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
        reduction="none", ignore_index=-100,
    ).reshape_as(labels)
    weights = 1.0 + (identifier_weight - 1.0) * identifier_mask.to(values.dtype)
    return _per_record_mean(values * weights, selected)


def schema_contrast_loss(question_state, schema_states, positive_indices,
                         negative_indices):
    if question_state.ndim == 2 and question_state.shape[0] == 1:
        question_state = question_state.squeeze(0)
    if question_state.ndim != 1 or schema_states.ndim != 2:
        raise ValueError("Schema contrast expects a question vector and a schema matrix")
    if question_state.shape[0] != schema_states.shape[-1]:
        raise ValueError("Question and schema representation widths differ")
    if len(positive_indices) != len(negative_indices):
        raise ValueError("Each positive must have one corresponding negative list")
    zero = (question_state.sum() + schema_states.sum()) * 0.0
    if not len(positive_indices):
        return zero
    question = F.normalize(question_state.float(), dim=-1)
    schema = F.normalize(schema_states.float(), dim=-1)
    values = []
    for positive, negatives in zip(positive_indices, negative_indices):
        positive = int(positive)
        negatives = [int(index) for index in negatives]
        if not 0 <= positive < schema.shape[0]:
            raise ValueError("Positive schema index is out of range")
        if any(index < 0 or index >= schema.shape[0] for index in negatives):
            raise ValueError("Negative schema index is out of range")
        if positive in negatives or len(set(negatives)) != len(negatives):
            raise ValueError("Schema negatives must be distinct and exclude their positive")
        if not negatives:
            continue
        indices = torch.tensor([positive] + negatives, device=schema.device)
        scores = schema.index_select(0, indices) @ question
        values.append(-F.log_softmax(scores, dim=-1)[0])
    return torch.stack(values).mean() if values else zero


def ast_regularization_loss(node_states, node_labels, edge_indices, edge_labels,
                            node_classifier, edge_classifier):
    if node_states.ndim != 2:
        raise ValueError("AST node states must have shape [nodes, width]")
    node_labels = torch.as_tensor(
        node_labels, dtype=torch.long, device=node_states.device
    )
    edge_indices = torch.as_tensor(
        edge_indices, dtype=torch.long, device=node_states.device
    ).reshape(-1, 2)
    edge_labels = torch.as_tensor(
        edge_labels, dtype=torch.long, device=node_states.device
    )
    if node_labels.shape != (node_states.shape[0],):
        raise ValueError("AST node labels do not match the node count")
    if edge_labels.shape != (edge_indices.shape[0],):
        raise ValueError("AST edge labels do not match the edge count")
    zero = node_states.sum() * 0.0
    if not node_states.shape[0]:
        if edge_indices.shape[0]:
            raise ValueError("An empty AST cannot contain edges")
        return zero
    node_loss = F.cross_entropy(node_classifier(node_states).float(), node_labels)
    if edge_indices.shape[0]:
        if (edge_indices < 0).any() or (edge_indices >= node_states.shape[0]).any():
            raise ValueError("AST edge references a missing node")
        parents = node_states.index_select(0, edge_indices[:, 0])
        children = node_states.index_select(0, edge_indices[:, 1])
        edge_logits = edge_classifier(torch.cat([parents, children], dim=-1)).float()
        edge_loss = F.cross_entropy(edge_logits, edge_labels)
    else:
        edge_loss = zero
    return 0.5 * (node_loss + edge_loss)
