import torch
from torch import nn
from torch.nn import functional as F
from .corpus import record_key
from .objectives import ast_regularization_loss, distillation_per_record, localized_repair_per_record, schema_contrast_loss, token_cross_entropy_per_record


def shift_labels(labels, bos_token_id, pad_token_id):
    values = labels.masked_fill(labels.eq(-100), pad_token_id)
    shifted = values.new_full(values.shape, pad_token_id)
    shifted[:, 0] = bos_token_id
    shifted[:, 1:] = values[:, :-1]
    return shifted, labels.ne(-100)


def mean_pool(states, positions):
    if not positions:
        return states.sum(0) * 0.0
    positions = torch.as_tensor(positions, device=states.device, dtype=torch.long)
    return states.index_select(0, positions).mean(0)


class TrainingObjective(nn.Module):
    def __init__(self, student, config):
        super().__init__()
        self.student = student
        self.config = config

    @torch.no_grad()
    def kid_prefixes(self, batch, unk_token_id):
        labels = batch["labels"]
        eligible = labels.ne(-100) & labels.ne(self.config.model.bos_token_id) & labels.ne(self.config.model.eos_token_id) & labels.ne(self.config.model.pad_token_id)
        selected = torch.zeros_like(eligible)
        for index, row in enumerate(eligible):
            positions = row.nonzero(as_tuple=True)[0]
            count = int(round(len(positions) * self.config.training.kid_mask_rate))
            if count:
                chosen = positions[torch.randperm(len(positions), device=positions.device)[:count]]
                selected[index, chosen] = True
        masked = labels.clone()
        masked[selected] = unk_token_id
        inputs, mask = shift_labels(masked, self.config.model.bos_token_id, self.config.model.pad_token_id)
        predictions = self.student(batch["input_ids"], batch["attention_mask"], inputs, mask)["logits"].argmax(-1)
        return torch.where(selected, predictions, labels)

    def forward(self, batch, teacher_logits=None, kd_prefixes=None, repairs=None):
        config = self.config
        labels = batch["labels"]
        inputs, mask = shift_labels(labels, config.model.bos_token_id, config.model.pad_token_id)
        base = self.student(batch["input_ids"], batch["attention_mask"], inputs, mask)
        sft = token_cross_entropy_per_record(base["logits"], labels)
        zero = sft * 0.0
        kd, repair, schema, ast = zero, zero, zero, zero
        if teacher_logits is not None:
            logits = base["logits"]
            if kd_prefixes is not None:
                inputs, mask = shift_labels(kd_prefixes, config.model.bos_token_id, config.model.pad_token_id)
                states = self.student.decode(inputs, base["encoder_states"], batch["attention_mask"], mask)
                logits = F.linear(states, self.student.token_embeddings.weight)
            kd = distillation_per_record(logits, teacher_logits, labels, config.training.temperature, reverse=config.training.method == "kid")
        repairs = repairs or {}
        chosen = [(index, repairs[record_key(record)]) for index, record in enumerate(batch["metadata"]) if record_key(record) in repairs]
        if chosen:
            indices = torch.tensor([index for index, _ in chosen], device=labels.device)
            length = max(len(item["targets"]["input_ids"]) for _, item in chosen)
            repaired_labels = labels.new_full((len(chosen), length), -100)
            edit_mask = labels.new_zeros((len(chosen), length), dtype=torch.bool)
            identifier_mask = torch.zeros_like(edit_mask)
            for row, (_, item) in enumerate(chosen):
                targets = item["targets"]
                size = len(targets["input_ids"])
                repaired_labels[row, :size] = labels.new_tensor(targets["input_ids"])
                edit_mask[row, :size] = torch.as_tensor(targets["edit_mask"], device=labels.device, dtype=torch.bool)
                identifier_mask[row, :size] = torch.as_tensor(targets["identifier_mask"], device=labels.device, dtype=torch.bool)
            inputs, mask = shift_labels(repaired_labels, config.model.bos_token_id, config.model.pad_token_id)
            states = self.student.decode(inputs, base["encoder_states"].index_select(0, indices), batch["attention_mask"].index_select(0, indices), mask)
            logits = F.linear(states, self.student.token_embeddings.weight)
            values = localized_repair_per_record(logits, repaired_labels, edit_mask, identifier_mask, config.training.identifier_weight, full_token=not config.training.edit_mask)
            repair = zero.index_add(0, indices, values)
            schema_values = []
            ast_values = []
            for row, (index, item) in enumerate(chosen):
                metadata = batch["metadata"][index]
                targets = item["targets"]
                available = metadata["schema_items"]
                item_indices = {value["id"]: i for i, value in enumerate(available)}
                if available and metadata["question_positions"] and config.training.schema_weight:
                    question = self.student.question_projection(mean_pool(base["encoder_states"][index], metadata["question_positions"]))
                    vectors = torch.stack([mean_pool(base["encoder_states"][index], value["positions"]) for value in available])
                    vectors = self.student.schema_projection(vectors)
                    positives = [value for value in targets["schema_positives"] if value in item_indices]
                    negatives = targets.get("schema_negatives", {})
                    schema_values.append(schema_contrast_loss(question, vectors, [item_indices[value] for value in positives], [[item_indices[n] for n in negatives.get(value, []) if n in item_indices] for value in positives]))
                else:
                    schema_values.append(states[row].sum() * 0.0)
                nodes = targets["ast_nodes"]
                if nodes and config.training.ast_weight:
                    pooled = torch.stack([mean_pool(states[row], node["positions"]) for node in nodes])
                    edges = targets["ast_edges"]
                    ast_values.append(ast_regularization_loss(pooled, [node["label"] for node in nodes], [[edge["parent"], edge["child"]] for edge in edges], [edge["label"] for edge in edges], self.student.node_classifier, self.student.edge_classifier))
                else:
                    ast_values.append(states[row].sum() * 0.0)
            schema = zero.index_add(0, indices, torch.stack(schema_values))
            ast = zero.index_add(0, indices, torch.stack(ast_values))
        total = sft + config.training.kd_weight * kd + config.training.repair_weight * repair + config.training.schema_weight * schema + config.training.ast_weight * ast
        return {"total": total, "sft": sft, "kd": kd, "repair": repair, "schema": schema, "ast": ast}

