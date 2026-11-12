from types import SimpleNamespace

import pytest
import torch

from redisql.config import TeacherConfig
from redisql.teacher import FrozenTeacher, clean_sql_response, repair_prompt, tokenizer_identity


class TeacherTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, decoded="```sql\nSELECT 1\n```", reverse_vocabulary=False):
        self.decoded = decoded
        self.reverse_vocabulary = reverse_vocabulary
        self.prompt = None
        self.decoded_ids = None

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        self.prompt = text
        return {"input_ids": torch.tensor([[1, 4, 5]]), "attention_mask": torch.ones((1, 3), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        self.decoded_ids = ids.tolist()
        return self.decoded

    def get_vocab(self):
        values = [(f"token_{index}", index) for index in range(16)]
        if self.reverse_vocabulary:
            values.reverse()
        return dict(values)


class ToyCausalModel(torch.nn.Module):
    def __init__(self, weight=0.0, context=64):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight))
        self.config = SimpleNamespace(max_position_embeddings=context, _commit_hash=None)
        self.last_input_ids = None
        self.last_attention_mask = None
        self.generation_arguments = None

    def forward(self, input_ids, attention_mask, use_cache=False):
        self.last_input_ids = input_ids.detach().clone()
        self.last_attention_mask = attention_mask.detach().clone()
        logits = torch.nn.functional.one_hot(input_ids.remainder(16), num_classes=16).float() + self.weight
        return SimpleNamespace(logits=logits)

    def generate(self, **kwargs):
        self.generation_arguments = kwargs
        extension = torch.tensor([[11, 12]], device=kwargs["input_ids"].device)
        return torch.cat([kwargs["input_ids"], extension], dim=1)


def make_teacher(weight=0.0, context=64):
    tokenizer = TeacherTokenizer()
    model = ToyCausalModel(weight, context)
    config = TeacherConfig(model="unit-test-teacher", device="cpu", dtype="float32", max_new_tokens=8)
    return FrozenTeacher(config, tokenizer, model=model), tokenizer, model


def test_teacher_logits_align_each_sql_token_with_the_preceding_position():
    teacher, _, model = make_teacher()
    inputs = torch.tensor([[1, 4, 5], [1, 8, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    labels = torch.tensor([[6, 7, 2, -100], [9, 2, -100, -100]])
    result = teacher.logits(inputs, mask, labels)
    assert result.shape == (2, 4, 16)
    assert result[0, :3].argmax(-1).tolist() == [5, 6, 7]
    assert result[1, :2].argmax(-1).tolist() == [8, 9]
    assert torch.equal(result[0, 3], torch.zeros(16))
    assert torch.equal(result[1, 2:], torch.zeros((2, 16)))
    assert model.last_input_ids.tolist() == [[1, 4, 5, 6, 7, 2], [1, 8, 9, 2, 0, 0]]
    assert model.last_attention_mask.tolist() == [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]
    assert not result.requires_grad
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_teacher_context_overflow_and_empty_sources_fail_explicitly():
    teacher, _, _ = make_teacher(context=4)
    with pytest.raises(ValueError, match="context length exceeded"):
        teacher.logits(torch.tensor([[1, 4, 5]]), torch.ones((1, 3), dtype=torch.long), torch.tensor([[6, 2]]))
    with pytest.raises(ValueError, match="must not be empty"):
        teacher.logits(torch.tensor([[0]]), torch.tensor([[0]]), torch.tensor([[2]]))


@pytest.mark.parametrize(("input_ids", "attention_mask", "labels"), [
    ([1, 4], [1, 1], [[2]]),
    ([[1, 4]], [[1]], [[2]]),
    ([[1, 4]], [[1, 1]], [[2], [2]]),
    ([[1, 4]], [[1, 1]], [2]),
])
def test_teacher_rejects_inconsistent_tensor_shapes_before_forward(input_ids, attention_mask, labels):
    teacher, _, model = make_teacher()
    with pytest.raises(ValueError, match="inconsistent shapes"):
        teacher.logits(torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels))
    assert model.last_input_ids is None


def test_repair_prompt_does_not_expose_unused_reference_fields():
    feedback = {"category": "denotation", "message": "Executable query; result mismatch", "gold_sql": "SECRET_GOLD_SQL", "rows": [["SECRET_RESULT_ROW"]]}
    prompt = repair_prompt("How many accounts?", "account(id)", "SELECT id FROM account", feedback)
    assert "SECRET_GOLD_SQL" not in prompt
    assert "SECRET_RESULT_ROW" not in prompt
    assert "How many accounts?" in prompt
    assert "SELECT id FROM account" in prompt
    assert prompt.startswith("[INST]") and prompt.endswith("[/INST]")


@pytest.mark.parametrize(("response", "expected"), [
    ("  \n```sql\nSELECT 1;\n``` \n", "SELECT 1;"),
    ("```SQLite\nSELECT 2\n```", "SELECT 2"),
    ("```\nSELECT 3\n```", "SELECT 3"),
    ("```sql\nSELECT 1\n```\nExplanation", "```sql\nSELECT 1\n```\nExplanation"),
    (" SELECT '```' ", "SELECT '```'"),
])
def test_response_cleaning_removes_only_outer_whitespace_and_a_whole_fence(response, expected):
    assert clean_sql_response(response) == expected


def test_repair_decodes_only_new_tokens_with_one_greedy_proposal():
    teacher, tokenizer, model = make_teacher()
    result = teacher.repair("How many?", "account(id)", "SELECT id FROM account", {"category": "denotation", "message": "Executable query; result mismatch"})
    assert result == "SELECT 1"
    assert tokenizer.decoded_ids == [11, 12]
    assert model.generation_arguments["do_sample"] is False
    assert model.generation_arguments["num_beams"] == 1
    assert model.generation_arguments["num_return_sequences"] == 1
    assert model.generation_arguments["max_new_tokens"] == 8


def test_repair_refuses_to_silently_truncate_overlong_prompts():
    teacher, _, _ = make_teacher(context=10)
    with pytest.raises(ValueError, match="prompt exceeds teacher context"):
        teacher.repair("Q", "T(c)", "SELECT c FROM T", {"category": "schema", "message": "Unresolved identifier: c"})


def test_tokenizer_identity_is_order_independent_and_includes_special_ids():
    first = TeacherTokenizer()
    second = TeacherTokenizer(reverse_vocabulary=True)
    assert tokenizer_identity(first) == tokenizer_identity(second)
    second.pad_token_id = 3
    assert tokenizer_identity(first) != tokenizer_identity(second)


def test_injected_teacher_identity_distinguishes_weights_and_mutations():
    first, _, model = make_teacher(weight=0.0)
    second, _, _ = make_teacher(weight=1.0)
    identity_before = first.identity
    assert identity_before != second.identity
    with torch.no_grad():
        model.weight.add_(2)
    assert identity_before != first.identity
