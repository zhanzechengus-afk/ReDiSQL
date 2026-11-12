from pathlib import Path
import re
import torch
from .runtime import resolve_device
from .storage import file_fingerprint, fingerprint, state_fingerprint


def clean_sql_response(value):
    value = value.strip()
    match = re.fullmatch(r"```(?:sql|sqlite)?[ \t]*\n(.*?)\n?```", value, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else value


def repair_prompt(question, schema, draft, feedback):
    return (
        "[INST] Correct the SQL draft using only the question, listed schema, and validation feedback. "
        "Preserve the requested meaning. Return one corrected SQL query without explanation.\n"
        f"Question: {question}\nSchema: {schema}\nDraft: {draft}\n"
        f"Failure: {feedback['category']}\nFeedback: {feedback['message']} [/INST]"
    )


def tokenizer_identity(tokenizer):
    backend = getattr(tokenizer, "backend_tokenizer", None)
    value = backend.to_str() if backend is not None else sorted(tokenizer.get_vocab().items())
    return fingerprint({"vocabulary": value, "bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id})


class FrozenTeacher:
    def __init__(self, config, tokenizer, model=None):
        self.config = config
        self.tokenizer = tokenizer
        self.device = resolve_device(config.device)
        self.model = model
        self.injected_model = model is not None
        if model is not None:
            self.model.to(self.device).eval().requires_grad_(False)

    def _load(self):
        if self.model is None:
            from transformers import AutoModelForCausalLM
            dtype = getattr(torch, self.config.dtype) if self.device.type == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model,
                revision=self.config.revision,
                local_files_only=self.config.local_files_only,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(self.device).eval().requires_grad_(False)
        return self.model

    @property
    def identity(self):
        model = self._load()
        resolved = getattr(model.config, "_commit_hash", None)
        local = Path(self.config.model)
        local_files = {}
        if local.is_dir():
            for path in sorted(local.rglob("*")):
                if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".model"}:
                    local_files[str(path.relative_to(local))] = file_fingerprint(path)
        parameter = next(model.parameters(), None)
        identity = {"model": self.config.model, "revision": resolved or self.config.revision, "local_files": local_files, "type": type(model).__name__, "dtype": str(parameter.dtype) if parameter is not None else None, "device_type": self.device.type, "attention": getattr(model.config, "_attn_implementation", None), "torch": torch.__version__}
        if self.injected_model:
            identity["weights"] = state_fingerprint(model)
        return identity

    @torch.no_grad()
    def logits(self, input_ids, attention_mask, labels):
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape or labels.ndim != 2 or labels.shape[0] != input_ids.shape[0]:
            raise ValueError("teacher input, attention mask, and labels have inconsistent shapes")
        model = self._load()
        sources = [ids[mask.bool()].tolist() for ids, mask in zip(input_ids, attention_mask)]
        targets = [row[row.ne(-100)].tolist() for row in labels]
        if any(not source for source in sources):
            raise ValueError("teacher inputs must not be empty")
        lengths = [len(source) + len(target) for source, target in zip(sources, targets)]
        maximum = max(lengths)
        if maximum > getattr(model.config, "max_position_embeddings", maximum):
            raise ValueError("teacher context length exceeded")
        packed = torch.full((len(sources), maximum), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(packed)
        for index, (source, target) in enumerate(zip(sources, targets)):
            ids = source + target
            packed[index, :len(ids)] = torch.tensor(ids, device=self.device)
            mask[index, :len(ids)] = 1
        predictions = model(input_ids=packed, attention_mask=mask, use_cache=False).logits
        result = predictions.new_zeros((labels.shape[0], labels.shape[1], predictions.shape[-1]))
        for index, (source, target) in enumerate(zip(sources, targets)):
            start = len(source) - 1
            result[index, :len(target)] = predictions[index, start:start + len(target)]
        return result.to(input_ids.device)

    @torch.no_grad()
    def repair(self, question, schema, draft, feedback):
        model = self._load()
        prompt = repair_prompt(question, schema, draft, feedback)
        values = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        values = {key: value.to(self.device) for key, value in values.items()}
        maximum = getattr(model.config, "max_position_embeddings", None)
        if maximum is not None and values["input_ids"].shape[1] + self.config.max_new_tokens > maximum:
            raise ValueError("repair prompt exceeds teacher context length")
        generated = model.generate(
            **values,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(generated[0, values["input_ids"].shape[1]:], skip_special_tokens=True)
        return clean_sql_response(text)

