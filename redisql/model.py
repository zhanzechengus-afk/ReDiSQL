import dataclasses
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ReDiSQLModel(nn.Module):
    def __init__(self, config, auxiliary=True):
        super().__init__()
        self.config = config
        self.auxiliary = auxiliary
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if min(config.max_input_length, config.max_output_length) < 1:
            raise ValueError("Sequence limits must be positive")
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.source_positions = nn.Embedding(config.max_input_length, config.d_model)
        self.target_positions = nn.Embedding(config.max_output_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.encoder_layers, norm=None,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.decoder_layers, norm=None
        )
        self.reset_parameters()
        if auxiliary:
            device = self.token_embeddings.weight.device
            devices = [device.index] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                self.question_projection = nn.Linear(config.d_model, config.d_model)
                self.schema_projection = nn.Linear(config.d_model, config.d_model)
                self.node_classifier = nn.Linear(config.d_model, 9)
                self.edge_classifier = nn.Linear(config.d_model * 2, 6)
                for module in (
                    self.question_projection, self.schema_projection,
                    self.node_classifier, self.edge_classifier,
                ):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _embed(self, token_ids, positions):
        if token_ids.ndim != 2:
            raise ValueError("Token IDs must have shape [batch, sequence]")
        if token_ids.shape[1] > positions.num_embeddings:
            raise ValueError("Sequence exceeds the configured position limit")
        position_ids = torch.arange(token_ids.shape[1], device=token_ids.device)
        states = self.token_embeddings(token_ids) * math.sqrt(self.config.d_model)
        return self.embedding_dropout(states + positions(position_ids).unsqueeze(0))

    def encode(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("Encoder attention mask shape does not match input IDs")
        if not attention_mask.bool().any(dim=-1).all():
            raise ValueError("Each encoder input must contain a non-padding token")
        return self.encoder(
            self._embed(input_ids, self.source_positions),
            src_key_padding_mask=~attention_mask.bool(),
        )

    def decode(self, decoder_input_ids, encoder_states, attention_mask,
               decoder_attention_mask=None):
        if decoder_attention_mask is None:
            decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id)
        if decoder_attention_mask.shape != decoder_input_ids.shape:
            raise ValueError("Decoder attention mask shape does not match input IDs")
        length = decoder_input_ids.shape[1]
        causal_mask = torch.ones(
            length, length, device=decoder_input_ids.device, dtype=torch.bool
        ).triu(1)
        states = self.decoder(
            self._embed(decoder_input_ids, self.target_positions),
            encoder_states,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~decoder_attention_mask.bool(),
            memory_key_padding_mask=~attention_mask.bool(),
        )
        return states

    def forward(self, input_ids, attention_mask, decoder_input_ids,
                decoder_attention_mask=None):
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        encoder_states = self.encode(input_ids, attention_mask)
        decoder_states = self.decode(
            decoder_input_ids, encoder_states, attention_mask, decoder_attention_mask
        )
        return {
            "logits": F.linear(decoder_states, self.token_embeddings.weight),
            "encoder_states": encoder_states,
            "decoder_states": decoder_states,
        }

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=None,
                 num_beams=1):
        if num_beams < 1 or num_beams > self.config.vocab_size:
            raise ValueError("num_beams must be between 1 and vocab_size")
        max_new_tokens = (
            self.config.max_output_length
            if max_new_tokens is None else max_new_tokens
        )
        if not 0 <= max_new_tokens <= self.config.max_output_length:
            raise ValueError("max_new_tokens exceeds the configured output limit")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        was_training = self.training
        self.eval()
        try:
            memory = self.encode(input_ids, attention_mask)
            if num_beams == 1:
                return self._greedy(memory, attention_mask, max_new_tokens)
            sequences = [
                self._beam_search(
                    memory[i:i + 1], attention_mask[i:i + 1],
                    max_new_tokens, num_beams
                )
                for i in range(input_ids.shape[0])
            ]
            result = input_ids.new_full(
                (len(sequences), max(sequence.shape[0] for sequence in sequences)),
                self.config.pad_token_id,
            )
            for index, sequence in enumerate(sequences):
                result[index, :sequence.shape[0]] = sequence
            return result
        finally:
            self.train(was_training)

    def _greedy(self, memory, attention_mask, max_new_tokens):
        sequence = torch.full(
            (memory.shape[0], 1), self.config.bos_token_id,
            dtype=torch.long, device=memory.device,
        )
        finished = torch.zeros(memory.shape[0], dtype=torch.bool, device=memory.device)
        for _ in range(max_new_tokens):
            states = self.decode(
                sequence, memory, attention_mask,
                decoder_attention_mask=sequence.ne(self.config.pad_token_id),
            )
            token = self._generation_log_probs(states[:, -1]).argmax(-1)
            token = torch.where(finished, self.config.pad_token_id, token)
            sequence = torch.cat([sequence, token.unsqueeze(-1)], dim=-1)
            finished |= token.eq(self.config.eos_token_id)
            if finished.all():
                break
        return sequence

    def _generation_log_probs(self, states):
        logits = F.linear(states, self.token_embeddings.weight).float()
        values = F.log_softmax(logits, dim=-1)
        for token_id in {self.config.pad_token_id, self.config.bos_token_id}:
            if token_id != self.config.eos_token_id:
                values[..., token_id] = -torch.inf
        return values

    @torch.no_grad()
    def generate_candidates(self, input_ids, attention_mask=None,
                            max_new_tokens=None, num_beams=5):
        if num_beams < 1 or num_beams > self.config.vocab_size:
            raise ValueError("num_beams must be between 1 and vocab_size")
        max_new_tokens = (
            self.config.max_output_length
            if max_new_tokens is None else max_new_tokens
        )
        if not 0 <= max_new_tokens <= self.config.max_output_length:
            raise ValueError("max_new_tokens exceeds the configured output limit")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        was_training = self.training
        self.eval()
        try:
            memory = self.encode(input_ids, attention_mask)
            return [
                self._beam_search(
                    memory[i:i + 1], attention_mask[i:i + 1],
                    max_new_tokens, num_beams, return_all=True,
                )
                for i in range(input_ids.shape[0])
            ]
        finally:
            self.train(was_training)

    def _beam_search(self, memory, attention_mask, max_new_tokens, num_beams,
                     return_all=False):
        sequence = torch.full(
            (1,), self.config.bos_token_id, dtype=torch.long, device=memory.device
        )
        beams = [(0.0, sequence, False)]
        for _ in range(max_new_tokens):
            candidates = []
            for score, tokens, finished in beams:
                if finished:
                    candidates.append((score, tokens, True))
                    continue
                states = self.decode(
                    tokens.unsqueeze(0), memory, attention_mask,
                    decoder_attention_mask=tokens.unsqueeze(0).ne(self.config.pad_token_id),
                )
                log_probs = self._generation_log_probs(states[:, -1])
                values, indices = log_probs.topk(num_beams, dim=-1)
                for value, index in zip(values[0].tolist(), indices[0].tolist()):
                    if not math.isfinite(value):
                        continue
                    child = torch.cat([tokens, tokens.new_tensor([index])])
                    candidates.append((
                        score + value, child, index == self.config.eos_token_id
                    ))
            candidates.sort(key=lambda item: (-item[0], item[1].tolist()))
            beams = candidates[:num_beams]
            if all(item[2] for item in beams):
                break
        if return_all:
            return [
                {"input_ids": tokens.tolist(), "score": float(score)}
                for score, tokens, _ in beams
            ]
        return beams[0][1]

    def backbone_state_dict(self):
        auxiliary_names = (
            "question_projection.", "schema_projection.",
            "node_classifier.", "edge_classifier.",
        )
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.state_dict().items()
            if not name.startswith(auxiliary_names)
        }

    def export_backbone(self, path=None):
        state = self.backbone_state_dict()
        config = (
            dataclasses.asdict(self.config) if dataclasses.is_dataclass(self.config)
            else dict(vars(self.config))
        )
        artifact = {"config": config, "model": state, "auxiliary": False}
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(artifact, path)
        return artifact

    def count_parameters(self, trainable_only=False):
        return sum(
            parameter.numel() for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def parameter_counts(self):
        prefixes = (
            "question_projection.", "schema_projection.",
            "node_classifier.", "edge_classifier.",
        )
        auxiliary = sum(
            parameter.numel() for name, parameter in self.named_parameters()
            if name.startswith(prefixes)
        )
        total = self.count_parameters()
        return {"backbone": total - auxiliary, "auxiliary": auxiliary, "total": total}
