from __future__ import annotations

from typing import Any


try:  # Keep the core package importable without the optional Mamba dependencies.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in environments without torch
    torch = None
    nn = None


def require_torch():
    if torch is None or nn is None:
        raise RuntimeError(
            'PyTorch is required for the state encoder. Install with: pip install -e ".[mamba]"'
        )
    return torch, nn


_ModuleBase: Any = nn.Module if nn is not None else object


class ReferenceSelectiveStateBlock(_ModuleBase):
    """Small causal selective state block for the Windows/CPU research pilot."""

    def __init__(self, d_model: int, d_state: int, dropout: float):
        require_torch()
        super().__init__()
        self.input_state = nn.Linear(d_model, d_state)
        self.decay = nn.Linear(d_model, d_state)
        self.state_to_model = nn.Linear(d_state, d_model)
        self.output_gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_state = d_state

    def forward(self, inputs):
        batch = inputs.shape[0]
        state = inputs.new_zeros((batch, self.d_state))
        outputs = []
        for step in range(inputs.shape[1]):
            current = inputs[:, step, :]
            decay = torch.sigmoid(self.decay(current))
            candidate = torch.tanh(self.input_state(current))
            state = decay * state + (1.0 - decay) * candidate
            readout = self.state_to_model(state)
            gated = torch.sigmoid(self.output_gate(current)) * readout
            outputs.append(self.norm(current + self.dropout(gated)))
        return torch.stack(outputs, dim=1)


class ReferenceSelectiveStateEncoder(_ModuleBase):
    """Causal state encoder; deliberately excludes Graph and explicit feature gates."""

    def __init__(
        self,
        input_dim: int,
        *,
        d_model: int,
        d_state: int,
        layers: int,
        embedding_dim: int,
        dropout: float,
    ):
        require_torch()
        super().__init__()
        self.input_dim = input_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim * 2, d_model), nn.SiLU(), nn.LayerNorm(d_model)
        )
        self.blocks = nn.ModuleList([
            ReferenceSelectiveStateBlock(d_model, d_state, dropout) for _ in range(layers)
        ])
        self.embedding_head = nn.Sequential(
            nn.Linear(d_model, embedding_dim), nn.Tanh()
        )
        self.reconstruction_head = nn.Linear(d_model, input_dim)

    def hidden_sequence(self, values, valid_mask):
        hidden = self.input_projection(torch.cat([values, valid_mask], dim=-1))
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def forward(self, values, valid_mask):
        hidden = self.hidden_sequence(values, valid_mask)
        embedding = self.embedding_head(hidden[:, -1, :])
        reconstruction = self.reconstruction_head(hidden)
        return embedding, reconstruction

    def encode(self, values, valid_mask):
        return self.forward(values, valid_mask)[0]


def encoder_from_config(input_dim: int, config):
    if config.backend != "torch_reference":  # pragma: no cover - schema prevents this
        raise ValueError(f"unsupported encoder backend: {config.backend}")
    return ReferenceSelectiveStateEncoder(
        input_dim,
        d_model=config.d_model,
        d_state=config.d_state,
        layers=config.layers,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    )
