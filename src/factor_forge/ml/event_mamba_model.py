from __future__ import annotations

import math
from typing import Any

from .mamba_state_model import encoder_from_config, require_torch


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


_ModuleBase: Any = nn.Module if nn is not None else object


class EventMambaSensitivityModel(_ModuleBase):
    """Causal sequence encoder with named event-factor sensitivity axes."""

    def __init__(self, input_dim, factor_dim, template_count, encoder_config, head_config):
        require_torch()
        super().__init__()
        self.backbone = encoder_from_config(input_dim, encoder_config)
        self.template_embedding = nn.Embedding(template_count, head_config.template_embedding_dim)
        context_dim = encoder_config.d_model + head_config.template_embedding_dim + 1
        self.context = nn.Sequential(
            nn.Linear(context_dim, encoder_config.d_model), nn.SiLU(),
            nn.LayerNorm(encoder_config.d_model),
        )
        self.beta_head = nn.Sequential(nn.Linear(encoder_config.d_model, factor_dim), nn.Tanh())
        self.intercept_head = nn.Linear(encoder_config.d_model, 1)
        self.residual_embedding_dim = head_config.residual_embedding_dim
        if self.residual_embedding_dim:
            self.residual_head = nn.Sequential(
                nn.Linear(encoder_config.d_model, self.residual_embedding_dim), nn.Tanh()
            )
            self.residual_score = nn.Linear(self.residual_embedding_dim, 1)
        else:
            self.residual_head = None
            self.residual_score = None
        self.factor_dim = factor_dim

    def forward(self, values, valid_mask, template_index, severity, factor_values):
        hidden = self.backbone.hidden_sequence(values, valid_mask)
        reconstruction = self.backbone.reconstruction_head(hidden)
        template = self.template_embedding(template_index)
        context = self.context(torch.cat([hidden[:, -1], template, severity[:, None]], dim=1))
        beta = self.beta_head(context)
        gated = beta * factor_values
        prediction = self.intercept_head(context).squeeze(1) + gated.sum(dim=1) / math.sqrt(self.factor_dim)
        if self.residual_head is not None:
            residual = self.residual_head(context)
            prediction = prediction + self.residual_score(residual).squeeze(1)
        else:
            residual = context.new_empty((len(context), 0))
        return {
            "prediction": prediction, "beta": beta, "gated": gated,
            "residual_embedding": residual, "reconstruction": reconstruction,
        }


def event_mamba_from_config(input_dim, factor_dim, template_count, config):
    return EventMambaSensitivityModel(
        input_dim, factor_dim, template_count, config.encoder, config.event_mamba,
    )
