from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .event_mamba_model import event_mamba_from_config
from .mamba_state_model import require_torch


@dataclass(frozen=True)
class EventMambaFitResult:
    model: object
    history: pd.DataFrame
    checkpoint_path: Path
    checkpoint_sha256: str
    device: str


def fit_event_mamba(store, train, valid, config, *, checkpoint_path) -> EventMambaFitResult:
    torch, _ = require_torch()
    seed = config.training.random_seeds[0]
    _seed(seed, torch)
    device = _device(config.training.device, torch)
    model = event_mamba_from_config(
        len(store.feature_names), len(config.event.factor_basis),
        len(config.event_templates), config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    rows, best_loss, best_state, bad_epochs = [], float("inf"), None, 0
    for epoch in range(config.training.epochs):
        train_loss = _epoch(model, store, train, config, device, optimizer, seed + epoch * 997)
        valid_loss = _epoch(model, store, valid, config, device, None, seed + 10000)
        rows.append({"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss})
        if valid_loss < best_loss - 1e-8:
            best_loss, best_state, bad_epochs = valid_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.training.patience:
                break
    if best_state is None:
        raise RuntimeError("Event-Mamba produced no valid checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state, "feature_names": store.feature_names,
        "factor_names": config.event.factor_basis,
        "template_paths": [str(path) for path in config.event_templates],
        "encoder_config": config.encoder.model_dump(mode="json"),
        "head_config": config.event_mamba.model_dump(mode="json"),
        "best_valid_loss": best_loss,
    }, checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return EventMambaFitResult(
        model=model, history=pd.DataFrame(rows), checkpoint_path=checkpoint_path,
        checkpoint_sha256=digest, device=device,
    )


def encode_event_features(model, store, events, config, *, device):
    torch, _ = require_torch()
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in _batches(events, config.training.batch_size, shuffle=False, seed=0):
            values, mask = store.take(batch["sample_position"].to_numpy(int))
            result = model(
                torch.from_numpy(values).to(device), torch.from_numpy(mask).to(device),
                torch.from_numpy(batch["template_index"].to_numpy(np.int64)).to(device),
                torch.from_numpy(batch["severity"].to_numpy(np.float32)).to(device),
                torch.from_numpy(batch[config.event.factor_basis].to_numpy(np.float32)).to(device),
            )
            outputs.append({
                name: result[name].detach().cpu().numpy().astype(np.float32)
                for name in ("prediction", "beta", "gated", "residual_embedding")
            })
    if not outputs:
        return {"prediction": np.empty(0), "beta": np.empty((0, len(config.event.factor_basis))),
                "gated": np.empty((0, len(config.event.factor_basis))),
                "residual_embedding": np.empty((0, config.event_mamba.residual_embedding_dim))}
    return {name: np.concatenate([row[name] for row in outputs], axis=0) for name in outputs[0]}


def _epoch(model, store, events, config, device, optimizer, seed):
    torch, _ = require_torch()
    training = optimizer is not None
    model.train(training)
    losses = []
    for batch in _batches(events, config.training.batch_size, shuffle=training, seed=seed):
        values_np, mask_np = store.take(batch["sample_position"].to_numpy(int))
        values = torch.from_numpy(values_np).to(device)
        mask = torch.from_numpy(mask_np).to(device)
        corrupted, corrupted_mask, masked = _mask(values, mask, config.encoder.mask_probability, seed, torch)
        result = model(
            corrupted, corrupted_mask,
            torch.from_numpy(batch["template_index"].to_numpy(np.int64)).to(device),
            torch.from_numpy(batch["severity"].to_numpy(np.float32)).to(device),
            torch.from_numpy(batch[config.event.factor_basis].to_numpy(np.float32)).to(device),
        )
        target = torch.from_numpy(batch["target"].to_numpy(np.float32)).to(device)
        weights = torch.from_numpy(batch["sample_weight"].to_numpy(np.float32)).to(device)
        supervised = torch.nn.functional.smooth_l1_loss(
            result["prediction"], target, reduction="none"
        )
        supervised = (supervised * weights).sum() / weights.sum().clamp_min(1e-6)
        reconstruction = (result["reconstruction"] - values).square().masked_select(masked).mean()
        beta_l1 = result["beta"].abs().mean()
        loss = (
            config.event_mamba.supervised_loss_weight * supervised
            + config.event_mamba.reconstruction_loss_weight * reconstruction
            + config.event_mamba.beta_l1_weight * beta_l1
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _mask(values, mask, probability, seed, torch):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_values = torch.rand(mask.shape, generator=generator).to(values.device)
    masked = (random_values < probability) & mask.bool()
    if not masked.any():
        masked[:, -1, 0] = mask[:, -1, 0].bool()
    return values.masked_fill(masked, 0.0), mask.masked_fill(masked, 0.0), masked


def _batches(frame, batch_size, *, shuffle, seed):
    order = np.arange(len(frame))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield frame.iloc[order[start:start + batch_size]]


def _seed(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device(requested, torch):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device=cuda but CUDA is unavailable")
    return "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"
