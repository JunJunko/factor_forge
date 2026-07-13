from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .mamba_state_model import encoder_from_config, require_torch


@dataclass(frozen=True)
class EncoderFitResult:
    model: object
    history: pd.DataFrame
    checkpoint_path: Path
    checkpoint_sha256: str
    seed: int
    device: str


def fit_reference_encoder(
    store,
    train_positions: np.ndarray,
    valid_positions: np.ndarray,
    encoder_config,
    training_config,
    *,
    seed: int,
    checkpoint_path: str | Path,
) -> EncoderFitResult:
    torch, _ = require_torch()
    _seed_everything(seed, torch)
    device = _device(training_config.device, torch)
    model = encoder_from_config(len(store.feature_names), encoder_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    train_positions = _subsample(train_positions, training_config.max_train_samples, seed)
    valid_positions = _subsample(valid_positions, training_config.max_valid_samples, seed + 1)
    if len(train_positions) == 0 or len(valid_positions) == 0:
        raise ValueError("encoder train and valid segments must contain sequence samples")

    rows, best_loss, best_state, bad_epochs = [], float("inf"), None, 0
    for epoch in range(training_config.epochs):
        train_loss = _epoch(
            model, store, train_positions, training_config.batch_size,
            encoder_config.mask_probability, device, optimizer, seed + epoch * 1009,
        )
        valid_loss = _epoch(
            model, store, valid_positions, training_config.batch_size,
            encoder_config.mask_probability, device, None, seed + 10_000,
        )
        rows.append({"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss})
        if valid_loss < best_loss - 1e-8:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= training_config.patience:
                break
    if best_state is None:  # pragma: no cover - finite first epoch should always win
        raise RuntimeError("encoder training produced no finite checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "input_dim": len(store.feature_names),
        "feature_names": store.feature_names,
        "encoder_config": encoder_config.model_dump(mode="json"),
        "seed": seed,
        "best_valid_loss": best_loss,
    }, checkpoint_path)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    return EncoderFitResult(
        model=model, history=pd.DataFrame(rows), checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash, seed=seed, device=device,
    )


def encode_sequences(model, store, positions: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    torch, _ = require_torch()
    model.eval()
    output = []
    with torch.no_grad():
        for batch_positions in _batches(positions, batch_size, shuffle=False, seed=0):
            values, mask = store.take(batch_positions)
            embedding = model.encode(
                torch.from_numpy(values).to(device), torch.from_numpy(mask).to(device)
            )
            output.append(embedding.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(output, axis=0) if output else np.empty((0, 0), dtype=np.float32)


def _epoch(model, store, positions, batch_size, mask_probability, device, optimizer, seed):
    torch, _ = require_torch()
    training = optimizer is not None
    model.train(training)
    losses = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for batch_positions in _batches(positions, batch_size, shuffle=training, seed=seed):
        values_np, valid_np = store.take(batch_positions)
        values = torch.from_numpy(values_np).to(device)
        valid = torch.from_numpy(valid_np).to(device)
        random_values = torch.rand(valid.shape, generator=generator).to(device)
        masked = (random_values < mask_probability) & valid.bool()
        if not masked.any():
            masked[:, -1, 0] = valid[:, -1, 0].bool()
        corrupted_values = values.masked_fill(masked, 0.0)
        corrupted_valid = valid.masked_fill(masked, 0.0)
        _, reconstruction = model(corrupted_values, corrupted_valid)
        squared = (reconstruction - values).square()
        denominator = masked.sum().clamp_min(1)
        loss = squared.masked_select(masked).sum() / denominator
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _batches(positions, batch_size, *, shuffle, seed):
    values = np.asarray(positions, dtype=np.int64).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(values)
    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]


def _subsample(positions, maximum, seed):
    values = np.asarray(positions, dtype=np.int64)
    if maximum is None or len(values) <= maximum:
        return values
    chosen = np.random.default_rng(seed).choice(values, size=maximum, replace=False)
    return np.sort(chosen)


def _seed_everything(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device(requested, torch):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device=cuda but CUDA is unavailable")
    return "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"
