"""Run the pinned TabPFN regression model on engineered features."""

import hashlib
from pathlib import Path

import numpy as np


def verify_checkpoint(path: Path, expected_sha256: str):
    with path.open("rb") as checkpoint:
        actual_sha256 = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Checkpoint SHA-256 mismatch for {path}. Use the pinned model revision.")


def predict_log_prices(train, log_prices, other, checkpoint, device, settings):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    torch.set_num_threads(settings["torch_threads"])
    torch.manual_seed(settings["random_state"])
    np.random.seed(settings["random_state"])

    model = TabPFNRegressor.create_default_for_version(
        ModelVersion(settings["version"]),
        model_path=checkpoint,
        device=device,
        n_estimators=settings["n_estimators"],
        inference_precision=getattr(torch, settings["inference_precision"]),
        random_state=settings["random_state"],
        n_preprocessing_jobs=settings["preprocessing_jobs"],
        ignore_pretraining_limits=settings["ignore_pretraining_limits"],
        show_progress_bar=False,
    )
    center = float(log_prices.mean())
    model.fit(train, log_prices - center)
    predictions = model.predict(other, output_type=settings["output_type"])
    predictions = np.asarray(predictions, dtype=float).reshape(-1) + center
    if len(predictions) != len(other) or not np.isfinite(predictions).all():
        raise ValueError("Model output must contain one finite log prediction per input row.")
    return predictions
