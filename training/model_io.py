from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from training.lstm_predictor import (
    LSTMNextWindowPredictor,
    build_lstm_next_window_predictor,
)
from training.scalers import SequenceScaler


ARTIFACT_VERSION = "1.0"


def save_predictor_model(
    model: LSTMNextWindowPredictor,
    output_path: str | Path,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """
    Сохраняет внешний продукт нейросети:
    - веса модели
    - конфиг модели
    - дополнительные метаданные (scaler, feature_columns, history_length и т.д.)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "state_dict": model.state_dict(),
        "model_config": model.get_model_config(),
        "extra_metadata": extra_metadata or {},
    }

    torch.save(payload, output_path)


def load_predictor_bundle(
    model_path: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Загружает полный bundle внешнего продукта нейросети.
    """
    model_path = Path(model_path)
    payload = torch.load(model_path, map_location=device)

    model_config = payload["model_config"]
    model = build_lstm_next_window_predictor(**model_config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    extra_metadata = payload.get("extra_metadata", {})

    x_scaler_payload = extra_metadata.get("x_scaler")
    y_scaler_payload = extra_metadata.get("y_scaler")

    x_scaler = (
        SequenceScaler.from_dict(x_scaler_payload)
        if x_scaler_payload is not None
        else None
    )
    y_scaler = (
        SequenceScaler.from_dict(y_scaler_payload)
        if y_scaler_payload is not None
        else None
    )

    bundle = {
        "artifact_version": payload.get("artifact_version", "unknown"),
        "model": model,
        "model_config": model_config,
        "feature_columns": extra_metadata.get("feature_columns"),
        "history_length": extra_metadata.get("history_length"),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "metadata": extra_metadata,
    }

    return bundle


def load_predictor_model(
    model_path: str | Path,
    device: str = "cpu",
) -> LSTMNextWindowPredictor:
    """
    Обратная совместимость:
    если нужен только объект модели без полного bundle.
    """
    bundle = load_predictor_bundle(
        model_path=model_path,
        device=device,
    )
    return bundle["model"]