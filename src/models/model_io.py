from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.lstm_predictor import LSTMNextWindowPredictor, build_lstm_next_window_predictor


def save_predictor_model(
    model: LSTMNextWindowPredictor,
    output_path: str | Path,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """
    Сохраняет обученную модель и её конфигурацию.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "model_config": model.get_model_config(),
        "extra_metadata": extra_metadata or {},
    }

    torch.save(payload, output_path)



def load_predictor_model(
    model_path: str | Path,
    device: str = "cpu",
) -> LSTMNextWindowPredictor:
    """
    Загружает сохранённую модель предиктора для рабочего проекта.
    """
    model_path = Path(model_path)
    payload = torch.load(model_path, map_location=device)

    model_config = payload["model_config"]
    model = build_lstm_next_window_predictor(**model_config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    return model
