from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from src.features.sequence_features import (
    DEFAULT_HISTORY_LENGTH,
    get_predictor_feature_columns,
)
from training.scalers import inverse_transform_y, transform_X


def validate_history_for_inference(
    history_df: pd.DataFrame,
    feature_columns: list[str],
    history_length: int = DEFAULT_HISTORY_LENGTH,
) -> None:
    """
    Проверяет, что история окон подходит для инференса.
    """
    required_columns = {
        "device_id",
        "interface_name",
        "window_start",
        "window_end",
        *feature_columns,
    }

    missing = sorted(required_columns - set(history_df.columns))
    if missing:
        raise ValueError(
            "В history_df отсутствуют обязательные колонки: "
            f"{missing}"
        )

    if history_df.empty:
        raise ValueError("history_df пустой, инференс невозможен.")

    if len(history_df) < history_length:
        raise ValueError(
            f"Недостаточно окон для инференса: "
            f"len(history_df)={len(history_df)} < history_length={history_length}"
        )

    latest_history = history_df.sort_values("window_start").tail(history_length)

    if latest_history[feature_columns].isna().any().any():
        raise ValueError(
            "В последних окнах истории есть пропуски в признаках, "
            "инференс невозможен."
        )


def prepare_sequence_array_for_inference(
    history_df: pd.DataFrame,
    feature_columns: list[str],
    history_length: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Готовит последние history_length окон для инференса.
    """
    work_df = history_df.copy()
    work_df["window_start"] = pd.to_datetime(work_df["window_start"], errors="coerce")
    work_df["window_end"] = pd.to_datetime(work_df["window_end"], errors="coerce")
    work_df = work_df.dropna(subset=["window_start", "window_end"])
    work_df = work_df.sort_values("window_start").reset_index(drop=True)

    validate_history_for_inference(
        history_df=work_df,
        feature_columns=feature_columns,
        history_length=history_length,
    )

    latest_history_df = work_df.tail(history_length).copy()

    x_array = latest_history_df[feature_columns].to_numpy(dtype=np.float32)
    x_array = np.expand_dims(x_array, axis=0)

    return x_array, latest_history_df


def normalize_input_sequence(
    x_array: np.ndarray,
    x_scaler,
) -> np.ndarray:
    """
    Нормализует входную последовательность.
    """
    if x_scaler is None:
        return x_array.astype(np.float32)

    return transform_X(x_array, x_scaler)


@torch.no_grad()
def predict_next_window_features(
    predictor_bundle: dict[str, Any],
    input_sequence: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Прогоняет последовательность через модель и возвращает
    денормализованный прогнозный вектор окна.
    """
    model = predictor_bundle["model"]
    y_scaler = predictor_bundle.get("y_scaler")

    model = model.to(device)
    model.eval()

    input_tensor = torch.tensor(input_sequence, dtype=torch.float32, device=device)
    y_pred = model(input_tensor)

    if y_pred.ndim != 2 or y_pred.shape[0] != 1:
        raise ValueError(
            f"Ожидался выход формы [1, feature_count], получено {tuple(y_pred.shape)}"
        )

    y_pred_np = y_pred.detach().cpu().numpy().astype(np.float32)

    if y_scaler is not None:
        y_pred_np = inverse_transform_y(y_pred_np, y_scaler)

    return y_pred_np[0]


def _clamp_predicted_value(feature_name: str, value: float) -> float:
    """
    Ограничивает явно невозможные отрицательные значения.
    """
    non_negative_features = {
        "status_change_count",
        "down_seconds_total",
        "errors_total_delta",
        "discards_total_delta",
        "packet_loss_avg_pct",
        "packet_loss_max_pct",
        "latency_avg_ms",
        "latency_max_ms",
        "utilization_in_avg_pct",
        "utilization_out_avg_pct",
        "utilization_peak_pct",
        "device_cpu_avg_pct",
        "device_memory_avg_pct",
    }

    if feature_name in non_negative_features:
        return max(0.0, float(value))

    return float(value)


def _postprocess_predicted_vector(
    predicted_vector: np.ndarray,
    feature_columns: list[str],
) -> dict[str, float]:
    """
    Преобразует выходной вектор модели в словарь признаков следующего окна.
    """
    if len(predicted_vector) != len(feature_columns):
        raise ValueError(
            "Длина predicted_vector не совпадает с числом feature_columns: "
            f"{len(predicted_vector)} != {len(feature_columns)}"
        )

    result: dict[str, float] = {}
    for feature_name, value in zip(feature_columns, predicted_vector):
        result[f"predicted_{feature_name}"] = _clamp_predicted_value(
            feature_name=feature_name,
            value=float(value),
        )

    return result


def build_predicted_next_window(
    latest_history_df: pd.DataFrame,
    predicted_vector: np.ndarray,
    feature_columns: list[str],
) -> dict[str, Any]:
    """
    Собирает объект predicted_next_window.
    """
    if latest_history_df.empty:
        raise ValueError("latest_history_df пустой, predicted_next_window собрать нельзя.")

    latest_history_df = latest_history_df.sort_values("window_start").reset_index(drop=True)
    last_row = latest_history_df.iloc[-1]

    last_window_start = pd.Timestamp(last_row["window_start"])
    last_window_end = pd.Timestamp(last_row["window_end"])
    window_delta = last_window_end - last_window_start

    predicted_next_window_start = last_window_end
    predicted_next_window_end = last_window_end + window_delta

    predicted_features = _postprocess_predicted_vector(
        predicted_vector=predicted_vector,
        feature_columns=feature_columns,
    )

    predicted_window = {
        "device_id": last_row.get("device_id"),
        "device_name": last_row.get("device_name", pd.NA),
        "device_vendor": last_row.get("device_vendor", pd.NA),
        "device_model": last_row.get("device_model", pd.NA),
        "interface_id": last_row.get("interface_id", pd.NA),
        "interface_name": last_row.get("interface_name"),
        "interface_role": last_row.get("interface_role", pd.NA),
        "predicted_next_window_start": predicted_next_window_start,
        "predicted_next_window_end": predicted_next_window_end,
        "history_length_used": int(len(latest_history_df)),
        **predicted_features,
    }

    return predicted_window


def predict_next_window_from_history(
    predictor_bundle: dict[str, Any],
    history_df: pd.DataFrame,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Полный inference-проход через bundle.
    """
    feature_columns = predictor_bundle.get("feature_columns")
    history_length = predictor_bundle.get("history_length")
    x_scaler = predictor_bundle.get("x_scaler")

    if feature_columns is None:
        feature_columns = get_predictor_feature_columns()

    if history_length is None:
        history_length = DEFAULT_HISTORY_LENGTH

    x_array, latest_history_df = prepare_sequence_array_for_inference(
        history_df=history_df,
        feature_columns=feature_columns,
        history_length=history_length,
    )

    x_array = normalize_input_sequence(
        x_array=x_array,
        x_scaler=x_scaler,
    )

    predicted_vector = predict_next_window_features(
        predictor_bundle=predictor_bundle,
        input_sequence=x_array,
        device=device,
    )

    predicted_next_window = build_predicted_next_window(
        latest_history_df=latest_history_df,
        predicted_vector=predicted_vector,
        feature_columns=feature_columns,
    )

    return predicted_next_window