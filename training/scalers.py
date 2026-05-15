from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


EPS = 1e-8


@dataclass
class SequenceScaler:
    """
    Простой standard scaler для numpy-массивов.

    Поддерживает:
    - X формы [num_samples, history_length, feature_count]
    - y формы [num_samples, feature_count]
    """

    mean: np.ndarray
    std: np.ndarray
    feature_axis: int

    def transform(self, array: np.ndarray) -> np.ndarray:
        return (array - self.mean) / np.maximum(self.std, EPS)

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        return array * np.maximum(self.std, EPS) + self.mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_axis": self.feature_axis,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SequenceScaler":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            feature_axis=int(payload["feature_axis"]),
        )


def fit_x_scaler(X_train: np.ndarray) -> SequenceScaler:
    """
    Считает scaler для X по train-части.

    X_train.shape = [num_samples, history_length, feature_count]
    Нормализация идёт по признакам, агрегируя все samples и timesteps.
    """
    if X_train.ndim != 3:
        raise ValueError(f"Ожидается X_train.ndim == 3, получено {X_train.ndim}")

    mean = X_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = X_train.std(axis=(0, 1), keepdims=True).astype(np.float32)

    return SequenceScaler(
        mean=mean,
        std=std,
        feature_axis=2,
    )


def fit_y_scaler(y_train: np.ndarray) -> SequenceScaler:
    """
    Считает scaler для y по train-части.

    y_train.shape = [num_samples, feature_count]
    """
    if y_train.ndim != 2:
        raise ValueError(f"Ожидается y_train.ndim == 2, получено {y_train.ndim}")

    mean = y_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = y_train.std(axis=0, keepdims=True).astype(np.float32)

    return SequenceScaler(
        mean=mean,
        std=std,
        feature_axis=1,
    )


def fit_sequence_scalers(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[SequenceScaler, SequenceScaler]:
    """
    Возвращает:
    - x_scaler
    - y_scaler
    """
    x_scaler = fit_x_scaler(X_train)
    y_scaler = fit_y_scaler(y_train)
    return x_scaler, y_scaler


def transform_X(X: np.ndarray, x_scaler: SequenceScaler) -> np.ndarray:
    """
    Нормализует X.
    """
    transformed = x_scaler.transform(X)
    return transformed.astype(np.float32)


def transform_y(y: np.ndarray, y_scaler: SequenceScaler) -> np.ndarray:
    """
    Нормализует y.
    """
    transformed = y_scaler.transform(y)
    return transformed.astype(np.float32)


def inverse_transform_y(y_pred: np.ndarray, y_scaler: SequenceScaler) -> np.ndarray:
    """
    Возвращает y из нормализованного пространства обратно в исходную шкалу.
    """
    restored = y_scaler.inverse_transform(y_pred)
    return restored.astype(np.float32)