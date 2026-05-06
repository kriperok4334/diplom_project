from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_HISTORY_LENGTH = 5


def get_predictor_feature_columns() -> list[str]:
    """
    Возвращает фиксированный список признаков одного interface_window,
    которые используются как вход и цель для LSTM-предиктора этапа 2.
    """
    return [
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
    ]


def validate_interface_windows_df(
    interface_windows_df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """
    Проверяет, что таблица окон содержит все обязательные колонки,
    нужные для построения последовательностного датасета.
    """
    required_columns = {
        "device_id",
        "interface_name",
        "window_start",
        "window_end",
        *feature_columns,
    }

    missing = sorted(required_columns - set(interface_windows_df.columns))
    if missing:
        raise ValueError(
            "В interface_windows_df отсутствуют обязательные колонки: "
            f"{missing}"
        )

    if interface_windows_df.empty:
        raise ValueError("interface_windows_df пустой, последовательности построить нельзя.")


def _has_missing_values(df: pd.DataFrame, feature_columns: list[str]) -> bool:
    """
    Проверяет, есть ли пропуски в признаках конкретной последовательности.
    """
    return bool(df[feature_columns].isna().any().any())


def _build_sample_metadata(
    target_df: pd.DataFrame,
    input_slice_df: pd.DataFrame,
    target_row: pd.Series,
    history_length: int,
) -> dict[str, Any]:
    """
    Формирует метаданные для одного sequence sample.
    """
    first_row = input_slice_df.iloc[0]
    last_row = input_slice_df.iloc[-1]

    return {
        "device_id": str(target_row["device_id"]),
        "interface_name": str(target_row["interface_name"]),
        "history_length": history_length,
        "sequence_first_window_start": first_row["window_start"],
        "sequence_last_window_end": last_row["window_end"],
        "target_window_start": target_row["window_start"],
        "target_window_end": target_row["window_end"],
        "target_index_in_group": int(target_row.name),
        "group_window_count": int(len(target_df)),
    }


def build_sequence_samples_for_target(
    target_df: pd.DataFrame,
    feature_columns: list[str],
    history_length: int = DEFAULT_HISTORY_LENGTH,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    """
    Строит последовательности для одного интерфейса.

    На входе должен быть DataFrame только для одного device_id + interface_name,
    уже содержащий окна interface_window.

    Возвращает:
    - список X-примеров формы [history_length, feature_count]
    - список y-примеров формы [feature_count]
    - список metadata-словарей
    """
    if target_df.empty:
        return [], [], []

    work_df = target_df.copy()
    work_df["window_start"] = pd.to_datetime(work_df["window_start"], errors="coerce")
    work_df["window_end"] = pd.to_datetime(work_df["window_end"], errors="coerce")

    work_df = work_df.dropna(subset=["window_start", "window_end"])
    work_df = work_df.sort_values("window_start").reset_index(drop=True)

    min_required_windows = history_length + 1
    if len(work_df) < min_required_windows:
        return [], [], []

    x_samples: list[np.ndarray] = []
    y_samples: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []

    for target_pos in range(history_length, len(work_df)):
        input_start = target_pos - history_length
        input_end = target_pos

        input_slice_df = work_df.iloc[input_start:input_end].copy()
        target_row = work_df.iloc[target_pos].copy()

        if _has_missing_values(input_slice_df, feature_columns):
            continue

        if target_row[feature_columns].isna().any():
            continue

        x_sample = input_slice_df[feature_columns].to_numpy(dtype=np.float32)
        y_sample = target_row[feature_columns].to_numpy(dtype=np.float32)

        metadata = _build_sample_metadata(
            target_df=work_df,
            input_slice_df=input_slice_df,
            target_row=target_row,
            history_length=history_length,
        )

        x_samples.append(x_sample)
        y_samples.append(y_sample)
        metadata_rows.append(metadata)

    return x_samples, y_samples, metadata_rows


def build_lstm_dataset(
    interface_windows_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    history_length: int = DEFAULT_HISTORY_LENGTH,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Главная функция подготовки датасета для LSTM.

    Шаги:
    1. Проверяет входную таблицу.
    2. Группирует окна по device_id + interface_name.
    3. Для каждой группы строит последовательности длины history_length + 1.
    4. Возвращает:
       - X формы [num_samples, history_length, feature_count]
       - y формы [num_samples, feature_count]
       - metadata_df с описанием каждого sample
    """
    if feature_columns is None:
        feature_columns = get_predictor_feature_columns()

    validate_interface_windows_df(interface_windows_df, feature_columns)

    work_df = interface_windows_df.copy()
    work_df["window_start"] = pd.to_datetime(work_df["window_start"], errors="coerce")
    work_df["window_end"] = pd.to_datetime(work_df["window_end"], errors="coerce")

    all_x_samples: list[np.ndarray] = []
    all_y_samples: list[np.ndarray] = []
    all_metadata_rows: list[dict[str, Any]] = []

    grouped = work_df.groupby(["device_id", "interface_name"], sort=True)

    for (_, _), group_df in grouped:
        x_samples, y_samples, metadata_rows = build_sequence_samples_for_target(
            target_df=group_df,
            feature_columns=feature_columns,
            history_length=history_length,
        )

        all_x_samples.extend(x_samples)
        all_y_samples.extend(y_samples)
        all_metadata_rows.extend(metadata_rows)

    feature_count = len(feature_columns)

    if not all_x_samples:
        empty_x = np.empty((0, history_length, feature_count), dtype=np.float32)
        empty_y = np.empty((0, feature_count), dtype=np.float32)
        empty_metadata_df = pd.DataFrame(
            columns=[
                "device_id",
                "interface_name",
                "history_length",
                "sequence_first_window_start",
                "sequence_last_window_end",
                "target_window_start",
                "target_window_end",
                "target_index_in_group",
                "group_window_count",
            ]
        )
        return empty_x, empty_y, empty_metadata_df

    X = np.stack(all_x_samples).astype(np.float32)
    y = np.stack(all_y_samples).astype(np.float32)
    metadata_df = pd.DataFrame(all_metadata_rows)

    return X, y, metadata_df
