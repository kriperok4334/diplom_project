from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loaders import load_all_input_tables
from src.data.source_adapters import adapt_generic_telemetry_to_normalized
from src.features.interface_features import build_interface_windows_dataset
from src.features.sequence_features import (
    DEFAULT_HISTORY_LENGTH,
    build_lstm_dataset,
    get_predictor_feature_columns,
)
from training.training_utils import TrainConfig, train_lstm_predictor


METRICS_PATH = Path("data/raw/interface_metrics.csv")
EVENTS_PATH = Path("data/raw/interface_events.csv")
CONTEXT_PATH = Path("data/raw/device_context.csv")

MODEL_OUTPUT_PATH = Path("artifacts/lstm_next_window_predictor.pt")
WINDOW_SIZE_MINUTES = 5



def normalize_training_input(
    metrics_path: str | Path | None,
    events_path: str | Path | None,
    context_path: str | Path | None,
):
    """
    Подготовительный вход обучения.
    Обучение находится вне рабочего проекта и может использовать свой adapter.
    """
    return load_all_input_tables(
        metrics_path=metrics_path,
        events_path=events_path,
        context_path=context_path,
        adapter=lambda metrics_df, events_df, context_df: adapt_generic_telemetry_to_normalized(
            metrics_df=metrics_df,
            events_df=events_df,
            context_df=context_df,
            source_name="training_input",
        ),
    )



def get_available_targets(normalized_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает все доступные пары device_id + interface_name.
    """
    required_cols = {"device_id", "interface_name"}
    missing = required_cols - set(normalized_metrics_df.columns)
    if missing:
        raise ValueError(
            f"В normalized_metrics_df отсутствуют обязательные колонки: {sorted(missing)}"
        )

    targets_df = (
        normalized_metrics_df[["device_id", "interface_name"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["device_id", "interface_name"])
        .reset_index(drop=True)
    )
    return targets_df



def build_windows_spec_for_target(
    normalized_metrics_df: pd.DataFrame,
    device_id: str,
    interface_name: str,
    window_size_minutes: int,
) -> list[dict[str, Any]]:
    """
    Строит список окон анализа для одного объекта.
    """
    interface_df = normalized_metrics_df.loc[
        (normalized_metrics_df["device_id"].astype("string") == str(device_id))
        & (normalized_metrics_df["interface_name"].astype("string") == str(interface_name))
    ].copy()

    if interface_df.empty:
        return []

    interface_df["timestamp"] = pd.to_datetime(interface_df["timestamp"], errors="coerce")
    interface_df = interface_df.dropna(subset=["timestamp"]).sort_values("timestamp")

    if interface_df.empty:
        return []

    min_ts = interface_df["timestamp"].min()
    max_ts = interface_df["timestamp"].max()

    if pd.isna(min_ts) or pd.isna(max_ts):
        return []

    window_delta = pd.Timedelta(minutes=window_size_minutes)

    windows_spec: list[dict[str, Any]] = []
    current_start = min_ts

    while current_start <= max_ts:
        current_end = current_start + window_delta

        windows_spec.append(
            {
                "device_id": str(device_id),
                "interface_name": str(interface_name),
                "window_start": current_start,
                "window_end": current_end,
            }
        )

        current_start = current_end

    return windows_spec



def build_windows_spec_for_all_targets(
    normalized_metrics_df: pd.DataFrame,
    window_size_minutes: int,
) -> list[dict[str, Any]]:
    """
    Строит список окон для всех доступных объектов.
    """
    targets_df = get_available_targets(normalized_metrics_df)

    all_windows_spec: list[dict[str, Any]] = []

    for _, row in targets_df.iterrows():
        device_id = str(row["device_id"])
        interface_name = str(row["interface_name"])

        target_windows = build_windows_spec_for_target(
            normalized_metrics_df=normalized_metrics_df,
            device_id=device_id,
            interface_name=interface_name,
            window_size_minutes=window_size_minutes,
        )
        all_windows_spec.extend(target_windows)

    return all_windows_spec



def print_training_summary(
    windows_df: pd.DataFrame,
    X,
    y,
    metadata_df: pd.DataFrame,
) -> None:
    """
    Печатает краткую сводку по датасету обучения.
    """
    unique_targets = windows_df[["device_id", "interface_name"]].drop_duplicates().shape[0]

    print("=== TRAINING DATA SUMMARY ===")
    print(f"Targets available: {unique_targets}")
    print(f"Windows available: {len(windows_df)}")
    print(f"LSTM samples built: {len(X)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Metadata rows: {len(metadata_df)}")



def main() -> None:
    """
    Подготовительный training pipeline:
    1. Загружаем и нормализуем телеметрию.
    2. Строим interface_window dataset.
    3. Строим последовательности для LSTM.
    4. Обучаем модель.
    5. Сохраняем модель в artifacts/.

    Этот файл относится к отдельному контуру подготовки инструмента,
    а не к рабочему проекту напрямую.
    """
    normalized_metrics_df, normalized_events_df, normalized_context_df = normalize_training_input(
        metrics_path=METRICS_PATH,
        events_path=EVENTS_PATH,
        context_path=CONTEXT_PATH,
    )

    windows_spec = build_windows_spec_for_all_targets(
        normalized_metrics_df=normalized_metrics_df,
        window_size_minutes=WINDOW_SIZE_MINUTES,
    )

    if not windows_spec:
        print("Не удалось сформировать окна анализа для обучения.")
        return

    windows_df = build_interface_windows_dataset(
        metrics_df=normalized_metrics_df,
        events_df=normalized_events_df,
        context_df=normalized_context_df,
        windows_spec=windows_spec,
    )

    if windows_df.empty:
        print("Не удалось собрать interface_window dataset для обучения.")
        return

    feature_columns = get_predictor_feature_columns()

    X, y, metadata_df = build_lstm_dataset(
        interface_windows_df=windows_df,
        feature_columns=feature_columns,
        history_length=DEFAULT_HISTORY_LENGTH,
    )

    if len(X) == 0:
        print("Не удалось построить LSTM-датасет: недостаточно окон или есть пропуски.")
        return

    print_training_summary(
        windows_df=windows_df,
        X=X,
        y=y,
        metadata_df=metadata_df,
    )

    train_config = TrainConfig(
        batch_size=16,
        num_epochs=10,
        learning_rate=1e-3,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        device="cpu",
    )

    train_result = train_lstm_predictor(
        X=X,
        y=y,
        metadata_df=metadata_df,
        config=train_config,
        model_output_path=MODEL_OUTPUT_PATH,
    )

    print("\n=== TRAINING FINISHED ===")
    print(f"Best val loss: {train_result['history'].get('best_val_loss')}")
    print(f"Test loss: {train_result['history'].get('test_loss')}")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
