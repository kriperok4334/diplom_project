from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loaders import load_all_input_tables
from src.features.interface_features import build_interface_windows_dataset
from src.features.sequence_features import (
    DEFAULT_HISTORY_LENGTH,
    build_lstm_dataset,
    get_predictor_feature_columns,
)
from src.inference.realtime_pipeline import run_realtime_cycle
from src.models.train_predictor import TrainConfig, train_lstm_predictor


METRICS_PATH = Path("data/raw/interface_metrics.csv")
EVENTS_PATH = Path("data/raw/interface_events.csv")
CONTEXT_PATH = Path("data/raw/device_context.csv")

# Источник можно менять: "generic", "zabbix", "noc"
SOURCE_TYPE = "generic"
SOURCE_NAME = "generic_csv"

MODEL_OUTPUT_PATH = Path("artifacts/lstm_next_window_predictor.pt")

WINDOW_SIZE_MINUTES = 5


def get_available_targets(normalized_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает все доступные уникальные пары device_id + interface_name.
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
    Строит список окон для одной пары device_id + interface_name
    на основе normalized_metrics_df.
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
    Строит список окон для всех доступных интерфейсов.
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


def print_data_summary(
    windows_df: pd.DataFrame,
    X,
    y,
    metadata_df: pd.DataFrame,
) -> None:
    """
    Краткая сводка по данным этапа 2.
    """
    unique_targets = windows_df[["device_id", "interface_name"]].drop_duplicates().shape[0]

    print("=== DATA SUMMARY ===")
    print(f"Targets available: {unique_targets}")
    print(f"Windows available: {len(windows_df)}")
    print(f"LSTM samples built: {len(X)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Metadata rows: {len(metadata_df)}")


def print_realtime_result(result: dict[str, Any]) -> None:
    """
    Печатает один результат realtime-цикла.
    """
    print("\n=== REALTIME RESULT ===")
    print(f"Device: {result.get('device_name')}")
    print(f"Interface: {result.get('interface_name')}")
    print(
        f"Current window: {result.get('current_window_start')} -> "
        f"{result.get('current_window_end')}"
    )
    print(f"Current state: {result.get('current_state_label')}")
    print(f"Current problem: {result.get('current_problem_type_label')}")
    print(f"Current comment: {result.get('current_comment')}")

    if result.get("predicted_next_window_start") is not None:
        print(
            f"Predicted next window: {result.get('predicted_next_window_start')} -> "
            f"{result.get('predicted_next_window_end')}"
        )
        print(f"Predicted next state: {result.get('predicted_next_state_label')}")
        print(f"Predicted next problem: {result.get('predicted_next_problem_type_label')}")
        print(f"Predicted next comment: {result.get('predicted_next_comment')}")
    else:
        print("Predicted next window: insufficient history")

    print(f"History length used: {result.get('history_length_used')}")


def main() -> None:
    """
    Актуальная точка входа проекта для текущего этапа разработки.

    Алгоритм:
    1. Загружаем и нормализуем телеметрию.
    2. Строим interface_window dataset.
    3. Готовим LSTM-датасет.
    4. Обучаем предиктор.
    5. Запускаем псевдо-real-time цикл по окнам.
    """
    normalized_metrics_df, normalized_events_df, normalized_context_df = load_all_input_tables(
        metrics_path=METRICS_PATH,
        events_path=EVENTS_PATH,
        context_path=CONTEXT_PATH,
        source_type=SOURCE_TYPE,
        source_name=SOURCE_NAME,
    )

    windows_spec = build_windows_spec_for_all_targets(
        normalized_metrics_df=normalized_metrics_df,
        window_size_minutes=WINDOW_SIZE_MINUTES,
    )

    if not windows_spec:
        print("Не удалось сформировать окна анализа. Проверь входные данные.")
        return

    windows_df = build_interface_windows_dataset(
        metrics_df=normalized_metrics_df,
        events_df=normalized_events_df,
        context_df=normalized_context_df,
        windows_spec=windows_spec,
    )

    if windows_df.empty:
        print("Не удалось собрать interface_window dataset.")
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

    print_data_summary(
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

    predictor_model = train_result["model"]

    history_store: dict[tuple[str, str], list[dict[str, Any]]] = {}

    windows_records = (
        windows_df.sort_values(["device_id", "interface_name", "window_start"])
        .to_dict(orient="records")
    )

    print("\n=== REALTIME DEMO START ===")

    max_demo_cycles = min(len(windows_records), 12)

    for i in range(max_demo_cycles):
        current_window = windows_records[i]

        realtime_result = run_realtime_cycle(
            current_window=current_window,
            history_store=history_store,
            predictor_model=predictor_model,
            feature_columns=feature_columns,
            history_length=DEFAULT_HISTORY_LENGTH,
            prediction_model_version="predictor_v1",
            device="cpu",
        )

        print_realtime_result(realtime_result)

    print("\n=== MODEL SAVED ===")
    print(MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()
