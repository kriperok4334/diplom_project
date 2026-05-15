from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loaders import load_all_input_tables
from src.data.source_adapters import adapt_tabular_telemetry_to_normalized
from src.features.interface_features import build_interface_windows_dataset
from src.inference.realtime_pipeline import run_realtime_cycle
from training.model_io import load_predictor_bundle


METRICS_PATH = Path("data/raw/interface_metrics.csv")
EVENTS_PATH = Path("data/raw/interface_events.csv")
CONTEXT_PATH = Path("data/raw/device_context.csv")

MODEL_PATH = Path("artifacts/lstm_next_window_predictor.pt")

WINDOW_SIZE_MINUTES = 5


def normalize_project_input(
    metrics_df: pd.DataFrame,
    events_df: pd.DataFrame,
    context_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Нормализация табличной телеметрии рабочего проекта.
    """
    return adapt_tabular_telemetry_to_normalized(
        metrics_df=metrics_df,
        events_df=events_df,
        context_df=context_df,
        source_name="project_runtime_input",
    )


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
    Строит окна анализа для одной пары device_id + interface_name.
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
    Строит окна анализа для всех интерфейсов.
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
    Точка входа готового проекта.
    """
    if not MODEL_PATH.exists():
        print("Файл обученной модели не найден:")
        print(MODEL_PATH)
        print("Сначала нужно отдельно обучить предиктор.")
        return

    normalized_metrics_df, normalized_events_df, normalized_context_df = load_all_input_tables(
        metrics_path=METRICS_PATH,
        events_path=EVENTS_PATH,
        context_path=CONTEXT_PATH,
        adapter=normalize_project_input,
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

    predictor_bundle = load_predictor_bundle(
        model_path=MODEL_PATH,
        device="cpu",
    )

    print("=== MODEL BUNDLE LOADED ===")
    print(f"Artifact version: {predictor_bundle.get('artifact_version')}")
    print(f"History length: {predictor_bundle.get('history_length')}")
    print(f"Feature columns: {predictor_bundle.get('feature_columns')}")
    print(f"Model config: {predictor_bundle.get('model_config')}")

    history_store: dict[tuple[str, str], list[dict[str, Any]]] = {}

    windows_records = (
        windows_df.sort_values(["device_id", "interface_name", "window_start"])
        .to_dict(orient="records")
    )

    print("=== PROJECT RUN START ===")
    print(f"Windows prepared: {len(windows_records)}")

    max_demo_cycles = min(len(windows_records), 12)

    for i in range(max_demo_cycles):
        current_window = windows_records[i]

        realtime_result = run_realtime_cycle(
            current_window=current_window,
            history_store=history_store,
            predictor_bundle=predictor_bundle,
            prediction_model_version="predictor_v1",
            device="cpu",
        )

        print_realtime_result(realtime_result)


if __name__ == "__main__":
    main()