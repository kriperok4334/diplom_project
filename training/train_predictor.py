from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from training.cesnet_dataset import load_cesnet_normalized_tables
from training.dataset_cache import (
    build_cache_key,
    build_cache_signature,
    cache_exists,
    load_prepared_dataset,
    print_cache_summary,
    save_prepared_dataset,
)
from training.training_utils import TrainConfig, train_lstm_predictor

from src.features.interface_features import build_interface_windows_dataset
from src.features.sequence_features import (
    DEFAULT_HISTORY_LENGTH,
    build_lstm_dataset,
)


AGG_ROOT = Path("training/data/agg_10_minutes")
TIMES_PATH = Path("training/data/times_10_minutes.csv")

MODEL_OUTPUT_PATH = Path("artifacts/lstm_next_window_predictor.pt")

WINDOW_SIZE_MINUTES = 10
MAX_FILES = 50

MODEL_ARTIFACT_NAME = "lstm_next_window_predictor"
HISTORY_LENGTH = DEFAULT_HISTORY_LENGTH

CESNET_FEATURE_COLUMNS = [
    "latency_avg_ms",
    "latency_max_ms",
    "utilization_in_avg_pct",
    "utilization_out_avg_pct",
    "utilization_peak_pct",
]


def get_available_targets(normalized_metrics_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"device_id", "interface_name"}
    missing = required_cols - set(normalized_metrics_df.columns)
    if missing:
        raise ValueError(
            f"В normalized_metrics_df отсутствуют обязательные колонки: {sorted(missing)}"
        )

    return (
        normalized_metrics_df[["device_id", "interface_name"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["device_id", "interface_name"])
        .reset_index(drop=True)
    )


def build_windows_spec_for_target(
    normalized_metrics_df: pd.DataFrame,
    device_id: str,
    interface_name: str,
    window_size_minutes: int,
) -> list[dict[str, Any]]:
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
    targets_df = get_available_targets(normalized_metrics_df)
    all_windows_spec: list[dict[str, Any]] = []

    print("\n=== STEP 2: BUILD WINDOWS SPEC ===")
    print(f"Targets available for windows: {len(targets_df)}")

    for idx, row in enumerate(targets_df.itertuples(index=False), start=1):
        target_windows = build_windows_spec_for_target(
            normalized_metrics_df=normalized_metrics_df,
            device_id=str(row.device_id),
            interface_name=str(row.interface_name),
            window_size_minutes=window_size_minutes,
        )
        all_windows_spec.extend(target_windows)

        if idx % 10 == 0 or idx == len(targets_df):
            print(
                f"  windows spec progress: {idx}/{len(targets_df)} | "
                f"windows so far: {len(all_windows_spec)}"
            )

    return all_windows_spec


def print_training_summary(
    windows_df: pd.DataFrame,
    X,
    y,
    metadata_df: pd.DataFrame,
) -> None:
    unique_targets = windows_df[["device_id", "interface_name"]].drop_duplicates().shape[0]

    print("\n=== TRAINING DATA SUMMARY ===")
    print(f"Targets available: {unique_targets}")
    print(f"Windows available: {len(windows_df)}")
    print(f"LSTM samples built: {len(X)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Metadata rows: {len(metadata_df)}")


def main() -> None:
    feature_columns = CESNET_FEATURE_COLUMNS
    history_length = HISTORY_LENGTH

    cache_signature = build_cache_signature(
        max_files=MAX_FILES,
        window_size_minutes=WINDOW_SIZE_MINUTES,
        history_length=history_length,
        feature_columns=feature_columns,
        extra_params={
            "dataset_name": "cesnet_timeseries24",
            "artifact_name": MODEL_ARTIFACT_NAME,
        },
    )

    cache_key = build_cache_key(
        max_files=MAX_FILES,
        window_size_minutes=WINDOW_SIZE_MINUTES,
        history_length=history_length,
        feature_columns=feature_columns,
        extra_params={
            "dataset_name": "cesnet_timeseries24",
            "artifact_name": MODEL_ARTIFACT_NAME,
        },
    )

    if cache_exists(cache_key):
        print("\n=== LOAD PREPARED DATASET FROM CACHE ===")
        windows_df, X, y, metadata_df, loaded_signature = load_prepared_dataset(cache_key)
        print_cache_summary(cache_key, loaded_signature)
    else:
        normalized_metrics_df, normalized_events_df, normalized_context_df = load_cesnet_normalized_tables(
            agg_root=AGG_ROOT,
            times_path=TIMES_PATH,
            max_files=MAX_FILES,
        )

        print("\n=== STEP 1: LOAD NORMALIZED TRAINING TABLES ===")
        print(f"Normalized metrics rows: {len(normalized_metrics_df)}")
        print(f"Normalized events rows: {len(normalized_events_df)}")
        print(f"Normalized context rows: {len(normalized_context_df)}")

        windows_spec = build_windows_spec_for_all_targets(
            normalized_metrics_df=normalized_metrics_df,
            window_size_minutes=WINDOW_SIZE_MINUTES,
        )

        if not windows_spec:
            print("Не удалось сформировать окна анализа для обучения.")
            return

        print(f"Windows spec rows: {len(windows_spec)}")

        print("\n=== STEP 3: BUILD INTERFACE WINDOWS DATASET ===")
        windows_df = build_interface_windows_dataset(
            metrics_df=normalized_metrics_df,
            events_df=normalized_events_df,
            context_df=normalized_context_df,
            windows_spec=windows_spec,
        )

        if windows_df.empty:
            print("Не удалось собрать interface_window dataset для обучения.")
            return

        print(f"Interface windows dataset rows: {len(windows_df)}")

        print("\n=== STEP 4: BUILD LSTM DATASET ===")
        X, y, metadata_df = build_lstm_dataset(
            interface_windows_df=windows_df,
            feature_columns=feature_columns,
            history_length=history_length,
        )

        if len(X) == 0:
            print("Не удалось построить LSTM-датасет: недостаточно окон или есть пропуски.")
            return

        save_prepared_dataset(
            cache_key=cache_key,
            windows_df=windows_df,
            X=X,
            y=y,
            metadata_df=metadata_df,
            signature=cache_signature,
        )

        print_cache_summary(cache_key, cache_signature)

    print_training_summary(
        windows_df=windows_df,
        X=X,
        y=y,
        metadata_df=metadata_df,
    )

    train_config = TrainConfig(
        batch_size=32,
        num_epochs=50,
        learning_rate=5e-4,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("\n=== STEP 5: TRAIN MODEL ===")
    print(f"Training device: {train_config.device}")
    print(f"Batch size: {train_config.batch_size}")
    print(f"Epochs: {train_config.num_epochs}")
    print(f"History length: {history_length}")
    print(f"Feature columns: {feature_columns}")

    train_result = train_lstm_predictor(
        X=X,
        y=y,
        metadata_df=metadata_df,
        config=train_config,
        model_output_path=MODEL_OUTPUT_PATH,
        feature_columns=feature_columns,
        history_length=history_length,
    )

    print("\n=== TRAINING FINISHED ===")
    print(f"Best val loss: {train_result['history'].get('best_val_loss')}")
    print(f"Test loss: {train_result['history'].get('test_loss')}")
    print(f"History length saved: {train_result.get('history_length')}")
    print(f"Feature columns saved: {train_result.get('feature_columns')}")
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()