from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders import load_all_input_tables
from src.features.interface_features import (
    build_interface_window,
    build_interface_windows_dataset,
)
from src.models.baseline import (
    evaluate_interface_window,
    evaluate_interface_windows_dataset,
)


METRICS_PATH = Path("data/raw/interface_metrics.csv")
EVENTS_PATH = Path("data/raw/interface_events.csv")
CONTEXT_PATH = Path("data/raw/device_context.csv")

DEFAULT_DEVICE_ID = "mt-rb1100-01"
DEFAULT_INTERFACE_NAME = "ether1"


def run_single_window_demo(
    metrics_df: pd.DataFrame,
    events_df: pd.DataFrame,
    context_df: pd.DataFrame,
) -> None:
    """
    Демонстрация для одного окна.
    """
    window_start = pd.Timestamp("2026-04-09 10:00:00")
    window_end = pd.Timestamp("2026-04-09 10:10:00")

    interface_window = build_interface_window(
        metrics_df=metrics_df,
        events_df=events_df,
        context_df=context_df,
        device_id=DEFAULT_DEVICE_ID,
        interface_name=DEFAULT_INTERFACE_NAME,
        window_start=window_start,
        window_end=window_end,
    )

    baseline_result = evaluate_interface_window(interface_window)

    print("=== SINGLE INTERFACE WINDOW ===")
    for key, value in interface_window.items():
        print(f"{key}: {value}")

    print("\n=== SINGLE BASELINE RESULT ===")
    for key, value in baseline_result.items():
        print(f"{key}: {value}")


def run_multi_window_demo(
    metrics_df: pd.DataFrame,
    events_df: pd.DataFrame,
    context_df: pd.DataFrame,
) -> None:
    """
    Демонстрация для нескольких окон подряд.
    """
    windows_spec = [
        {
            "device_id": DEFAULT_DEVICE_ID,
            "interface_name": DEFAULT_INTERFACE_NAME,
            "window_start": "2026-04-09 10:00:00",
            "window_end": "2026-04-09 10:05:00",
        },
        {
            "device_id": DEFAULT_DEVICE_ID,
            "interface_name": DEFAULT_INTERFACE_NAME,
            "window_start": "2026-04-09 10:05:00",
            "window_end": "2026-04-09 10:10:00",
        },
    ]

    windows_df = build_interface_windows_dataset(
        metrics_df=metrics_df,
        events_df=events_df,
        context_df=context_df,
        windows_spec=windows_spec,
    )

    baseline_results = evaluate_interface_windows_dataset(windows_df)

    print("\n=== MULTI WINDOW DATASET ===")
    print(windows_df[[
        "record_id",
        "device_id",
        "interface_name",
        "window_start",
        "window_end",
        "status_change_count",
        "errors_total_delta",
        "packet_loss_avg_pct",
        "latency_avg_ms",
    ]])

    print("\n=== MULTI WINDOW BASELINE RESULTS ===")
    print(pd.DataFrame(baseline_results))


def main() -> None:
    metrics_df, events_df, context_df = load_all_input_tables(
        metrics_path=METRICS_PATH,
        events_path=EVENTS_PATH,
        context_path=CONTEXT_PATH,
    )

    run_single_window_demo(metrics_df, events_df, context_df)
    run_multi_window_demo(metrics_df, events_df, context_df)


if __name__ == "__main__":
    main()