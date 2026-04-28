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

REPORT_PATH = Path("artifacts/interface_analysis_report.txt")


def save_txt_report(
    results: list[dict],
    windows: list[dict],
    output_path: str | Path,
) -> None:
    """
    Сохраняет простой текстовый отчёт по результатам анализа.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("INTERFACE ANALYSIS REPORT")
    lines.append("")

    for i, (window, result) in enumerate(zip(windows, results), start=1):
        lines.append("-" * 40)
        lines.append(f"Window #{i}")
        lines.append(f"Device: {window.get('device_name')}")
        lines.append(f"Model: {window.get('device_model')}")
        lines.append(f"Interface: {window.get('interface_name')}")
        lines.append(f"Window start: {window.get('window_start')}")
        lines.append(f"Window end: {window.get('window_end')}")
        lines.append(f"State: {result.get('state_label')}")
        lines.append(f"Problem type: {result.get('problem_type_label')}")
        lines.append(f"Matched rules: {', '.join(result.get('matched_rule_ids', []))}")
        lines.append("Comment:")
        lines.append(str(result.get("comment_template", "")))
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_single_window_demo(
    metrics_df: pd.DataFrame,
    events_df: pd.DataFrame,
    context_df: pd.DataFrame,
) -> tuple[dict, dict]:
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

    return interface_window, baseline_result


def run_multi_window_demo(
    metrics_df: pd.DataFrame,
    events_df: pd.DataFrame,
    context_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
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
    print(
        windows_df[
            [
                "record_id",
                "device_id",
                "interface_name",
                "window_start",
                "window_end",
                "status_change_count",
                "errors_total_delta",
                "packet_loss_avg_pct",
                "latency_avg_ms",
            ]
        ]
    )

    print("\n=== MULTI WINDOW BASELINE RESULTS ===")
    print(pd.DataFrame(baseline_results))

    return windows_df, baseline_results


def main() -> None:
    metrics_df, events_df, context_df = load_all_input_tables(
        metrics_path=METRICS_PATH,
        events_path=EVENTS_PATH,
        context_path=CONTEXT_PATH,
    )

    # Одиночный прогон оставляем для наглядной проверки.
    run_single_window_demo(metrics_df, events_df, context_df)

    # Основной результат для отчёта — набор окон.
    windows_df, baseline_results = run_multi_window_demo(
        metrics_df, events_df, context_df
    )

    windows_records = windows_df.to_dict(orient="records")
    save_txt_report(
        results=baseline_results,
        windows=windows_records,
        output_path=REPORT_PATH,
    )

    print(f"\n=== TXT REPORT SAVED ===")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
