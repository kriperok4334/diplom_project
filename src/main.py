from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loaders import load_all_input_tables
from src.features.interface_features import build_interface_windows_dataset
from src.models.baseline import evaluate_interface_windows_dataset


METRICS_PATH = Path("data/raw/interface_metrics.csv")
EVENTS_PATH = Path("data/raw/interface_events.csv")
CONTEXT_PATH = Path("data/raw/device_context.csv")

REPORT_PATH = Path("artifacts/interface_analysis_report.txt")

# Размер окна анализа для этапа 1
WINDOW_SIZE_MINUTES = 5


def get_available_targets(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает все доступные уникальные пары device_id + interface_name,
    которые можно анализировать.
    """
    required_cols = {"device_id", "interface_name"}
    missing = required_cols - set(metrics_df.columns)
    if missing:
        raise ValueError(
            f"В metrics_df отсутствуют обязательные колонки: {sorted(missing)}"
        )

    targets_df = (
        metrics_df[["device_id", "interface_name"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["device_id", "interface_name"])
        .reset_index(drop=True)
    )

    return targets_df


def build_windows_spec_for_target(
    metrics_df: pd.DataFrame,
    device_id: str,
    interface_name: str,
    window_size_minutes: int,
) -> list[dict[str, Any]]:
    """
    Строит список окон для одной пары device_id + interface_name.
    """
    interface_df = metrics_df.loc[
        (metrics_df["device_id"].astype("string") == str(device_id))
        & (metrics_df["interface_name"].astype("string") == str(interface_name))
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

    # Добавляем небольшую дельту, чтобы последнее окно тоже попало в цикл
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
    metrics_df: pd.DataFrame,
    window_size_minutes: int,
) -> list[dict[str, Any]]:
    """
    Строит список окон для всех доступных интерфейсов.
    """
    targets_df = get_available_targets(metrics_df)

    all_windows_spec: list[dict[str, Any]] = []

    for _, row in targets_df.iterrows():
        device_id = str(row["device_id"])
        interface_name = str(row["interface_name"])

        target_windows = build_windows_spec_for_target(
            metrics_df=metrics_df,
            device_id=device_id,
            interface_name=interface_name,
            window_size_minutes=window_size_minutes,
        )

        all_windows_spec.extend(target_windows)

    return all_windows_spec


def save_txt_report(
    results: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Сохраняет простой текстовый отчёт по результатам анализа этапа 1.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("INTERFACE ANALYSIS REPORT")
    lines.append("")

    if not windows or not results:
        lines.append("No analysis results available.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    for i, (window, result) in enumerate(zip(windows, results), start=1):
        lines.append("-" * 50)
        lines.append(f"Window #{i}")
        lines.append(f"Device ID: {window.get('device_id')}")
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


def print_console_summary(
    windows_df: pd.DataFrame,
    results: list[dict[str, Any]],
) -> None:
    """
    Печатает краткую сводку по результатам анализа.
    """
    if windows_df.empty or not results:
        print("Анализ не выполнен: окна не были сформированы.")
        return

    results_df = pd.DataFrame(results)

    unique_targets = (
        windows_df[["device_id", "interface_name"]]
        .drop_duplicates()
        .shape[0]
    )

    print("=== ANALYSIS SUMMARY ===")
    print(f"Targets analyzed: {unique_targets}")
    print(f"Windows analyzed: {len(windows_df)}")

    if "state_label" in results_df.columns:
        print("\nState distribution:")
        print(results_df["state_label"].value_counts(dropna=False))

    if "problem_type_label" in results_df.columns:
        print("\nProblem type distribution:")
        print(results_df["problem_type_label"].value_counts(dropna=False))


def main() -> None:
    """
    Универсальная точка входа для этапа 1.

    Алгоритм:
    1. Загружаем и нормализуем подготовленные данные.
    2. Находим все доступные device_id + interface_name.
    3. Формируем окна по каждому интерфейсу.
    4. Собираем dataset из interface_window.
    5. Применяем baseline-анализ.
    6. Сохраняем txt-отчёт.
    7. Печатаем краткую сводку в консоль.
    """
    metrics_df, events_df, context_df = load_all_input_tables(
        metrics_path=METRICS_PATH,
        events_path=EVENTS_PATH,
        context_path=CONTEXT_PATH,
    )

    targets_df = get_available_targets(metrics_df)

    if targets_df.empty:
        print("Во входных данных не найдено ни одной пары device_id + interface_name.")
        return

    print("=== AVAILABLE TARGETS ===")
    print(targets_df.to_string(index=False))

    windows_spec = build_windows_spec_for_all_targets(
        metrics_df=metrics_df,
        window_size_minutes=WINDOW_SIZE_MINUTES,
    )

    if not windows_spec:
        print("Не удалось сформировать окна анализа.")
        print("Проверь наличие корректных timestamp и данных в metrics.")
        return

    windows_df = build_interface_windows_dataset(
        metrics_df=metrics_df,
        events_df=events_df,
        context_df=context_df,
        windows_spec=windows_spec,
    )

    if windows_df.empty:
        print("Не удалось собрать dataset из interface_window.")
        return

    baseline_results = evaluate_interface_windows_dataset(windows_df)
    windows_records = windows_df.to_dict(orient="records")

    save_txt_report(
        results=baseline_results,
        windows=windows_records,
        output_path=REPORT_PATH,
    )

    print_console_summary(
        windows_df=windows_df,
        results=baseline_results,
    )

    print("\n=== REPORT SAVED ===")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
