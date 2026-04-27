from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders import load_all_input_tables
from src.features.interface_features import build_interface_window
from src.models.baseline import evaluate_interface_window


# Пути к тестовым входным данным.
# Пока фиксируем их как простые CSV в data/raw/.
METRICS_PATH = Path("data/raw/interface_metrics.csv")
EVENTS_PATH = Path("data/raw/interface_events.csv")
CONTEXT_PATH = Path("data/raw/device_context.csv")

# Наш стартовый стенд.
DEFAULT_DEVICE_ID = "mt-rb1100-01"
DEFAULT_INTERFACE_NAME = "ether1"

# Базовое окно анализа: 10 минут.
DEFAULT_WINDOW_START = pd.Timestamp("2026-04-09 10:00:00")
DEFAULT_WINDOW_END = pd.Timestamp("2026-04-09 10:10:00")


def main() -> None:
    """
    Первый сквозной запуск:
    1. Загружаем таблицы.
    2. Собираем один interface_window.
    3. Прогоняем его через baseline.
    4. Печатаем результат.
    """
    metrics_df, events_df, context_df = load_all_input_tables(
        metrics_path=METRICS_PATH,
        events_path=EVENTS_PATH,
        context_path=CONTEXT_PATH,
    )

    interface_window = build_interface_window(
        metrics_df=metrics_df,
        events_df=events_df,
        context_df=context_df,
        device_id=DEFAULT_DEVICE_ID,
        interface_name=DEFAULT_INTERFACE_NAME,
        window_start=DEFAULT_WINDOW_START,
        window_end=DEFAULT_WINDOW_END,
    )

    baseline_result = evaluate_interface_window(interface_window)

    print("=== INTERFACE WINDOW ===")
    for key, value in interface_window.items():
        print(f"{key}: {value}")

    print("\n=== BASELINE RESULT ===")
    for key, value in baseline_result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()