from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd


NormalizedTables = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

TelemetryAdapter = Callable[
    [pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None],
    NormalizedTables,
]


def _read_csv(path: str | Path) -> pd.DataFrame:
    """
    Безопасное чтение CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return pd.read_csv(path)



def load_metrics_csv(path: str | Path) -> pd.DataFrame:
    """
    Загружает сырую таблицу метрик.
    """
    return _read_csv(path)



def load_events_csv(path: str | Path) -> pd.DataFrame:
    """
    Загружает сырую таблицу событий.
    """
    return _read_csv(path)



def load_context_csv(path: str | Path) -> pd.DataFrame:
    """
    Загружает сырую таблицу контекста.
    """
    return _read_csv(path)



def load_raw_input_tables(
    metrics_path: str | Path | None = None,
    events_path: str | Path | None = None,
    context_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загружает сырые входные таблицы без какой-либо классификации источника.

    Если путь не передан, возвращается пустая таблица.
    """
    metrics_df = load_metrics_csv(metrics_path) if metrics_path is not None else pd.DataFrame()
    events_df = load_events_csv(events_path) if events_path is not None else pd.DataFrame()
    context_df = load_context_csv(context_path) if context_path is not None else pd.DataFrame()

    return metrics_df, events_df, context_df



def normalize_input_tables(
    metrics_df: pd.DataFrame | None = None,
    events_df: pd.DataFrame | None = None,
    context_df: pd.DataFrame | None = None,
    adapter: TelemetryAdapter | None = None,
) -> NormalizedTables:
    """
    Приводит входные таблицы к внутреннему нормализованному формату проекта.

    Важно:
    - loaders.py не определяет тип источника;
    - нормализация делегируется переданному adapter;
    - если adapter не передан, это считается ошибкой архитектуры.
    """
    if adapter is None:
        raise ValueError(
            "Не передан adapter для нормализации телеметрии. "
            "loaders.py работает только с универсальным входом через внешний адаптер."
        )

    metrics_df = metrics_df if metrics_df is not None else pd.DataFrame()
    events_df = events_df if events_df is not None else pd.DataFrame()
    context_df = context_df if context_df is not None else pd.DataFrame()

    normalized_metrics_df, normalized_events_df, normalized_context_df = adapter(
        metrics_df,
        events_df,
        context_df,
    )

    return (
        normalized_metrics_df,
        normalized_events_df,
        normalized_context_df,
    )



def load_and_normalize_input_tables(
    metrics_path: str | Path | None = None,
    events_path: str | Path | None = None,
    context_path: str | Path | None = None,
    adapter: TelemetryAdapter | None = None,
) -> NormalizedTables:
    """
    Полный универсальный входной сценарий:
    1. загрузить сырые таблицы;
    2. передать их во внешний адаптер нормализации;
    3. вернуть внутренние нормализованные таблицы проекта.
    """
    metrics_df, events_df, context_df = load_raw_input_tables(
        metrics_path=metrics_path,
        events_path=events_path,
        context_path=context_path,
    )

    return normalize_input_tables(
        metrics_df=metrics_df,
        events_df=events_df,
        context_df=context_df,
        adapter=adapter,
    )



def load_all_input_tables(
    metrics_path: str | Path | None = None,
    events_path: str | Path | None = None,
    context_path: str | Path | None = None,
    adapter: TelemetryAdapter | None = None,
) -> NormalizedTables:
    """
    Главная точка входа data-layer.

    Возвращает:
    - normalized_metrics_df
    - normalized_events_df
    - normalized_context_df

    Никакой ручной классификации источника здесь нет.
    """
    return load_and_normalize_input_tables(
        metrics_path=metrics_path,
        events_path=events_path,
        context_path=context_path,
        adapter=adapter,
    )
