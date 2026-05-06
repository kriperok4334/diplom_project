from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from src.data.source_adapters import (
    adapt_generic_telemetry_to_normalized,
    adapt_noc_to_normalized,
    adapt_zabbix_to_normalized,
)


SourceType = Literal["generic", "zabbix", "noc"]


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
    Загружает сырую таблицу метрик из CSV.
    """
    return _read_csv(path)


def load_events_csv(path: str | Path) -> pd.DataFrame:
    """
    Загружает сырую таблицу событий из CSV.
    """
    return _read_csv(path)


def load_context_csv(path: str | Path) -> pd.DataFrame:
    """
    Загружает сырую таблицу контекста из CSV.
    """
    return _read_csv(path)


def normalize_source_tables(
    metrics_df: pd.DataFrame | None = None,
    events_df: pd.DataFrame | None = None,
    context_df: pd.DataFrame | None = None,
    source_type: SourceType = "generic",
    source_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Приводит внешний источник данных к внутреннему нормализованному формату проекта.

    Возвращает:
    - normalized_metrics_df
    - normalized_events_df
    - normalized_context_df
    """
    metrics_df = metrics_df if metrics_df is not None else pd.DataFrame()
    events_df = events_df if events_df is not None else pd.DataFrame()
    context_df = context_df if context_df is not None else pd.DataFrame()

    if source_type == "zabbix":
        return adapt_zabbix_to_normalized(
            metrics_df=metrics_df,
            events_df=events_df,
            context_df=context_df,
        )

    if source_type == "noc":
        return adapt_noc_to_normalized(
            metrics_df=metrics_df,
            events_df=events_df,
            context_df=context_df,
        )

    return adapt_generic_telemetry_to_normalized(
        metrics_df=metrics_df,
        events_df=events_df,
        context_df=context_df,
        source_name=source_name or "generic",
    )


def load_and_normalize_from_csv(
    metrics_path: str | Path | None = None,
    events_path: str | Path | None = None,
    context_path: str | Path | None = None,
    source_type: SourceType = "generic",
    source_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загружает CSV-таблицы и сразу приводит их к внутреннему нормализованному формату.

    Можно передавать не все пути:
    - если какого-то файла нет на текущем этапе, вместо него будет пустая таблица.
    """
    metrics_df = load_metrics_csv(metrics_path) if metrics_path is not None else pd.DataFrame()
    events_df = load_events_csv(events_path) if events_path is not None else pd.DataFrame()
    context_df = load_context_csv(context_path) if context_path is not None else pd.DataFrame()

    return normalize_source_tables(
        metrics_df=metrics_df,
        events_df=events_df,
        context_df=context_df,
        source_type=source_type,
        source_name=source_name,
    )


def load_all_input_tables(
    metrics_path: str | Path | None = None,
    events_path: str | Path | None = None,
    context_path: str | Path | None = None,
    source_type: SourceType = "generic",
    source_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Главная совместимая точка входа для data-layer.

    Возвращает уже не "сырые широкие таблицы", а внутренние нормализованные таблицы:
    - normalized_metrics_df
    - normalized_events_df
    - normalized_context_df

    Это делает loaders.py независимым от конкретного источника данных.
    """
    return load_and_normalize_from_csv(
        metrics_path=metrics_path,
        events_path=events_path,
        context_path=context_path,
        source_type=source_type,
        source_name=source_name,
    )


def load_from_zabbix_csv(
    metrics_path: str | Path | None = None,
    events_path: str | Path | None = None,
    context_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Удобный враппер для Zabbix CSV.
    """
    return load_and_normalize_from_csv(
        metrics_path=metrics_path,
        events_path=events_path,
        context_path=context_path,
        source_type="zabbix",
    )


def load_from_noc_csv(
    metrics_path: str | Path | None = None,
    events_path: str | Path | None = None,
    context_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Удобный враппер для NOC CSV.
    """
    return load_and_normalize_from_csv(
        metrics_path=metrics_path,
        events_path=events_path,
        context_path=context_path,
        source_type="noc",
    )


def load_from_generic_csv(
    metrics_path: str | Path | None = None,
    events_path: str | Path | None = None,
    context_path: str | Path | None = None,
    source_name: str = "generic",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Удобный враппер для произвольного CSV-источника.
    """
    return load_and_normalize_from_csv(
        metrics_path=metrics_path,
        events_path=events_path,
        context_path=context_path,
        source_type="generic",
        source_name=source_name,
    )
