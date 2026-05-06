from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.telemetry_schema import (
    get_normalized_context_required_columns,
    get_normalized_events_required_columns,
    get_normalized_metrics_required_columns,
    get_core_metric_names,
)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит имена колонок к нижнему регистру и удаляет лишние пробелы.
    """
    result = df.copy()
    result.columns = [str(col).strip().lower() for col in result.columns]
    return result


def _ensure_columns(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """
    Гарантирует наличие всех обязательных колонок.
    Отсутствующие колонки создаются и заполняются pd.NA.
    """
    result = df.copy()
    for col in required_columns:
        if col not in result.columns:
            result[col] = pd.NA
    return result


def _normalize_status_value(value: Any) -> Any:
    """
    Унификация статусов интерфейса/устройства.
    """
    if pd.isna(value):
        return pd.NA

    value_str = str(value).strip().lower()

    mapping = {
        "1": "up",
        "2": "down",
        "up": "up",
        "down": "down",
        "true": "up",
        "false": "down",
        "available": "up",
        "unavailable": "down",
    }

    return mapping.get(value_str, value_str)


def _normalize_severity(value: Any) -> Any:
    """
    Унификация severity событий.
    """
    if pd.isna(value):
        return pd.NA

    value_str = str(value).strip().lower()

    mapping = {
        "info": "info",
        "information": "info",
        "warning": "warning",
        "warn": "warning",
        "average": "warning",
        "high": "high",
        "critical": "critical",
        "disaster": "critical",
    }

    return mapping.get(value_str, value_str)


def _normalize_event_status(value: Any) -> Any:
    """
    Унификация статуса события.
    """
    if pd.isna(value):
        return pd.NA

    value_str = str(value).strip().lower()

    mapping = {
        "problem": "problem",
        "alert": "problem",
        "active": "problem",
        "recovery": "recovery",
        "resolved": "recovery",
        "ok": "recovery",
        "info": "info",
    }

    return mapping.get(value_str, value_str)


def _finalize_normalized_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Финальная обработка normalized_metrics_df.
    """
    required_columns = get_normalized_metrics_required_columns()

    result = _standardize_columns(df)
    result = _ensure_columns(result, required_columns)

    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["metric_name"] = result["metric_name"].astype("string").str.strip().str.lower()

    result["device_name"] = result["device_name"].astype("string")
    result["device_vendor"] = result["device_vendor"].astype("string")
    result["device_model"] = result["device_model"].astype("string")
    result["interface_name"] = result["interface_name"].astype("string")
    result["interface_role"] = result["interface_role"].astype("string")
    result["metric_unit"] = result["metric_unit"].astype("string")
    result["source_name"] = result["source_name"].astype("string")

    status_metric_names = {"oper_status", "admin_status", "device_availability"}

    # Сначала храним metric_value как object, чтобы не потерять строковые статусы.
    result["metric_value"] = result["metric_value"].astype(object)

    # Для числовых метрик приводим к numeric.
    numeric_mask = ~result["metric_name"].isin(status_metric_names)
    result.loc[numeric_mask, "metric_value"] = pd.to_numeric(
        result.loc[numeric_mask, "metric_value"],
        errors="coerce",
    )

    # Для статусных метрик нормализуем строковое представление.
    result.loc[~numeric_mask, "metric_value"] = (
        result.loc[~numeric_mask, "metric_value"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(_normalize_status_value)
    )

    result = result.dropna(subset=["timestamp", "device_id", "interface_name", "metric_name"])
    result = result.drop_duplicates().sort_values(
        ["device_id", "interface_name", "metric_name", "timestamp"]
    ).reset_index(drop=True)

    return result


def _finalize_normalized_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Финальная обработка normalized_events_df.
    """
    required_columns = get_normalized_events_required_columns()

    result = _standardize_columns(df)
    result = _ensure_columns(result, required_columns)

    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["event_type"] = result["event_type"].astype("string").str.strip().str.lower()
    result["severity"] = result["severity"].map(_normalize_severity)
    result["event_status"] = result["event_status"].map(_normalize_event_status)

    result["device_name"] = result["device_name"].astype("string")
    result["device_vendor"] = result["device_vendor"].astype("string")
    result["device_model"] = result["device_model"].astype("string")
    result["interface_name"] = result["interface_name"].astype("string")
    result["source_name"] = result["source_name"].astype("string")

    result = result.dropna(subset=["timestamp", "device_id", "interface_name", "event_type"])
    result = result.drop_duplicates().sort_values(
        ["device_id", "interface_name", "timestamp"]
    ).reset_index(drop=True)

    return result


def _finalize_normalized_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Финальная обработка normalized_context_df.
    """
    required_columns = get_normalized_context_required_columns()

    result = _standardize_columns(df)
    result = _ensure_columns(result, required_columns)

    result["interface_index"] = pd.to_numeric(result["interface_index"], errors="coerce")
    result["interface_speed_mbps"] = pd.to_numeric(result["interface_speed_mbps"], errors="coerce")

    result["device_name"] = result["device_name"].astype("string")
    result["device_vendor"] = result["device_vendor"].astype("string")
    result["device_model"] = result["device_model"].astype("string")
    result["interface_name"] = result["interface_name"].astype("string")
    result["interface_role"] = result["interface_role"].astype("string")
    result["neighbor_device"] = result["neighbor_device"].astype("string")
    result["source_name"] = result["source_name"].astype("string")

    result = result.dropna(subset=["device_id", "interface_name"])
    result = result.drop_duplicates().reset_index(drop=True)

    return result


def adapt_wide_metrics_to_normalized(
    metrics_df: pd.DataFrame,
    source_name: str = "generic",
) -> pd.DataFrame:
    """
    Преобразует широкую таблицу метрик в normalized_metrics_df.

    Ожидается, что во входе могут присутствовать колонки вида:
    - oper_status
    - admin_status
    - in_traffic_bps
    - out_traffic_bps
    - in_errors
    - out_errors
    - in_discards
    - out_discards
    - packet_loss_pct
    - latency_ms
    - device_cpu_pct
    - device_memory_pct
    - device_availability
    - device_uptime_sec
    """
    result = _standardize_columns(metrics_df)

    id_columns = [
        "timestamp",
        "device_id",
        "device_name",
        "device_vendor",
        "device_model",
        "interface_name",
        "interface_role",
    ]

    available_id_columns = [col for col in id_columns if col in result.columns]

    metric_candidates = [col for col in get_core_metric_names() if col in result.columns]

    if not metric_candidates:
        empty_df = pd.DataFrame(columns=get_normalized_metrics_required_columns())
        return _finalize_normalized_metrics(empty_df)

    melted = result.melt(
        id_vars=available_id_columns,
        value_vars=metric_candidates,
        var_name="metric_name",
        value_name="metric_value",
    )

    melted["metric_unit"] = pd.NA
    melted["source_name"] = source_name

    return _finalize_normalized_metrics(melted)


def adapt_wide_events_to_normalized(
    events_df: pd.DataFrame,
    source_name: str = "generic",
) -> pd.DataFrame:
    """
    Приводит таблицу событий к normalized_events_df.
    """
    result = _standardize_columns(events_df)

    rename_map = {
        "time": "timestamp",
        "datetime": "timestamp",
        "host": "device_name",
        "device": "device_name",
        "vendor": "device_vendor",
        "model": "device_model",
        "interface": "interface_name",
        "level": "severity",
        "status": "event_status",
        "event": "event_type",
    }
    result = result.rename(columns=rename_map)
    result["source_name"] = source_name

    return _finalize_normalized_events(result)


def adapt_wide_context_to_normalized(
    context_df: pd.DataFrame,
    source_name: str = "generic",
) -> pd.DataFrame:
    """
    Приводит таблицу контекста к normalized_context_df.
    """
    result = _standardize_columns(context_df)

    rename_map = {
        "host": "device_name",
        "device": "device_name",
        "vendor": "device_vendor",
        "model": "device_model",
        "interface": "interface_name",
        "ifindex": "interface_index",
        "role": "interface_role",
        "speed": "interface_speed_mbps",
        "neighbor": "neighbor_device",
    }
    result = result.rename(columns=rename_map)
    result["source_name"] = source_name

    return _finalize_normalized_context(result)


def adapt_zabbix_to_normalized(
    metrics_df: pd.DataFrame | None = None,
    events_df: pd.DataFrame | None = None,
    context_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Адаптер под Zabbix-данные.
    Пока использует generic-wide адаптацию.
    Позже сюда можно добавить zabbix-specific mapping.
    """
    normalized_metrics_df = adapt_wide_metrics_to_normalized(
        metrics_df if metrics_df is not None else pd.DataFrame(),
        source_name="zabbix",
    )
    normalized_events_df = adapt_wide_events_to_normalized(
        events_df if events_df is not None else pd.DataFrame(),
        source_name="zabbix",
    )
    normalized_context_df = adapt_wide_context_to_normalized(
        context_df if context_df is not None else pd.DataFrame(),
        source_name="zabbix",
    )

    return normalized_metrics_df, normalized_events_df, normalized_context_df


def adapt_noc_to_normalized(
    metrics_df: pd.DataFrame | None = None,
    events_df: pd.DataFrame | None = None,
    context_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Адаптер под NOC-данные.
    Пока использует generic-wide адаптацию.
    """
    normalized_metrics_df = adapt_wide_metrics_to_normalized(
        metrics_df if metrics_df is not None else pd.DataFrame(),
        source_name="noc",
    )
    normalized_events_df = adapt_wide_events_to_normalized(
        events_df if events_df is not None else pd.DataFrame(),
        source_name="noc",
    )
    normalized_context_df = adapt_wide_context_to_normalized(
        context_df if context_df is not None else pd.DataFrame(),
        source_name="noc",
    )

    return normalized_metrics_df, normalized_events_df, normalized_context_df


def adapt_generic_telemetry_to_normalized(
    metrics_df: pd.DataFrame | None = None,
    events_df: pd.DataFrame | None = None,
    context_df: pd.DataFrame | None = None,
    source_name: str = "generic",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Универсальный адаптер для произвольной телеметрии,
    если она уже близка к нашей широкой форме.
    """
    normalized_metrics_df = adapt_wide_metrics_to_normalized(
        metrics_df if metrics_df is not None else pd.DataFrame(),
        source_name=source_name,
    )
    normalized_events_df = adapt_wide_events_to_normalized(
        events_df if events_df is not None else pd.DataFrame(),
        source_name=source_name,
    )
    normalized_context_df = adapt_wide_context_to_normalized(
        context_df if context_df is not None else pd.DataFrame(),
        source_name=source_name,
    )

    return normalized_metrics_df, normalized_events_df, normalized_context_df
