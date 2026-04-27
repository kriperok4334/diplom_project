from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


# Стандартные имена колонок, к которым мы хотим привести входные таблицы.
METRICS_REQUIRED_COLUMNS = [
    "timestamp",
    "device_id",
    "device_name",
    "interface_name",
    "interface_index",
    "oper_status",
    "admin_status",
    "in_traffic_bps",
    "out_traffic_bps",
    "in_errors",
    "out_errors",
    "in_discards",
    "out_discards",
    "packet_loss_pct",
    "latency_ms",
    "device_cpu_pct",
    "device_memory_pct",
    "device_availability",
    "device_uptime_sec",
]

EVENTS_REQUIRED_COLUMNS = [
    "timestamp",
    "device_id",
    "device_name",
    "interface_name",
    "event_type",
    "severity",
    "event_status",
]

CONTEXT_REQUIRED_COLUMNS = [
    "device_id",
    "device_name",
    "device_vendor",
    "device_model",
    "interface_name",
    "interface_index",
    "interface_role",
    "interface_speed_mbps",
    "neighbor_device",
]


def _read_csv(path: str | Path) -> pd.DataFrame:
    """Безопасное чтение CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return pd.read_csv(path)


def _standardize_columns(
    df: pd.DataFrame,
    rename_map: Dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Приводит названия колонок к нижнему регистру и применяет карту переименования.
    """
    result = df.copy()
    result.columns = [str(col).strip().lower() for col in result.columns]

    if rename_map:
        normalized_map = {str(k).strip().lower(): v for k, v in rename_map.items()}
        result = result.rename(columns=normalized_map)

    return result


def _ensure_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> pd.DataFrame:
    """
    Гарантирует наличие обязательных колонок.
    Если колонка отсутствует, создаёт её и заполняет pd.NA.
    """
    result = df.copy()
    for col in required_columns:
        if col not in result.columns:
            result[col] = pd.NA
    return result


def load_interface_metrics_from_csv(path: str | Path) -> pd.DataFrame:
    """Загружает сырые метрики интерфейсов из CSV."""
    return _read_csv(path)


def load_interface_events_from_csv(path: str | Path) -> pd.DataFrame:
    """Загружает события по интерфейсам из CSV."""
    return _read_csv(path)


def load_device_context_from_csv(path: str | Path) -> pd.DataFrame:
    """Загружает контекст устройств и интерфейсов из CSV."""
    return _read_csv(path)


def normalize_interface_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит таблицу метрик к стандартному виду.
    """
    rename_map = {
        "time": "timestamp",
        "datetime": "timestamp",
        "hostid": "device_id",
        "host": "device_name",
        "device": "device_name",
        "interface": "interface_name",
        "ifname": "interface_name",
        "ifindex": "interface_index",
        "operstatus": "oper_status",
        "adminstatus": "admin_status",
        "inbps": "in_traffic_bps",
        "outbps": "out_traffic_bps",
        "inerrors": "in_errors",
        "outerrors": "out_errors",
        "indiscards": "in_discards",
        "outdiscards": "out_discards",
        "packetloss": "packet_loss_pct",
        "loss_pct": "packet_loss_pct",
        "latency": "latency_ms",
        "cpu": "device_cpu_pct",
        "memory": "device_memory_pct",
        "availability": "device_availability",
        "uptime": "device_uptime_sec",
    }

    result = _standardize_columns(df, rename_map=rename_map)
    result = _ensure_columns(result, METRICS_REQUIRED_COLUMNS)

    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")

    numeric_columns = [
        "interface_index",
        "in_traffic_bps",
        "out_traffic_bps",
        "in_errors",
        "out_errors",
        "in_discards",
        "out_discards",
        "packet_loss_pct",
        "latency_ms",
        "device_cpu_pct",
        "device_memory_pct",
        "device_uptime_sec",
    ]
    for col in numeric_columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result["device_availability"] = result["device_availability"].astype("string").str.strip().str.lower()
    result["oper_status"] = result["oper_status"].astype("string").str.strip().str.lower()
    result["admin_status"] = result["admin_status"].astype("string").str.strip().str.lower()

    result = result.drop_duplicates().sort_values(["device_id", "interface_name", "timestamp"]).reset_index(drop=True)
    return result


def normalize_interface_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит таблицу событий к стандартному виду.
    """
    rename_map = {
        "time": "timestamp",
        "datetime": "timestamp",
        "hostid": "device_id",
        "host": "device_name",
        "device": "device_name",
        "interface": "interface_name",
        "event": "event_type",
        "level": "severity",
        "status": "event_status",
    }

    result = _standardize_columns(df, rename_map=rename_map)
    result = _ensure_columns(result, EVENTS_REQUIRED_COLUMNS)

    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["event_type"] = result["event_type"].astype("string").str.strip().str.lower()
    result["severity"] = result["severity"].astype("string").str.strip().str.lower()
    result["event_status"] = result["event_status"].astype("string").str.strip().str.lower()

    result = result.drop_duplicates().sort_values(["device_id", "interface_name", "timestamp"]).reset_index(drop=True)
    return result


def normalize_device_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит таблицу контекста устройств/интерфейсов к стандартному виду.
    """
    rename_map = {
        "hostid": "device_id",
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

    result = _standardize_columns(df, rename_map=rename_map)
    result = _ensure_columns(result, CONTEXT_REQUIRED_COLUMNS)

    result["interface_index"] = pd.to_numeric(result["interface_index"], errors="coerce")
    result["interface_speed_mbps"] = pd.to_numeric(result["interface_speed_mbps"], errors="coerce")

    # Фиксируем наш стартовый стенд, если модель/вендор не указаны во входе.
    result["device_vendor"] = result["device_vendor"].fillna("MikroTik")
    result["device_model"] = result["device_model"].fillna("RB1100AHx4 Dude Edition")

    result = result.drop_duplicates().reset_index(drop=True)
    return result


def load_all_input_tables(
    metrics_path: str | Path,
    events_path: str | Path,
    context_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загружает и нормализует все входные таблицы.
    """
    metrics_df = normalize_interface_metrics(load_interface_metrics_from_csv(metrics_path))
    events_df = normalize_interface_events(load_interface_events_from_csv(events_path))
    context_df = normalize_device_context(load_device_context_from_csv(context_path))
    return metrics_df, events_df, context_df
