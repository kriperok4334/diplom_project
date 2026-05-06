from __future__ import annotations

from typing import Final


# =========================
# Внутренние таблицы проекта
# =========================

NORMALIZED_METRICS_REQUIRED_COLUMNS: Final[list[str]] = [
    "timestamp",
    "device_id",
    "device_name",
    "device_vendor",
    "device_model",
    "interface_name",
    "interface_role",
    "metric_name",
    "metric_value",
    "metric_unit",
    "source_name",
]

NORMALIZED_EVENTS_REQUIRED_COLUMNS: Final[list[str]] = [
    "timestamp",
    "device_id",
    "device_name",
    "device_vendor",
    "device_model",
    "interface_name",
    "event_type",
    "severity",
    "event_status",
    "source_name",
]

NORMALIZED_CONTEXT_REQUIRED_COLUMNS: Final[list[str]] = [
    "device_id",
    "device_name",
    "device_vendor",
    "device_model",
    "interface_name",
    "interface_index",
    "interface_role",
    "interface_speed_mbps",
    "neighbor_device",
    "source_name",
]


# =========================
# Внутренний стандарт metric_name
# =========================

CORE_METRIC_NAMES: Final[list[str]] = [
    # Интерфейсные статусы
    "oper_status",
    "admin_status",
    # Интерфейсный трафик
    "in_traffic_bps",
    "out_traffic_bps",
    # Ошибки и discard
    "in_errors",
    "out_errors",
    "in_discards",
    "out_discards",
    # Качество канала
    "packet_loss_pct",
    "latency_ms",
    # Контекст устройства
    "device_cpu_pct",
    "device_memory_pct",
    "device_availability",
    "device_uptime_sec",
]


# =========================
# Внутренний стандарт event_type
# =========================

CORE_EVENT_TYPES: Final[list[str]] = [
    "interface_down",
    "interface_up",
    "interface_recovered",
    "high_packet_loss",
    "high_errors",
    "high_utilization",
    "device_unavailable",
]


# =========================
# Ядро аналитических признаков окна
# Под них строится baseline и LSTM
# =========================

INTERFACE_WINDOW_CORE_FEATURES: Final[list[str]] = [
    "status_change_count",
    "down_seconds_total",
    "errors_total_delta",
    "discards_total_delta",
    "packet_loss_avg_pct",
    "packet_loss_max_pct",
    "latency_avg_ms",
    "latency_max_ms",
    "utilization_in_avg_pct",
    "utilization_out_avg_pct",
    "utilization_peak_pct",
    "device_cpu_avg_pct",
    "device_memory_avg_pct",
]


# =========================
# Обязательные поля interface_window
# Без них окно не имеет смысла
# =========================

INTERFACE_WINDOW_REQUIRED_COLUMNS: Final[list[str]] = [
    "window_start",
    "window_end",
    "device_id",
    "interface_name",
]


# =========================
# Вспомогательные функции
# =========================

def get_normalized_metrics_required_columns() -> list[str]:
    return list(NORMALIZED_METRICS_REQUIRED_COLUMNS)


def get_normalized_events_required_columns() -> list[str]:
    return list(NORMALIZED_EVENTS_REQUIRED_COLUMNS)


def get_normalized_context_required_columns() -> list[str]:
    return list(NORMALIZED_CONTEXT_REQUIRED_COLUMNS)


def get_core_metric_names() -> list[str]:
    return list(CORE_METRIC_NAMES)


def get_core_event_types() -> list[str]:
    return list(CORE_EVENT_TYPES)


def get_interface_window_core_features() -> list[str]:
    return list(INTERFACE_WINDOW_CORE_FEATURES)


def get_interface_window_required_columns() -> list[str]:
    return list(INTERFACE_WINDOW_REQUIRED_COLUMNS)
