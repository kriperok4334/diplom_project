from __future__ import annotations

from typing import Any

import pandas as pd


SCHEMA_VERSION = "1.0"
OBJECT_TYPE = "interface"


def filter_interface_metrics(
    metrics_df: pd.DataFrame,
    device_id: str,
    interface_name: str,
) -> pd.DataFrame:
    """Оставляет метрики одного интерфейса одного устройства."""
    mask = (
        (metrics_df["device_id"].astype("string") == str(device_id))
        & (metrics_df["interface_name"].astype("string") == str(interface_name))
    )
    return metrics_df.loc[mask].copy()


def filter_interface_events(
    events_df: pd.DataFrame,
    device_id: str,
    interface_name: str,
) -> pd.DataFrame:
    """Оставляет события одного интерфейса одного устройства."""
    mask = (
        (events_df["device_id"].astype("string") == str(device_id))
        & (events_df["interface_name"].astype("string") == str(interface_name))
    )
    return events_df.loc[mask].copy()


def slice_time_window(
    df: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Вырезает строки, попадающие в заданное окно времени."""
    if df.empty:
        return df.copy()

    result = df.copy()
    result[time_col] = pd.to_datetime(result[time_col], errors="coerce")

    mask = (result[time_col] >= window_start) & (result[time_col] < window_end)
    return result.loc[mask].sort_values(time_col).reset_index(drop=True)


def _safe_last_value(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return pd.NA
    return non_null.iloc[-1]


def _safe_delta(series: pd.Series) -> float:
    non_null = pd.to_numeric(series, errors="coerce").dropna()
    if non_null.empty:
        return 0.0
    if len(non_null) == 1:
        return float(non_null.iloc[0])
    return float(non_null.iloc[-1] - non_null.iloc[0])


def compute_status_features(
    metrics_window_df: pd.DataFrame,
    events_window_df: pd.DataFrame | None = None,
) -> dict:
    """Считает признаки статуса интерфейса."""
    if metrics_window_df.empty:
        return {
            "oper_status_last": pd.NA,
            "admin_status_last": pd.NA,
            "status_change_count": 0,
            "down_seconds_total": 0,
        }

    oper_series = metrics_window_df["oper_status"].astype("string").fillna("unknown")
    admin_series = metrics_window_df["admin_status"].astype("string").fillna("unknown")

    status_change_count = int((oper_series != oper_series.shift(1)).sum() - 1)
    status_change_count = max(status_change_count, 0)

    down_mask = oper_series.str.lower() == "down"
    if len(metrics_window_df) >= 2:
        deltas = metrics_window_df["timestamp"].diff().dt.total_seconds().dropna()
        step_seconds = int(deltas.median()) if not deltas.empty else 0
    else:
        step_seconds = 0
    down_seconds_total = int(down_mask.sum() * step_seconds)

    return {
        "oper_status_last": _safe_last_value(oper_series),
        "admin_status_last": _safe_last_value(admin_series),
        "status_change_count": status_change_count,
        "down_seconds_total": down_seconds_total,
    }


def compute_traffic_features(
    metrics_window_df: pd.DataFrame,
    interface_speed_mbps: float | None = None,
) -> dict:
    """Считает признаки трафика и утилизации."""
    if metrics_window_df.empty:
        return {
            "in_traffic_avg_bps": 0.0,
            "out_traffic_avg_bps": 0.0,
            "in_traffic_max_bps": 0.0,
            "out_traffic_max_bps": 0.0,
            "utilization_in_avg_pct": 0.0,
            "utilization_out_avg_pct": 0.0,
            "utilization_peak_pct": 0.0,
            "traffic_asymmetry_ratio": 0.0,
        }

    in_traffic = pd.to_numeric(metrics_window_df["in_traffic_bps"], errors="coerce").fillna(0)
    out_traffic = pd.to_numeric(metrics_window_df["out_traffic_bps"], errors="coerce").fillna(0)

    in_avg = float(in_traffic.mean())
    out_avg = float(out_traffic.mean())
    in_max = float(in_traffic.max())
    out_max = float(out_traffic.max())

    capacity_bps = None
    if interface_speed_mbps is not None and pd.notna(interface_speed_mbps):
        capacity_bps = float(interface_speed_mbps) * 1_000_000

    if capacity_bps and capacity_bps > 0:
        utilization_in_avg_pct = (in_avg / capacity_bps) * 100
        utilization_out_avg_pct = (out_avg / capacity_bps) * 100
        utilization_peak_pct = (max(in_max, out_max) / capacity_bps) * 100
    else:
        utilization_in_avg_pct = 0.0
        utilization_out_avg_pct = 0.0
        utilization_peak_pct = 0.0

    traffic_asymmetry_ratio = float(in_avg / out_avg) if out_avg > 0 else 0.0

    return {
        "in_traffic_avg_bps": in_avg,
        "out_traffic_avg_bps": out_avg,
        "in_traffic_max_bps": in_max,
        "out_traffic_max_bps": out_max,
        "utilization_in_avg_pct": utilization_in_avg_pct,
        "utilization_out_avg_pct": utilization_out_avg_pct,
        "utilization_peak_pct": utilization_peak_pct,
        "traffic_asymmetry_ratio": traffic_asymmetry_ratio,
    }


def compute_error_features(metrics_window_df: pd.DataFrame) -> dict:
    """Считает признаки ошибок и discard."""
    if metrics_window_df.empty:
        return {
            "in_errors_delta": 0.0,
            "out_errors_delta": 0.0,
            "in_discards_delta": 0.0,
            "out_discards_delta": 0.0,
            "errors_total_delta": 0.0,
            "discards_total_delta": 0.0,
            "error_burst_flag": False,
        }

    in_errors_delta = _safe_delta(metrics_window_df["in_errors"])
    out_errors_delta = _safe_delta(metrics_window_df["out_errors"])
    in_discards_delta = _safe_delta(metrics_window_df["in_discards"])
    out_discards_delta = _safe_delta(metrics_window_df["out_discards"])

    errors_total_delta = in_errors_delta + out_errors_delta
    discards_total_delta = in_discards_delta + out_discards_delta

    error_burst_flag = bool(errors_total_delta >= 50)

    return {
        "in_errors_delta": in_errors_delta,
        "out_errors_delta": out_errors_delta,
        "in_discards_delta": in_discards_delta,
        "out_discards_delta": out_discards_delta,
        "errors_total_delta": errors_total_delta,
        "discards_total_delta": discards_total_delta,
        "error_burst_flag": error_burst_flag,
    }


def compute_quality_features(metrics_window_df: pd.DataFrame) -> dict:
    """Считает признаки качества канала."""
    if metrics_window_df.empty:
        return {
            "packet_loss_avg_pct": 0.0,
            "packet_loss_max_pct": 0.0,
            "latency_avg_ms": 0.0,
            "latency_max_ms": 0.0,
            "jitter_avg_ms": pd.NA,
        }

    packet_loss = pd.to_numeric(metrics_window_df["packet_loss_pct"], errors="coerce").fillna(0)
    latency = pd.to_numeric(metrics_window_df["latency_ms"], errors="coerce").fillna(0)

    jitter_avg_ms = float(latency.diff().abs().dropna().mean()) if len(latency) > 1 else pd.NA

    return {
        "packet_loss_avg_pct": float(packet_loss.mean()),
        "packet_loss_max_pct": float(packet_loss.max()),
        "latency_avg_ms": float(latency.mean()),
        "latency_max_ms": float(latency.max()),
        "jitter_avg_ms": jitter_avg_ms,
    }


def compute_event_features(events_window_df: pd.DataFrame) -> dict:
    """Считает событийные признаки по интерфейсу."""
    if events_window_df.empty:
        return {
            "alert_count_total": 0,
            "alert_count_critical": 0,
            "recovery_event_count": 0,
            "flap_event_count": 0,
            "event_types_seen": [],
        }

    event_types = events_window_df["event_type"].astype("string").fillna("").str.lower()
    severity = events_window_df["severity"].astype("string").fillna("").str.lower()
    event_status = events_window_df["event_status"].astype("string").fillna("").str.lower()

    return {
        "alert_count_total": int(len(events_window_df)),
        "alert_count_critical": int(severity.isin(["critical", "high"]).sum()),
        "recovery_event_count": int(event_status.eq("recovery").sum()),
        "flap_event_count": int(event_types.str.contains("down|up|flap", regex=True).sum()),
        "event_types_seen": sorted(set(event_types.tolist()) - {""}),
    }


def compute_device_context_features(
    metrics_df: pd.DataFrame,
    device_id: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> dict:
    """Собирает контекст устройства за то же окно."""
    device_df = metrics_df.loc[metrics_df["device_id"].astype("string") == str(device_id)].copy()
    device_window_df = slice_time_window(device_df, window_start, window_end)

    if device_window_df.empty:
        return {
            "device_cpu_avg_pct": 0.0,
            "device_cpu_max_pct": 0.0,
            "device_memory_avg_pct": 0.0,
            "device_availability_flag": False,
            "problematic_interfaces_count": 0,
            "device_uptime_sec": pd.NA,
        }

    cpu = pd.to_numeric(device_window_df["device_cpu_pct"], errors="coerce").fillna(0)
    memory = pd.to_numeric(device_window_df["device_memory_pct"], errors="coerce").fillna(0)
    availability = (
        device_window_df["device_availability"]
        .astype("string")
        .str.lower()
        .isin(["true", "1", "up", "available", "yes"])
    )

    grouped = device_window_df.groupby("interface_name", sort=False)

    problematic_interfaces = 0
    for _, g in grouped:
        has_packet_loss = (
            pd.to_numeric(g["packet_loss_pct"], errors="coerce").fillna(0).mean() >= 5
        )
        has_errors = (_safe_delta(g["in_errors"]) + _safe_delta(g["out_errors"])) >= 20
        has_down = g["oper_status"].astype("string").str.lower().eq("down").any()

        if has_packet_loss or has_errors or has_down:
            problematic_interfaces += 1

    return {
        "device_cpu_avg_pct": float(cpu.mean()),
        "device_cpu_max_pct": float(cpu.max()),
        "device_memory_avg_pct": float(memory.mean()),
        "device_availability_flag": bool(availability.any()),
        "problematic_interfaces_count": int(problematic_interfaces),
        "device_uptime_sec": _safe_last_value(device_window_df["device_uptime_sec"]),
    }


def _get_context_row(
    context_df: pd.DataFrame,
    device_id: str,
    interface_name: str,
) -> pd.Series:
    mask = (
        (context_df["device_id"].astype("string") == str(device_id))
        & (context_df["interface_name"].astype("string") == str(interface_name))
    )
    rows = context_df.loc[mask]
    if rows.empty:
        return pd.Series(
            {
                "device_name": pd.NA,
                "device_vendor": "MikroTik",
                "device_model": "RB1100AHx4 Dude Edition",
                "interface_index": pd.NA,
                "interface_role": pd.NA,
                "interface_speed_mbps": pd.NA,
                "neighbor_device": pd.NA,
            }
        )
    return rows.iloc[0]


def build_interface_window(
    metrics_df: pd.DataFrame,
    events_df: pd.DataFrame,
    context_df: pd.DataFrame,
    device_id: str,
    interface_name: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> dict:
    """
    Собирает полный объект interface_window.
    """
    interface_metrics_df = filter_interface_metrics(metrics_df, device_id, interface_name)
    interface_events_df = filter_interface_events(events_df, device_id, interface_name)

    metrics_window_df = slice_time_window(interface_metrics_df, window_start, window_end)
    events_window_df = slice_time_window(interface_events_df, window_start, window_end)

    context_row = _get_context_row(context_df, device_id, interface_name)
    interface_speed_mbps = context_row.get("interface_speed_mbps", pd.NA)

    status_features = compute_status_features(metrics_window_df, events_window_df)
    traffic_features = compute_traffic_features(metrics_window_df, interface_speed_mbps=interface_speed_mbps)
    error_features = compute_error_features(metrics_window_df)
    quality_features = compute_quality_features(metrics_window_df)
    event_features = compute_event_features(events_window_df)
    device_features = compute_device_context_features(metrics_df, device_id, window_start, window_end)

    device_name = context_row.get("device_name", pd.NA)
    if pd.isna(device_name) and not interface_metrics_df.empty:
        device_name = _safe_last_value(interface_metrics_df["device_name"])

    record_id = f"{device_id}-{interface_name}-{window_start.isoformat()}-{window_end.isoformat()}"

    return {
        "record_id": record_id,
        "object_type": OBJECT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "device_id": str(device_id),
        "device_name": device_name,
        "device_vendor": context_row.get("device_vendor", "MikroTik"),
        "device_model": context_row.get("device_model", "RB1100AHx4 Dude Edition"),
        "interface_id": f"{device_id}-{interface_name}",
        "interface_name": str(interface_name),
        "interface_index": context_row.get("interface_index", pd.NA),
        "interface_role": context_row.get("interface_role", pd.NA),
        "interface_speed_mbps": interface_speed_mbps,
        "neighbor_device": context_row.get("neighbor_device", pd.NA),
        "window_start": window_start,
        "window_end": window_end,
        "window_size_sec": int((window_end - window_start).total_seconds()),
        "aggregation_step_sec": 60,
        **status_features,
        **traffic_features,
        **error_features,
        **quality_features,
        **event_features,
        **device_features,
    }


def build_interface_windows_dataset(
    metrics_df: pd.DataFrame,
    events_df: pd.DataFrame,
    context_df: pd.DataFrame,
    windows_spec: list[dict],
) -> pd.DataFrame:
    """Собирает датасет из множества interface_window."""
    rows: list[dict] = []

    for spec in windows_spec:
        row = build_interface_window(
            metrics_df=metrics_df,
            events_df=events_df,
            context_df=context_df,
            device_id=spec["device_id"],
            interface_name=spec["interface_name"],
            window_start=pd.Timestamp(spec["window_start"]),
            window_end=pd.Timestamp(spec["window_end"]),
        )
        rows.append(row)

    return pd.DataFrame(rows)
