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
    """
    Оставляет нормализованные метрики одного интерфейса одного устройства.
    """
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
    """
    Оставляет события одного интерфейса одного устройства.
    """
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
    """
    Вырезает строки, попадающие в заданное окно времени.
    """
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
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    if len(numeric) == 1:
        return float(numeric.iloc[0])
    return float(numeric.iloc[-1] - numeric.iloc[0])


def _extract_metric_series(
    metrics_window_df: pd.DataFrame,
    metric_name: str,
) -> pd.Series:
    """
    Возвращает series значений одной метрики в пределах окна.
    """
    if metrics_window_df.empty:
        return pd.Series(dtype="object")

    metric_df = metrics_window_df.loc[
        metrics_window_df["metric_name"].astype("string") == str(metric_name)
    ].copy()

    if metric_df.empty:
        return pd.Series(dtype="object")

    metric_df = metric_df.sort_values("timestamp")
    return metric_df["metric_value"]


def _extract_metric_frame(
    metrics_window_df: pd.DataFrame,
    metric_name: str,
) -> pd.DataFrame:
    """
    Возвращает DataFrame одной метрики с timestamp и metric_value.
    """
    if metrics_window_df.empty:
        return pd.DataFrame(columns=["timestamp", "metric_value"])

    metric_df = metrics_window_df.loc[
        metrics_window_df["metric_name"].astype("string") == str(metric_name)
    ].copy()

    if metric_df.empty:
        return pd.DataFrame(columns=["timestamp", "metric_value"])

    metric_df["timestamp"] = pd.to_datetime(metric_df["timestamp"], errors="coerce")
    metric_df = metric_df.sort_values("timestamp")
    return metric_df[["timestamp", "metric_value"]].reset_index(drop=True)


def compute_status_features(
    metrics_window_df: pd.DataFrame,
    events_window_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Считает признаки статуса интерфейса из нормализованной телеметрии.
    """
    oper_status_df = _extract_metric_frame(metrics_window_df, "oper_status")
    admin_status_series = _extract_metric_series(metrics_window_df, "admin_status")

    if oper_status_df.empty:
        return {
            "oper_status_last": pd.NA,
            "admin_status_last": pd.NA,
            "status_change_count": 0,
            "down_seconds_total": 0,
        }

    oper_series = oper_status_df["metric_value"].astype("string").str.lower().fillna("unknown")

    status_change_count = int((oper_series != oper_series.shift(1)).sum() - 1)
    status_change_count = max(status_change_count, 0)

    if len(oper_status_df) >= 2:
        deltas = oper_status_df["timestamp"].diff().dt.total_seconds().dropna()
        step_seconds = int(deltas.median()) if not deltas.empty else 0
    else:
        step_seconds = 0

    down_mask = oper_series.eq("down")
    down_seconds_total = int(down_mask.sum() * step_seconds)

    return {
        "oper_status_last": _safe_last_value(oper_series),
        "admin_status_last": _safe_last_value(admin_status_series.astype("string").str.lower()),
        "status_change_count": status_change_count,
        "down_seconds_total": down_seconds_total,
    }


def compute_traffic_features(
    metrics_window_df: pd.DataFrame,
    interface_speed_mbps: float | None = None,
) -> dict[str, float]:
    """
    Считает признаки трафика и утилизации.
    """
    in_traffic = pd.to_numeric(
        _extract_metric_series(metrics_window_df, "in_traffic_bps"),
        errors="coerce",
    ).fillna(0)

    out_traffic = pd.to_numeric(
        _extract_metric_series(metrics_window_df, "out_traffic_bps"),
        errors="coerce",
    ).fillna(0)

    if in_traffic.empty and out_traffic.empty:
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

    in_avg = float(in_traffic.mean()) if not in_traffic.empty else 0.0
    out_avg = float(out_traffic.mean()) if not out_traffic.empty else 0.0
    in_max = float(in_traffic.max()) if not in_traffic.empty else 0.0
    out_max = float(out_traffic.max()) if not out_traffic.empty else 0.0

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


def compute_error_features(metrics_window_df: pd.DataFrame) -> dict[str, float | bool]:
    """
    Считает признаки ошибок и discard.
    """
    in_errors_delta = _safe_delta(_extract_metric_series(metrics_window_df, "in_errors"))
    out_errors_delta = _safe_delta(_extract_metric_series(metrics_window_df, "out_errors"))
    in_discards_delta = _safe_delta(_extract_metric_series(metrics_window_df, "in_discards"))
    out_discards_delta = _safe_delta(_extract_metric_series(metrics_window_df, "out_discards"))

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


def compute_quality_features(metrics_window_df: pd.DataFrame) -> dict[str, Any]:
    """
    Считает признаки качества канала.
    """
    packet_loss = pd.to_numeric(
        _extract_metric_series(metrics_window_df, "packet_loss_pct"),
        errors="coerce",
    ).fillna(0)

    latency = pd.to_numeric(
        _extract_metric_series(metrics_window_df, "latency_ms"),
        errors="coerce",
    ).fillna(0)

    if packet_loss.empty and latency.empty:
        return {
            "packet_loss_avg_pct": 0.0,
            "packet_loss_max_pct": 0.0,
            "latency_avg_ms": 0.0,
            "latency_max_ms": 0.0,
            "jitter_avg_ms": pd.NA,
        }

    jitter_avg_ms = float(latency.diff().abs().dropna().mean()) if len(latency) > 1 else pd.NA

    return {
        "packet_loss_avg_pct": float(packet_loss.mean()) if not packet_loss.empty else 0.0,
        "packet_loss_max_pct": float(packet_loss.max()) if not packet_loss.empty else 0.0,
        "latency_avg_ms": float(latency.mean()) if not latency.empty else 0.0,
        "latency_max_ms": float(latency.max()) if not latency.empty else 0.0,
        "jitter_avg_ms": jitter_avg_ms,
    }


def compute_event_features(events_window_df: pd.DataFrame) -> dict[str, Any]:
    """
    Считает событийные признаки по интерфейсу.
    """
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
) -> dict[str, Any]:
    """
    Собирает контекст устройства за то же окно из normalized_metrics_df.
    """
    device_df = metrics_df.loc[
        metrics_df["device_id"].astype("string") == str(device_id)
    ].copy()

    device_window_df = slice_time_window(device_df, window_start, window_end)

    cpu = pd.to_numeric(
        _extract_metric_series(device_window_df, "device_cpu_pct"),
        errors="coerce",
    ).fillna(0)

    memory = pd.to_numeric(
        _extract_metric_series(device_window_df, "device_memory_pct"),
        errors="coerce",
    ).fillna(0)

    availability_series = (
        _extract_metric_series(device_window_df, "device_availability")
        .astype("string")
        .str.lower()
    )

    if availability_series.empty:
        device_availability_flag = False
    else:
        device_availability_flag = bool(
            availability_series.isin(["true", "1", "up", "available", "yes"]).any()
        )

    grouped = device_window_df.groupby("interface_name", sort=False)

    problematic_interfaces = 0
    for _, g in grouped:
        packet_loss = pd.to_numeric(
            _extract_metric_series(g, "packet_loss_pct"),
            errors="coerce",
        ).fillna(0)

        has_packet_loss = float(packet_loss.mean()) >= 5 if not packet_loss.empty else False
        has_errors = (
            _safe_delta(_extract_metric_series(g, "in_errors"))
            + _safe_delta(_extract_metric_series(g, "out_errors"))
        ) >= 20

        oper_status = (
            _extract_metric_series(g, "oper_status")
            .astype("string")
            .str.lower()
        )
        has_down = oper_status.eq("down").any() if not oper_status.empty else False

        if has_packet_loss or has_errors or has_down:
            problematic_interfaces += 1

    uptime_series = _extract_metric_series(device_window_df, "device_uptime_sec")

    return {
        "device_cpu_avg_pct": float(cpu.mean()) if not cpu.empty else 0.0,
        "device_cpu_max_pct": float(cpu.max()) if not cpu.empty else 0.0,
        "device_memory_avg_pct": float(memory.mean()) if not memory.empty else 0.0,
        "device_availability_flag": device_availability_flag,
        "problematic_interfaces_count": int(problematic_interfaces),
        "device_uptime_sec": _safe_last_value(uptime_series),
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
                "device_vendor": pd.NA,
                "device_model": pd.NA,
                "interface_index": pd.NA,
                "interface_role": pd.NA,
                "interface_speed_mbps": pd.NA,
                "neighbor_device": pd.NA,
            }
        )

    return rows.iloc[0]


def _get_device_name_from_metrics(
    metrics_df: pd.DataFrame,
    device_id: str,
    interface_name: str,
) -> Any:
    subset = metrics_df.loc[
        (metrics_df["device_id"].astype("string") == str(device_id))
        & (metrics_df["interface_name"].astype("string") == str(interface_name))
    ].copy()

    if subset.empty or "device_name" not in subset.columns:
        return pd.NA

    subset = subset.dropna(subset=["device_name"])
    if subset.empty:
        return pd.NA

    return subset.iloc[-1]["device_name"]


def build_interface_window(
    metrics_df: pd.DataFrame,
    events_df: pd.DataFrame,
    context_df: pd.DataFrame,
    device_id: str,
    interface_name: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> dict[str, Any]:
    """
    Собирает полный interface_window из нормализованной телеметрии.
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
    if pd.isna(device_name):
        device_name = _get_device_name_from_metrics(metrics_df, device_id, interface_name)

    record_id = f"{device_id}-{interface_name}-{window_start.isoformat()}-{window_end.isoformat()}"

    return {
        "record_id": record_id,
        "object_type": OBJECT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "device_id": str(device_id),
        "device_name": device_name,
        "device_vendor": context_row.get("device_vendor", pd.NA),
        "device_model": context_row.get("device_model", pd.NA),
        "interface_id": f"{device_id}-{interface_name}",
        "interface_name": str(interface_name),
        "interface_index": context_row.get("interface_index", pd.NA),
        "interface_role": context_row.get("interface_role", pd.NA),
        "interface_speed_mbps": interface_speed_mbps,
        "neighbor_device": context_row.get("neighbor_device", pd.NA),
        "window_start": pd.Timestamp(window_start),
        "window_end": pd.Timestamp(window_end),
        "window_size_sec": int((pd.Timestamp(window_end) - pd.Timestamp(window_start)).total_seconds()),
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
    windows_spec: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Собирает датасет из множества interface_window.
    """
    rows: list[dict[str, Any]] = []

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
