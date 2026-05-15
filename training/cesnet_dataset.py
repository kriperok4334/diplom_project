from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.source_adapters import (
    _finalize_normalized_context,
    _finalize_normalized_events,
    _finalize_normalized_metrics,
    _standardize_columns,
)


MAX_ROWS_PER_SERIES = 300


def _extract_series_id_from_path(csv_path: Path, root_dir: Path) -> str:
    """
    Формирует устойчивый идентификатор ряда из пути файла.

    Пример:
    agg_10_minutes/1/3.csv -> 1_3
    """
    relative_path = csv_path.relative_to(root_dir)
    parts = list(relative_path.parts)

    if len(parts) >= 2:
        folder_name = parts[-2]
        file_stem = Path(parts[-1]).stem
        return f"{folder_name}_{file_stem}"

    return csv_path.stem


def iter_cesnet_csv_files(agg_root: str | Path) -> list[Path]:
    """
    Возвращает список всех CSV внутри agg_10_minutes.
    """
    agg_root = Path(agg_root)

    if not agg_root.exists():
        raise FileNotFoundError(f"CESNET directory not found: {agg_root}")

    csv_files = sorted(agg_root.rglob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in: {agg_root}")

    return csv_files


def build_cesnet_normalized_metrics(
    agg_df: pd.DataFrame,
    times_df: pd.DataFrame,
    series_id: str,
    source_name: str = "cesnet_timeseries24",
) -> pd.DataFrame:
    """
    Преобразует один CESNET agg CSV в normalized_metrics_df.
    """
    agg_df = _standardize_columns(agg_df)
    times_df = _standardize_columns(times_df)

    if "id_time" not in agg_df.columns:
        raise ValueError("CESNET agg dataframe must contain 'id_time'")

    if "id_time" not in times_df.columns or "time" not in times_df.columns:
        raise ValueError("times dataframe must contain 'id_time' and 'time'")

    merged = agg_df.merge(
        times_df[["id_time", "time"]],
        on="id_time",
        how="left",
    )

    merged = merged.rename(columns={"time": "timestamp"})

    merged["timestamp"] = pd.to_datetime(
        merged["timestamp"],
        errors="coerce",
    )

    merged = merged.dropna(subset=["timestamp"]).sort_values("timestamp")

    if MAX_ROWS_PER_SERIES is not None:
        merged = merged.tail(MAX_ROWS_PER_SERIES).reset_index(drop=True)

    merged["device_id"] = f"cesnet_{series_id}"
    merged["device_name"] = merged["device_id"]
    merged["device_vendor"] = "CESNET"
    merged["device_model"] = "TimeSeries24"
    merged["interface_name"] = series_id
    merged["interface_role"] = "WAN"

    cesnet_metric_map = {
        "n_packets": "in_traffic_bps",
        "n_bytes": "out_traffic_bps",
        "avg_duration": "latency_ms",
        "n_flows": "cesnet_n_flows",
        "n_dest_asn": "cesnet_n_dest_asn",
        "n_dest_ports": "cesnet_n_dest_ports",
        "n_dest_ip": "cesnet_n_dest_ip",
        "tcp_udp_ratio_packets": "cesnet_tcp_udp_ratio_packets",
        "tcp_udp_ratio_bytes": "cesnet_tcp_udp_ratio_bytes",
        "dir_ratio_packets": "cesnet_dir_ratio_packets",
        "dir_ratio_bytes": "cesnet_dir_ratio_bytes",
        "avg_ttl": "cesnet_avg_ttl",
    }

    metric_columns = [
        col for col in cesnet_metric_map
        if col in merged.columns
    ]

    if not metric_columns:
        return _finalize_normalized_metrics(pd.DataFrame())

    id_columns = [
        "timestamp",
        "device_id",
        "device_name",
        "device_vendor",
        "device_model",
        "interface_name",
        "interface_role",
    ]

    melted = merged.melt(
        id_vars=id_columns,
        value_vars=metric_columns,
        var_name="metric_name",
        value_name="metric_value",
    )

    melted["metric_name"] = melted["metric_name"].map(cesnet_metric_map)
    melted["metric_unit"] = pd.NA
    melted["source_name"] = source_name

    return _finalize_normalized_metrics(melted)


def build_empty_events_df() -> pd.DataFrame:
    """
    CESNET не содержит готовых event logs.
    Возвращаем пустой normalized_events_df.
    """
    return _finalize_normalized_events(pd.DataFrame())


def build_cesnet_context_df(
    series_ids: list[str],
    source_name: str = "cesnet_timeseries24",
) -> pd.DataFrame:
    """
    Создаёт минимальный normalized_context_df.
    """
    rows = []

    for series_id in series_ids:
        rows.append(
            {
                "device_id": f"cesnet_{series_id}",
                "device_name": f"cesnet_{series_id}",
                "device_vendor": "CESNET",
                "device_model": "TimeSeries24",
                "interface_name": series_id,
                "interface_role": "WAN",
                "interface_index": pd.NA,
                "interface_speed_mbps": pd.NA,
                "neighbor_device": pd.NA,
                "source_name": source_name,
            }
        )

    context_df = pd.DataFrame(rows)

    return _finalize_normalized_context(context_df)


def load_cesnet_normalized_tables(
    agg_root: str | Path,
    times_path: str | Path,
    max_files: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загружает CESNET-TimeSeries24 и возвращает:
    - normalized_metrics_df
    - normalized_events_df
    - normalized_context_df
    """
    agg_root = Path(agg_root)
    times_path = Path(times_path)

    if not times_path.exists():
        raise FileNotFoundError(f"times file not found: {times_path}")

    times_df = pd.read_csv(times_path)

    csv_files = iter_cesnet_csv_files(agg_root)

    if max_files is not None:
        csv_files = csv_files[:max_files]

    all_metrics = []
    series_ids = []

    print("\n=== LOADING CESNET DATASET ===")
    print(f"CSV files detected: {len(csv_files)}")

    for idx, csv_path in enumerate(csv_files, start=1):
        series_id = _extract_series_id_from_path(
            csv_path=csv_path,
            root_dir=agg_root,
        )

        series_ids.append(series_id)

        print(f"[{idx}/{len(csv_files)}] Loading: {series_id}")

        agg_df = pd.read_csv(csv_path)

        normalized_metrics_df = build_cesnet_normalized_metrics(
            agg_df=agg_df,
            times_df=times_df,
            series_id=series_id,
        )

        all_metrics.append(normalized_metrics_df)

    if all_metrics:
        normalized_metrics_df = pd.concat(
            all_metrics,
            ignore_index=True,
        )
    else:
        normalized_metrics_df = pd.DataFrame()

    normalized_events_df = build_empty_events_df()

    normalized_context_df = build_cesnet_context_df(
        series_ids=series_ids,
    )

    print("\n=== CESNET NORMALIZATION COMPLETE ===")
    print(f"Normalized metrics rows: {len(normalized_metrics_df)}")
    print(f"Normalized context rows: {len(normalized_context_df)}")

    return (
        normalized_metrics_df,
        normalized_events_df,
        normalized_context_df,
    )