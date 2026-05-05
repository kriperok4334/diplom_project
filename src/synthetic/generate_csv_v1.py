from __future__ import annotations

from pathlib import Path
from typing import Any
import random

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("data/synthetic")
DEFAULT_START_TIME = pd.Timestamp("2026-04-09 10:00:00")
DEFAULT_DURATION_MINUTES = 60
DEFAULT_STEP_SECONDS = 60


def build_time_index(
    start_time: pd.Timestamp,
    duration_minutes: int,
    step_seconds: int,
) -> pd.DatetimeIndex:
    """
    Создаёт временную сетку для генерации метрик.
    """
    periods = max(int((duration_minutes * 60) / step_seconds), 1)
    return pd.date_range(start=start_time, periods=periods, freq=f"{step_seconds}s")


def generate_device_context_rows(
    device_config: dict[str, Any],
    interfaces_config: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Генерирует строки для device_context.csv.
    """
    rows: list[dict[str, Any]] = []

    for iface in interfaces_config:
        rows.append(
            {
                "device_id": device_config["device_id"],
                "device_name": device_config["device_name"],
                "device_vendor": device_config["device_vendor"],
                "device_model": device_config["device_model"],
                "interface_name": iface["interface_name"],
                "interface_index": iface["interface_index"],
                "interface_role": iface["interface_role"],
                "interface_speed_mbps": iface["interface_speed_mbps"],
                "neighbor_device": iface.get("neighbor_device"),
            }
        )

    return rows


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _normal_jitter(rng: random.Random, base: float, rel: float = 0.05) -> float:
    delta = base * rel
    return max(0.0, base + rng.uniform(-delta, delta))


def generate_normal_metrics_series(
    device_config: dict[str, Any],
    interface_config: dict[str, Any],
    time_index: pd.DatetimeIndex,
    step_seconds: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Генерирует базовую нормальную телеметрию для интерфейса.
    """
    rows: list[dict[str, Any]] = []

    in_errors = 0
    out_errors = 0
    in_discards = 0
    out_discards = 0
    uptime_start = int(device_config.get("device_uptime_start_sec", 864000))

    base_in = float(interface_config.get("base_in_traffic_bps", 120_000_000))
    base_out = float(interface_config.get("base_out_traffic_bps", 60_000_000))
    base_latency = float(interface_config.get("base_latency_ms", 8.0))
    base_cpu = float(device_config.get("base_cpu_pct", 25.0))
    base_memory = float(device_config.get("base_memory_pct", 45.0))

    for i, ts in enumerate(time_index):
        in_traffic = _normal_jitter(rng, base_in, rel=0.10)
        out_traffic = _normal_jitter(rng, base_out, rel=0.10)
        latency = _normal_jitter(rng, base_latency, rel=0.15)
        packet_loss = rng.choice([0.0, 0.0, 0.0, 0.1, 0.2])
        cpu = _normal_jitter(rng, base_cpu, rel=0.08)
        memory = _normal_jitter(rng, base_memory, rel=0.05)

        # В норме счётчики могут расти очень медленно или не расти вовсе.
        in_errors += rng.choice([0, 0, 0, 1])
        out_errors += rng.choice([0, 0, 0, 1])
        in_discards += rng.choice([0, 0, 0, 1])
        out_discards += rng.choice([0, 0, 0, 1])

        rows.append(
            {
                "timestamp": ts,
                "device_id": device_config["device_id"],
                "device_name": device_config["device_name"],
                "interface_name": interface_config["interface_name"],
                "interface_index": interface_config["interface_index"],
                "oper_status": "up",
                "admin_status": "up",
                "in_traffic_bps": round(in_traffic, 2),
                "out_traffic_bps": round(out_traffic, 2),
                "in_errors": in_errors,
                "out_errors": out_errors,
                "in_discards": in_discards,
                "out_discards": out_discards,
                "packet_loss_pct": round(packet_loss, 2),
                "latency_ms": round(latency, 2),
                "device_cpu_pct": round(cpu, 2),
                "device_memory_pct": round(memory, 2),
                "device_availability": "true",
                "device_uptime_sec": uptime_start + i * step_seconds,
            }
        )

    return rows


def apply_high_utilization_scenario(
    metrics_rows: list[dict[str, Any]],
    interface_config: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Поднимает трафик до зоны высокой утилизации.
    """
    capacity_bps = float(interface_config["interface_speed_mbps"]) * 1_000_000
    target_in = capacity_bps * rng.uniform(0.78, 0.92)
    target_out = capacity_bps * rng.uniform(0.55, 0.80)

    for row in metrics_rows:
        row["in_traffic_bps"] = round(_normal_jitter(rng, target_in, rel=0.03), 2)
        row["out_traffic_bps"] = round(_normal_jitter(rng, target_out, rel=0.04), 2)
        row["latency_ms"] = round(row["latency_ms"] + rng.uniform(5, 20), 2)
        row["device_cpu_pct"] = round(_clamp(float(row["device_cpu_pct"]) + rng.uniform(8, 18), 0, 100), 2)

    return metrics_rows


def apply_packet_loss_scenario(
    metrics_rows: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Вносит деградацию по потерям и задержке.
    """
    n = len(metrics_rows)
    start_idx = max(n // 3, 0)

    for i, row in enumerate(metrics_rows):
        if i >= start_idx:
            growth = (i - start_idx + 1) / max(n - start_idx, 1)
            row["packet_loss_pct"] = round(rng.uniform(3.0, 12.0) * growth + rng.uniform(1.0, 4.0), 2)
            row["latency_ms"] = round(float(row["latency_ms"]) + rng.uniform(10, 35), 2)

    return metrics_rows


def apply_interface_errors_scenario(
    metrics_rows: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Вносит рост ошибок и discards.
    """
    in_errors_counter = 0
    out_errors_counter = 0
    in_discards_counter = 0
    out_discards_counter = 0

    for row in metrics_rows:
        in_errors_counter += rng.randint(3, 12)
        out_errors_counter += rng.randint(0, 4)
        in_discards_counter += rng.randint(1, 8)
        out_discards_counter += rng.randint(0, 3)

        row["in_errors"] = in_errors_counter
        row["out_errors"] = out_errors_counter
        row["in_discards"] = in_discards_counter
        row["out_discards"] = out_discards_counter
        row["packet_loss_pct"] = round(float(row["packet_loss_pct"]) + rng.uniform(0.5, 2.5), 2)
        row["latency_ms"] = round(float(row["latency_ms"]) + rng.uniform(2, 10), 2)

    return metrics_rows


def apply_flapping_scenario(
    metrics_rows: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Вносит нестабильность статуса интерфейса.
    """
    n = len(metrics_rows)
    if n < 4:
        return metrics_rows

    flap_points = sorted(set([
        max(1, n // 3),
        max(2, n // 2),
    ]))

    for idx in flap_points:
        if idx < n:
            metrics_rows[idx]["oper_status"] = "down"
            metrics_rows[idx]["in_traffic_bps"] = 0.0
            metrics_rows[idx]["out_traffic_bps"] = 0.0
            metrics_rows[idx]["packet_loss_pct"] = round(rng.uniform(20.0, 60.0), 2)
            metrics_rows[idx]["latency_ms"] = round(float(metrics_rows[idx]["latency_ms"]) + rng.uniform(30, 90), 2)

    return metrics_rows


def apply_down_scenario(
    metrics_rows: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Вносит длительное состояние down в конце окна.
    """
    n = len(metrics_rows)
    start_idx = max(n // 2, 0)

    for i in range(start_idx, n):
        metrics_rows[i]["oper_status"] = "down"
        metrics_rows[i]["in_traffic_bps"] = 0.0
        metrics_rows[i]["out_traffic_bps"] = 0.0
        metrics_rows[i]["packet_loss_pct"] = round(rng.uniform(40.0, 100.0), 2)
        metrics_rows[i]["latency_ms"] = round(float(metrics_rows[i]["latency_ms"]) + rng.uniform(40, 120), 2)

    return metrics_rows


def apply_scenario(
    metrics_rows: list[dict[str, Any]],
    interface_config: dict[str, Any],
    scenario_name: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Применяет выбранный сценарий к базовой серии.
    """
    scenario = scenario_name.strip().lower()

    if scenario == "normal":
        return metrics_rows
    if scenario == "high_utilization":
        return apply_high_utilization_scenario(metrics_rows, interface_config, rng)
    if scenario == "packet_loss":
        return apply_packet_loss_scenario(metrics_rows, rng)
    if scenario == "interface_errors":
        return apply_interface_errors_scenario(metrics_rows, rng)
    if scenario == "flapping":
        return apply_flapping_scenario(metrics_rows, rng)
    if scenario == "down":
        return apply_down_scenario(metrics_rows, rng)

    raise ValueError(f"Неизвестный сценарий: {scenario_name}")


def generate_events_from_metrics(
    metrics_rows: list[dict[str, Any]],
    scenario_name: str,
) -> list[dict[str, Any]]:
    """
    Строит события по уже сгенерированным метрикам и сценарию.
    """
    if not metrics_rows:
        return []

    events: list[dict[str, Any]] = []
    device_id = metrics_rows[0]["device_id"]
    device_name = metrics_rows[0]["device_name"]
    interface_name = metrics_rows[0]["interface_name"]
    interface_capacity_bps = 1_000_000_000

    previous_status = None
    high_loss_emitted = False
    high_util_emitted = False
    high_errors_emitted = False

    for row in metrics_rows:
        timestamp = row["timestamp"]
        current_status = str(row["oper_status"]).lower()

        if previous_status is None:
            previous_status = current_status

        if previous_status == "up" and current_status == "down":
            events.append(
                {
                    "timestamp": timestamp,
                    "device_id": device_id,
                    "device_name": device_name,
                    "interface_name": interface_name,
                    "event_type": "interface_down",
                    "severity": "high",
                    "event_status": "problem",
                }
            )

        if previous_status == "down" and current_status == "up":
            events.append(
                {
                    "timestamp": timestamp,
                    "device_id": device_id,
                    "device_name": device_name,
                    "interface_name": interface_name,
                    "event_type": "interface_recovered",
                    "severity": "info",
                    "event_status": "recovery",
                }
            )

        if float(row["packet_loss_pct"]) >= 5 and not high_loss_emitted:
            events.append(
                {
                    "timestamp": timestamp,
                    "device_id": device_id,
                    "device_name": device_name,
                    "interface_name": interface_name,
                    "event_type": "high_packet_loss",
                    "severity": "critical" if float(row["packet_loss_pct"]) >= 20 else "high",
                    "event_status": "problem",
                }
            )
            high_loss_emitted = True

        if (
            float(row["in_traffic_bps"]) > 0
            and float(row["in_traffic_bps"]) >= 0.75 * interface_capacity_bps
            and not high_util_emitted
        ):
            events.append(
                {
                    "timestamp": timestamp,
                    "device_id": device_id,
                    "device_name": device_name,
                    "interface_name": interface_name,
                    "event_type": "high_utilization",
                    "severity": "high",
                    "event_status": "problem",
                }
            )
            high_util_emitted = True

        if (int(row["in_errors"]) + int(row["out_errors"])) >= 30 and not high_errors_emitted:
            events.append(
                {
                    "timestamp": timestamp,
                    "device_id": device_id,
                    "device_name": device_name,
                    "interface_name": interface_name,
                    "event_type": "high_error_rate",
                    "severity": "high",
                    "event_status": "problem",
                }
            )
            high_errors_emitted = True

        previous_status = current_status

    _ = scenario_name
    return events


def generate_interface_dataset(
    device_config: dict[str, Any],
    interface_config: dict[str, Any],
    scenario_name: str,
    start_time: pd.Timestamp,
    duration_minutes: int,
    step_seconds: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Генерирует метрики и события по одному интерфейсу.
    """
    time_index = build_time_index(
        start_time=start_time,
        duration_minutes=duration_minutes,
        step_seconds=step_seconds,
    )

    metrics_rows = generate_normal_metrics_series(
        device_config=device_config,
        interface_config=interface_config,
        time_index=time_index,
        step_seconds=step_seconds,
        rng=rng,
    )
    metrics_rows = apply_scenario(
        metrics_rows=metrics_rows,
        interface_config=interface_config,
        scenario_name=scenario_name,
        rng=rng,
    )
    event_rows = generate_events_from_metrics(
        metrics_rows=metrics_rows,
        scenario_name=scenario_name,
    )

    return metrics_rows, event_rows


def generate_synthetic_csv_bundle(
    output_dir: str | Path,
    devices_config: list[dict[str, Any]],
    start_time: pd.Timestamp,
    duration_minutes: int,
    step_seconds: int,
    random_seed: int = 42,
) -> dict[str, Path]:
    """
    Генерирует полный комплект CSV:
    - device_context.csv
    - interface_metrics.csv
    - interface_events.csv
    """
    rng = random.Random(random_seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_context_rows: list[dict[str, Any]] = []
    all_metrics_rows: list[dict[str, Any]] = []
    all_event_rows: list[dict[str, Any]] = []

    for device in devices_config:
        device_base = {
            "device_id": device["device_id"],
            "device_name": device["device_name"],
            "device_vendor": device.get("device_vendor", "MikroTik"),
            "device_model": device.get("device_model", "RB1100AHx4 Dude Edition"),
            "base_cpu_pct": device.get("base_cpu_pct", 25.0),
            "base_memory_pct": device.get("base_memory_pct", 45.0),
            "device_uptime_start_sec": device.get("device_uptime_start_sec", 864000),
        }

        interfaces = device["interfaces"]
        all_context_rows.extend(generate_device_context_rows(device_base, interfaces))

        for iface in interfaces:
            scenario_name = iface.get("scenario_name", "normal")
            metrics_rows, event_rows = generate_interface_dataset(
                device_config=device_base,
                interface_config=iface,
                scenario_name=scenario_name,
                start_time=start_time,
                duration_minutes=duration_minutes,
                step_seconds=step_seconds,
                rng=rng,
            )
            all_metrics_rows.extend(metrics_rows)
            all_event_rows.extend(event_rows)

    context_df = pd.DataFrame(all_context_rows)
    metrics_df = pd.DataFrame(all_metrics_rows)
    events_df = pd.DataFrame(all_event_rows)

    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(["device_id", "interface_name", "timestamp"]).reset_index(drop=True)
    if not events_df.empty:
        events_df = events_df.sort_values(["device_id", "interface_name", "timestamp"]).reset_index(drop=True)
    if not context_df.empty:
        context_df = context_df.sort_values(["device_id", "interface_index"]).reset_index(drop=True)

    context_path = output_path / "device_context.csv"
    metrics_path = output_path / "interface_metrics.csv"
    events_path = output_path / "interface_events.csv"

    context_df.to_csv(context_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    events_df.to_csv(events_path, index=False)

    return {
        "device_context": context_path,
        "interface_metrics": metrics_path,
        "interface_events": events_path,
    }


def build_default_devices_config() -> list[dict[str, Any]]:
    """
    Стартовая конфигурация синтетического стенда.
    Имена устройств шаблонные:
    - device_id: r1, r2, ...
    - device_name: router1, router2, ...
    """
    return [
        {
            "device_id": "r1",
            "device_name": "router1",
            "device_vendor": "MikroTik",
            "device_model": "RB1100AHx4 Dude Edition",
            "interfaces": [
                {
                    "interface_name": "ether1",
                    "interface_index": 1,
                    "interface_role": "WAN",
                    "interface_speed_mbps": 1000,
                    "neighbor_device": "isp1",
                    "scenario_name": "normal",
                    "base_in_traffic_bps": 180_000_000,
                    "base_out_traffic_bps": 90_000_000,
                },
                {
                    "interface_name": "ether2",
                    "interface_index": 2,
                    "interface_role": "uplink",
                    "interface_speed_mbps": 1000,
                    "neighbor_device": "router2",
                    "scenario_name": "packet_loss",
                    "base_in_traffic_bps": 220_000_000,
                    "base_out_traffic_bps": 100_000_000,
                },
            ],
        },
        {
            "device_id": "r2",
            "device_name": "router2",
            "device_vendor": "MikroTik",
            "device_model": "RB1100AHx4 Dude Edition",
            "interfaces": [
                {
                    "interface_name": "ether1",
                    "interface_index": 1,
                    "interface_role": "WAN",
                    "interface_speed_mbps": 1000,
                    "neighbor_device": "router1",
                    "scenario_name": "interface_errors",
                    "base_in_traffic_bps": 140_000_000,
                    "base_out_traffic_bps": 70_000_000,
                },
                {
                    "interface_name": "ether2",
                    "interface_index": 2,
                    "interface_role": "backup_uplink",
                    "interface_speed_mbps": 1000,
                    "neighbor_device": "router3",
                    "scenario_name": "flapping",
                    "base_in_traffic_bps": 90_000_000,
                    "base_out_traffic_bps": 45_000_000,
                },
            ],
        },
        {
            "device_id": "r3",
            "device_name": "router3",
            "device_vendor": "MikroTik",
            "device_model": "RB1100AHx4 Dude Edition",
            "interfaces": [
                {
                    "interface_name": "ether1",
                    "interface_index": 1,
                    "interface_role": "WAN",
                    "interface_speed_mbps": 1000,
                    "neighbor_device": "router2",
                    "scenario_name": "high_utilization",
                    "base_in_traffic_bps": 300_000_000,
                    "base_out_traffic_bps": 180_000_000,
                },
                {
                    "interface_name": "ether2",
                    "interface_index": 2,
                    "interface_role": "transit",
                    "interface_speed_mbps": 1000,
                    "neighbor_device": "core1",
                    "scenario_name": "down",
                    "base_in_traffic_bps": 110_000_000,
                    "base_out_traffic_bps": 60_000_000,
                },
            ],
        },
    ]


def main() -> None:
    devices_config = build_default_devices_config()

    saved_paths = generate_synthetic_csv_bundle(
        output_dir=DEFAULT_OUTPUT_DIR,
        devices_config=devices_config,
        start_time=DEFAULT_START_TIME,
        duration_minutes=DEFAULT_DURATION_MINUTES,
        step_seconds=DEFAULT_STEP_SECONDS,
        random_seed=42,
    )

    context_df = pd.read_csv(saved_paths["device_context"])
    metrics_df = pd.read_csv(saved_paths["interface_metrics"])
    events_df = pd.read_csv(saved_paths["interface_events"])

    print("=== SYNTHETIC CSV GENERATED ===")
    for name, path in saved_paths.items():
        print(f"{name}: {path}")

    print("\n=== SUMMARY ===")
    print(f"devices: {context_df['device_id'].nunique() if not context_df.empty else 0}")
    print(f"interfaces: {len(context_df)}")
    print(f"metric rows: {len(metrics_df)}")
    print(f"event rows: {len(events_df)}")

    print("\nFiles are saved in data/synthetic/")
    print("They are format-compatible with the main project input.")
    print("If needed, copy them manually into data/raw/ before running the main pipeline.")


if __name__ == "__main__":
    main()
