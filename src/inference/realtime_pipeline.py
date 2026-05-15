from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.features.sequence_features import DEFAULT_HISTORY_LENGTH
from src.inference.predict_next_window import predict_next_window_from_history
from src.models.baseline import evaluate_interface_window


def get_history_key(window: dict[str, Any]) -> tuple[str, str]:
    """
    Возвращает ключ истории для одного интерфейса.
    """
    return (
        str(window.get("device_id")),
        str(window.get("interface_name")),
    )


def update_window_history(
    history_store: dict[tuple[str, str], list[dict[str, Any]]],
    new_window: dict[str, Any],
    history_length: int = DEFAULT_HISTORY_LENGTH,
) -> list[dict[str, Any]]:
    """
    Обновляет историю окон для одного интерфейса.
    """
    key = get_history_key(new_window)

    if key not in history_store:
        history_store[key] = []

    history_store[key].append(new_window)

    history_store[key] = sorted(
        history_store[key],
        key=lambda row: pd.Timestamp(row["window_start"]),
    )

    if len(history_store[key]) > history_length:
        history_store[key] = history_store[key][-history_length:]

    return history_store[key]


def analyze_current_window(
    current_window: dict[str, Any],
) -> dict[str, Any]:
    """
    Анализирует текущее окно с помощью baseline-логики.
    """
    baseline_result = evaluate_interface_window(current_window)

    return {
        "current_state_label": baseline_result.get("state_label"),
        "current_problem_type_label": baseline_result.get("problem_type_label"),
        "current_comment": baseline_result.get("comment_template"),
        "current_matched_rule_ids": baseline_result.get("matched_rule_ids", []),
    }


def convert_predicted_next_window_for_baseline(
    predicted_next_window: dict[str, Any],
    last_known_window: dict[str, Any],
) -> dict[str, Any]:
    """
    Преобразует predicted_next_window в структуру, совместимую с baseline.py.
    """
    predicted_window_start = predicted_next_window["predicted_next_window_start"]
    predicted_window_end = predicted_next_window["predicted_next_window_end"]

    window_size_sec = int(
        (pd.Timestamp(predicted_window_end) - pd.Timestamp(predicted_window_start)).total_seconds()
    )

    converted = {
        "record_id": (
            f"{last_known_window.get('device_id')}-"
            f"{last_known_window.get('interface_name')}-"
            f"{pd.Timestamp(predicted_window_start).isoformat()}-"
            f"{pd.Timestamp(predicted_window_end).isoformat()}"
        ),
        "object_type": "interface",
        "schema_version": last_known_window.get("schema_version", "1.0"),
        "device_id": last_known_window.get("device_id"),
        "device_name": last_known_window.get("device_name"),
        "device_vendor": last_known_window.get("device_vendor"),
        "device_model": last_known_window.get("device_model"),
        "interface_id": last_known_window.get("interface_id"),
        "interface_name": last_known_window.get("interface_name"),
        "interface_index": last_known_window.get("interface_index"),
        "interface_role": last_known_window.get("interface_role"),
        "interface_speed_mbps": last_known_window.get("interface_speed_mbps"),
        "neighbor_device": last_known_window.get("neighbor_device"),
        "window_start": predicted_window_start,
        "window_end": predicted_window_end,
        "window_size_sec": window_size_sec,
        "aggregation_step_sec": last_known_window.get("aggregation_step_sec", 60),
        "oper_status_last": last_known_window.get("oper_status_last", "up"),
        "admin_status_last": last_known_window.get("admin_status_last", "up"),
        "status_change_count": predicted_next_window.get("predicted_status_change_count", 0.0),
        "down_seconds_total": predicted_next_window.get("predicted_down_seconds_total", 0.0),
        "errors_total_delta": predicted_next_window.get("predicted_errors_total_delta", 0.0),
        "discards_total_delta": predicted_next_window.get("predicted_discards_total_delta", 0.0),
        "error_burst_flag": predicted_next_window.get("predicted_errors_total_delta", 0.0) >= 50,
        "packet_loss_avg_pct": predicted_next_window.get("predicted_packet_loss_avg_pct", 0.0),
        "packet_loss_max_pct": predicted_next_window.get("predicted_packet_loss_max_pct", 0.0),
        "latency_avg_ms": predicted_next_window.get("predicted_latency_avg_ms", 0.0),
        "latency_max_ms": predicted_next_window.get("predicted_latency_max_ms", 0.0),
        "utilization_in_avg_pct": predicted_next_window.get("predicted_utilization_in_avg_pct", 0.0),
        "utilization_out_avg_pct": predicted_next_window.get("predicted_utilization_out_avg_pct", 0.0),
        "utilization_peak_pct": predicted_next_window.get("predicted_utilization_peak_pct", 0.0),
        "device_availability_flag": last_known_window.get("device_availability_flag", True),
        "device_cpu_avg_pct": predicted_next_window.get("predicted_device_cpu_avg_pct", 0.0),
        "device_memory_avg_pct": predicted_next_window.get("predicted_device_memory_avg_pct", 0.0),
        "in_traffic_avg_bps": last_known_window.get("in_traffic_avg_bps", 0.0),
        "out_traffic_avg_bps": last_known_window.get("out_traffic_avg_bps", 0.0),
        "in_traffic_max_bps": last_known_window.get("in_traffic_max_bps", 0.0),
        "out_traffic_max_bps": last_known_window.get("out_traffic_max_bps", 0.0),
        "traffic_asymmetry_ratio": last_known_window.get("traffic_asymmetry_ratio", 0.0),
        "in_errors_delta": last_known_window.get("in_errors_delta", 0.0),
        "out_errors_delta": last_known_window.get("out_errors_delta", 0.0),
        "in_discards_delta": last_known_window.get("in_discards_delta", 0.0),
        "out_discards_delta": last_known_window.get("out_discards_delta", 0.0),
        "jitter_avg_ms": last_known_window.get("jitter_avg_ms", pd.NA),
        "alert_count_total": last_known_window.get("alert_count_total", 0),
        "alert_count_critical": last_known_window.get("alert_count_critical", 0),
        "recovery_event_count": last_known_window.get("recovery_event_count", 0),
        "flap_event_count": last_known_window.get("flap_event_count", 0),
        "event_types_seen": last_known_window.get("event_types_seen", []),
        "device_cpu_max_pct": predicted_next_window.get("predicted_device_cpu_avg_pct", 0.0),
        "problematic_interfaces_count": last_known_window.get("problematic_interfaces_count", 0),
        "device_uptime_sec": last_known_window.get("device_uptime_sec", pd.NA),
    }

    return converted


def analyze_predicted_next_window(
    predicted_next_window: dict[str, Any],
    last_known_window: dict[str, Any],
) -> dict[str, Any]:
    """
    Интерпретирует прогнозное окно через baseline-логику.
    """
    baseline_ready_window = convert_predicted_next_window_for_baseline(
        predicted_next_window=predicted_next_window,
        last_known_window=last_known_window,
    )

    baseline_result = evaluate_interface_window(baseline_ready_window)

    return {
        "predicted_next_state_label": baseline_result.get("state_label"),
        "predicted_next_problem_type_label": baseline_result.get("problem_type_label"),
        "predicted_next_comment": baseline_result.get("comment_template"),
        "predicted_next_matched_rule_ids": baseline_result.get("matched_rule_ids", []),
        "predicted_next_window_baseline_ready": baseline_ready_window,
    }


def build_realtime_interface_analysis_result(
    current_window: dict[str, Any],
    current_analysis: dict[str, Any],
    predicted_next_window: dict[str, Any] | None,
    predicted_analysis: dict[str, Any] | None,
    prediction_model_version: str = "predictor_v1",
    history_length_used: int | None = None,
) -> dict[str, Any]:
    """
    Собирает итоговый объект результата.
    """
    result = {
        "device_id": current_window.get("device_id"),
        "device_name": current_window.get("device_name"),
        "device_vendor": current_window.get("device_vendor"),
        "device_model": current_window.get("device_model"),
        "interface_id": current_window.get("interface_id"),
        "interface_name": current_window.get("interface_name"),
        "interface_role": current_window.get("interface_role"),
        "current_window_start": current_window.get("window_start"),
        "current_window_end": current_window.get("window_end"),
        "current_window_size_sec": current_window.get("window_size_sec"),
        **current_analysis,
        "history_length_used": history_length_used,
        "prediction_model_version": prediction_model_version,
        "prediction_timestamp": pd.Timestamp(datetime.utcnow()),
    }

    if predicted_next_window is not None:
        result.update(predicted_next_window)

    if predicted_analysis is not None:
        result.update(predicted_analysis)

    return result


def run_realtime_cycle(
    current_window: dict[str, Any],
    history_store: dict[tuple[str, str], list[dict[str, Any]]],
    predictor_bundle: dict[str, Any] | None = None,
    prediction_model_version: str = "predictor_v1",
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Один полный цикл обработки.
    """
    current_analysis = analyze_current_window(current_window)

    if predictor_bundle is not None:
        history_length = predictor_bundle.get("history_length") or DEFAULT_HISTORY_LENGTH
    else:
        history_length = DEFAULT_HISTORY_LENGTH

    updated_history = update_window_history(
        history_store=history_store,
        new_window=current_window,
        history_length=history_length,
    )

    predicted_next_window = None
    predicted_analysis = None

    if predictor_bundle is not None and len(updated_history) >= history_length:
        history_df = pd.DataFrame(updated_history)

        predicted_next_window = predict_next_window_from_history(
            predictor_bundle=predictor_bundle,
            history_df=history_df,
            device=device,
        )

        predicted_analysis = analyze_predicted_next_window(
            predicted_next_window=predicted_next_window,
            last_known_window=current_window,
        )

    realtime_result = build_realtime_interface_analysis_result(
        current_window=current_window,
        current_analysis=current_analysis,
        predicted_next_window=predicted_next_window,
        predicted_analysis=predicted_analysis,
        prediction_model_version=prediction_model_version,
        history_length_used=len(updated_history),
    )

    return realtime_result