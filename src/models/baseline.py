from __future__ import annotations

from typing import Any


PROBLEM_TYPE_PRIORITY = [
    "down",
    "flapping",
    "packet_loss",
    "interface_errors",
    "high_utilization",
    "device_side_issue",
    "unknown",
    "none",
]

SEVERITY_PRIORITY = {
    "critical": 3,
    "degraded": 2,
    "warning": 1,
    "normal": 0,
}


def _rule(
    rule_id: str,
    severity: str,
    problem_type: str,
    comment_template: str,
) -> dict[str, str]:
    return {
        "rule_id": rule_id,
        "severity": severity,
        "problem_type": problem_type,
        "comment_template": comment_template,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Безопасно приводит значение к float.
    """
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    """
    Безопасно приводит значение к bool.
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    value_str = str(value).strip().lower()
    if value_str in {"true", "1", "yes", "up", "available"}:
        return True
    if value_str in {"false", "0", "no", "down", "unavailable"}:
        return False

    return default


def _safe_str(value: Any, default: str = "") -> str:
    """
    Безопасно приводит значение к строке.
    """
    if value is None:
        return default
    return str(value).strip()


def _normalize_window_for_baseline(interface_window: dict[str, Any]) -> dict[str, Any]:
    """
    Нормализует окно перед применением baseline-правил.

    Здесь мы:
    - гарантируем наличие ключевых полей;
    - приводим типы к ожидаемым;
    - не меняем смысл признаков.
    """
    normalized = dict(interface_window)

    # Строковые поля
    normalized["device_name"] = _safe_str(normalized.get("device_name"), "unknown_device")
    normalized["interface_name"] = _safe_str(normalized.get("interface_name"), "unknown_interface")
    normalized["oper_status_last"] = _safe_str(normalized.get("oper_status_last"), "").lower()
    normalized["admin_status_last"] = _safe_str(normalized.get("admin_status_last"), "").lower()

    # Числовые поля
    numeric_fields = [
        "status_change_count",
        "flap_event_count",
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
    for field in numeric_fields:
        normalized[field] = _safe_float(normalized.get(field), 0.0)

    normalized["device_availability_flag"] = _safe_bool(
        normalized.get("device_availability_flag"),
        default=True,
    )

    # Если флаг burst не передан, вычисляем его из errors_total_delta.
    if "error_burst_flag" not in normalized or normalized.get("error_burst_flag") is None:
        normalized["error_burst_flag"] = normalized["errors_total_delta"] >= 50
    else:
        normalized["error_burst_flag"] = _safe_bool(normalized.get("error_burst_flag"), default=False)

    return normalized


def check_critical_rules(interface_window: dict[str, Any]) -> list[dict[str, str]]:
    window = _normalize_window_for_baseline(interface_window)
    matched: list[dict[str, str]] = []

    if window["device_availability_flag"] is False:
        matched.append(
            _rule(
                "C1_DEVICE_UNAVAILABLE",
                "critical",
                "device_side_issue",
                "Устройство недоступно в рассматриваемом окне. Локальная оценка интерфейса ограничена.",
            )
        )

    if window["admin_status_last"] == "up" and window["oper_status_last"] == "down":
        matched.append(
            _rule(
                "C2_INTERFACE_DOWN",
                "critical",
                "down",
                "Интерфейс административно включён, но фактически находится в состоянии down.",
            )
        )

    if window["status_change_count"] >= 3 or window["flap_event_count"] >= 3:
        matched.append(
            _rule(
                "C3_STRONG_FLAPPING",
                "critical",
                "flapping",
                "Наблюдается выраженная нестабильность интерфейса: за окно зафиксированы многократные смены состояния.",
            )
        )

    if window["packet_loss_avg_pct"] >= 20 or window["packet_loss_max_pct"] >= 40:
        matched.append(
            _rule(
                "C4_CRITICAL_PACKET_LOSS",
                "critical",
                "packet_loss",
                "Зафиксированы критически высокие потери пакетов.",
            )
        )

    if window["errors_total_delta"] >= 100 or (
        bool(window["error_burst_flag"]) and window["errors_total_delta"] >= 50
    ):
        matched.append(
            _rule(
                "C5_CRITICAL_ERRORS",
                "critical",
                "interface_errors",
                "Зафиксирован критический рост ошибок на интерфейсе.",
            )
        )

    return matched


def check_degraded_rules(interface_window: dict[str, Any]) -> list[dict[str, str]]:
    window = _normalize_window_for_baseline(interface_window)
    matched: list[dict[str, str]] = []

    if window["status_change_count"] in [1, 2]:
        matched.append(
            _rule(
                "D1_MODERATE_FLAPPING",
                "degraded",
                "flapping",
                "Интерфейс нестабилен: в течение окна наблюдались смены состояния.",
            )
        )

    if window["packet_loss_avg_pct"] >= 5 or window["packet_loss_max_pct"] >= 10:
        matched.append(
            _rule(
                "D2_PACKET_LOSS_DEGRADED",
                "degraded",
                "packet_loss",
                "Наблюдаются устойчивые потери пакетов, что указывает на деградацию качества канала.",
            )
        )

    if window["errors_total_delta"] >= 20 or window["discards_total_delta"] >= 20:
        matched.append(
            _rule(
                "D3_ERRORS_DEGRADED",
                "degraded",
                "interface_errors",
                "На интерфейсе наблюдается рост ошибок или отброшенных пакетов.",
            )
        )

    if window["latency_avg_ms"] >= 50 or window["latency_max_ms"] >= 100:
        matched.append(
            _rule(
                "D4_HIGH_LATENCY",
                "degraded",
                "unknown",
                "Наблюдается повышенная задержка на канале.",
            )
        )

    if (
        window["utilization_peak_pct"] >= 90
        or window["utilization_in_avg_pct"] >= 80
        or window["utilization_out_avg_pct"] >= 80
    ):
        matched.append(
            _rule(
                "D5_HIGH_UTILIZATION",
                "degraded",
                "high_utilization",
                "Интерфейс работает с высокой загрузкой и близок к насыщению.",
            )
        )

    return matched


def check_warning_rules(interface_window: dict[str, Any]) -> list[dict[str, str]]:
    window = _normalize_window_for_baseline(interface_window)
    matched: list[dict[str, str]] = []

    if 0 < window["packet_loss_avg_pct"] < 5:
        matched.append(
            _rule(
                "W1_LIGHT_PACKET_LOSS",
                "warning",
                "packet_loss",
                "На интерфейсе зафиксированы небольшие потери пакетов.",
            )
        )

    if (0 < window["errors_total_delta"] < 20) or (0 < window["discards_total_delta"] < 20):
        matched.append(
            _rule(
                "W2_LIGHT_ERRORS",
                "warning",
                "interface_errors",
                "На интерфейсе появились ошибки или отброшенные пакеты, но их объём пока невелик.",
            )
        )

    if 75 <= window["utilization_peak_pct"] < 90:
        matched.append(
            _rule(
                "W3_ELEVATED_UTILIZATION",
                "warning",
                "high_utilization",
                "Интерфейс периодически работает под повышенной нагрузкой.",
            )
        )

    if window["device_cpu_avg_pct"] >= 75:
        matched.append(
            _rule(
                "W4_DEVICE_CPU_WARNING",
                "warning",
                "device_side_issue",
                "Средняя загрузка CPU устройства повышена.",
            )
        )

    return matched


def select_state_label(matched_rules: list[dict[str, str]]) -> str:
    """
    Выбирает итоговый класс состояния.
    """
    if not matched_rules:
        return "normal"

    best_rule = max(matched_rules, key=lambda x: SEVERITY_PRIORITY.get(x["severity"], -1))
    return best_rule["severity"]


def select_problem_type(matched_rules: list[dict[str, str]]) -> str:
    """
    Выбирает итоговый тип проблемы по приоритету.
    """
    if not matched_rules:
        return "none"

    problem_types = {rule["problem_type"] for rule in matched_rules}
    for problem_type in PROBLEM_TYPE_PRIORITY:
        if problem_type in problem_types:
            return problem_type
    return "unknown"


def build_comment_template(
    state_label: str,
    problem_type_label: str,
    interface_window: dict[str, Any],
    matched_rule_ids: list[str] | None = None,
) -> str:
    """
    Строит понятный комментарий без генеративной логики.
    """
    window = _normalize_window_for_baseline(interface_window)

    interface_name = window.get("interface_name", "unknown_interface")
    device_name = window.get("device_name", "unknown_device")

    if problem_type_label == "down":
        return (
            f"Интерфейс {interface_name} на устройстве {device_name} административно включён, "
            "но фактически находится в состоянии down. Требуется проверить линк, порт и соседнее устройство."
        )

    if problem_type_label == "flapping":
        return (
            f"Интерфейс {interface_name} на устройстве {device_name} нестабилен: "
            "за окно наблюдались смены состояния. Рекомендуется проверить физический линк."
        )

    if problem_type_label == "packet_loss":
        return (
            f"На интерфейсе {interface_name} устройства {device_name} обнаружены потери пакетов. "
            "Требуется проверить качество канала, нагрузку и состояние смежного участка сети."
        )

    if problem_type_label == "interface_errors":
        return (
            f"На интерфейсе {interface_name} устройства {device_name} растёт число ошибок "
            "или отброшенных пакетов. Следует проверить кабель, порт и counters интерфейса."
        )

    if problem_type_label == "high_utilization":
        return (
            f"Интерфейс {interface_name} устройства {device_name} работает под высокой нагрузкой. "
            "Следует проверить структуру трафика и риск насыщения канала."
        )

    if problem_type_label == "device_side_issue":
        return (
            f"Для устройства {device_name} наблюдаются признаки общей проблемы, "
            "которая может влиять на состояние интерфейса."
        )

    if state_label == "normal":
        return (
            f"Интерфейс {interface_name} на устройстве {device_name} работает в штатном режиме, "
            "выраженных признаков деградации не обнаружено."
        )

    suffix = ""
    if matched_rule_ids:
        suffix = f" Сработали правила: {', '.join(matched_rule_ids)}."

    return (
        f"Для интерфейса {interface_name} на устройстве {device_name} "
        f"обнаружено состояние {state_label} с типом проблемы {problem_type_label}.{suffix}"
    )


def evaluate_interface_window(interface_window: dict[str, Any]) -> dict[str, Any]:
    """
    Главная функция baseline-оценки одного interface_window.
    """
    window = _normalize_window_for_baseline(interface_window)

    critical_rules = check_critical_rules(window)
    degraded_rules = check_degraded_rules(window) if not critical_rules else []
    warning_rules = check_warning_rules(window) if not critical_rules and not degraded_rules else []

    matched_rules = critical_rules + degraded_rules + warning_rules

    state_label = select_state_label(matched_rules)
    problem_type_label = select_problem_type(matched_rules)
    matched_rule_ids = [rule["rule_id"] for rule in matched_rules]

    comment_template = build_comment_template(
        state_label=state_label,
        problem_type_label=problem_type_label,
        interface_window=window,
        matched_rule_ids=matched_rule_ids,
    )

    return {
        "record_id": window.get("record_id"),
        "state_label": state_label,
        "problem_type_label": problem_type_label,
        "comment_template": comment_template,
        "matched_rule_ids": matched_rule_ids,
    }


def evaluate_interface_windows_dataset(windows_df_or_list) -> list[dict[str, Any]]:
    """
    Применяет baseline к набору interface_window.
    """
    if hasattr(windows_df_or_list, "to_dict"):
        windows = windows_df_or_list.to_dict(orient="records")
    else:
        windows = list(windows_df_or_list)

    return [evaluate_interface_window(window) for window in windows]
