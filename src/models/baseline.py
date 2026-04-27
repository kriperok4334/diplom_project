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


def check_critical_rules(interface_window: dict[str, Any]) -> list[dict[str, str]]:
    matched: list[dict[str, str]] = []

    if interface_window.get("device_availability_flag") is False:
        matched.append(
            _rule(
                "C1_DEVICE_UNAVAILABLE",
                "critical",
                "device_side_issue",
                "Устройство недоступно в рассматриваемом окне. Локальная оценка интерфейса ограничена.",
            )
        )

    if (
        str(interface_window.get("admin_status_last", "")).lower() == "up"
        and str(interface_window.get("oper_status_last", "")).lower() == "down"
    ):
        matched.append(
            _rule(
                "C2_INTERFACE_DOWN",
                "critical",
                "down",
                "Интерфейс административно включён, но фактически находится в состоянии down.",
            )
        )

    if (
        interface_window.get("status_change_count", 0) >= 3
        or interface_window.get("flap_event_count", 0) >= 3
    ):
        matched.append(
            _rule(
                "C3_STRONG_FLAPPING",
                "critical",
                "flapping",
                "Наблюдается выраженная нестабильность интерфейса: за окно зафиксированы многократные смены состояния.",
            )
        )

    if (
        interface_window.get("packet_loss_avg_pct", 0) >= 20
        or interface_window.get("packet_loss_max_pct", 0) >= 40
    ):
        matched.append(
            _rule(
                "C4_CRITICAL_PACKET_LOSS",
                "critical",
                "packet_loss",
                "Зафиксированы критически высокие потери пакетов.",
            )
        )

    if (
        interface_window.get("errors_total_delta", 0) >= 100
        or (
            bool(interface_window.get("error_burst_flag", False))
            and interface_window.get("errors_total_delta", 0) >= 50
        )
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
    matched: list[dict[str, str]] = []

    if interface_window.get("status_change_count", 0) in [1, 2]:
        matched.append(
            _rule(
                "D1_MODERATE_FLAPPING",
                "degraded",
                "flapping",
                "Интерфейс нестабилен: в течение окна наблюдались смены состояния.",
            )
        )

    if (
        interface_window.get("packet_loss_avg_pct", 0) >= 5
        or interface_window.get("packet_loss_max_pct", 0) >= 10
    ):
        matched.append(
            _rule(
                "D2_PACKET_LOSS_DEGRADED",
                "degraded",
                "packet_loss",
                "Наблюдаются устойчивые потери пакетов, что указывает на деградацию качества канала.",
            )
        )

    if (
        interface_window.get("errors_total_delta", 0) >= 20
        or interface_window.get("discards_total_delta", 0) >= 20
    ):
        matched.append(
            _rule(
                "D3_ERRORS_DEGRADED",
                "degraded",
                "interface_errors",
                "На интерфейсе наблюдается рост ошибок или отброшенных пакетов.",
            )
        )

    if (
        interface_window.get("latency_avg_ms", 0) >= 50
        or interface_window.get("latency_max_ms", 0) >= 100
    ):
        matched.append(
            _rule(
                "D4_HIGH_LATENCY",
                "degraded",
                "unknown",
                "Наблюдается повышенная задержка на канале.",
            )
        )

    if (
        interface_window.get("utilization_peak_pct", 0) >= 90
        or interface_window.get("utilization_in_avg_pct", 0) >= 80
        or interface_window.get("utilization_out_avg_pct", 0) >= 80
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
    matched: list[dict[str, str]] = []

    packet_loss_avg_pct = interface_window.get("packet_loss_avg_pct", 0)
    if 0 < packet_loss_avg_pct < 5:
        matched.append(
            _rule(
                "W1_LIGHT_PACKET_LOSS",
                "warning",
                "packet_loss",
                "На интерфейсе зафиксированы небольшие потери пакетов.",
            )
        )

    errors_total_delta = interface_window.get("errors_total_delta", 0)
    discards_total_delta = interface_window.get("discards_total_delta", 0)
    if (0 < errors_total_delta < 20) or (0 < discards_total_delta < 20):
        matched.append(
            _rule(
                "W2_LIGHT_ERRORS",
                "warning",
                "interface_errors",
                "На интерфейсе появились ошибки или отброшенные пакеты, но их объём пока невелик.",
            )
        )

    utilization_peak_pct = interface_window.get("utilization_peak_pct", 0)
    if 75 <= utilization_peak_pct < 90:
        matched.append(
            _rule(
                "W3_ELEVATED_UTILIZATION",
                "warning",
                "high_utilization",
                "Интерфейс периодически работает под повышенной нагрузкой.",
            )
        )

    if interface_window.get("device_cpu_avg_pct", 0) >= 75:
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
    """Выбирает итоговый класс состояния."""
    if not matched_rules:
        return "normal"

    best_rule = max(matched_rules, key=lambda x: SEVERITY_PRIORITY.get(x["severity"], -1))
    return best_rule["severity"]


def select_problem_type(matched_rules: list[dict[str, str]]) -> str:
    """Выбирает итоговый тип проблемы по приоритету."""
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
    """Строит человеко-понятный комментарий."""
    interface_name = interface_window.get("interface_name", "unknown_interface")
    device_name = interface_window.get("device_name", "unknown_device")

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
    critical_rules = check_critical_rules(interface_window)
    degraded_rules = check_degraded_rules(interface_window) if not critical_rules else []
    warning_rules = check_warning_rules(interface_window) if not critical_rules and not degraded_rules else []

    matched_rules = critical_rules + degraded_rules + warning_rules

    state_label = select_state_label(matched_rules)
    problem_type_label = select_problem_type(matched_rules)
    matched_rule_ids = [rule["rule_id"] for rule in matched_rules]

    comment_template = build_comment_template(
        state_label=state_label,
        problem_type_label=problem_type_label,
        interface_window=interface_window,
        matched_rule_ids=matched_rule_ids,
    )

    return {
        "record_id": interface_window.get("record_id"),
        "state_label": state_label,
        "problem_type_label": problem_type_label,
        "comment_template": comment_template,
        "matched_rule_ids": matched_rule_ids,
    }


def evaluate_interface_windows_dataset(windows_df_or_list) -> list[dict[str, Any]]:
    """Применяет baseline к набору interface_window."""
    if hasattr(windows_df_or_list, "to_dict"):
        windows = windows_df_or_list.to_dict(orient="records")
    else:
        windows = list(windows_df_or_list)

    return [evaluate_interface_window(window) for window in windows]
