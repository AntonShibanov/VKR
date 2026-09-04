"""
Проверка финальной формулировки гипотезы: снижение среднего времени отклика
(против всех бейзлайнов) и снижение доли катастрофических решений
(против "умных" одношаговых методов).
"""
import pandas as pd
from src.stats import bootstrap_reduction_ci

RL = "rl_dqn"                                                          # ключ метода RL в CSV
BLIND_BASELINES = ["round_robin", "hash"]                                 # методы, не использующие состояние системы
SMART_BASELINES = ["cost_threshold", "gbm_static", "bandit_lints"]         # одношаговые адаптивные методы
ALL_BASELINES = BLIND_BASELINES + SMART_BASELINES                           # объединённый список всех бейзлайнов
PRACTICAL_THRESHOLD = 0.10                                                    # порог практической значимости эффекта (10%)
SIGNIFICANCE_THRESHOLD = 0.0                                                     # порог статистической значимости (просто > 0)


def _series(summary, method, column):
    """Извлечь отсортированный по run_id массив значений колонки для заданного метода."""
    return summary[summary.method == method].sort_values("run_id")[column].values


def main():
    summary = pd.read_csv("results/episode_summary.csv")                          # читаем итоговую таблицу метрик
    rows = []                                                                        # список строк будущей итоговой таблицы

    # --- условие (1): среднее время отклика ---
    for base in ALL_BASELINES:                                                        # для каждого базового метода
        threshold = PRACTICAL_THRESHOLD if base in BLIND_BASELINES else SIGNIFICANCE_THRESHOLD  # разный порог для разных групп
        ci = bootstrap_reduction_ci(_series(summary, RL, "mean_latency_ms"),
                                     _series(summary, base, "mean_latency_ms"), threshold=threshold)  # бутстрэп-ДИ эффекта
        rows.append({
            "metric": "mean_latency", "baseline": base,
            "reduction_mean": round(ci["reduction_mean"], 3),                            # среднее снижение
            "ci_low": round(ci["ci_low"], 3), "ci_high": round(ci["ci_high"], 3),           # границы 95% ДИ
            "threshold": threshold, "p_value": round(ci["p_value"], 4),                       # порог и p-value
            "confirmed": ci["ci_low"] > threshold,                                              # подтверждён ли эффект
        })

    # --- условие (2): доля катастрофических решений — только против "умных" методов ---
    for base in SMART_BASELINES:                                                          # только против одношаговых адаптивных методов
        ci = bootstrap_reduction_ci(_series(summary, RL, "catastrophic_rate"),
                                     _series(summary, base, "catastrophic_rate"),
                                     threshold=PRACTICAL_THRESHOLD)                            # бутстрэп-ДИ снижения доли катастроф
        rows.append({
            "metric": "catastrophic_rate", "baseline": base,
            "reduction_mean": round(ci["reduction_mean"], 3),
            "ci_low": round(ci["ci_low"], 3), "ci_high": round(ci["ci_high"], 3),
            "threshold": PRACTICAL_THRESHOLD, "p_value": round(ci["p_value"], 4),
            "confirmed": ci["ci_low"] > PRACTICAL_THRESHOLD,
        })

    # --- диагностика (не входит в условия подтверждения гипотезы) ---
    for base in ALL_BASELINES:                                                              # P95 считается только для справки
        ci = bootstrap_reduction_ci(_series(summary, RL, "p95_latency_ms"),
                                     _series(summary, base, "p95_latency_ms"), threshold=0.0)
        rows.append({
            "metric": "p95_latency_diagnostic", "baseline": base,
            "reduction_mean": round(ci["reduction_mean"], 3),
            "ci_low": round(ci["ci_low"], 3), "ci_high": round(ci["ci_high"], 3),
            "threshold": None, "p_value": round(ci["p_value"], 4), "confirmed": None,           # confirmed=None: не проверяем условие
        })

    out = pd.DataFrame(rows)                                                                       # собираем таблицу из всех строк
    out.to_csv("results/hypothesis_test.csv", index=False)                                            # сохраняем результат в CSV
    print(out.to_string(index=False))                                                                    # печатаем таблицу в консоль

    part1 = out[(out.metric == "mean_latency")]                                                              # строки условия (1)
    part2 = out[(out.metric == "catastrophic_rate")]                                                           # строки условия (2)
    print(f"\nУсловие (1) выполнено полностью: {bool(part1['confirmed'].all())}")                                 # проверка первого условия
    print(f"Условие (2) выполнено полностью: {bool(part2['confirmed'].all())}")                                     # проверка второго условия
    print(f"Гипотеза подтверждена целиком: {bool(part1['confirmed'].all() and part2['confirmed'].all())}")            # общий вывод


if __name__ == "__main__":
    main()