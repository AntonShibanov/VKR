"""
Статистическая проверка эффекта методом парного бутстрэпа по прогонам.
"""
import numpy as np


def bootstrap_reduction_ci(rl_values, baseline_values, threshold=0.0, n_boot=5000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)                                         # генератор случайных чисел бутстрэпа
    rl, bl = np.asarray(rl_values), np.asarray(baseline_values)                  # значения метрики RL и бейзлайна по прогонам
    n = len(rl)                                                                   # число прогонов
    boot = np.empty(n_boot)                                                        # массив для бутстрэп-повторов
    for i in range(n_boot):                                                          # выполняем n_boot бутстрэп-повторов
        idx = rng.integers(0, n, size=n)                                              # случайные индексы ПРОГОНОВ (с повторением)
        boot[i] = 1 - rl[idx].mean() / bl[idx].mean()                                  # относительное снижение метрики RL против бейзлайна
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])              # границы 95%-го доверительного интервала
    p_value = float((1 + np.sum(boot <= threshold)) / (n_boot + 1))                       # доля повторов не превысивших порог (p-value)
    return {"reduction_mean": float(boot.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": p_value}                                                              # итоговый словарь результатов


def select_best_baseline(metric_by_baseline):
    return min(metric_by_baseline, key=lambda name: np.mean(metric_by_baseline[name]))         # бейзлайн с наименьшим средним значением метрики