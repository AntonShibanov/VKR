"""
Статистическая проверка гипотезы (обновлённая формулировка, раздел 8 статьи):
 (1) парный бутстрэп-ДИ снижения СРЕДНЕГО времени отклика RL относительно КАЖДОГО
     базового метода (порог эффекта — 10%, см. analyze_results.py);
 (2) парный бутстрэп-ДИ снижения P95-задержки RL относительно ЛУЧШЕГО (по P95)
     базового метода (порог эффекта — 10%).

Парность обеспечена общим seed workload'а для всех методов в рамках одного прогона:
бутстрэп ресэмплирует индексы ПРОГОНОВ (run_id), а не отдельные запросы, поэтому
сравнение остаётся согласованным по конкретной реализации потока запросов и дрейфа.
"""
import numpy as np


def bootstrap_reduction_ci(rl_values, baseline_values, threshold=0.0, n_boot=5000, seed=0, alpha=0.05):
    """
    Относительное снижение метрики (mean_latency_ms или p95_latency_ms) RL vs бейзлайн,
    оценённое парным бутстрэпом по прогонам (run_id).

    Дополнительно считается односторонний бутстрэп-p-value для проверки H0: истинное
    снижение <= threshold, против H1: снижение > threshold. p_value — доля бутстрэп-
    повторов, где снижение оказалось <= threshold (чем меньше p_value, тем сильнее
    подтверждение того, что реальный эффект превышает порог). Поправка Лапласа (+1/+1)
    исключает p_value=0 при малом числе "неудачных" бутстрэп-повторов.
    """
    rng = np.random.default_rng(seed)
    rl, bl = np.asarray(rl_values), np.asarray(baseline_values)
    n = len(rl)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)              # парный бутстрэп по прогонам (run_id)
        boot[i] = 1 - rl[idx].mean() / bl[idx].mean()
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_value = float((1 + np.sum(boot <= threshold)) / (n_boot + 1))
    return {"reduction_mean": float(boot.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": p_value}


def select_best_baseline(metric_by_baseline):
    """Выбрать лучший (минимальный по среднему значению метрики, напр. P95) базовый метод."""
    return min(metric_by_baseline, key=lambda name: np.mean(metric_by_baseline[name]))