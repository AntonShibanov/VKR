"""
Проверка гипотезы исследования (обновлённая формулировка, раздел 8 статьи) по итогам
эксперимента. Гипотеза подтверждена, если выполнены ОБА условия:
 (1) снижение СРЕДНЕГО времени отклика RL относительно КАЖДОГО базового метода —
     нижняя граница 95% бутстрэп-ДИ превышает 10%;
 (2) снижение P95-задержки RL относительно ЛУЧШЕГО (по P95) базового метода —
     нижняя граница 95% бутстрэп-ДИ превышает 10%.
"""
import pandas as pd
from src.stats import bootstrap_reduction_ci, select_best_baseline

RL = "rl_dqn"
BASELINES = ["round_robin", "hash", "cost_threshold", "gbm_static", "bandit_lints"]
EFFECT_THRESHOLD = 0.10


def main():
    summary = pd.read_csv("results/episode_summary.csv")
    rl = summary[summary.method == RL].sort_values("run_id")
    rows = []

    # --- условие (1): среднее время отклика — RL должен превосходить КАЖДЫЙ бейзлайн ---
    for base in BASELINES:
        bl = summary[summary.method == base].sort_values("run_id")
        ci = bootstrap_reduction_ci(rl.mean_latency_ms.values, bl.mean_latency_ms.values,
                                     threshold=EFFECT_THRESHOLD)
        rows.append({
            "metric": "mean_latency", "baseline": base,
            "reduction_mean": round(ci["reduction_mean"], 3),
            "ci_low": round(ci["ci_low"], 3), "ci_high": round(ci["ci_high"], 3),
            "threshold": EFFECT_THRESHOLD, "p_value": round(ci["p_value"], 4),
            "confirmed": ci["ci_low"] > EFFECT_THRESHOLD,
        })

    # --- условие (2): P95-задержка — RL сравнивается только с ЛУЧШИМ (по P95) бейзлайном ---
    p95_by_baseline = {b: summary[summary.method == b].sort_values("run_id").p95_latency_ms.values
                        for b in BASELINES}
    best_base = select_best_baseline(p95_by_baseline)
    ci_p95 = bootstrap_reduction_ci(rl.p95_latency_ms.values, p95_by_baseline[best_base],
                                     threshold=EFFECT_THRESHOLD)
    rows.append({
        "metric": "p95_latency", "baseline": f"best={best_base}",
        "reduction_mean": round(ci_p95["reduction_mean"], 3),
        "ci_low": round(ci_p95["ci_low"], 3), "ci_high": round(ci_p95["ci_high"], 3),
        "threshold": EFFECT_THRESHOLD, "p_value": round(ci_p95["p_value"], 4),
        "confirmed": ci_p95["ci_low"] > EFFECT_THRESHOLD,
    })

    out = pd.DataFrame(rows)
    out.to_csv("results/hypothesis_test.csv", index=False)
    print(out.to_string(index=False))

    overall_confirmed = bool(out["confirmed"].all())
    if overall_confirmed == 'True':
        print(f"\nГипотеза подтверждена целиком (оба условия): {overall_confirmed}")
    else:
        print(f"\nГипотеза не подтверждена: {overall_confirmed}")


if __name__ == "__main__":
    main()