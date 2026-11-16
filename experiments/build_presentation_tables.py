"""
Формирует итоговые CSV-таблицы на основе результатов прогонов.
"""
import os
import numpy as np
import pandas as pd
from src.stats import bootstrap_reduction_ci          # уже используется в вашем hypothesis_test-скрипте

RL = "rl_dqn"
BLIND_BASELINES = ["round_robin", "hash"]
SMART_BASELINES = ["cost_threshold", "gbm_static", "bandit_lints"]
ALL_BASELINES = BLIND_BASELINES + SMART_BASELINES

LABELS_RU = {
    "round_robin": "Round-Robin", "hash": "Hash", "cost_threshold": "Cost-threshold",
    "gbm_static": "GBM (статическая модель)", "bandit_lints": "Бандит (LinTS)", "rl_dqn": "RL-агент",
}

OUTLIER_RUN_MULTIPLIER = 2.0        # "аномальный прогон" -- см. Табл. В
CATASTROPHIC_COL = "catastrophic_rate_abs"     # фиксированный порог 2000 мс -- см. Табл. Б
OUT_DIR = "results/tables"


def _series(df, method, column):
    """Отсортированный по run_id массив значений колонки для метода."""
    return df[df.method == method].sort_values("run_id")[column].values


def _verdict(ci, threshold):
    """Единая логика вердикта 'Есть/Нет (паритет)/Нет (эффект есть, но мал)/Нет (эффект обратный)'."""
    if ci["ci_low"] > threshold:
        return "Есть"
    if ci["ci_low"] <= 0 <= ci["ci_high"]:
        return "Нет (паритет)"
    if ci["ci_high"] < 0:
        return "Нет (эффект обратный)"
    return "Нет (эффект есть, но мал)"


def build_table_a(df):
    """Таблица А: снижение среднего времени отклика RL vs каждый базис."""
    rows = []
    for base in ALL_BASELINES:
        threshold = 0.10 if base in BLIND_BASELINES else 0.0
        ci = bootstrap_reduction_ci(_series(df, RL, "mean_latency_ms"),
                                     _series(df, base, "mean_latency_ms"), threshold=threshold)
        rows.append({
            "Базовый метод": LABELS_RU[base],
            "Снижение (%)": round(100 * ci["reduction_mean"], 1),
            "95% ДИ низ": round(100 * ci["ci_low"], 1),
            "95% ДИ верх": round(100 * ci["ci_high"], 1),
            "Значимость": _verdict(ci, threshold),
        })
    return pd.DataFrame(rows)


def build_table_b(df):
    """Таблица Б: доля катастрофических ЗАПРОСОВ (абс. порог 2000 мс) --
    RL vs каждый из трёх одношаговых адаптивных методов, плюс собственные
    абсолютные значения у каждого метода (включая RL)."""
    rows = []
    for base in SMART_BASELINES:
        ci = bootstrap_reduction_ci(_series(df, RL, CATASTROPHIC_COL),
                                     _series(df, base, CATASTROPHIC_COL), threshold=0.10)
        rows.append({
            "Базовый метод": LABELS_RU[base],
            "catastrophic_rate RL, %": round(100 * df.loc[df.method == RL, CATASTROPHIC_COL].mean(), 2),
            "catastrophic_rate базиса, %": round(100 * df.loc[df.method == base, CATASTROPHIC_COL].mean(), 2),
            "Снижение (%)": round(100 * ci["reduction_mean"], 1),
            "95% ДИ низ": round(100 * ci["ci_low"], 1),
            "95% ДИ верх": round(100 * ci["ci_high"], 1),
            "Значимость": _verdict(ci, 0.10),
        })
    return pd.DataFrame(rows)


def build_table_v(df, methods=None):
    """Таблица В: устойчивость по прогонам -- CV среднего времени
    отклика и число АНОМАЛЬНЫХ прогонов (не "катастрофических" -- см. разбор
    терминологии; порог здесь определяется ОТНОСИТЕЛЬНО собственной медианы
    метода, а не абсолютным значением 2000 мс, как в Таблице Б)."""
    methods = methods or [RL, "cost_threshold", "gbm_static", "bandit_lints"]
    rows = []
    for m in methods:
        vals = _series(df, m, "mean_latency_ms")
        median = np.median(vals)
        n_outliers = int((vals > OUTLIER_RUN_MULTIPLIER * median).sum())
        rows.append({
            "Метод": LABELS_RU[m],
            "Коэфф. вариации, %": round(100 * vals.std(ddof=1) / vals.mean(), 1),
            f"Аномальных прогонов (> {OUTLIER_RUN_MULTIPLIER:.0f}x медианы метода)": f"{n_outliers} из {len(vals)}",
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv("results/episode_summary.csv")

    table_a = build_table_a(df)
    table_b = build_table_b(df)
    table_v = build_table_v(df)

    table_a.to_csv(os.path.join(OUT_DIR, "table_a_mean_latency.csv"), index=False)
    table_b.to_csv(os.path.join(OUT_DIR, "table_b_catastrophic_rate_abs.csv"), index=False)
    table_v.to_csv(os.path.join(OUT_DIR, "table_v_stability.csv"), index=False)

    print("=== Таблица А: снижение среднего времени отклика ===")
    print(table_a.to_string(index=False))
    print("\n=== Таблица Б: доля катастрофических запросов (порог 2000 мс) ===")
    print(table_b.to_string(index=False))
    print("\n=== Таблица В: устойчивость по прогонам ===")
    print(table_v.to_string(index=False))
    print(f"\nCSV сохранены в {OUT_DIR}/")


if __name__ == "__main__":
    main()