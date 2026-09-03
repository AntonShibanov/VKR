"""
Оценка необходимого числа прогонов (n_runs) через анализ статистической мощности.

Дизайн парный (все методы видят один и тот же поток запросов в рамках прогона), поэтому
мощность оценивается для ПАРНОГО одностороннего t-теста на разностях "бейзлайн − RL",
отдельно для двух условий обновлённой гипотезы:
 (1) среднее время отклика — берётся НАИХУДШИЙ (наименьший) эффект среди всех бейзлайнов
     (это самое трудное для обнаружения сравнение -> определяет нижнюю границу n);
 (2) P95-задержка — эффект относительно ЛУЧШЕГО (по P95) бейзлайна.
Итоговое n_runs = max(n_mean, n_p95), с разумными пределами 20..40.
"""
import numpy as np
import yaml
from statsmodels.stats.power import TTestPower

from src.offline_logs import build_oracle_log, build_bandit_log
from src.workload import WorkloadGenerator
from experiments.run_experiment import run_episode, make_routers, CFG

N_PILOT = 8
BASELINES = ["round_robin", "hash", "cost_threshold", "gbm_static", "bandit_lints"]


def pilot_samples(cfg, seed0):
    """Прогнать RL и все бейзлайны на N_PILOT парных сидах, собрать mean/P95 по прогонам."""
    mean_lat = {m: [] for m in BASELINES + ["rl_dqn"]}
    p95_lat = {m: [] for m in BASELINES + ["rl_dqn"]}
    for i in range(N_PILOT):
        seed = seed0 + i
        offline_q, _ = WorkloadGenerator(cfg["offline"]["n_offline_queries"], seed=seed,
                                          drift_enabled=False).generate()
        trans = build_oracle_log(offline_q, seed=seed, env_kwargs=cfg["nodes"])
        bandit_log = build_bandit_log(offline_q, seed=seed, env_kwargs=cfg["nodes"],
                                       bandit_alpha=cfg["bandit"]["alpha"])
        routers = make_routers(cfg, trans, bandit_log, seed)
        for name, router in routers.items():
            online = name in {"bandit_lints", "rl_dqn"}
            lat, *_ = run_episode(router, seed, cfg, online_learn=online)
            mean_lat[name].append(np.mean(lat))
            p95_lat[name].append(np.percentile(lat, 95))
    return {k: np.array(v) for k, v in mean_lat.items()}, {k: np.array(v) for k, v in p95_lat.items()}


def paired_effect_size(rl_values, baseline_values):
    """Cohen's d для парных разностей (baseline - RL)."""
    diff = baseline_values - rl_values
    sd = diff.std(ddof=1)
    return diff.mean() / sd if sd > 0 else 0.0


def required_n_paired(effect_size, power=0.8, alpha=0.05):
    if effect_size <= 0:
        return 999  # на пилоте эффект не в пользу RL -> сигнал пересмотреть постановку, а не подбирать n
    n = TTestPower().solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="larger")
    return int(np.ceil(n))


def main():
    mean_lat, p95_lat = pilot_samples(CFG, seed0=CFG["seed0"] + 5000)

    # условие (1): худший эффект среди бейзлайнов по среднему времени отклика
    mean_effects = {b: paired_effect_size(mean_lat["rl_dqn"], mean_lat[b]) for b in BASELINES}
    worst_baseline = min(mean_effects, key=mean_effects.get)
    n_mean = required_n_paired(mean_effects[worst_baseline])

    # условие (2): эффект относительно лучшего (по P95) бейзлайна
    best_baseline = min(BASELINES, key=lambda b: p95_lat[b].mean())
    p95_effect = paired_effect_size(p95_lat["rl_dqn"], p95_lat[best_baseline])
    n_p95 = required_n_paired(p95_effect)

    n_final = int(np.clip(max(n_mean, n_p95), 20, 40))

    print(f"Среднее время отклика: наихудший бейзлайн '{worst_baseline}', "
          f"d={mean_effects[worst_baseline]:.2f} -> n={n_mean}")
    print(f"P95-задержка: лучший бейзлайн '{best_baseline}', d={p95_effect:.2f} -> n={n_p95}")
    print(f"Итоговое n_runs = {n_final}")

    cfg = dict(CFG); cfg["n_runs"] = n_final
    with open("configs/config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()