"""
Пилотный прогон для оценки требуемого числа полных прогонов
(n_runs), обеспечивающего заданную статистическую мощность.
"""
import numpy as np
import yaml
from statsmodels.stats.power import TTestPower

from src.offline_logs import build_oracle_log, build_bandit_log
from src.workload import WorkloadGenerator
from experiments.run_experiment import run_episode, make_routers, CFG

N_PILOT = 8                                                              # число пилотных прогонов для оценки размера эффекта
BASELINES = ["round_robin", "hash", "cost_threshold", "gbm_static", "bandit_lints"]  # все базовые методы сравнения


def pilot_samples(cfg, seed0):
    """Прогнать все методы на N_PILOT сидах и собрать среднее/P95 задержки по каждому прогону."""
    mean_lat = {m: [] for m in BASELINES + ["rl_dqn"]}                     # накопитель средней задержки по методам
    p95_lat = {m: [] for m in BASELINES + ["rl_dqn"]}                        # накопитель P95-задержки по методам
    for i in range(N_PILOT):                                                  # для каждого пилотного прогона
        seed = seed0 + i                                                         # уникальный seed
        offline_q, _ = WorkloadGenerator(cfg["offline"]["n_offline_queries"], seed=seed,
                                          drift_enabled=False).generate()          # офлайн-запросы для этого seed
        trans = build_oracle_log(offline_q, seed=seed, env_kwargs=cfg["nodes"])       # оракульный лог для RL
        bandit_log = build_bandit_log(offline_q, seed=seed, env_kwargs=cfg["nodes"],
                                       bandit_alpha=cfg["bandit"]["alpha"])              # лог бандита для GBM
        routers = make_routers(cfg, trans, bandit_log, seed)                              # создаём все стратегии
        for name, router in routers.items():                                                # для каждой стратегии
            online = name in {"bandit_lints", "rl_dqn"}                                       # дообучается ли метод онлайн
            lat, *_ = run_episode(router, seed, cfg, online_learn=online)                       # прогоняем эпизод
            mean_lat[name].append(np.mean(lat))                                                   # средняя задержка этого прогона
            p95_lat[name].append(np.percentile(lat, 95))                                            # P95-задержка этого прогона
    return {k: np.array(v) for k, v in mean_lat.items()}, {k: np.array(v) for k, v in p95_lat.items()}  # переводим в numpy


def paired_effect_size(rl_values, baseline_values):
    """Cohen's d для парных разностей (бейзлайн минус RL) — мера размера эффекта."""
    diff = baseline_values - rl_values                                       # парная разность по прогонам
    sd = diff.std(ddof=1)                                                       # выборочное стандартное отклонение разности
    return diff.mean() / sd if sd > 0 else 0.0                                    # Cohen's d = среднее разности / её std


def required_n_paired(effect_size, power=0.8, alpha=0.05):
    """Вычислить минимальное число прогонов, необходимое для обнаружения данного эффекта."""
    if effect_size <= 0:                                                         # если эффект не в пользу RL
        return 999                                                                 # сигнал, что нужно пересмотреть постановку эксперимента
    n = TTestPower().solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative="larger")  # расчёт мощности
    return int(np.ceil(n))                                                          # округляем вверх до целого числа прогонов


def main():
    mean_lat, p95_lat = pilot_samples(CFG, seed0=CFG["seed0"] + 5000)               # запускаем пилот на "чужих" сидах (не пересекаются с основными)

    mean_effects = {b: paired_effect_size(mean_lat["rl_dqn"], mean_lat[b]) for b in BASELINES}  # эффект RL против каждого бейзлайна
    worst_baseline = min(mean_effects, key=mean_effects.get)                           # находим наихудший (самый маленький) эффект
    n_mean = required_n_paired(mean_effects[worst_baseline])                             # требуемое n для этого худшего случая

    best_baseline = min(BASELINES, key=lambda b: p95_lat[b].mean())                        # бейзлайн с наилучшим P95
    p95_effect = paired_effect_size(p95_lat["rl_dqn"], p95_lat[best_baseline])               # эффект RL против лучшего по P95
    n_p95 = required_n_paired(p95_effect)                                                     # требуемое n для этого сравнения

    n_final = int(np.clip(max(n_mean, n_p95), 20, 30))                                          # итоговое n, ограниченное диапазоном 20..30

    print(f"Среднее время отклика: наихудший бейзлайн '{worst_baseline}', "
          f"d={mean_effects[worst_baseline]:.2f} -> n={n_mean}")                                    # диагностический вывод
    print(f"P95-задержка: лучший бейзлайн '{best_baseline}', d={p95_effect:.2f} -> n={n_p95}")
    print(f"Итоговое n_runs = {n_final}")

    cfg = dict(CFG); cfg["n_runs"] = n_final                                                          # обновляем n_runs в конфиге
    with open("configs/config.yaml", "w", encoding="utf-8") as f:                                        # перезаписываем файл конфига
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()