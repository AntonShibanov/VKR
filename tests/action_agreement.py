"""
Диагностика: доля выбора действия COL каждым бейзлайном (без online-обучения
здесь, только для базовой сверки поведения политик).
"""
import numpy as np
import yaml
from src.workload import WorkloadGenerator
from src.baselines import RoundRobinRouter, HashRouter, CostThresholdRouter
from src.bandit import LinTSRouter
from src.environment import STATE_DIM
from experiments.run_experiment import _make_env, CFG


def collect_actions(router, seed, cfg):
    """Прогнать один эпизод для заданного маршрутизатора и собрать список выбранных действий."""
    env = _make_env(seed, cfg)                                                # создаём среду
    queries, _ = WorkloadGenerator(cfg["workload"]["n_queries"], cfg["workload"]["phase_len"],
                                    seed, drift_enabled=bool(cfg["workload"].get("drift_enabled", 1))).generate()
    actions = []                                                                # список выбранных действий по шагам
    for q in queries:                                                             # проходим по всем запросам
        valid = env.valid_actions(q)                                                # допустимые действия
        a = router.route(q, env, valid)                                              # маршрутизатор выбирает действие
        env.step(q, a)                                                                 # выполняем шаг среды (без сохранения метрик)
        actions.append(a)                                                                # запоминаем действие
    return np.array(actions)


def main():
    seed = CFG["seed0"]                                                          # берём базовый seed из конфига
    baselines = {                                                                   # два бейзлайна для сравнения
        "cost_threshold": CostThresholdRouter(),
        "bandit_lints": LinTSRouter(dim=STATE_DIM, alpha=CFG["bandit"]["alpha"], seed=seed),
    }
    ref_actions = {name: collect_actions(r, seed, CFG) for name, r in baselines.items()}  # собираем действия для каждого
    for name, acts in ref_actions.items():
        print(f"{name}: доля действия COL = {acts.mean():.3f}")                           # печатаем долю выбора COL


if __name__ == "__main__":
    main()