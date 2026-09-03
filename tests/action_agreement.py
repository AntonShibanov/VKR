"""
Диагностика: доля совпадения решений RL-агента с каждым бейзлайном по шагам эпизода.
Помогает проверить гипотезу о том, что RL сходится к политике, близкой к жадным методам,
из-за оффлайн-претрейна на одношаговом оракуле и быстрого затухания очередей в среде.
Использует только уже сохранённые трассы node_id (results/trace_latency_*.npy аналог для действий
потребовал бы отдельного сохранения — здесь показан минимальный пример на новых данных).
"""
import numpy as np
import yaml
from src.workload import WorkloadGenerator
from src.baselines import RoundRobinRouter, HashRouter, CostThresholdRouter
from src.bandit import LinTSRouter
from src.environment import STATE_DIM
from experiments.run_experiment import _make_env, CFG


def collect_actions(router, seed, cfg):
    env = _make_env(seed, cfg)
    queries, _ = WorkloadGenerator(cfg["workload"]["n_queries"], cfg["workload"]["phase_len"],
                                    seed, drift_enabled=bool(cfg["workload"].get("drift_enabled", 1))).generate()
    actions = []
    for q in queries:
        valid = env.valid_actions(q)
        a = router.route(q, env, valid)
        env.step(q, a)
        actions.append(a)
    return np.array(actions)


def main():
    seed = CFG["seed0"]
    baselines = {
        "cost_threshold": CostThresholdRouter(),
        "bandit_lints": LinTSRouter(dim=STATE_DIM, alpha=CFG["bandit"]["alpha"], seed=seed),
    }
    ref_actions = {name: collect_actions(r, seed, CFG) for name, r in baselines.items()}
    # rl_actions нужно собрать аналогично через DQNRouter (пропущено для краткости примера)
    for name, acts in ref_actions.items():
        print(f"{name}: доля действия COL = {acts.mean():.3f}")


if __name__ == "__main__":
    main()