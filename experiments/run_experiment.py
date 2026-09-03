"""
Главный эксперимент. Дополнение к предыдущей версии: среда теперь строится с явной
топологией узлов (cfg["nodes"]), а деградация одного узла (cfg["degradation"])
назначается ДЕТЕРМИНИРОВАННО по номеру запроса -> одинаково для всех методов и всех
прогонов, что сохраняет честность парного сравнения.
"""
import os
import csv
import yaml
import numpy as np

from src.workload import WorkloadGenerator
from src.environment import HybridDBEnv, STATE_DIM, ROW, COL
from src.baselines import RoundRobinRouter, HashRouter, CostThresholdRouter
from src.bandit import LinTSRouter
from src.gbm_router import GBMRouter
from src.dqn_agent import DQNRouter
from src.offline_logs import build_oracle_log, build_bandit_log
from src.metrics import compute_metrics

CFG = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
ONLINE_METHODS = {"bandit_lints", "rl_dqn"}                     # дообучаются по ходу эпизода
TRACE_METHODS = {"rl_dqn", "cost_threshold", "round_robin"}


def _make_env(seed, cfg):
    env = HybridDBEnv(seed=seed, reward_weights=cfg.get("reward"), **cfg["nodes"])
    d = cfg["degradation"]
    if d["enabled"]:
        events = [d] + list(d.get("extra", []))   # основное событие + дополнительные
        for ev in events:
            engine = ROW if ev["engine"] == "row" else COL
            env.schedule_degradation(engine, ev["node_idx"], ev["start_query"], ev["end_query"],
                                      health_factor=ev.get("health_factor", 1.0), is_down=ev.get("is_down", False),
                                      ramp_queries=ev.get("ramp_queries", 0))
    return env


def run_episode(router, seed, cfg, online_learn=False, external_thresholds=None):
    env = _make_env(seed, cfg)
    if external_thresholds is not None:
        env.set_external_thresholds(external_thresholds)
    queries, phases = WorkloadGenerator(
        cfg["workload"]["n_queries"], seed=seed,
        drift_enabled=bool(cfg["workload"].get("drift_enabled", 1)),
        phases=cfg["workload"].get("phases"),
    ).generate()
    latencies, imbalances, node_ids, network_ms, processing_ms = [], [], [], [], []
    rewards, epsilons = [], []
    for i, q in enumerate(queries):
        valid = env.valid_actions(q)
        a = router.route(q, env, valid)
        epsilons.append(getattr(router, "last_eps", np.nan))   # NaN для методов без эпсилон-разведки
        lat, r, imb, info = env.step(q, a)
        latencies.append(lat); imbalances.append(imb); rewards.append(r)
        node_ids.append(info["node_id"])
        network_ms.append(info["network_ms"])
        processing_ms.append(info["processing_ms"])
        if online_learn:
            next_q = queries[i + 1] if i + 1 < len(queries) else q
            if hasattr(router, "update"):
                router.update(q, env, a, r)
            if hasattr(router, "observe"):
                router.observe(a, r, lat, next_q, env)
    return latencies, phases, imbalances, node_ids, network_ms, processing_ms, rewards, epsilons


def make_routers(cfg, offline_transitions, bandit_log, seed):
    bandit_X, bandit_actions, bandit_latencies = bandit_log
    gbm = GBMRouter(); gbm.fit(bandit_X, bandit_actions, bandit_latencies)   # лог собран бандитом
    dqn = DQNRouter(dim=STATE_DIM, gamma=cfg["dqn"]["gamma"], lr=cfg["dqn"]["lr"],
                    eps_base=cfg["dqn"]["eps_base"], eps_drift=cfg["dqn"]["eps_drift"],
                    eps_start=cfg["dqn"]["eps_start"], eps_decay_steps=cfg["dqn"]["eps_decay_steps"],
                    target_sync_every=cfg["dqn"]["target_sync_every"],
                    buffer_size=cfg["dqn"]["buffer_size"], batch_size=cfg["dqn"]["batch_size"],
                    hidden_size=cfg["dqn"]["hidden_size"], n_hidden_layers=cfg["dqn"]["n_hidden_layers"],
                    safety_enabled=bool(cfg["dqn"]["safety_enabled"]),
                    drift_exploration_enabled=bool(cfg["dqn"].get("drift_exploration_enabled", True)),
                    grad_steps_per_update=cfg["dqn"]["grad_steps_per_update"],
                    train_interval=cfg["dqn"].get("train_interval", 1),
                    per_alpha=cfg["dqn"]["per_alpha"], per_beta_start=cfg["dqn"]["per_beta_start"],
                    per_beta_frames=cfg["dqn"]["per_beta_frames"], per_eps=cfg["dqn"]["per_eps"],
                    fallback=CostThresholdRouter(), seed=seed)
    # Претрейн отключается флагом pretrain_enabled: без него агент стартует со случайных
    # весов Q-сети и учится исключительно на своём онлайн-опыте (buffer заполняется только
    # через observe()), что позволяет оценить вклад тёплого старта в итоговую политику.
    if bool(cfg["dqn"].get("pretrain_enabled", True)):
        dqn.pretrain(offline_transitions, grad_steps=cfg["dqn"]["pretrain_grad_steps"])
    return {
        "round_robin": RoundRobinRouter(),
        "hash": HashRouter(),
        "cost_threshold": CostThresholdRouter(),
        "gbm_static": gbm,
        # отдельный, "чистый" экземпляр бандита — не путать с бандитом, который собирал лог для GBM
        "bandit_lints": LinTSRouter(dim=STATE_DIM, alpha=cfg["bandit"]["alpha"], seed=seed),
        "rl_dqn": dqn,
    }


def build_reference_thresholds(seed, cfg):
    """
    Строит ОБЩИЙ профиль адаптивного порога reward'а: прогоняет тот же поток запросов
    через нейтральную детерминированную эвристику (cost_threshold) на ОТДЕЛЬНОМ
    экземпляре среды и записывает её собственный адаптивный порог на каждом шаге.
    Этот профиль затем передаётся ВСЕМ методам одинаково (см. env.set_external_thresholds),
    устраняя петлю, при которой метод, ухудшающий свою латентность, поднимал себе порог
    и тем самым ослаблял собственный штраф (см. подробное объяснение в environment.py).
    """
    ref_env = _make_env(seed, cfg)
    ref_router = CostThresholdRouter()
    queries, _ = WorkloadGenerator(
        cfg["workload"]["n_queries"], seed=seed,
        drift_enabled=bool(cfg["workload"].get("drift_enabled", 1)),
        phases=cfg["workload"].get("phases"),
    ).generate()
    thresholds = []
    for q in queries:
        valid = ref_env.valid_actions(q)
        a = ref_router.route(q, ref_env, valid)
        _, _, _, info = ref_env.step(q, a)
        thresholds.append(info["adaptive_threshold_ms"])
    return np.array(thresholds)


def main():
    os.makedirs("results", exist_ok=True)
    n_runs, seed0 = CFG["n_runs"], CFG["seed0"]
    summary_rows, sla_counts, node_usage = [], {}, {}

    for run_id in range(n_runs):
        seed = seed0 + run_id
        reference_thresholds = build_reference_thresholds(seed, CFG)   # общий для всех методов профиль порога
        offline_q, _ = WorkloadGenerator(CFG["offline"]["n_offline_queries"], seed=seed,
            drift_enabled=False).generate()
        # Оракульный лог строится только если претрейн включён — иначе это бесполезная работа
        offline_transitions = (build_oracle_log(offline_q, seed=seed, env_kwargs=CFG["nodes"])
                                if bool(CFG["dqn"].get("pretrain_enabled", True)) else [])
        bandit_log = build_bandit_log(offline_q, seed=seed, env_kwargs=CFG["nodes"],
                                       bandit_alpha=CFG["bandit"]["alpha"])                       # реалистичный лог для GBM
        routers = make_routers(CFG, offline_transitions, bandit_log, seed)

        for name, router in routers.items():
            lat, phases, imb, node_ids, net_ms, proc_ms, rewards, eps = run_episode(
                router, seed, CFG, online_learn=name in ONLINE_METHODS,
                external_thresholds=reference_thresholds)
            m = compute_metrics(lat, CFG["sla_ms"])
            m.update({"method": name, "run_id": run_id, "mean_imbalance": float(np.mean(imb)),
                      "mean_network_ms": float(np.mean(net_ms)), "mean_processing_ms": float(np.mean(proc_ms)),
                      "mean_reward": float(np.mean(rewards))})
            summary_rows.append(m)

            k = int((np.array(lat) <= CFG["sla_ms"]).sum())
            acc = sla_counts.setdefault(name, [0, 0])
            acc[0] += k; acc[1] += len(lat)

            usage = node_usage.setdefault(name, {})              # частота использования узлов по всем прогонам
            for nid in node_ids:
                usage[nid] = usage.get(nid, 0) + 1

            if run_id == 0:                                        # трассы сохраняем для ВСЕХ методов, только прогон #0
                np.save(f"results/trace_latency_{name}.npy", np.array(lat))
                np.save(f"results/trace_network_{name}.npy", np.array(net_ms))
                np.save(f"results/trace_processing_{name}.npy", np.array(proc_ms))
                np.save(f"results/trace_reward_{name}.npy", np.array(rewards))
                if name == "rl_dqn":
                    np.save("results/trace_epsilon_rl_dqn.npy", np.array(eps))
                np.save("results/trace_phases.npy", phases)

        print(f"Прогон {run_id + 1}/{n_runs} завершён (seed={seed})")

    with open("results/episode_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader(); w.writerows(summary_rows)

    with open("results/sla_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "k_under_sla", "n_total"])
        for name, (k, n) in sla_counts.items():
            w.writerow([name, k, n])

    with open("results/node_usage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "node_id", "count"])
        for name, usage in node_usage.items():
            for node_id, count in usage.items():
                w.writerow([name, node_id, count])

    print(f"Готово: {n_runs} прогонов x {len(routers)} методов -> results/episode_summary.csv")


if __name__ == "__main__":
    main()