"""
Главный скрипт эксперимента: строит среду и все стратегии маршрутизации,
прогоняет n_runs независимых эпизодов и сохраняет метрики/трассы в results/.
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

CFG = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))    # загружаем весь конфиг в память один раз
ONLINE_METHODS = {"bandit_lints", "rl_dqn"}                              # методы, которые дообучаются во время эпизода
TRACE_METHODS = {"rl_dqn", "cost_threshold", "round_robin"}              # (не используется явно ниже — оставлено для справки)


def _make_env(seed, cfg):
    """Создать новый экземпляр среды с заданной топологией и запланировать все события деградации."""
    env = HybridDBEnv(seed=seed, reward_weights=cfg.get("reward"), **cfg["nodes"])  # среда со всеми параметрами узлов/reward
    d = cfg["degradation"]                                                            # секция деградации из конфига
    if d["enabled"]:                                                                    # если сценарий деградации включён
        events = [d] + list(d.get("extra", []))                                           # основное событие + дополнительные
        for ev in events:                                                                   # регистрируем каждое событие
            engine = ROW if ev["engine"] == "row" else COL                                    # переводим строку в код движка
            env.schedule_degradation(engine, ev["node_idx"], ev["start_query"], ev["end_query"],
                                      health_factor=ev.get("health_factor", 1.0), is_down=ev.get("is_down", False),
                                      ramp_queries=ev.get("ramp_queries", 0))                    # передаём все параметры события
    return env


def run_episode(router, seed, cfg, online_learn=False, external_thresholds=None):
    """Прогнать один полный эпизод (n_queries запросов) для одной стратегии маршрутизации."""
    env = _make_env(seed, cfg)                                                          # создаём среду для этого эпизода
    if external_thresholds is not None:                                                  # если передан общий профиль порога
        env.set_external_thresholds(external_thresholds)                                   # подключаем его к среде
    queries, phases = WorkloadGenerator(
        cfg["workload"]["n_queries"], seed=seed,
        drift_enabled=bool(cfg["workload"].get("drift_enabled", 1)),
        phases=cfg["workload"].get("phases"),
    ).generate()                                                                            # генерируем поток запросов
    latencies, imbalances, node_ids, network_ms, processing_ms = [], [], [], [], []           # накопители метрик по шагам
    rewards, epsilons = [], []                                                                 # накопители награды и эпсилон
    for i, q in enumerate(queries):                                                              # обрабатываем запросы по очереди
        valid = env.valid_actions(q)                                                              # допустимые действия для запроса
        a = router.route(q, env, valid)                                                             # маршрутизатор выбирает действие
        epsilons.append(getattr(router, "last_eps", np.nan))                                          # запоминаем эпсилон (если есть)
        lat, r, imb, info = env.step(q, a)                                                              # выполняем шаг среды
        latencies.append(lat); imbalances.append(imb); rewards.append(r)                                  # сохраняем метрики этого шага
        node_ids.append(info["node_id"])                                                                    # id реально использованного узла
        network_ms.append(info["network_ms"])                                                                # сетевая компонента задержки
        processing_ms.append(info["processing_ms"])                                                            # компонента обработки на узле
        if online_learn:                                                                                        # если метод дообучается онлайн
            next_q = queries[i + 1] if i + 1 < len(queries) else q                                                 # следующий запрос
            if hasattr(router, "update"):                                                                           # для контекстуального бандита
                router.update(q, env, a, r)
            if hasattr(router, "observe"):                                                                            # для RL-агента
                router.observe(a, r, lat, next_q, env)
    return latencies, phases, imbalances, node_ids, network_ms, processing_ms, rewards, epsilons                        # все собранные трассы


def make_routers(cfg, offline_transitions, bandit_log, seed):
    """Создать по одному экземпляру каждой стратегии маршрутизации для данного прогона."""
    bandit_X, bandit_actions, bandit_latencies = bandit_log                             # распаковываем офлайн-лог бандита
    gbm = GBMRouter(); gbm.fit(bandit_X, bandit_actions, bandit_latencies)                # создаём и обучаем GBM на этом логе
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
                    fallback=CostThresholdRouter(), seed=seed)                              # создаём RL-агента со всеми параметрами
    if bool(cfg["dqn"].get("pretrain_enabled", True)):                                        # если офлайн-претрейн включён
        dqn.pretrain(offline_transitions, grad_steps=cfg["dqn"]["pretrain_grad_steps"])          # обучаем агента на оракульном логе
    return {
        "round_robin": RoundRobinRouter(),                                                       # эвристика Round-Robin
        "hash": HashRouter(),                                                                       # эвристика хеширования
        "cost_threshold": CostThresholdRouter(),                                                     # эвристика по стоимости
        "gbm_static": gbm,                                                                             # статическая ML-модель
        "bandit_lints": LinTSRouter(dim=STATE_DIM, alpha=cfg["bandit"]["alpha"], seed=seed),             # отдельный "чистый" бандит
        "rl_dqn": dqn,                                                                                     # RL-агент
    }


def build_reference_thresholds(seed, cfg):
    """Построить общий профиль адаптивного порога reward'а на основе поведения нейтральной эвристики."""
    ref_env = _make_env(seed, cfg)                                                       # отдельный экземпляр среды
    ref_router = CostThresholdRouter()                                                     # нейтральная эвристика-маршрутизатор
    queries, _ = WorkloadGenerator(
        cfg["workload"]["n_queries"], seed=seed,
        drift_enabled=bool(cfg["workload"].get("drift_enabled", 1)),
        phases=cfg["workload"].get("phases"),
    ).generate()                                                                             # тот же поток запросов (тот же seed)
    thresholds = []                                                                            # список значений порога по шагам
    for q in queries:                                                                            # проходим по всем запросам
        valid = ref_env.valid_actions(q)
        a = ref_router.route(q, ref_env, valid)
        _, _, _, info = ref_env.step(q, a)
        thresholds.append(info["adaptive_threshold_ms"])                                            # сохраняем адаптивный порог этого шага
    return np.array(thresholds)                                                                       # возвращаем как единый профиль


def main():
    """Основная точка входа: n_runs прогонов x 6 методов, сохранение всех метрик в CSV."""
    os.makedirs("results", exist_ok=True)                                                # создаём папку для результатов
    n_runs, seed0 = CFG["n_runs"], CFG["seed0"]                                             # число прогонов и базовый seed
    summary_rows, sla_counts, node_usage = [], {}, {}                                         # накопители итоговых данных

    for run_id in range(n_runs):                                                                # цикл по прогонам
        seed = seed0 + run_id                                                                      # уникальный seed для этого прогона
        reference_thresholds = build_reference_thresholds(seed, CFG)                                 # общий профиль порога для этого прогона
        offline_q, _ = WorkloadGenerator(CFG["offline"]["n_offline_queries"], seed=seed,
            drift_enabled=False).generate()                                                            # офлайн-запросы (без дрейфа)
        offline_transitions = (build_oracle_log(offline_q, seed=seed, env_kwargs=CFG["nodes"])
                                if bool(CFG["dqn"].get("pretrain_enabled", True)) else [])                # оракульный лог (если нужен претрейн)
        bandit_log = build_bandit_log(offline_q, seed=seed, env_kwargs=CFG["nodes"],
                                       bandit_alpha=CFG["bandit"]["alpha"])                                # лог бандита (для GBM)
        routers = make_routers(CFG, offline_transitions, bandit_log, seed)                                  # создаём все стратегии

        for name, router in routers.items():                                                                  # для каждой стратегии
            lat, phases, imb, node_ids, net_ms, proc_ms, rewards, eps = run_episode(
                router, seed, CFG, online_learn=name in ONLINE_METHODS,
                external_thresholds=reference_thresholds)                                                        # прогоняем полный эпизод
            m = compute_metrics(lat, CFG["sla_ms"])                                                                 # считаем агрегированные метрики
            m.update({"method": name, "run_id": run_id, "mean_imbalance": float(np.mean(imb)),
                      "mean_network_ms": float(np.mean(net_ms)), "mean_processing_ms": float(np.mean(proc_ms)),
                      "mean_reward": float(np.mean(rewards))})                                                       # добавляем доп. поля
            summary_rows.append(m)                                                                                     # сохраняем строку итоговой таблицы

            k = int((np.array(lat) <= CFG["sla_ms"]).sum())                                                             # число запросов в пределах SLA
            acc = sla_counts.setdefault(name, [0, 0])                                                                     # накопитель по методу
            acc[0] += k; acc[1] += len(lat)                                                                                # обновляем счётчики

            usage = node_usage.setdefault(name, {})                                                                          # частота использования узлов по методу
            for nid in node_ids:                                                                                               # считаем, сколько раз выбирался каждый узел
                usage[nid] = usage.get(nid, 0) + 1

            if run_id == 0:                                                                                                     # трассы сохраняем только для первого прогона
                np.save(f"results/trace_latency_{name}.npy", np.array(lat))                                                       # трасса задержки
                np.save(f"results/trace_network_{name}.npy", np.array(net_ms))                                                      # трасса сетевой задержки
                np.save(f"results/trace_processing_{name}.npy", np.array(proc_ms))                                                    # трасса обработки на узле
                np.save(f"results/trace_reward_{name}.npy", np.array(rewards))                                                          # трасса награды
                if name == "rl_dqn":                                                                                                       # только для RL
                    np.save("results/trace_epsilon_rl_dqn.npy", np.array(eps))                                                              # трасса эпсилон
                np.save("results/trace_phases.npy", phases)                                                                                   # трасса фаз нагрузки

        print(f"Прогон {run_id + 1}/{n_runs} завершён (seed={seed})")                                                                          # индикатор прогресса

    with open("results/episode_summary.csv", "w", newline="") as f:                                                                              # сохраняем итоговую таблицу метрик
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader(); w.writerows(summary_rows)

    with open("results/sla_counts.csv", "w", newline="") as f:                                                                                     # сохраняем счётчики SLA
        w = csv.writer(f)
        w.writerow(["method", "k_under_sla", "n_total"])
        for name, (k, n) in sla_counts.items():
            w.writerow([name, k, n])

    with open("results/node_usage.csv", "w", newline="") as f:                                                                                       # сохраняем частоту использования узлов
        w = csv.writer(f)
        w.writerow(["method", "node_id", "count"])
        for name, usage in node_usage.items():
            for node_id, count in usage.items():
                w.writerow([name, node_id, count])

    print(f"Готово: {n_runs} прогонов x {len(routers)} методов -> results/episode_summary.csv")


if __name__ == "__main__":
    main()