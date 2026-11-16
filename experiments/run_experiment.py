"""
run_experiment.py
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

# загружаем весь конфиг в память один раз
CFG = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
# методы, которые дообучаются во время эпизода
ONLINE_METHODS = {"bandit_lints", "rl_dqn"}
# (не используется явно ниже — оставлено для справки)
TRACE_METHODS = {"rl_dqn", "cost_threshold", "round_robin"}


def _make_env(seed, cfg):
    """Создать новый экземпляр среды с заданной топологией и запланировать все
    события деградации."""
    env = HybridDBEnv(seed=seed, reward_weights=cfg.get(
        "reward"), **cfg["nodes"])  # среда со всеми параметрами узлов/reward
    d = cfg["degradation"]
    if d["enabled"]:
        events = [d] + list(d.get("extra", []))
        for ev in events:
            engine = ROW if ev["engine"] == "row" else COL
            env.schedule_degradation(engine, ev["node_idx"], ev["start_query"],
                                      ev["end_query"],
                                      health_factor=ev.get("health_factor", 1.0),
                                      is_down=ev.get("is_down", False),
                                      ramp_queries=ev.get("ramp_queries", 0))
    return env


def run_episode(router, seed, cfg, online_learn=False,
                 external_thresholds=None, warmup_queries=0,
                 progress_every=20000, progress_label=""):
    """Прогнать один полный эпизод (n_queries запросов) для одной стратегии
    маршрутизации.

    добавлен параметр warmup_queries (burn-in без оракула).
    Поток из cfg["workload"]["n_queries"] запросов генерируется и
    обрабатывается ОДНОЙ непрерывной средой env от начала до конца (env._t
    непрерывно растёт на всём протяжении — расписание деградации и профиль
    адаптивного порога, заданные в АБСОЛЮТНЫХ номерах запроса, остаются
    корректными). Для методов с online_learn=True route()/update()/observe()
    вызываются на КАЖДОМ шаге без исключения, включая первые warmup_queries
    запросов — то есть реальное онлайн-обучение не прерывается. Разница
    только в том, что результаты первых warmup_queries шагов НЕ попадают в
    latencies/imbalances/node_ids/network_ms/processing_ms/rewards/epsilons
    (а значит — не попадают ни в episode_summary.csv, ни в node_usage.csv,
    ни в sla_counts.csv, ни в trace_*.npy). Это НЕ оракульный тёплый старт
    (там требовалось копирование среды copy.deepcopy — физически
    невозможно в реальной системе): здесь агент учится на РЕАЛЬНОМ потоке,
    просто до начала окна измерения KPI — ровно то, что происходит при
    поэтапном вводе системы в эксплуатацию. Burn-in применяется ОДИНАКОВО
    ко всем методам (а не только к online-обучаемым) — это также убирает
    артефакт холодного старта очередей/кэша/лага репликации (все они
    стартуют с нулевых/дефолтных значений в _make_env) для ВСЕХ стратегий,
    а не только для RL."""
    env = _make_env(
        # создаём среду для этого эпизода
        seed, cfg)
    # если передан общий профиль порога
    if external_thresholds is not None:
        # подключаем его к среде
        env.set_external_thresholds(external_thresholds)
    queries, phases = WorkloadGenerator(
        cfg["workload"]["n_queries"], seed=seed,
        drift_enabled=bool(cfg["workload"].get("drift_enabled", 1)),
        phases=cfg["workload"].get("phases"),
        cost_distributions=cfg["workload"].get("cost_distributions"),
        # генерируем ЕДИНЫЙ непрерывный поток запросов (warmup + измеряемая
        # часть) — без разрыва фаз на границе burn-in
    ).generate()
    total = len(queries)
    latencies, imbalances, node_ids, network_ms, processing_ms = [
    ], [], [], [], []           # накопители метрик по шагам (только для записываемой части)
    # накопители награды и эпсилон (только для записываемой части)
    rewards, epsilons = [], []
    # обрабатываем запросы по очереди (ВСЕ n_queries, включая burn-in)
    for i, q in enumerate(queries):
        if progress_every and (i % progress_every == 0):
            phase_tag = "warmup" if i < warmup_queries else "measured"
            print(f"  [{progress_label}] шаг {i}/{total} ({phase_tag})")
        # допустимые действия для запроса
        valid = env.valid_actions(q)
        # маршрутизатор выбирает действие
        a = router.route(q, env, valid)
        # запоминаем эпсилон (если есть) ДО step(), как и раньше —
        # сохраняем во временную переменную, добавим в список ниже только
        # если шаг относится к измеряемой части эпизода
        current_eps = getattr(router, "last_eps", np.nan)
        # выполняем шаг среды (ВСЕГДА — и во время burn-in, и в измеряемой части)
        lat, r, imb, info = env.step(q, a)
        # --- ИЗМЕНЕНИЕ: запись в метрики только после окончания burn-in ---
        if i >= warmup_queries:
            latencies.append(lat)
            imbalances.append(imb)
            # сохраняем метрики этого шага
            rewards.append(r)
            # id реально использованного узла
            node_ids.append(info["node_id"])
            # сетевая компонента задержки
            network_ms.append(info["network_ms"])
            # компонента обработки на узле
            processing_ms.append(info["processing_ms"])
            epsilons.append(current_eps)
        # если метод дообучается онлайн — обучение происходит НА КАЖДОМ
        # шаге, независимо от того, идёт burn-in или измеряемая часть
        if online_learn:
            # следующий запрос
            next_q = queries[i + 1] if i + 1 < len(queries) else q
            # для контекстуального бандита
            if hasattr(router, "update"):
                router.update(q, env, a, r)
            # для RL-агента
            if hasattr(router, "observe"):
                router.observe(a, r, lat, next_q, env)
    # синхронно обрезаем массив фаз под записанную часть.
    # (нужно, чтобы длина trace_phases.npy совпадала с длиной
    # trace_latency_*.npy при построении графиков в make_plots.py)
    recorded_phases = phases[warmup_queries:] if warmup_queries > 0 else phases
    # все собранные трассы (только по измеряемой части эпизода)
    return latencies, recorded_phases, imbalances, node_ids, network_ms, processing_ms, rewards, epsilons


def make_routers(cfg, offline_transitions, bandit_log, seed):
    """Создать по одному экземпляру каждой стратегии маршрутизации для данного
    прогона."""
    bandit_X, bandit_actions, bandit_latencies = bandit_log
    # распаковываем офлайн-лог бандита
    gbm = GBMRouter()
    # создаём и обучаем GBM на этом логе
    gbm.fit(bandit_X, bandit_actions, bandit_latencies)
    dqn = DQNRouter(dim=STATE_DIM, gamma=cfg["dqn"]["gamma"],
                    lr=cfg["dqn"]["lr"],
                    eps_base=cfg["dqn"]["eps_base"],
                    eps_drift=cfg["dqn"]["eps_drift"],
                    eps_start=cfg["dqn"]["eps_start"],
                    eps_decay_steps=cfg["dqn"]["eps_decay_steps"],
                    target_sync_every=cfg["dqn"]["target_sync_every"],
                    buffer_size=cfg["dqn"]["buffer_size"],
                    batch_size=cfg["dqn"]["batch_size"],
                    hidden_size=cfg["dqn"]["hidden_size"],
                    n_hidden_layers=cfg["dqn"]["n_hidden_layers"],
                    safety_enabled=bool(cfg["dqn"]["safety_enabled"]),
                    drift_exploration_enabled=bool(
                        cfg["dqn"].get("drift_exploration_enabled", True)),
                    grad_steps_per_update=cfg["dqn"]["grad_steps_per_update"],
                    train_interval=cfg["dqn"].get("train_interval", 1),
                    per_alpha=cfg["dqn"]["per_alpha"],
                    per_beta_start=cfg["dqn"]["per_beta_start"],
                    per_beta_frames=cfg["dqn"]["per_beta_frames"],
                    per_eps=cfg["dqn"]["per_eps"],
                    tail_shield_enabled=bool(cfg["dqn"].get("tail_shield_enabled", True)),
                    # создаём RL-агента со всеми параметрами
                    fallback=CostThresholdRouter(), seed=seed)
    # если офлайн-претрейн включён (сейчас pretrain_enabled=0 — не применяется;
    # не путать с новым warmup_queries выше: это два разных, независимых
    # механизма — оракульный offline-претрейн и реалистичный online-burn-in)
    if bool(cfg["dqn"].get("pretrain_enabled", True)):
        # обучаем агента на оракульном логе
        dqn.pretrain(offline_transitions,
                     grad_steps=cfg["dqn"]["pretrain_grad_steps"])
    return {
        # эвристика Round-Robin
        "round_robin": RoundRobinRouter(),
        # эвристика хеширования
        "hash": HashRouter(),
        # эвристика по стоимости
        "cost_threshold": CostThresholdRouter(),
        # статическая ML-модель
        "gbm_static": gbm,
        # отдельный "чистый" бандит
        "bandit_lints": LinTSRouter(dim=STATE_DIM,
                                     alpha=cfg["bandit"]["alpha"], seed=seed),
        # RL-агент
        "rl_dqn": dqn,
    }


def build_reference_thresholds(seed, cfg):
    """Построить общий профиль адаптивного порога reward'а на основе поведения
    нейтральной эвристики.

    ПРИМЕЧАНИЕ: профиль строится по ВСЕМ cfg["workload"]["n_queries"] шагам
    (включая то, что в run_episode() будет burn-in частью) — это
    необходимо, чтобы длина массива thresholds совпадала с максимальным
    значением env._t (1..n_queries) в run_episode(), который тоже проходит
    по полному потоку. Правка warmup_queries НЕ требует изменений здесь."""
    ref_env = _make_env(
        # отдельный экземпляр среды
        seed, cfg)
    # нейтральная эвристика-маршрутизатор
    ref_router = CostThresholdRouter()
    queries, _ = WorkloadGenerator(
        cfg["workload"]["n_queries"], seed=seed,
        drift_enabled=bool(cfg["workload"].get("drift_enabled", 1)),
        phases=cfg["workload"].get("phases"),
        cost_distributions=cfg["workload"].get("cost_distributions"),
        # тот же поток запросов (тот же seed)
    ).generate()
    # список значений порога по шагам
    thresholds = []
    # проходим по всем запросам
    for q in queries:
        valid = ref_env.valid_actions(q)
        a = ref_router.route(q, ref_env, valid)
        _, _, _, info = ref_env.step(q, a)
        # сохраняем адаптивный порог этого шага
        thresholds.append(info["adaptive_threshold_ms"])
    # возвращаем как единый профиль
    return np.array(thresholds)


def main():
    """Основная точка входа: n_runs прогонов x 6 методов, сохранение всех
    метрик в CSV."""
    os.makedirs(
        # создаём папку для результатов
        "results", exist_ok=True)
    # число прогонов и базовый seed
    n_runs, seed0 = CFG["n_runs"], CFG["seed0"]
    # --- ИЗМЕНЕНИЕ: читаем warmup_queries из конфига (0 = поведение как раньше) ---
    warmup_queries = CFG["workload"].get("warmup_queries", 0)
    # накопители итоговых данных
    summary_rows, sla_counts, node_usage = [], {}, {}

    # цикл по прогонам
    for run_id in range(n_runs):
        # уникальный seed для этого прогона
        seed = seed0 + run_id
        # общий профиль порога для этого прогона
        reference_thresholds = build_reference_thresholds(seed, CFG)
        offline_q, _ = WorkloadGenerator(CFG["offline"]["n_offline_queries"],
                                          seed=seed,
                                          # офлайн-запросы (без дрейфа)
                                          drift_enabled=False).generate()
        offline_transitions = (build_oracle_log(offline_q, seed=seed,
                                                  env_kwargs=CFG["nodes"])
                                # оракульный лог (если нужен претрейн)
                                if bool(CFG["dqn"].get("pretrain_enabled",
                                                        True)) else [])
        bandit_log = build_bandit_log(offline_q, seed=seed,
                                       env_kwargs=CFG["nodes"],
                                       # лог бандита (для GBM)
                                       bandit_alpha=CFG["bandit"]["alpha"])
        # создаём все стратегии
        routers = make_routers(CFG, offline_transitions, bandit_log, seed)

        # для каждой стратегии
        for name, router in routers.items():
            lat, phases, imb, node_ids, net_ms, proc_ms, rewards, eps = run_episode(
                router, seed, CFG, online_learn=name in ONLINE_METHODS,
                # прогоняем полный эпизод (warmup + измеряемая часть)
                external_thresholds=reference_thresholds,
                # --- ИЗМЕНЕНИЕ: передаём warmup_queries одинаково для ВСЕХ методов ---
                warmup_queries=warmup_queries,
                progress_label=f"run {run_id+1}/{n_runs} | {name}")
            # считаем агрегированные метрики (только по измеряемой части,
            # т.к. lat уже не содержит warmup-шагов — см. run_episode())
            m = compute_metrics(lat, CFG["sla_ms"])
            m.update({"method": name, "run_id": run_id, "mean_imbalance":
                      float(np.mean(imb)),
                      "mean_network_ms": float(np.mean(net_ms)),
                      "mean_processing_ms": float(np.mean(proc_ms)),
                      # добавляем доп. поля
                      "mean_reward": float(np.mean(rewards))})
            # сохраняем строку итоговой таблицы
            summary_rows.append(m)

            # число запросов в пределах SLA
            k = int((np.array(lat) <= CFG["sla_ms"]).sum())
            # накопитель по методу
            acc = sla_counts.setdefault(name, [0, 0])
            acc[0] += k
            # обновляем счётчики
            acc[1] += len(lat)

            # частота использования узлов по методу
            usage = node_usage.setdefault(name, {})
            # считаем, сколько раз выбирался каждый узел
            for nid in node_ids:
                usage[nid] = usage.get(nid, 0) + 1

            # трассы сохраняем только для первого прогона
            if run_id == 0:
                # трасса задержки (только измеряемая часть — len == n_queries - warmup_queries)
                np.save(f"results/trace_latency_{name}.npy", np.array(lat))
                # трасса сетевой задержки
                np.save(f"results/trace_network_{name}.npy", np.array(net_ms))
                # трасса обработки на узле
                np.save(
                    f"results/trace_processing_{name}.npy", np.array(proc_ms))
                # трасса награды
                np.save(f"results/trace_reward_{name}.npy", np.array(rewards))
                # только для RL
                if name == "rl_dqn":
                    # трасса эпсилон
                    np.save("results/trace_epsilon_rl_dqn.npy", np.array(eps))
                # трасса фаз нагрузки (уже обрезана до измеряемой части внутри run_episode())
                np.save("results/trace_phases.npy", phases)

        # индикатор прогресса
        print(f"Прогон {run_id + 1}/{n_runs} завершён (seed={seed})")

    # сохраняем итоговую таблицу метрик
    with open("results/episode_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)
    # сохраняем счётчики SLA
    with open("results/sla_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "k_under_sla", "n_total"])
        for name, (k, n) in sla_counts.items():
            w.writerow([name, k, n])
    # сохраняем частоту использования узлов
    with open("results/node_usage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "node_id", "count"])
        for name, usage in node_usage.items():
            for node_id, count in usage.items():
                w.writerow([name, node_id, count])

    print(
        f"Готово: {n_runs} прогонов x {len(routers)} методов -> "
        f"results/episode_summary.csv (burn-in={warmup_queries}, "
        f"измеряемых запросов на прогон={CFG['workload']['n_queries'] - warmup_queries})")


if __name__ == "__main__":
    main()