"""Офлайн-логи для претрейна (см. предыдущую версию); добавлен env_kwargs для передачи
топологии узлов (число/пропускная способность/RTT), чтобы офлайн-лог соответствовал
той же среде, на которой агент будет оцениваться."""

import copy
import numpy as np
from src.environment import HybridDBEnv, ROW, COL, STATE_DIM
from src.baselines import CostThresholdRouter
from src.bandit import LinTSRouter


def build_trajectory_log(queries, seed=0, env_kwargs=None):
    env = HybridDBEnv(seed=seed, **(env_kwargs or {}))
    router = CostThresholdRouter()
    transitions = []
    for i, q in enumerate(queries):
        valid = env.valid_actions(q)
        x = env.state_vector(q)
        a = router.route(q, env, valid)
        _, r, _, _ = env.step(q, a)
        next_q = queries[i + 1] if i + 1 < len(queries) else q
        transitions.append((x, a, r, env.state_vector(next_q)))
    return transitions


def build_bandit_log(queries, seed=0, env_kwargs=None, bandit_alpha=1.0):
    """
    Реалистичный офлайн-лог для обучения GBM: вместо оракульного двойного исполнения
    (build_shadow_dataset — требовало параллельного клонирования среды, невозможного
    в реальной системе) поток офлайн-запросов реально маршрутизируется контекстуальным
    бандитом (LinTS), который параллельно обучается на своих же наблюдениях — так, как
    это происходило бы в проде ДО внедрения GBM. Бандит выбран как логирующая политика,
    поскольку среди эвристик он показал наилучшую эффективность (см. предыдущий анализ).

    В отличие от оракульного датасета, здесь известен ТОЛЬКО исход фактически выбранного
    движка — контрфактический исход альтернативы недоступен. Это воспроизводит реальное
    ограничение (selection bias, невозможность двойного исполнения write-запросов),
    из-за которого GBM ниже обучается как ДВА регрессора, а не как классификатор
    с оракульными метками (см. gbm_router.py).
    """
    env = HybridDBEnv(seed=seed, **(env_kwargs or {}))
    bandit = LinTSRouter(dim=STATE_DIM, alpha=bandit_alpha, seed=seed)
    X, actions, latencies = [], [], []
    for q in queries:
        valid = env.valid_actions(q)
        x = env.state_vector(q)
        a = bandit.route(q, env, valid)
        lat, r, _, _ = env.step(q, a)
        bandit.update(q, env, a, r)
        X.append(x); actions.append(a); latencies.append(lat)
    return np.array(X), np.array(actions), np.array(latencies)

def build_oracle_log(queries, seed=0, env_kwargs=None):
    """
    Офлайн-лог "оракульных" переходов для тёплого старта RL (замена build_trajectory_log,
    где действия брались из посредственной эвристики cost-threshold). На каждом шаге
    сравниваются гипотетические задержки ROW/COL на НЕЗАВИСИМЫХ КОПИЯХ текущего состояния,
    выбирается меньшая, действие применяется к основной среде, которая продолжает реально
    эволюционировать. Это даёт RL почти оптимальную точку старта — ту же информацию,
    что делает gbm_static сильным бейзлайном, но в виде RL-переходов (s, a, r, s').
    """
    env_kwargs = env_kwargs or {}
    oracle_env = HybridDBEnv(seed=seed, **env_kwargs)
    transitions = []
    for i, q in enumerate(queries):
        valid = oracle_env.valid_actions(q)
        x = oracle_env.state_vector(q)
        if len(valid) == 1:
            a = valid[0]
        else:
            probe_row, probe_col = copy.deepcopy(oracle_env), copy.deepcopy(oracle_env)
            lat_row, _, _, _ = probe_row.step(q, ROW)
            lat_col, _, _, _ = probe_col.step(q, COL)
            a = ROW if lat_row <= lat_col else COL
        _, r, _, _ = oracle_env.step(q, a)
        next_q = queries[i + 1] if i + 1 < len(queries) else q
        transitions.append((x, a, r, oracle_env.state_vector(next_q)))
    return transitions