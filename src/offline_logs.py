"""
Построение офлайн-логов: build_oracle_log (для тёплого старта RL, использует
клонирование среды — допустимо только офлайн) и build_bandit_log (реалистичный
лог для GBM, без клонирования, собранный контекстуальным бандитом).
"""
import copy                                             # для создания независимых копий среды в офлайн
import numpy as np
from src.environment import HybridDBEnv, ROW, COL, STATE_DIM
from src.baselines import CostThresholdRouter
from src.bandit import LinTSRouter


def build_trajectory_log(queries, seed=0, env_kwargs=None):
    """Устаревший вариант: лог на основе исторической эвристики cost-threshold (оставлен для совместимости)."""
    env = HybridDBEnv(seed=seed, **(env_kwargs or {}))                        # создаём среду с заданной топологией
    router = CostThresholdRouter()                                             # используем эвристику как логирующую политику
    transitions = []                                                            # список собранных переходов
    for i, q in enumerate(queries):                                              # проходим по всем офлайн-запросам
        valid = env.valid_actions(q)                                              # допустимые действия для запроса
        x = env.state_vector(q)                                                    # текущее состояние
        a = router.route(q, env, valid)                                             # действие, выбранное эвристикой
        _, r, _, _ = env.step(q, a)                                                   # выполняем шаг, получаем награду
        next_q = queries[i + 1] if i + 1 < len(queries) else q                          # следующий запрос (или тот же, если последний)
        transitions.append((x, a, r, env.state_vector(next_q)))                          # сохраняем переход (s,a,r,s')
    return transitions


def build_bandit_log(queries, seed=0, env_kwargs=None, bandit_alpha=1.0):
    """Реалистичный лог для GBM: поток реально маршрутизируется бандитом, который параллельно обучается."""
    env = HybridDBEnv(seed=seed, **(env_kwargs or {}))                          # создаём среду
    bandit = LinTSRouter(dim=STATE_DIM, alpha=bandit_alpha, seed=seed)            # создаём логирующего бандита
    X, actions, latencies = [], [], []                                             # три параллельных списка результатов
    for q in queries:                                                               # проходим по всем офлайн-запросам
        valid = env.valid_actions(q)                                                  # допустимые действия
        x = env.state_vector(q)                                                        # текущее состояние
        a = bandit.route(q, env, valid)                                                 # действие, выбранное бандитом
        lat, r, _, _ = env.step(q, a)                                                     # выполняем шаг, получаем задержку и награду
        bandit.update(q, env, a, r)                                                        # бандит обучается на своём же наблюдении
        X.append(x); actions.append(a); latencies.append(lat)                                # сохраняем признаки, действие, задержку
    return np.array(X), np.array(actions), np.array(latencies)                                 # возвращаем как numpy-массивы


def build_oracle_log(queries, seed=0, env_kwargs=None):
    """Оракульный лог для тёплого старта RL: сравнение гипотетических исходов ROW/COL на клонах среды."""
    env_kwargs = env_kwargs or {}                                                # словарь параметров среды (или пустой)
    oracle_env = HybridDBEnv(seed=seed, **env_kwargs)                             # основная среда, которая реально эволюционирует
    transitions = []                                                                # список собранных переходов
    for i, q in enumerate(queries):                                                  # проходим по офлайн-запросам
        valid = oracle_env.valid_actions(q)                                            # допустимые действия
        x = oracle_env.state_vector(q)                                                   # текущее состояние
        if len(valid) == 1:                                                                # если выбор предопределён
            a = valid[0]
        else:
            probe_row, probe_col = copy.deepcopy(oracle_env), copy.deepcopy(oracle_env)      # две независимые копии среды
            lat_row, _, _, _ = probe_row.step(q, ROW)                                           # гипотетический исход при выборе ROW
            lat_col, _, _, _ = probe_col.step(q, COL)                                            # гипотетический исход при выборе COL
            a = ROW if lat_row <= lat_col else COL                                                # выбираем действие с меньшей задержкой
        _, r, _, _ = oracle_env.step(q, a)                                                          # применяем выбранное действие к РЕАЛЬНОЙ среде
        next_q = queries[i + 1] if i + 1 < len(queries) else q                                        # следующий запрос
        transitions.append((x, a, r, oracle_env.state_vector(next_q)))                                  # сохраняем переход
    return transitions