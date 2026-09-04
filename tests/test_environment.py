"""
Санити-тесты симулятора среды (pytest).
"""
import numpy as np
from src.environment import HybridDBEnv, ROW, COL
from src.workload import Query
import pytest


def test_strict_consistency_routes_to_row_only():
    """Строгие транзакции должны иметь только один допустимый вариант — ROW."""
    env = HybridDBEnv(seed=0)
    q = Query(qtype="transactional", cost=5.0, strict_consistency=True)
    assert env.valid_actions(q) == [ROW]


def test_latency_is_positive_and_finite():
    """Задержка должна быть положительным конечным числом, а её компоненты должны суммироваться в неё."""
    env = HybridDBEnv(seed=0)
    q = Query(qtype="analytic", cost=50.0, strict_consistency=False)
    lat, reward, imbalance, info = env.step(q, COL)
    assert lat > 0 and np.isfinite(lat) and np.isfinite(reward)                       # проверка корректности значений
    assert info["node_id"].startswith("col_")                                          # запрос действительно ушёл на COL-узел
    assert info["network_ms"] + info["processing_ms"] == pytest.approx(lat, rel=1e-6)    # сумма компонент = полной задержке


def test_degraded_node_increases_latency():
    """Деградация узла должна заметно увеличивать среднюю задержку на этом кластере."""
    q = Query(qtype="analytic", cost=10.0, strict_consistency=False)
    env_normal, env_degraded = HybridDBEnv(seed=2), HybridDBEnv(seed=2)                     # два одинаковых экземпляра среды
    env_degraded.schedule_degradation(COL, node_idx=0, start_q=0, end_q=100, health_factor=10.0)  # деградация только во втором

    lat_normal = np.mean([env_normal.step(q, COL)[0] for _ in range(5)])                     # средняя задержка без деградации
    lat_degraded = np.mean([env_degraded.step(q, COL)[0] for _ in range(5)])                    # средняя задержка с деградацией
    assert lat_degraded > lat_normal                                                              # деградация должна ухудшать задержку


def test_full_cluster_down_causes_failover_or_timeout():
    """Полный отказ COL-кластера при неполном отказе ROW должен привести к failover, а не к таймауту."""
    env = HybridDBEnv(seed=0, n_col_nodes=2)
    env.schedule_degradation(COL, 0, 0, 100, is_down=True)                                        # оба COL-узла отказывают
    env.schedule_degradation(COL, 1, 0, 100, is_down=True)
    q = Query(qtype="analytic", cost=10.0, strict_consistency=False)
    lat, _, _, info = env.step(q, COL)                                                              # пытаемся направить на COL
    assert lat < HybridDBEnv.TIMEOUT_MS                                                                # это не таймаут (ROW жив)
    assert info["node_id"].startswith("row_")                                                            # запрос реально ушёл на ROW