"""Санити-тесты симулятора, включая новую логику узлов и деградации."""
import numpy as np
from src.environment import HybridDBEnv, ROW, COL
from src.workload import Query
import pytest


def test_strict_consistency_routes_to_row_only():
    env = HybridDBEnv(seed=0)
    q = Query(qtype="transactional", cost=5.0, strict_consistency=True)
    assert env.valid_actions(q) == [ROW]


def test_latency_is_positive_and_finite():
    env = HybridDBEnv(seed=0)
    q = Query(qtype="analytic", cost=50.0, strict_consistency=False)
    lat, reward, imbalance, info = env.step(q, COL)
    assert lat > 0 and np.isfinite(lat) and np.isfinite(reward)
    assert info["node_id"].startswith("col_")
    assert info["network_ms"] + info["processing_ms"] == pytest.approx(lat, rel=1e-6)


def test_degraded_node_increases_latency():
    """Деградация узла (health_factor=10) должна заметно увеличивать задержку на этом кластере."""
    q = Query(qtype="analytic", cost=10.0, strict_consistency=False)
    env_normal, env_degraded = HybridDBEnv(seed=2), HybridDBEnv(seed=2)
    env_degraded.schedule_degradation(COL, node_idx=0, start_q=0, end_q=100, health_factor=10.0)

    lat_normal = np.mean([env_normal.step(q, COL)[0] for _ in range(5)])
    lat_degraded = np.mean([env_degraded.step(q, COL)[0] for _ in range(5)])
    assert lat_degraded > lat_normal


def test_full_cluster_down_causes_failover_or_timeout():
    """Если оба COL-узла отказали, а запрос не строгий -> должен произойти failover на ROW."""
    env = HybridDBEnv(seed=0, n_col_nodes=2)
    env.schedule_degradation(COL, 0, 0, 100, is_down=True)
    env.schedule_degradation(COL, 1, 0, 100, is_down=True)
    q = Query(qtype="analytic", cost=10.0, strict_consistency=False)
    lat, _, _, info = env.step(q, COL)
    assert lat < HybridDBEnv.TIMEOUT_MS       # не таймаут, т.к. ROW-кластер жив -> сработал failover
    assert info["node_id"].startswith("row_")  # failover действительно ушёл на ROW-кластер