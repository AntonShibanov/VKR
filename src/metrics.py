"""Расчёт метрик производительности по журналу задержек одного прогона."""
import numpy as np


def compute_metrics(latencies, sla_ms=200.0):
    lat = np.asarray(latencies)
    median = np.median(lat)
    return {
        "mean_latency_ms": float(lat.mean()),
        "p95_latency_ms": float(np.percentile(lat, 95)),
        "p99_latency_ms": float(np.percentile(lat, 99)),
        "pct_under_sla": float((lat <= sla_ms).mean()),
        "catastrophic_rate": float((lat > 10 * median).mean()),  # замедление более чем на порядок
    }