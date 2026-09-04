"""Расчёт метрик производительности по журналу задержек одного прогона."""
import numpy as np


def compute_metrics(latencies, sla_ms=200.0):
    lat = np.asarray(latencies)                                            # приводим список задержек к numpy-массиву
    median = np.median(lat)                                                  # медиана — эталон "типичной" задержки
    return {
        "mean_latency_ms": float(lat.mean()),                                 # среднее время отклика
        "p95_latency_ms": float(np.percentile(lat, 95)),                        # 95-й перцентиль (хвостовая задержка)
        "p99_latency_ms": float(np.percentile(lat, 99)),                         # 99-й перцентиль
        "pct_under_sla": float((lat <= sla_ms).mean()),                            # доля запросов в пределах SLA
        "catastrophic_rate": float((lat > 10 * median).mean()),                     # доля "катастрофических" (>10x медианы) задержек
    }