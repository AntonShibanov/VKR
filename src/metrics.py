"""Расчёт метрик производительности по журналу задержек одного прогона."""
import numpy as np

# --- НОВОЕ: абсолютный (не самоотносительный) порог "катастрофы" ---
# В отличие от catastrophic_rate (latency > 10*median ЭТОГО ЖЕ метода),
# этот порог ОДИНАКОВ для всех методов -- устраняет артефакт, при котором
# метод с более высокой типичной (median) задержкой получает более мягкий
# порог "катастрофы", чем метод с низкой типичной задержкой (см.
# diagnose_catastrophic_queue_health.py: ~20% разница в median между
# rl_dqn и bandit_lints давала ~27% разницы в catastrophic_rate при
# практически совпадающем по времени множестве экстремальных событий --
# 75.57% катастроф bandit_lints совпадали по step_abs с катастрофами rl_dqn).
# Значение 2000 мс -- 10x от SLA (sla_ms=200 по умолчанию в config.yaml),
# т.е. та же "кратность 10x", что и в относительном определении, но
# ЗАКРЕПЛЁННАЯ заранее, а не плавающая вместе с качеством метода.
# Это инженерный выбор для ДИАГНОСТИКИ/сравнения, НЕ замена основной
# метрики catastrophic_rate -- обе метрики теперь сохраняются параллельно.
CATASTROPHIC_ABS_THRESHOLD_MS = 2000.0


def compute_metrics(latencies, sla_ms=200.0, catastrophic_abs_ms=None):
    lat = np.asarray(latencies)                                            # приводим список задержек к numpy-массиву
    median = np.median(lat)                                                  # медиана — эталон "типичной" задержки
    # --- НОВОЕ: единый порог "катастрофы" для всех методов (см. комментарий выше) ---
    abs_threshold = (catastrophic_abs_ms if catastrophic_abs_ms is not None
                      else CATASTROPHIC_ABS_THRESHOLD_MS)
    return {
        "mean_latency_ms": float(lat.mean()),                                 # среднее время отклика
        "p95_latency_ms": float(np.percentile(lat, 95)),                        # 95-й перцентиль (хвостовая задержка)
        "p99_latency_ms": float(np.percentile(lat, 99)),                         # 99-й перцентиль
        "pct_under_sla": float((lat <= sla_ms).mean()),                            # доля запросов в пределах SLA
        "catastrophic_rate": float((lat > 10 * median).mean()),                     # доля "катастрофических" (>10x СОБСТВЕННОЙ медианы) задержек -- самоотносительное определение, зависит от типичной latency метода
        # --- НОВОЕ: то же самое, но с ОДНИМ И ТЕМ ЖЕ порогом для ВСЕХ методов ---
        "catastrophic_rate_abs": float((lat > abs_threshold).mean()),
        # --- НОВОЕ: сама медиана -- чтобы расхождение типичной latency между
        # методами было видно напрямую в episode_summary.csv, а не только
        # в отдельной диагностике ---
        "median_latency_ms": float(median),
    }