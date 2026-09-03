"""
Синтетическая IoT-нагрузка со сменой режимов (дрейф через фазовые переходы, раздел 9 статьи).

Список фаз и их параметры (длина, микс типов запросов, множитель стоимости аналитики,
подпись для графиков) теперь задаются в configs/config.yaml -> workload.phases, а не
зашиты в коде. Это позволяет варьировать каждый период дрейфа независимо (разную длину,
разный микс запросов, разную "тяжесть" аналитики) без правок кода.
"""
import numpy as np
from dataclasses import dataclass

# Дефолтный набор фаз — используется, если configs/config.yaml не содержит workload.phases
# (обратная совместимость со старыми конфигами и с офлайн-логами/тестами без явного списка).
DEFAULT_PHASES = [
    dict(name="steady", length=6000, mix=(0.70, 0.20, 0.10), cost_scale=1.0,
         label_ru="Стабильный режим"),
    dict(name="insert_burst", length=6000, mix=(0.92, 0.05, 0.03), cost_scale=1.0,
         label_ru="Всплеск вставок"),
    dict(name="analytic_saturation", length=6000, mix=(0.40, 0.55, 0.05), cost_scale=1.0,
         label_ru="Насыщение аналитикой"),
    dict(name="scale_shift", length=6000, mix=(0.55, 0.30, 0.15), cost_scale=3.0,
         label_ru="Рост масштаба данных"),
]


@dataclass
class Query:
    qtype: str                 # 'insert' | 'analytic' | 'transactional'
    cost: float                 # условная "стоимость" (проекция кол-ва строк/сложности)
    strict_consistency: bool    # True для транзакций/биллинга (не допускают eventual consistency)


class WorkloadGenerator:
    """
    Поток запросов, разбитый на фазы с разной долей вставок/аналитики/транзакций.

    phases — список словарей {name, length, mix=(p_insert, p_analytic, p_tx), cost_scale,
    label_ru}. Каждая фаза применяется ровно "length" запросов, затем список циклически
    повторяется до достижения n_queries. drift_enabled=False фиксирует нагрузку на первой
    фазе списка (обычно "steady") на весь эпизод.
    """

    def __init__(self, n_queries=1500, phase_len=None, seed=0, drift_enabled=True, phases=None):
        self.n_queries = n_queries
        self.rng = np.random.default_rng(seed)
        self.drift_enabled = drift_enabled
        self.phases = phases if phases is not None else self._phases_from_legacy(phase_len)

    @staticmethod
    def _phases_from_legacy(phase_len):
        """Обратная совместимость со старым API (n_queries, phase_len, seed): при отсутствии
        явного списка фаз использует DEFAULT_PHASES с единой длиной phase_len для всех фаз."""
        if phase_len is None:
            return DEFAULT_PHASES
        return [{**p, "length": phase_len} for p in DEFAULT_PHASES]

    def _phase_for(self, i):
        if not self.drift_enabled:
            return self.phases[0]
        total = sum(p["length"] for p in self.phases)
        pos = i % total
        acc = 0
        for p in self.phases:
            acc += p["length"]
            if pos < acc:
                return p
        return self.phases[-1]   # защита от погрешности округления

    def generate(self):
        queries, phase_names = [], []
        for i in range(self.n_queries):
            phase = self._phase_for(i)
            p_ins, p_an, p_tx = phase["mix"]
            qtype = self.rng.choice(["insert", "analytic", "transactional"], p=[p_ins, p_an, p_tx])
            if qtype == "insert":
                cost, strict = self.rng.lognormal(1.0, 0.3), False
            elif qtype == "analytic":
                cost, strict = self.rng.lognormal(3.5, 0.6) * phase.get("cost_scale", 1.0), False
            else:
                cost, strict = self.rng.lognormal(1.5, 0.4), True
            queries.append(Query(qtype, float(cost), strict))
            phase_names.append(phase["name"])
        return queries, np.array(phase_names)