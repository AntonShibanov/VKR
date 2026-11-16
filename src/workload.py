"""
workload.py
Генератор потока запросов с фазами дрейфа нагрузки (микс типов запросов,
их длина и "тяжесть" задаются в config.yaml -> workload.phases).
"""
import numpy as np
from dataclasses import dataclass

# Дефолтный набор фаз — используется, если явный список фаз не задан в конфиге
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

DEFAULT_COST_DISTRIBUTIONS = {
    "insert": {"mu": 1.0, "sigma": 0.3},
    "analytic": {"mu": 3.5, "sigma": 0.6},
    "transactional": {"mu": 1.5, "sigma": 0.4},
}


@dataclass
class Query:
    qtype: str                    # тип запроса: 'insert' | 'analytic' | 'transactional'
    # условная "стоимость" (объём/сложность) запроса
    cost: float
    # требует ли запрос строгой согласованности (нельзя направить на COL)
    strict_consistency: bool


class WorkloadGenerator:
    def __init__(self, n_queries=1500, phase_len=None, seed=0, drift_enabled=True, phases=None, cost_distributions=None):
        # общее число запросов в эпизоде
        self.n_queries = n_queries
        # генератор случайных чисел для типов/стоимостей
        self.rng = np.random.default_rng(seed)
        # включён ли переход между фазами
        self.drift_enabled = drift_enabled
        self.phases = phases if phases is not None else self._phases_from_legacy(
            phase_len)  # список фаз
        self.cost_distributions = cost_distributions

    @staticmethod
    def _phases_from_legacy(phase_len):
        """Обратная совместимость: если явный список фаз не передан, используем DEFAULT_PHASES
        с единой длиной phase_len (если она указана)."""
        if phase_len is None:
            return DEFAULT_PHASES
        return [{**p, "length": phase_len} for p in DEFAULT_PHASES]

    def _phase_for(self, i):
        """Определить, к какой фазе относится i-й запрос."""
        if not self.drift_enabled:                                         # если дрейф отключён
            # всегда используем первую фазу (обычно "steady")
            return self.phases[0]
        # суммарная длина одного полного цикла фаз
        total = sum(p["length"] for p in self.phases)
        # позиция запроса внутри текущего цикла
        pos = i % total
        # накопленная длина уже пройденных фаз
        acc = 0
        # ищем, в какую фазу попадает позиция pos
        for p in self.phases:
            acc += p["length"]
            if pos < acc:
                return p
        # защита от погрешности округления
        return self.phases[-1]

    def generate(self):
        """Сгенерировать список запросов и параллельный массив названий фаз."""
        cd = self.cost_distributions or DEFAULT_COST_DISTRIBUTIONS   # откалиброванные или синтетические дефолты
        # результирующие списки
        queries, phase_names = [], []
        # генерируем запросы один за другим
        for i in range(self.n_queries):
            # определяем текущую фазу
            phase = self._phase_for(i)
            # доли типов запроса в этой фазе
            p_ins, p_an, p_tx = phase["mix"]
            qtype = self.rng.choice(["insert", "analytic", "transactional"], p=[
                p_ins, p_an, p_tx]
            )  # случайный тип
            if qtype == "insert":
                cost, strict = self.rng.lognormal(
                    cd["insert"]["mu"], cd["insert"]["sigma"]), False
            elif qtype == "analytic":
                cost = self.rng.lognormal(
                    cd["analytic"]["mu"], cd["analytic"]["sigma"]) * phase.get("cost_scale", 1.0)
                strict = False
            else:
                cost, strict = self.rng.lognormal(
                    cd["transactional"]["mu"], cd["transactional"]["sigma"]), True
            # добавляем сформированный запрос
            queries.append(Query(qtype, float(cost), strict))
            # запоминаем имя фазы для этого запроса
            phase_names.append(phase["name"])
            # возвращаем список запросов и массив фаз
        return queries, np.array(phase_names)
