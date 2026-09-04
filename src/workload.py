"""
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


@dataclass
class Query:
    qtype: str                    # тип запроса: 'insert' | 'analytic' | 'transactional'
    cost: float                    # условная "стоимость" (объём/сложность) запроса
    strict_consistency: bool       # требует ли запрос строгой согласованности (нельзя направить на COL)


class WorkloadGenerator:
    def __init__(self, n_queries=1500, phase_len=None, seed=0, drift_enabled=True, phases=None):
        self.n_queries = n_queries                                        # общее число запросов в эпизоде
        self.rng = np.random.default_rng(seed)                              # генератор случайных чисел для типов/стоимостей
        self.drift_enabled = drift_enabled                                    # включён ли переход между фазами
        self.phases = phases if phases is not None else self._phases_from_legacy(phase_len)  # список фаз

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
            return self.phases[0]                                             # всегда используем первую фазу (обычно "steady")
        total = sum(p["length"] for p in self.phases)                          # суммарная длина одного полного цикла фаз
        pos = i % total                                                          # позиция запроса внутри текущего цикла
        acc = 0                                                                   # накопленная длина уже пройденных фаз
        for p in self.phases:                                                      # ищем, в какую фазу попадает позиция pos
            acc += p["length"]
            if pos < acc:
                return p
        return self.phases[-1]                                                      # защита от погрешности округления

    def generate(self):
        """Сгенерировать список запросов и параллельный массив названий фаз."""
        queries, phase_names = [], []                                           # результирующие списки
        for i in range(self.n_queries):                                           # генерируем запросы один за другим
            phase = self._phase_for(i)                                             # определяем текущую фазу
            p_ins, p_an, p_tx = phase["mix"]                                         # доли типов запроса в этой фазе
            qtype = self.rng.choice(["insert", "analytic", "transactional"], p=[p_ins, p_an, p_tx])  # случайный тип
            if qtype == "insert":
                cost, strict = self.rng.lognormal(1.0, 0.3), False                    # лог-нормальная стоимость вставки
            elif qtype == "analytic":
                cost, strict = self.rng.lognormal(3.5, 0.6) * phase.get("cost_scale", 1.0), False  # стоимость аналитики
            else:
                cost, strict = self.rng.lognormal(1.5, 0.4), True                       # транзакция требует строгой согласованности
            queries.append(Query(qtype, float(cost), strict))                             # добавляем сформированный запрос
            phase_names.append(phase["name"])                                              # запоминаем имя фазы для этого запроса
        return queries, np.array(phase_names)                                                # возвращаем список запросов и массив фаз