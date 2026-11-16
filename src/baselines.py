"""Эвристические бейзлайны: Round-Robin, хеширование, стоимостной порог (правило AQD)."""
import hashlib
from src.environment import ROW, COL


class RoundRobinRouter:
    def __init__(self):
        self._i = 0                                     # внутренний счётчик обработанных запросов

    def route(self, query, env, valid_actions):
        a = valid_actions[self._i % len(valid_actions)]  # циклический выбор из допустимых действий
        self._i += 1                                       # увеличиваем счётчик для следующего вызова
        return a


class HashRouter:
    def __init__(self):
        self._i = 0                                        # порядковый номер запроса (используется в хеше)

    def route(self, query, env, valid_actions):
        self._i += 1                                          # увеличиваем номер запроса
        h = int(hashlib.md5(str(self._i).encode()).hexdigest(), 16)  # вычисляем хеш номера запроса
        return valid_actions[h % len(valid_actions)]              # выбираем действие по остатку от деления хеша


class CostThresholdRouter:
    def __init__(self, threshold=20.0):
        self.threshold = threshold                             # порог стоимости, разделяющий ROW и COL

    def route(self, query, env, valid_actions):
        if COL not in valid_actions:                             # если COL недопустим (строгая согласованность)
            return ROW
        return COL if query.cost > self.threshold else ROW        # дорогие запросы — в COL, дешёвые — в ROW

class AllColRouter:
    """Диагностический бейзлайн: всегда COL, если допустимо; иначе ROW.
    Используется для проверки гипотезы, что контекстуальный бандит (LinTS)
    после интенсификации сценария (нелинейная очередь + стохастическая
    деградация) фактически сошёлся к вырожденной политике "почти всегда
    COL", а не к содержательно нетривиальному правилу, зависящему от
    состояния. Если метрики AllColRouter близки к bandit_lints — гипотеза
    подтверждена."""
    def route(self, query, env, valid_actions):
        return COL if COL in valid_actions else ROW