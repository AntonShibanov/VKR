"""Эвристические бейзлайны: Round-Robin, хеширование, стоимостной порог (правило AQD [34])."""
import hashlib
from src.environment import ROW, COL


class RoundRobinRouter:
    def __init__(self):
        self._i = 0

    def route(self, query, env, valid_actions):
        a = valid_actions[self._i % len(valid_actions)]
        self._i += 1
        return a


class HashRouter:
    """Имитация детерминированного шардинга: маршрут по хешу порядкового номера запроса."""
    def __init__(self):
        self._i = 0

    def route(self, query, env, valid_actions):
        self._i += 1
        h = int(hashlib.md5(str(self._i).encode()).hexdigest(), 16)
        return valid_actions[h % len(valid_actions)]


class CostThresholdRouter:
    """Правило «стоимость выше порога -> колоночный движок» (промышленный бейзлайн AQD)."""
    def __init__(self, threshold=20.0):
        self.threshold = threshold

    def route(self, query, env, valid_actions):
        if COL not in valid_actions:
            return ROW
        return COL if query.cost > self.threshold else ROW