"""
Статическая ML-модель (два независимых регрессора задержки — ROW и COL),
обученная на реалистичном (не оракульном) логе, собранном бандитом.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from src.environment import ROW, COL

MIN_SAMPLES_PER_ENGINE = 10          # минимальное число наблюдений движка, чтобы обучить его регрессор


class GBMRouter:
    def __init__(self):
        self.models = {                                               # два независимых регрессора — по одному на движок
            ROW: HistGradientBoostingRegressor(max_depth=4, max_iter=100, random_state=0),
            COL: HistGradientBoostingRegressor(max_depth=4, max_iter=100, random_state=1),
        }
        self._fitted = {ROW: False, COL: False}                        # флаги: обучен ли регрессор каждого движка

    def fit(self, X, actions, latencies):
        X, actions, latencies = np.asarray(X), np.asarray(actions), np.asarray(latencies)  # приводим к numpy-массивам
        for engine in (ROW, COL):                                       # обучаем регрессор для каждого движка отдельно
            mask = actions == engine                                     # отбираем только те строки, где выбирался этот движок
            if mask.sum() >= MIN_SAMPLES_PER_ENGINE:                       # если наблюдений достаточно
                self.models[engine].fit(X[mask], latencies[mask])           # обучаем регрессор на этом подмножестве
                self._fitted[engine] = True                                  # отмечаем, что модель обучена

    def route(self, query, env, valid_actions):
        if len(valid_actions) == 1:                                       # если выбор предопределён (строгая согласованность)
            return valid_actions[0]
        x = env.state_vector(query).reshape(1, -1)                          # признаки текущего запроса в формате для sklearn
        preds = {a: self.models[a].predict(x)[0] for a in valid_actions if self._fitted[a]}  # предсказанные задержки по движкам
        if not preds:                                                        # если ни одна модель не обучена
            return valid_actions[0]                                            # безопасный дефолт
        return min(preds, key=preds.get)                                        # выбираем движок с минимальной предсказанной задержкой