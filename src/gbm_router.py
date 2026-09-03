"""
Статическая ML-модель (Gradient Boosting) — обучается офлайн один раз, без дообучения.

Обновлено: раньше модель обучалась как классификатор на оракульных метках "лучший
движок" (метка требовала двойного исполнения запроса на обоих движках параллельно —
невозможно в реальной системе). Теперь модель обучается на РЕАЛИСТИЧНОМ логе,
собранном контекстуальным бандитом (build_bandit_log): для каждого исторического
запроса известен исход только ФАКТИЧЕСКИ выбранного движка. Поэтому вместо одного
классификатора используются ДВА независимых регрессора задержки (по одному на
движок), каждый обучен только на подмножестве лога, где соответствующий движок был
выбран бандитом. Маршрутизация — argmin предсказанной задержки среди допустимых
действий. Так честно воспроизводится типичное ограничение off-policy обучения на
исторических логах (селективное смещение выборки, отсутствие контрфактических меток).
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from src.environment import ROW, COL

MIN_SAMPLES_PER_ENGINE = 10  # минимум наблюдений движка в логе, чтобы обучить его регрессор


class GBMRouter:
    def __init__(self):
        self.models = {
            ROW: HistGradientBoostingRegressor(max_depth=4, max_iter=100, random_state=0),
            COL: HistGradientBoostingRegressor(max_depth=4, max_iter=100, random_state=1),
        }
        self._fitted = {ROW: False, COL: False}

    def fit(self, X, actions, latencies):
        """X — признаки состояния; actions — движок, реально выбранный бандитом;
        latencies — наблюдённая (не контрфактическая) задержка этого выбора."""
        X, actions, latencies = np.asarray(X), np.asarray(actions), np.asarray(latencies)
        for engine in (ROW, COL):
            mask = actions == engine
            if mask.sum() >= MIN_SAMPLES_PER_ENGINE:
                self.models[engine].fit(X[mask], latencies[mask])
                self._fitted[engine] = True

    def route(self, query, env, valid_actions):
        if len(valid_actions) == 1:
            return valid_actions[0]
        x = env.state_vector(query).reshape(1, -1)
        preds = {a: self.models[a].predict(x)[0] for a in valid_actions if self._fitted[a]}
        if not preds:
            return valid_actions[0]   # ни один регрессор не обучен -> безопасный дефолт
        return min(preds, key=preds.get)