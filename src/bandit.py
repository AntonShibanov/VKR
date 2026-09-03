"""
Линейный контекстуальный бандит с сэмплированием Томпсона (аналог AQD/Bao, разделы 5, 6.2).
Ограничение, обсуждаемое в статье: бандит предполагает независимость испытаний и не
учитывает, что выбранное действие меняет контекст следующего запроса (в отличие от RL).
"""
import numpy as np


class LinTSRouter:
    def __init__(self, n_actions=2, dim=11, alpha=1.0, seed=0):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.A = [alpha * np.eye(dim) for _ in range(n_actions)]  # обратная ковариация по действию
        self.b = [np.zeros(dim) for _ in range(n_actions)]

    def route(self, query, env, valid_actions):
        x = env.state_vector(query)
        scores = {}
        for a in valid_actions:
            A_inv = np.linalg.inv(self.A[a])
            mu = A_inv @ self.b[a]
            theta = self.rng.multivariate_normal(mu, A_inv)     # сэмпл из апостериорного распределения
            scores[a] = x @ theta
        return max(scores, key=scores.get)

    def update(self, query, env, action, reward):
        x = env.state_vector(query)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x