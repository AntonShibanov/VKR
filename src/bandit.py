"""
Линейный контекстуальный бандит с сэмплированием Томпсона (аналог AQD/Bao).
"""
import numpy as np


class LinTSRouter:
    def __init__(self, n_actions=2, dim=11, alpha=1.0, seed=0):
        self.n_actions = n_actions                                     # число возможных действий (движков)
        self.rng = np.random.default_rng(seed)                          # генератор случайных чисел для сэмплирования
        self.A = [alpha * np.eye(dim) for _ in range(n_actions)]         # "обратная ковариация" по каждому действию (априор)
        self.b = [np.zeros(dim) for _ in range(n_actions)]                # накопленный вектор наблюдений по каждому действию

    def route(self, query, env, valid_actions):
        x = env.state_vector(query)                                      # признаки текущего запроса и состояния среды
        scores = {}                                                       # оценки полезности каждого допустимого действия
        for a in valid_actions:
            A_inv = np.linalg.inv(self.A[a])                                # обратная матрица A для действия a
            mu = A_inv @ self.b[a]                                          # текущая оценка среднего вектора весов
            theta = self.rng.multivariate_normal(mu, A_inv)                 # сэмпл весов из апостериорного распределения
            scores[a] = x @ theta                                           # предсказанная награда = скалярное произведение
        return max(scores, key=scores.get)                                   # выбираем действие с максимальной сэмплированной оценкой

    def update(self, query, env, action, reward):
        x = env.state_vector(query)                                          # признаки запроса на момент выполнения
        self.A[action] += np.outer(x, x)                                      # обновляем матрицу A для выбранного действия
        self.b[action] += reward * x                                          # обновляем накопленный вектор наблюдений

