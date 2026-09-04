"""
RL-агент (DQN) для маршрутизации запросов: Q-сеть, приоритизированный
replay-буфер на sum-tree, эпсилон-разведка с расписанием, safety-откат,
Double DQN, target-сеть.
"""
import numpy as np
import torch
import torch.nn as nn
from collections import deque


class QNet(nn.Module):
    """Полносвязная нейросеть, аппроксимирующая Q(s,a) — ценность действия a в состоянии s."""
    def __init__(self, dim, n_actions=2, hidden_size=32, n_hidden_layers=2):
        super().__init__()                              # инициализация базового класса nn.Module
        layers, in_dim = [], dim                         # список слоёв и текущая входная размерность
        for _ in range(n_hidden_layers):                 # строим n_hidden_layers скрытых слоёв
            layers += [nn.Linear(in_dim, hidden_size), nn.ReLU()]  # линейный слой + нелинейность ReLU
            in_dim = hidden_size                         # следующий слой принимает hidden_size входов
        layers.append(nn.Linear(in_dim, n_actions))      # выходной слой: по одному значению на каждое действие
        self.net = nn.Sequential(*layers)                # собираем все слои в единый последовательный модуль

    def forward(self, x):
        return self.net(x)                               # прямой проход: вход x -> вектор Q-значений действий


class SumTree:
    """Дерево отрезков для хранения приоритетов PER — позволяет сэмплировать и обновлять за O(log n)."""
    def __init__(self, capacity):
        self.capacity = capacity                          # максимальное число листьев (= размер буфера)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)  # массив узлов дерева (внутренние + листья)
        self.write = 0                                     # позиция записи следующего перехода (кольцевой указатель)
        self.n_entries = 0                                  # текущее число реально записанных переходов

    def total(self):
        return self.tree[0]                                 # сумма всех приоритетов = значение в корне дерева

    def add(self, priority):
        """Записать новый приоритет в следующий слот по кольцу; вернуть индекс этого слота."""
        slot = self.write                                    # запоминаем текущую позицию записи
        leaf_idx = slot + self.capacity - 1                   # индекс соответствующего листа в массиве tree
        self.update(leaf_idx, priority)                       # записываем приоритет и обновляем дерево
        self.write = (self.write + 1) % self.capacity          # сдвигаем указатель записи по кольцу
        self.n_entries = min(self.n_entries + 1, self.capacity)  # увеличиваем счётчик заполненности (не выше capacity)
        return slot                                            # возвращаем индекс слота (нужен для хранилища данных)

    def update(self, leaf_idx, priority):
        """Изменить приоритет одного листа и распространить изменение вверх по дереву (итеративно)."""
        change = priority - self.tree[leaf_idx]                # разница между новым и старым приоритетом
        self.tree[leaf_idx] = priority                          # записываем новое значение в лист
        idx = leaf_idx                                          # начинаем подъём с этого листа
        while idx != 0:                                          # пока не дошли до корня
            idx = (idx - 1) // 2                                  # переходим к родительскому узлу
            self.tree[idx] += change                              # прибавляем разницу к сумме родителя

    def get(self, s):
        """Найти лист, соответствующий случайному числу s из диапазона [0, total()]."""
        idx = 0                                                   # начинаем поиск с корня дерева
        while True:
            left = 2 * idx + 1                                     # индекс левого потомка
            right = left + 1                                        # индекс правого потомка
            if left >= len(self.tree):                              # если потомков нет — дошли до листа
                break
            if s <= self.tree[left]:                                 # если s попадает в левое поддерево
                idx = left                                            # спускаемся влево
            else:
                s -= self.tree[left]                                  # иначе вычитаем сумму левого поддерева из s
                idx = right                                            # спускаемся вправо
        data_idx = idx - self.capacity + 1                              # переводим индекс листа в индекс данных буфера
        return idx, self.tree[idx], data_idx                            # индекс листа, его приоритет, индекс данных


class PrioritizedReplayBuffer:
    """Приоритизированный replay-буфер: хранит переходы (s,a,r,s') в numpy-массивах + приоритеты в sum-tree."""
    def __init__(self, capacity, state_dim, alpha=0.6, beta_start=0.4, beta_frames=20000,
                 eps=1e-3, seed=0):
        self.capacity, self.state_dim = capacity, state_dim         # ёмкость буфера и размерность состояния
        self.alpha, self.beta_start, self.beta_frames, self.eps = alpha, beta_start, beta_frames, eps  # гиперпараметры PER
        self.rng = np.random.default_rng(seed)                       # локальный генератор случайных чисел
        self.tree = SumTree(capacity)                                 # структура для хранения и сэмплирования приоритетов
        self.max_priority = 1.0                                        # текущий максимальный "сырой" приоритет (для новых переходов)
        self._frame = 0                                                  # счётчик вызовов sample() (для расписания beta)

        # Предвыделенные numpy-массивы для хранения самих переходов (без создания Python-объектов на каждый переход)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)       # массив состояний s
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)  # массив следующих состояний s'
        self.actions = np.zeros(capacity, dtype=np.int64)                     # массив выбранных действий a
        self.rewards = np.zeros(capacity, dtype=np.float32)                   # массив полученных наград r

    def __len__(self):
        return self.tree.n_entries                                      # текущее число сохранённых переходов

    def add(self, state, action, reward, next_state):
        """Добавить новый переход с максимальным текущим приоритетом (гарантирует его сэмплирование хотя бы раз)."""
        slot = self.tree.add(self.max_priority ** self.alpha)             # регистрируем приоритет, получаем слот записи
        self.states[slot] = state                                          # записываем состояние в этот слот
        self.actions[slot] = action                                          # записываем действие
        self.rewards[slot] = reward                                           # записываем награду
        self.next_states[slot] = next_state                                    # записываем следующее состояние

    def _beta(self):
        """Коэффициент коррекции importance-sampling смещения — линейно растёт от beta_start до 1.0."""
        self._frame += 1                                                        # увеличиваем счётчик кадров
        frac = min(1.0, self._frame / self.beta_frames)                         # доля пройденного пути роста beta
        return self.beta_start + frac * (1.0 - self.beta_start)                 # текущее значение beta

    def sample(self, batch_size):
        """Сэмплировать батч переходов пропорционально их приоритетам (PER)."""
        n = len(self)                                                            # текущее число переходов в буфере
        total = self.tree.total()                                                 # суммарный приоритет всех переходов
        segment = total / batch_size                                               # длина одного стратифицированного сегмента
        leaf_idx = np.empty(batch_size, dtype=np.int64)                             # массив индексов листьев дерева
        data_idx = np.empty(batch_size, dtype=np.int64)                              # массив индексов данных в буфере
        priorities = np.empty(batch_size, dtype=np.float64)                          # массив приоритетов сэмплированных переходов
        for i in range(batch_size):                                                   # для каждого элемента будущего батча
            lo, hi = segment * i, segment * (i + 1)                                    # границы i-го сегмента
            s = min(self.rng.uniform(lo, hi), total - 1e-6)                              # случайная точка внутри сегмента
            li, p, di = self.tree.get(s)                                                  # находим соответствующий лист
            leaf_idx[i], priorities[i], data_idx[i] = li, p, di                            # сохраняем найденные индексы

        probs = priorities / total                                                          # вероятности выбора каждого перехода
        weights = (n * probs) ** (-self._beta())                                             # веса importance-sampling
        weights /= weights.max()                                                              # нормировка весов (максимум = 1)

        # Одна векторизованная выборка по индексам + одно преобразование в тензоры PyTorch
        batch = {
            "states": torch.from_numpy(self.states[data_idx]),                                 # состояния сэмплированных переходов
            "actions": torch.from_numpy(self.actions[data_idx]),                                # действия
            "rewards": torch.from_numpy(self.rewards[data_idx]),                                 # награды
            "next_states": torch.from_numpy(self.next_states[data_idx]),                          # следующие состояния
        }
        return batch, leaf_idx, weights                                                             # батч, индексы для обновления приоритетов, веса

    def update_priorities(self, idx, td_errors):
        """Обновить приоритеты переходов по вычисленным TD-ошибкам (после шага обучения)."""
        raw_priorities = np.abs(td_errors) + self.eps                                                # приоритет = |TD-ошибка| + eps
        for leaf_idx, raw_p in zip(idx, raw_priorities):                                               # для каждого перехода в батче
            self.tree.update(leaf_idx, float(raw_p) ** self.alpha)                                       # обновляем приоритет в дереве
        self.max_priority = max(self.max_priority, float(raw_priorities.max()))                            # обновляем максимум для новых переходов


class DQNRouter:
    """Маршрутизатор на основе DQN: выбирает движок для запроса и обучается на собственном опыте."""
    def __init__(self, dim, gamma=0.9, lr=1e-3, buffer_size=10000, batch_size=256,
                 eps_base=0.05, eps_drift=0.25, eps_start=0.9, eps_decay_steps=5000,
                 target_sync_every=100, hidden_size=32, n_hidden_layers=2,
                 safety_enabled=True, drift_exploration_enabled=True, grad_steps_per_update=1,
                 train_interval=1,
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=20000, per_eps=1e-3,
                 fallback=None, seed=0):
        torch.manual_seed(seed)                                          # фиксируем seed PyTorch для воспроизводимости весов
        self.q = QNet(dim, hidden_size=hidden_size, n_hidden_layers=n_hidden_layers)         # основная (обучаемая) Q-сеть
        self.q_target = QNet(dim, hidden_size=hidden_size, n_hidden_layers=n_hidden_layers)  # целевая (стабилизирующая) сеть
        self.q_target.load_state_dict(self.q.state_dict())                # изначально веса target-сети = веса основной сети
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)           # оптимизатор Adam для обновления весов основной сети
        self.gamma, self.batch_size = gamma, batch_size                   # коэффициент дисконтирования и размер батча
        self.buffer = PrioritizedReplayBuffer(buffer_size, dim, alpha=per_alpha, beta_start=per_beta_start,
                                               beta_frames=per_beta_frames, eps=per_eps, seed=seed)  # PER-буфер переходов
        self.grad_steps_per_update = grad_steps_per_update                 # сколько градиентных шагов на один обучающий вызов
        self.train_interval = max(1, train_interval)                       # раз в сколько шагов среды запускать обучение
        self.eps_base, self.eps_drift = eps_base, eps_drift                # базовый эпсилон и эпсилон при обнаружении дрейфа
        self.eps_start, self.eps_decay_steps = eps_start, eps_decay_steps  # начальный эпсилон и длина фазы его затухания
        self.target_sync_every = target_sync_every                          # период синхронизации target-сети
        self.safety_enabled = bool(safety_enabled)                           # флаг: включён ли откат к эвристике при аномалиях
        self.drift_exploration_enabled = bool(drift_exploration_enabled)     # флаг: включён ли всплеск разведки при дрейфе
        self.fallback = fallback                                              # эвристика, используемая как аварийный резерв
        self.rng = np.random.default_rng(seed)                                # генератор случайных чисел для разведки/сэмплинга
        self._hist = deque(maxlen=50)                                          # скользящее окно последних 50 задержек
        self._cooldown = 0                                                      # число оставшихся шагов в режиме "откат к эвристике"
        self._online_steps = 0                                                   # счётчик выполненных онлайн-шагов
        self.last_eps = eps_start                                                 # последнее использованное значение эпсилон

    def _drift_detected(self):
        """Простая эвристика детекции дрейфа: сравнение среднего за последние 10 и предыдущие 40 наблюдений."""
        if len(self._hist) < 50:                                              # недостаточно данных для сравнения
            return False
        a = np.array(self._hist)                                               # переводим историю в numpy-массив
        return a[-10:].mean() > 1.5 * a[:-10].mean() + 1e-6                     # признак: последние 10 заметно хуже предыдущих

    def _current_eps(self):
        """Вычислить текущее значение эпсилон по расписанию затухания + возможный всплеск при дрейфе."""
        if self._online_steps < self.eps_decay_steps:                          # если фаза затухания ещё не закончилась
            frac = self._online_steps / max(self.eps_decay_steps, 1)            # доля пройденного пути затухания
            base = self.eps_start + frac * (self.eps_base - self.eps_start)     # линейная интерполяция eps_start -> eps_base
        else:
            base = self.eps_base                                                 # после затухания — фиксированное базовое значение

        at_minimum = base <= self.eps_base + 1e-9                                 # проверка: достигнут ли минимум эпсилон
        if self.drift_exploration_enabled and at_minimum and self._drift_detected():  # дрейф применяется только на минимуме
            return self.eps_drift                                                  # временно повышаем эпсилон
        return base                                                                 # иначе — обычное плановое значение

    def route(self, query, env, valid_actions):
        """Выбрать действие (движок) для текущего запроса."""
        self._last_x = env.state_vector(query)                                # запоминаем вектор состояния (для observe())
        self.last_eps = self._current_eps()                                    # сохраняем текущий эпсилон (для графиков)
        if self.safety_enabled and self._cooldown > 0:                          # если активен режим "откат к эвристике"
            self._cooldown -= 1                                                   # уменьшаем оставшееся время откaта
            return self.fallback.route(query, env, valid_actions)                  # используем эвристику вместо сети
        if self.rng.random() < self.last_eps:                                     # с вероятностью эпсилон — случайное действие
            return int(self.rng.choice(valid_actions))
        with torch.no_grad():                                                       # инференс без вычисления градиентов
            qv = self.q(torch.from_numpy(self._last_x)).numpy()                       # прогоняем состояние через Q-сеть
        return max(valid_actions, key=lambda a: qv[a])                                # выбираем действие с максимальным Q-значением

    def observe(self, action, reward, latency, next_query, next_env):
        """Обработать результат выполненного шага: сохранить переход и (возможно) обучиться."""
        self._hist.append(latency)                                              # добавляем задержку в скользящее окно
        if self.safety_enabled and len(self._hist) >= 10 and latency > 10 * np.median(self._hist):
            self._cooldown = 20                                                   # аномалия обнаружена -> временный откат к эвристике
        next_x = next_env.state_vector(next_query)                                 # вектор состояния для СЛЕДУЮЩЕГО запроса
        self.buffer.add(self._last_x, action, reward, next_x)                       # сохраняем переход (s,a,r,s') в буфер
        self._online_steps += 1                                                       # увеличиваем счётчик онлайн-шагов
        if len(self.buffer) >= self.batch_size and self._online_steps % self.train_interval == 0:
            for _ in range(self.grad_steps_per_update):                                 # делаем несколько шагов обучения
                self._train_step()
        if self._online_steps % self.target_sync_every == 0:                              # периодическая синхронизация
            self.q_target.load_state_dict(self.q.state_dict())                              # копируем веса основной сети в target

    def _train_step(self):
        """Один шаг обучения Q-сети на батче, сэмплированном из PER-буфера (Double DQN)."""
        batch, idx, is_weights = self.buffer.sample(self.batch_size)                # сэмплируем батч + индексы + веса IS
        xs, next_xs = batch["states"], batch["next_states"]                          # текущие и следующие состояния батча
        actions, rewards = batch["actions"], batch["rewards"]                         # действия и награды батча
        weights = torch.from_numpy(is_weights.astype(np.float32))                      # веса importance-sampling как тензор

        q_pred = self.q(xs).gather(1, actions.unsqueeze(1)).squeeze(1)                  # предсказанные Q-значения выбранных действий
        with torch.no_grad():                                                             # цель считается без градиентов
            next_actions = self.q(next_xs).argmax(dim=1)                                    # Double DQN: действие выбирает ОСНОВНАЯ сеть
            q_next = self.q_target(next_xs).gather(1, next_actions.unsqueeze(1)).squeeze(1)  # оценку даёт TARGET-сеть
        target = rewards + self.gamma * q_next                                              # цель Беллмана: r + gamma * Q(s',a')

        td_errors = (target - q_pred).detach().numpy()                                       # ошибка предсказания (для приоритетов PER)
        loss = (weights * (q_pred - target) ** 2).mean()                                      # взвешенная MSE-потеря
        self.opt.zero_grad(); loss.backward(); self.opt.step()                                  # обратный проход и шаг оптимизации

        self.buffer.update_priorities(idx, td_errors)                                            # обновляем приоритеты этих переходов

    def pretrain(self, transitions, grad_steps=20):
        """Офлайн-претрейн: заполнить буфер историческими переходами и сделать несколько шагов обучения заранее."""
        for x, a, r, nx in transitions:                                             # для каждого перехода из офлайн-лога
            self.buffer.add(x, a, r, nx)                                             # добавляем его в буфер (приоритет — автоматически максимальный)
        for _ in range(grad_steps):                                                   # выполняем заданное число шагов обучения
            if len(self.buffer) >= self.batch_size:                                     # только если в буфере достаточно данных
                self._train_step()
        self.q_target.load_state_dict(self.q.state_dict())                              # синхронизируем target-сеть после претрейна