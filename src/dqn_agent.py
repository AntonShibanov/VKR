"""
RL-агент (DQN) для маршрутизации запросов.
Реализует: (а) дисконтированную последовательную постановку (gamma>0) — отличие от бандита;
(б) офлайн-претрейн на логах исторической эвристики; (в) онлайн-дообучение с ограниченным
исследованием (1-5%) и повышением при обнаружении дрейфа; (г) safety-откат к эвристике
при аномальном всплеске задержки (см. раздел 9, механизмы CQR [36]).
"""
import numpy as np
import torch
import torch.nn as nn
from collections import deque


class QNet(nn.Module):
    """Полносвязная Q-сеть с настраиваемым числом и размером скрытых слоёв
    (параметры hidden_size/n_hidden_layers теперь задаются из configs/config.yaml)."""
    def __init__(self, dim, n_actions=2, hidden_size=32, n_hidden_layers=2):
        super().__init__()
        layers, in_dim = [], dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_size), nn.ReLU()]
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SumTree:
    """
    Бинарное дерево отрезков для хранения приоритетов (ИТЕРАТИВНАЯ реализация — без
    рекурсии). В Python вызов функции сам по себе дорог (создание кадра стека), а
    рекурсивные _propagate/_retrieve делали до log2(capacity)~14 вложенных вызовов на
    КАЖДЫЙ элемент батча при КАЖДОМ градиентном шаге — при grad_steps_per_update=10 и
    длинных эпизодах (150k+ шагов) это давало десятки миллионов лишних вызовов функций.
    Циклы while делают то же самое без накладных расходов на кадры стека.

    Дерево теперь хранит ТОЛЬКО приоритеты, не сами данные перехода — сами переходы
    лежат в предвыделенных numpy-массивах PrioritizedReplayBuffer (см. ниже), а "data"
    для листа — это просто числовой индекс слота в этих массивах.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write = 0        # позиция записи следующего перехода (кольцевой буфер по листьям)
        self.n_entries = 0

    def total(self):
        return self.tree[0]

    def add(self, priority):
        """Записать приоритет в следующий по кольцу слот; вернуть индекс слота (= индекс
        данных в numpy-хранилище буфера, вычисляется ДО инкремента self.write)."""
        slot = self.write
        leaf_idx = slot + self.capacity - 1
        self.update(leaf_idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return slot

    def update(self, leaf_idx, priority):
        """Итеративное распространение изменения приоритета вверх по дереву — без рекурсии."""
        change = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        idx = leaf_idx
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        """Итеративный поиск листа по значению s — без рекурсии."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):      # дошли до листа
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1  # индекс слота в numpy-хранилище буфера
        return idx, self.tree[idx], data_idx


class PrioritizedReplayBuffer:
    """
    Proportional Prioritized Experience Replay (Schaul et al., 2016) на основе SumTree.

    Переходы (s, a, r, s') хранятся в ПРЕДВЫДЕЛЕННЫХ numpy-массивах фиксированного
    размера (states/actions/rewards/next_states), а не как список Python-кортежей из
    отдельных torch.Tensor. Это устраняет два источника накладных расходов, которые
    оставались после первого перехода на sum-tree и не давали ускорения при длинных
    эпизодах:
      1) создание нового torch.Tensor на каждый переход при add() (150k+ раз за эпизод);
      2) сборку батча через список из 256 отдельных тензоров + torch.stack() на каждый
         вызов sample() (batch_size * grad_steps_per_update раз за эпизод) — заменено на
         ОДНУ операцию fancy-индексирования numpy-массива + ОДИН torch.from_numpy().

    Приоритеты вычисляются исключительно из внутренней TD-ошибки самого агента — эта
    оптимизация касается только скорости работы, не меняет ни данные, ни семантику PER.
    """
    def __init__(self, capacity, state_dim, alpha=0.6, beta_start=0.4, beta_frames=20000,
                 eps=1e-3, seed=0):
        self.capacity, self.state_dim = capacity, state_dim
        self.alpha, self.beta_start, self.beta_frames, self.eps = alpha, beta_start, beta_frames, eps
        self.rng = np.random.default_rng(seed)
        self.tree = SumTree(capacity)
        self.max_priority = 1.0   # "сырой" (до возведения в alpha) приоритет для новых переходов
        self._frame = 0

        # Предвыделенное хранилище переходов — заполняется по кольцу синхронно с SumTree
        # (slot, возвращаемый tree.add(), однозначно определяет позицию записи).
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)

    def __len__(self):
        return self.tree.n_entries

    def add(self, state, action, reward, next_state):
        """Новый переход получает максимальный текущий приоритет — гарантирует, что он
        будет хотя бы раз просэмплирован до того, как для него посчитают реальную TD-ошибку.
        state/next_state — обычные numpy-массивы (env.state_vector(...)), без промежуточного
        преобразования в torch.Tensor на этом шаге."""
        slot = self.tree.add(self.max_priority ** self.alpha)
        self.states[slot] = state
        self.actions[slot] = action
        self.rewards[slot] = reward
        self.next_states[slot] = next_state

    def _beta(self):
        """Коэффициент коррекции importance-sampling смещения: линейно растёт от
        beta_start до 1.0, чтобы в начале обучения не переоценивать редкие переходы."""
        self._frame += 1
        frac = min(1.0, self._frame / self.beta_frames)
        return self.beta_start + frac * (1.0 - self.beta_start)

    def sample(self, batch_size):
        n = len(self)
        total = self.tree.total()
        segment = total / batch_size            # стратифицированная выборка: 1 сегмент на элемент батча
        leaf_idx = np.empty(batch_size, dtype=np.int64)
        data_idx = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            s = min(self.rng.uniform(lo, hi), total - 1e-6)   # защита от погрешности округления на границе
            li, p, di = self.tree.get(s)
            leaf_idx[i], priorities[i], data_idx[i] = li, p, di

        probs = priorities / total
        weights = (n * probs) ** (-self._beta())
        weights /= weights.max()          # нормировка, чтобы веса не увеличивали эффективный шаг обучения

        # Одна векторизованная выборка по numpy-индексам + одно преобразование в torch —
        # вместо 256 отдельных Python-объектов и torch.stack() над ними.
        batch = {
            "states": torch.from_numpy(self.states[data_idx]),
            "actions": torch.from_numpy(self.actions[data_idx]),
            "rewards": torch.from_numpy(self.rewards[data_idx]),
            "next_states": torch.from_numpy(self.next_states[data_idx]),
        }
        return batch, leaf_idx, weights

    def update_priorities(self, idx, td_errors):
        raw_priorities = np.abs(td_errors) + self.eps
        for leaf_idx, raw_p in zip(idx, raw_priorities):
            self.tree.update(leaf_idx, float(raw_p) ** self.alpha)
        self.max_priority = max(self.max_priority, float(raw_priorities.max()))


class DQNRouter:
    def __init__(self, dim, gamma=0.9, lr=1e-3, buffer_size=10000, batch_size=256,
                 eps_base=0.05, eps_drift=0.25, eps_start=0.9, eps_decay_steps=5000,
                 target_sync_every=100, hidden_size=32, n_hidden_layers=2,
                 safety_enabled=True, drift_exploration_enabled=True, grad_steps_per_update=1,
                 train_interval=1,
                 per_alpha=0.6, per_beta_start=0.4, per_beta_frames=20000, per_eps=1e-3,
                 fallback=None, seed=0):
        torch.manual_seed(seed)
        self.q = QNet(dim, hidden_size=hidden_size, n_hidden_layers=n_hidden_layers)
        self.q_target = QNet(dim, hidden_size=hidden_size, n_hidden_layers=n_hidden_layers)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.batch_size = gamma, batch_size
        # PER на предвыделенных numpy-массивах (см. PrioritizedReplayBuffer выше);
        # при per_alpha=0 сэмплинг становится равномерным, т.е. поведение эквивалентно
        # обычному replay-буферу.
        self.buffer = PrioritizedReplayBuffer(buffer_size, dim, alpha=per_alpha, beta_start=per_beta_start,
                                               beta_frames=per_beta_frames, eps=per_eps, seed=seed)
        self.grad_steps_per_update = grad_steps_per_update   # число градиентных шагов НА КАЖДЫЙ вызов обучения
        # Разделяет частоту шагов среды и частоту обучения: обучение запускается раз в
        # train_interval шагов среды (а не на КАЖДОМ шаге, как раньше). Общее число
        # градиентных обновлений за эпизод = (episode_len / train_interval) * grad_steps_per_update.
        # При увеличении длины эпизода (30k -> 150k) это даёт прямой рычаг контроля суммарного
        # вычислительного бюджета без изменения качества/логики самого PER или сети.
        self.train_interval = max(1, train_interval)
        self.eps_base, self.eps_drift = eps_base, eps_drift
        self.eps_start, self.eps_decay_steps = eps_start, eps_decay_steps  # повышенная разведка в начале онлайн-фазы
        self.target_sync_every = target_sync_every
        self.safety_enabled = bool(safety_enabled)   # флаг: включён ли откат к эвристике при аномалиях
        self.drift_exploration_enabled = bool(drift_exploration_enabled)  # флаг: включён ли всплеск eps при дрейфе
        self.fallback = fallback        # эвристика для отката при аномалиях (используется только если safety_enabled)
        self.rng = np.random.default_rng(seed)
        self._hist = deque(maxlen=50)   # окно недавних задержек (детекция дрейфа/аномалий)
        self._cooldown = 0              # шаги, оставшиеся в режиме "откат к эвристике"
        self._online_steps = 0          # счётчик для расписания eps и синхронизации target-сети
        self.last_eps = eps_start        # текущее значение эпсилон (для построения кривой разведки)

    def _drift_detected(self):
        if len(self._hist) < 50:
            return False
        a = np.array(self._hist)
        return a[-10:].mean() > 1.5 * a[:-10].mean() + 1e-6

    def _current_eps(self):
        """Линейное затухание eps_start -> eps_base за eps_decay_steps онлайн-шагов
        (больше разведки в начале, когда претрейн-политика ещё не проверена на практике).

        Повышение при обнаружении дрейфа (eps_drift) применяется ТОЛЬКО когда базовое
        значение уже достигло минимума (base == eps_base)."""
        if self._online_steps < self.eps_decay_steps:
            frac = self._online_steps / max(self.eps_decay_steps, 1)
            base = self.eps_start + frac * (self.eps_base - self.eps_start)
        else:
            base = self.eps_base

        at_minimum = base <= self.eps_base + 1e-9
        if self.drift_exploration_enabled and at_minimum and self._drift_detected():
            return self.eps_drift
        return base

    def route(self, query, env, valid_actions):
        self._last_x = env.state_vector(query)    # numpy-массив; torch.Tensor создаётся только при реальной
                                                     # необходимости forward-pass (см. ниже), а не на каждом шаге
        self.last_eps = self._current_eps()       # сохраняем для построения кривой разведки
        if self.safety_enabled and self._cooldown > 0:   # предохранитель отключается флагом safety_enabled
            self._cooldown -= 1
            return self.fallback.route(query, env, valid_actions)
        if self.rng.random() < self.last_eps:
            return int(self.rng.choice(valid_actions))
        with torch.no_grad():
            qv = self.q(torch.from_numpy(self._last_x)).numpy()
        return max(valid_actions, key=lambda a: qv[a])

    def observe(self, action, reward, latency, next_query, next_env):
        self._hist.append(latency)
        if self.safety_enabled and len(self._hist) >= 10 and latency > 10 * np.median(self._hist):
            self._cooldown = 20                   # аномалия -> временный откат к эвристике (только если защита включена)
        next_x = next_env.state_vector(next_query)   # numpy-массив, без создания torch.Tensor на каждом шаге
        self.buffer.add(self._last_x, action, reward, next_x)
        self._online_steps += 1
        # Несколько градиентных шагов на 1 обучающий вызов (рекомендация 4), но сам вызов
        # теперь происходит не на КАЖДОМ шаге среды, а раз в train_interval шагов — иначе
        # при увеличении длины эпизода общее число обновлений сети растёт линейно и
        # становится основным источником замедления прогонов (см. пояснение в конструкторе).
        if len(self.buffer) >= self.batch_size and self._online_steps % self.train_interval == 0:
            for _ in range(self.grad_steps_per_update):
                self._train_step()
        if self._online_steps % self.target_sync_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())   # периодическая синхронизация target-сети (не зависит
                                                                    # от числа градиентных шагов — только от числа шагов среды)

    def _train_step(self):
        batch, idx, is_weights = self.buffer.sample(self.batch_size)   # PER: сэмплинг по приоритету + веса IS
        xs, next_xs = batch["states"], batch["next_states"]              # уже готовые torch.Tensor (см. sample())
        actions, rewards = batch["actions"], batch["rewards"]
        weights = torch.from_numpy(is_weights.astype(np.float32))

        q_pred = self.q(xs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN (рекомендация 2): действие для следующего состояния выбирает
            # ОНЛАЙН-сеть (self.q), а его Q-значение оценивает TARGET-сеть (self.q_target).
            next_actions = self.q(next_xs).argmax(dim=1)
            q_next = self.q_target(next_xs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + self.gamma * q_next

        td_errors = (target - q_pred).detach().numpy()
        loss = (weights * (q_pred - target) ** 2).mean()   # взвешивание по IS-весам PER
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        self.buffer.update_priorities(idx, td_errors)   # приоритет = |TD-ошибка| (см. PrioritizedReplayBuffer)

    def pretrain(self, transitions, grad_steps=20):
        """Офлайн-инициализация. transitions — список (x, a, r, nx) с x/nx как numpy-массивы
        (env.state_vector(...)); torch.Tensor создаётся только внутри _train_step() при
        сборке батча, не на каждый отдельный переход."""
        for x, a, r, nx in transitions:
            self.buffer.add(x, a, r, nx)   # PER: приоритет присваивается автоматически
        for _ in range(grad_steps):
            if len(self.buffer) >= self.batch_size:
                self._train_step()
        self.q_target.load_state_dict(self.q.state_dict())   # синхронизация target-сети после претрейна