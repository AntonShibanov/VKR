"""
environment.py
Симулятор гибридной БД с явной топологией узлов (ROW/COL), очередями,
деградацией и admission control. Центральный файл проекта — определяет,
что "видит" любая политика маршрутизации (state_vector) и как считается
итоговая задержка/награда (step).
"""
import numpy as np
from collections import deque
from dataclasses import dataclass

# числовые коды двух типов движков: 0=строчный, 1=колоночный
ROW, COL = 0, 1
# размерность вектора состояния, см. state_vector() ниже
STATE_DIM = 11


# автогенерация __init__/__repr__ для простого контейнера данных
@dataclass
class Node:
    """Один физический узел (сервер) внутри кластера ROW или COL."""
    engine: int                                       # тип движка узла: ROW или COL
    # пропускная способность узла (сколько очереди обслуживает за шаг)
    service_rate: float
    # базовая сетевая задержка до этого узла, мс
    base_rtt_ms: float
    # человекочитаемый идентификатор узла, например "row_0"
    node_id: str = ""
    # коэффициент замедления: 1.0=норма, >1.0=деградация
    health_factor: float = 1.0
    # True = узел полностью недоступен (отказ)
    is_down: bool = False
    # текущая длина очереди узла (условные единицы нагрузки)
    queue: float = 0.0
    # предельная ёмкость очереди (admission control); по умолчанию без ограничения
    max_queue: float = float("inf")
    # --- НОВОЕ (вариант 2: стохастическая деградация) ---
    # Внутреннее состояние "мерцающей" (случайной, не запланированной)
    # деградации узла — независимо от schedule_degradation(). Хранится на
    # самом узле, чтобы не заводить отдельный параллельный список событий:
    # _flicker_remaining — сколько шагов ещё действует текущий случайный
    # эпизод замедления (0 = сейчас неактивен);
    # _flicker_health_factor — целевой health_factor на время этого эпизода.
    _flicker_remaining: int = 0
    _flicker_health_factor: float = 1.0


class HybridDBEnv:
    """Основной класс среды: инкапсулирует состояние обоих кластеров и логику
    одного шага симуляции."""

    # Базовое (немодифицированное) время обслуживания запроса (мс) в
    # зависимости от типа запроса и движка.
    # Это ключевая "физика" модели — не должна подгоняться под желаемый
    # результат эксперимента.
    SERVICE_BASE = {
        # вставка: быстрее на ROW
        ("insert", ROW): 2.0, ("insert", COL): 6.0,
        # аналитика: значительно быстрее на COL
        ("analytic", ROW): 40.0, ("analytic", COL): 4.0,
        # транзакция: быстрее на ROW
        ("transactional", ROW): 3.0, ("transactional", COL): 25.0,
    }

    # "Типичная" (референсная) стоимость запроса КАЖДОГО типа, используемая
    # для нормировки cost_scaling_exponent в step() (см. ниже). Значения —
    # exp(mu) по исходным, преднамеренно сохранённым параметрам mu из
    # workload.cost_distributions (config.yaml: insert mu=1.0, analytic
    # mu=3.5, transactional mu=1.5).
    DEFAULT_COST_SCALING_REFERENCE = {
        "insert": 2.718281828459045,          # exp(1.0)
        "analytic": 33.11545195869231,        # exp(3.5)
        "transactional": 4.481689070338065,   # exp(1.5)
    }

    # --- ИЗМЕНЕНИЕ (исправление вырожденной "all_col"-политики) ---
    # Было: единое self.cost_load_reference (одно число) для ВСЕХ типов
    # запроса сразу — использовалось в step() как точка нормировки вклада
    # запроса в очередь (load_scales_with_cost). Из-за этого понижение
    # общего референса (чтобы вернуть insert'у реальную нагрузку на
    # очередь COL) неизбежно затронуло бы и analytic — а там референс=16.0
    # был СПЕЦИАЛЬНО подобран, чтобы скомпенсировать рост sigma аналитики
    # по калибровке Redset (0.6 -> 2.296) и не допустить срезания
    # реалистично-тяжёлого хвоста потолком очереди (см. комментарий в
    # config.yaml). Теперь — ОТДЕЛЬНЫЙ референс на каждый qtype, по той же
    # схеме, что уже применена к cost_scaling_reference:
    #   - "analytic" оставлен РОВНО 16.0 — калибровка Redset не затронута;
    #   - "insert"/"transactional" понижены до их собственной медианной
    #     стоимости (exp(mu), тот же якорь, что и в
    #     DEFAULT_COST_SCALING_REFERENCE) — это восстанавливает для них
    #     "типичный" запрос с arrival_load == ARRIVAL_LOAD (0.35), то есть
    #     реальное давление на очередь, если COL массово используется для
    #     дешёвых запросов (сейчас было 0.0595 << col_service_rate=0.25 —
    #     фактически "бесплатно"). Проверено Monte Carlo: доля запросов,
    #     упирающихся в потолок очереди, для insert/transactional остаётся
    #     пренебрежимо малой (<0.2%), а для analytic — НЕ изменилась
    #     (24.6%, совпадает с задокументированным фиксом).
    DEFAULT_COST_LOAD_REFERENCE = {
        "insert": 2.718281828459045,          # exp(1.0)
        "analytic": 16.0,                     # БЕЗ ИЗМЕНЕНИЙ — фикс после Redset-калибровки sigma
        "transactional": 4.481689070338065,   # exp(1.5)
    }

    # фиксированный вклад одного запроса в очередь узла (единиц) — значение
    # ПО УМОЛЧАНИЮ; реальное значение, используемое средой, хранится в
    # self.arrival_load и может быть переопределено через конструктор (см. ниже)
    ARRIVAL_LOAD = 0.35
    # "типичная" стоимость запроса — точка нормировки нагрузки
    COST_LOAD_REFERENCE = 5.0
    # ограничение сверху вклада запроса в очередь (в разах от ARRIVAL_LOAD) —
    # значение ПО УМОЛЧАНИЮ; реальное значение хранится в self.max_arrival_load_mult
    # и может быть переопределено через конструктор (см. ниже)
    MAX_ARRIVAL_LOAD_MULT = 10.0
    # фиксированная "катастрофическая" задержка при полном отказе обоих кластеров
    TIMEOUT_MS = 5000.0
    # размер скользящего окна истории задержек для адаптивного порога reward
    LAT_WINDOW = 200
    # нижняя граница адаптивного порога (не штрафовать шум при малой нагрузке)
    LAT_FLOOR_MS = 5.0
    # значение адаптивного порога до накопления достаточной истории
    INITIAL_THRESHOLD_MS = 50.0
    # --- НОВОЕ (вариант 1: нелинейная очередь) ---
    # Верхняя граница коэффициента загрузки rho = queue/capacity, используемая
    # в формуле M/M/1-подобной задержки очереди (см. step()). Ограничение
    # нужно, чтобы rho/(1-rho) не уходило в бесконечность вблизи переполнения
    # очереди; 0.95 выбрано как рабочая стартовая точка (rho/(1-rho)=19) —
    # это ДИЗАЙН-РЕШЕНИЕ интенсификации сценария, а не измеренная величина,
    # и подлежит подбору/валидации по итогам повторного прогона.
    QUEUE_RHO_CAP = 0.95
    # Если max_queue узла не ограничена (admission_control_enabled=False),
    # для вычисления rho используется эта условная "эффективная ёмкость"
    # вместо фактического max_queue=inf (иначе rho всегда было бы ~0).
    QUEUE_RHO_REFERENCE = 10.0
    # Веса компонентов reward по умолчанию (используются, если секция reward в
    # конфиге не задана)
    DEFAULT_REWARD_WEIGHTS = dict(
        imbalance=0.3, congestion=0.0, sla_violation=0.5, failover=1.0)

    def __init__(self, seed=0, n_row_nodes=2, n_col_nodes=2,
                 row_service_rate=0.5, col_service_rate=0.35,
                 row_rtt_ms=0.3, col_rtt_ms=2.0, reward_weights=None,
                 load_scales_with_cost=False, cost_load_reference=None,
                 admission_control_enabled=False, max_queue_row=float("inf"),
                 max_queue_col=float("inf"), reject_timeout_ms=600.0,
                 admission_reroute_penalty_ms=8.0, cost_scaling_exponent=0.3,
                 arrival_load=None, max_arrival_load_mult=None,
                 cost_scaling_reference=None, network_cost_scaling_exponent=0.0,
                 # --- НОВОЕ (вариант 1: нелинейная очередь) ---
                 # queue_nonlinear_enabled=False сохраняет прежнее (линейное)
                 # поведение по умолчанию — изменение полностью опционально.
                 queue_nonlinear_enabled=False, queue_rho_cap=None,
                 queue_rho_reference=None,
                 # --- НОВОЕ (вариант 2: стохастическая деградация) ---
                 # Выключено по умолчанию (stochastic_degradation_enabled=False)
                 # — не меняет поведение существующих экспериментов, пока не
                 # включено явно в конфиге.
                 stochastic_degradation_enabled=False,
                 stochastic_degradation_prob=0.0008,
                 stochastic_degradation_duration_range=(50, 300),
                 stochastic_degradation_health_range=(2.0, 6.0)):
        # собственный генератор случайных чисел для воспроизводимости
        self.rng = np.random.default_rng(seed)
        # Создаём список узлов ROW‐кластера: каждому присваивается уникальный
        # node_id вида "row_0", "row_1", ...
        self.row_nodes = [Node(ROW, row_service_rate, row_rtt_ms,
                                node_id=f"row_{i}",
                                max_queue=max_queue_row) for i in range(n_row_nodes)]
        # Аналогично для COL‐кластера
        self.col_nodes = [Node(COL, col_service_rate, col_rtt_ms,
                                node_id=f"col_{i}",
                                max_queue=max_queue_col) for i in range(n_col_nodes)]
        # флаг: включена ли защита от перегрузки очереди
        self.admission_control_enabled = bool(admission_control_enabled)
        # задержка при полном отказе (нет доступных узлов вообще)
        self.reject_timeout_ms = reject_timeout_ms
        # штраф за принудительное перенаправление
        self.admission_reroute_penalty_ms = admission_reroute_penalty_ms
        # текущая "прогретость" кэша колоночного кластера (0..1)
        self.col_cache_warmth = 0.3
        # текущий лаг репликации ROW -> COL (0..1)
        self.col_repl_lag = 0.0
        # счётчик выполненных шагов (номер текущего запроса)
        self._t = 0
        # список запланированных событий деградации узлов
        self._degradation_events = []
        # скользящее окно последних задержек (для адаптивного порога)
        self._lat_history = deque(maxlen=self.LAT_WINDOW)
        # общий (одинаковый для всех методов) профиль порога reward
        self._external_thresholds = None
        # Итоговые веса reward = дефолтные, перезаписанные тем, что явно
        # передано (если что‐то передано)
        self.reward_weights = {
            **self.DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        # включён ли масштабируемый вклад запроса в очередь
        self.load_scales_with_cost = bool(load_scales_with_cost)
        # точка нормировки стоимости
        #self.cost_load_reference = cost_load_reference or self.COST_LOAD_REFERENCE

        # --- ИЗМЕНЕНИЕ: self.cost_load_reference теперь словарь {qtype: ref} ---
        # Обратная совместимость: если передано ОДНО число (старое
        # поведение config.yaml -> nodes.cost_load_reference: 16.0) —
        # применяем его как общий референс для ВСЕХ типов (ровно как было
        # раньше). Если передан словарь — используем per-type значения,
        # дополняя отсутствующие ключи значениями по умолчанию
        # (DEFAULT_COST_LOAD_REFERENCE).
        if cost_load_reference is None:
            self.cost_load_reference = dict(self.DEFAULT_COST_LOAD_REFERENCE)
        elif isinstance(cost_load_reference, dict):
            self.cost_load_reference = {
                **self.DEFAULT_COST_LOAD_REFERENCE, **cost_load_reference}
        else:
            # скаляр — старое поведение, тот же референс для всех типов
            self.cost_load_reference = {
                k: float(cost_load_reference) for k in self.DEFAULT_COST_LOAD_REFERENCE}

        # Если arrival_load/max_arrival_load_mult явно не переданы — используем
        # прежние значения констант класса (поведение не отличается от старой
        # версии кода).
        self.arrival_load = arrival_load if arrival_load is not None else self.ARRIVAL_LOAD
        self.max_arrival_load_mult = (max_arrival_load_mult if max_arrival_load_mult is not None
                                       else self.MAX_ARRIVAL_LOAD_MULT)
        # верхний предел вклада запроса в очередь
        self.max_arrival_load = self.arrival_load * self.max_arrival_load_mult

        self.cost_scaling_exponent = cost_scaling_exponent
        # Итоговая таблица референсов = дефолтная, перезаписанная тем, что
        # явно передано (если что-то передано).
        self.cost_scaling_reference = {
            **self.DEFAULT_COST_SCALING_REFERENCE, **(cost_scaling_reference or {})}
        # Отдельный экспонент для сетевой компоненты задержки.
        self.network_cost_scaling_exponent = network_cost_scaling_exponent

        # --- НОВОЕ (вариант 1: нелинейная очередь) ---
        self.queue_nonlinear_enabled = bool(queue_nonlinear_enabled)
        self.queue_rho_cap = queue_rho_cap if queue_rho_cap is not None else self.QUEUE_RHO_CAP
        self.queue_rho_reference = (queue_rho_reference if queue_rho_reference is not None
                                     else self.QUEUE_RHO_REFERENCE)

        # --- НОВОЕ (вариант 2: стохастическая деградация) ---
        self.stochastic_degradation_enabled = bool(stochastic_degradation_enabled)
        self.stochastic_degradation_prob = float(stochastic_degradation_prob)
        self.stochastic_degradation_duration_range = tuple(stochastic_degradation_duration_range)
        self.stochastic_degradation_health_range = tuple(stochastic_degradation_health_range)

    def set_external_thresholds(self, thresholds):
        """Подключить общий профиль адаптивного порога, построенный нейтральной
        эвристикой (см. run_experiment.py),
        чтобы reward всех методов реагировал на дрейф нагрузки одинаково, а не
        на собственное качество маршрутизации."""
        self._external_thresholds = np.asarray(
            thresholds)   # сохраняем как numpy‐массив для быстрой индексации

    def schedule_degradation(self, engine, node_idx, start_q, end_q,
                              health_factor=1.0, is_down=False,
                              ramp_queries=0):
        """Зарегистрировать событие деградации конкретного узла в заданном
        диапазоне номеров запросов."""
        # Ограничиваем длину "рампы" половиной окна — иначе деградация
        # растянется дольше заявленного интервала
        ramp_queries = min(ramp_queries, max(0, (end_q - start_q) // 2))
        self._degradation_events.append(dict(engine=engine, node_idx=node_idx,
                                              start=start_q,
                                              end=end_q,
                                              health_factor=health_factor, is_down=is_down,
                                              ramp_queries=ramp_queries))  # сохраняем параметры события в список

    def _apply_scheduled_degradation(self):
        """Применить все зарегистрированные (запланированные заранее) события
        деградации к текущему моменту self._t."""
        for ev in self._degradation_events:                                   # проходим по всем событиям деградации
            node = (self.row_nodes if ev["engine"] == ROW else self.col_nodes)[
                ev["node_idx"]]  # находим целевой узел
            # активно ли событие СЕЙЧАС
            active = ev["start"] <= self._t < ev["end"]
            # полный отказ — мгновенный (без рампы)
            node.is_down = ev["is_down"] and active

            # длина рампы для health_factor
            ramp = ev.get("ramp_queries", 0)
            if ramp <= 0:                                                      # если рампа не задана — старое поведение
                # мгновенный шаг health_factor
                node.health_factor = ev["health_factor"] if active else 1.0
                # переходим к следующему событию
                continue
            # до начала окна деградации
            if self._t < ev["start"]:
                # узел полностью здоров
                node.health_factor = 1.0
            # фаза линейного подъёма деградации
            elif self._t < ev["start"] + ramp:
                # доля пройденного пути подъёма (0..1)
                frac = (self._t - ev["start"]) / ramp
                # линейная интерполяция от 1.0 до цели
                node.health_factor = 1.0 + frac * (ev["health_factor"] - 1.0)
            # фаза плато (полная деградация)
            elif self._t < ev["end"] - ramp:
                # держим целевое значение
                node.health_factor = ev["health_factor"]
            # фаза линейного спада деградации
            elif self._t < ev["end"]:
                # доля оставшегося пути спада (0..1)
                frac = (ev["end"] - self._t) / ramp
                # линейная интерполяция обратно к 1.0
                node.health_factor = 1.0 + frac * (ev["health_factor"] - 1.0)
            else:                                                              # после окончания окна
                # узел снова полностью здоров
                node.health_factor = 1.0

    def _apply_stochastic_degradation(self):
        """--- НОВОЕ (вариант 2 интенсификации сценария) ---
        Наложить на уже применённую запланированную деградацию (см.
        _apply_scheduled_degradation) НЕЗАВИСИМЫЙ случайный процесс
        кратковременных "мерцающих" замедлений узлов.

        Зачем: два фиксированных события деградации покрывают лишь ~27%
        эпизода (40 000 из 150 000 запросов); на остальных ~73% решение
        "тип запроса -> движок" совпадает с правильным почти всегда, и
        state-aware политика (RL/бандит) не имеет возможности показать
        преимущество перед простой эвристикой по стоимости. Этот метод
        создаёт воспроизводимый (по seed) поток случайных кратковременных
        замедлений РАВНОМЕРНО по всему эпизоду, не видимых политике иначе,
        чем через признаки row_degradation/col_degradation в state_vector —
        то есть принципиально недоступных CostThresholdRouter.

        Параметры запуска: вероятность начала эпизода замедления на каждом
        шаге для каждого (пока не деградирующего) узла, длительность и
        целевой health_factor сэмплируются равномерно из заданных диапазонов.
        Если на узел в этот момент уже действует запланированное событие
        (is_down=True) — "мерцание" на него не накладывается."""
        if not self.stochastic_degradation_enabled:
            return
        lo_dur, hi_dur = self.stochastic_degradation_duration_range
        lo_h, hi_h = self.stochastic_degradation_health_range
        for node in self.row_nodes + self.col_nodes:
            if node.is_down:
                # на полностью отказавший (по расписанию) узел мерцание не накладываем
                continue
            if node._flicker_remaining > 0:
                # активный эпизод мерцания — поддерживаем пониженный health_factor
                node._flicker_remaining -= 1
                # берём max с уже установленным расписанием health_factor —
                # так мерцание не "ослабляет" более сильную запланированную деградацию,
                # а лишь гарантирует НИЖНЮЮ ГРАНИЦУ замедления на время своего действия
                node.health_factor = max(node.health_factor, node._flicker_health_factor)
                if node._flicker_remaining == 0:
                    node._flicker_health_factor = 1.0
            elif self.rng.random() < self.stochastic_degradation_prob:
                # запускаем новый случайный эпизод замедления
                node._flicker_remaining = int(self.rng.integers(lo_dur, hi_dur + 1))
                node._flicker_health_factor = float(self.rng.uniform(lo_h, hi_h))
                node.health_factor = max(node.health_factor, node._flicker_health_factor)

    def _select_node(self, healthy):
        """Выбрать наименее загруженный узел среди переданных (здоровых, с
        запасом ёмкости)."""
        min_q = min(
            # минимальная длина очереди среди кандидатов
            n.queue for n in healthy)
        # все узлы с (почти) минимальной очередью
        candidates = [n for n in healthy if n.queue - min_q < 1e-6]
        # случайный выбор среди равных — избегаем смещения
        return candidates[self.rng.integers(len(candidates))]

    def _nodes_with_capacity(self, nodes):
        """Отфильтровать здоровые узлы, у которых очередь не превысила предел
        ёмкости (admission control)."""
        healthy = [
            # сначала отбрасываем полностью отказавшие узлы
            n for n in nodes if not n.is_down]
        # если защита от перегрузки выключена
        if not self.admission_control_enabled:
            # возвращаем всех здоровых без проверки очереди
            return healthy
        # иначе — только тех, у кого есть запас ёмкости
        return [n for n in healthy if n.queue < n.max_queue]

    def valid_actions(self, query):
        """Список допустимых действий (движков) для данного запроса."""
        # Запросы со строгой согласованностью (транзакции/биллинг) не могут
        # уйти на COL — там нет строгой консистентности
        return [ROW] if query.strict_consistency else [ROW, COL]

    @staticmethod
    def _cluster_features(nodes):
        """Агрегированные признаки состояния одного кластера (для вектора
        состояния агента)."""
        healthy = [
            # список живых узлов кластера
            n for n in nodes if not n.is_down]
        # доля живых узлов от общего числа
        healthy_frac = len(healthy) / len(nodes)
        # минимальная очередь; 50.0 — признак "все мертвы"
        min_queue = min((n.queue for n in healthy), default=50.0)
        # Средний коэффициент деградации по здоровым узлам — делает деградацию
        # НАБЛЮДАЕМОЙ для всех политик
        avg_health = float(
            np.mean([n.health_factor for n in healthy])) if healthy else 1.0
        # возвращаем три агрегированных признака
        return min_queue, healthy_frac, avg_health

    def state_vector(self, query):
        """Построить 11‐мерный вектор состояния — это то, что "видит" любая
        политика маршрутизации."""
        qcode = {"insert": 0, "analytic": 1, "transactional": 2}[
            query.qtype]  # числовой код типа запроса
        row_q, row_health, row_degradation = self._cluster_features(
            self.row_nodes)  # признаки ROW‐кластера
        col_q, col_health, col_degradation = self._cluster_features(
            self.col_nodes)  # признаки COL‐кластера
        return np.array([
            qcode,                              # тип текущего запроса (0/1/2)
            # логарифм стоимости запроса (сглаживает большой разброс значений)
            np.log1p(query.cost),
            # требует ли запрос строгой согласованности (0/1)
            float(query.strict_consistency),
            # минимальная очередь и доля здоровых узлов кластера ROW
            row_q, row_health,
            col_q, col_health,                  # то же для кластера COL
            self.col_cache_warmth,              # текущая прогретость колоночного кэша
            self.col_repl_lag,                  # текущий лаг репликации ROW -> COL
            # средний health_factor каждого кластера (видимость деградации)
            row_degradation, col_degradation,
            # приводим к float32 для совместимости с PyTorch
        ], dtype=np.float32)

    def step(self, query, action):
        """Выполнить один шаг симуляции: обработать один запрос выбранным
        действием (движком)."""
        self._t += 1                                            # увеличиваем счётчик выполненных запросов
        # обновляем health_factor/is_down всех узлов по расписанию
        self._apply_scheduled_degradation()
        # --- НОВОЕ (вариант 2) --- накладываем случайные "мерцающие" замедления
        self._apply_stochastic_degradation()

        # список узлов выбранного кластера
        nodes = self.row_nodes if action == ROW else self.col_nodes
        # живые узлы этого кластера
        healthy = [n for n in nodes if not n.is_down]
        # реально использованное действие и штраф за перенаправление
        used_action, failover_penalty = action, 0.0

        # --- Случай 1: весь запрошенный кластер физически недоступен (полный отказ) ---
        if not healthy:
            if query.strict_consistency:                         # для строгих транзакций нет запасного варианта
                info = dict(node_id="timeout", network_ms=0.0,
                            processing_ms=self.TIMEOUT_MS)
                # возвращаем таймаут с большим штрафом
                return self.TIMEOUT_MS, -10.0, 1.0, info
            # пробуем альтернативный кластер
            alt = self.col_nodes if action == ROW else self.row_nodes
            # живые узлы альтернативного кластера
            healthy = [n for n in alt if not n.is_down]
            if not healthy:                                       # если и он полностью недоступен
                info = dict(node_id="timeout", network_ms=0.0,
                            processing_ms=self.TIMEOUT_MS)
                # крайний случай — оба кластера недоступны
                return self.TIMEOUT_MS, -10.0, 1.0, info
            # переключаемся, штраф за failover
            used_action, failover_penalty = (
                COL if action == ROW else ROW), 15.0
            available = self._nodes_with_capacity(
                self.row_nodes if used_action == ROW else self.col_nodes)
        else:
            # узлы исходного кластера с запасом ёмкости
            available = self._nodes_with_capacity(nodes)

        # --- Случай 2: узел жив, но его очередь заполнена (admission control / backpressure) ---
        if not available:
            # альтернативный кластер
            alt_nodes = self.col_nodes if used_action == ROW else self.row_nodes
            alt_available = ([] if query.strict_consistency        # строгим запросам нельзя уходить на COL
                              else self._nodes_with_capacity(alt_nodes))
            if alt_available:                                       # если в альтернативном кластере есть место
                used_action = COL if used_action == ROW else ROW    # меняем используемое действие
                # берём доступные узлы альтернативы
                available = alt_available
                failover_penalty = max(
                    failover_penalty, self.admission_reroute_penalty_ms)  # фиксируем штраф
            else:                                                    # альтернативы тоже нет — отклоняем запрос
                reject_latency = self.reject_timeout_ms * \
                    self.rng.lognormal(
                        0.0, 0.15)  # случайная задержка отклонения
                info = dict(node_id="admission_rejected", network_ms=0.0,
                            processing_ms=float(reject_latency),
                            admission_rejected=True)
                reward = -(reject_latency / max(self.INITIAL_THRESHOLD_MS, 1.0)) \
                    - self.reward_weights["sla_violation"]           # награда штрафуется пропорционально задержке отклонения
                # возвращаем результат отклонённого запроса
                return reject_latency, reward, 0.0, info

        # выбираем конкретный узел (наименее загруженный)
        node = self._select_node(available)
        # базовое время с учётом деградации
        base = self.SERVICE_BASE[(
            query.qtype, used_action)] * node.health_factor
        # Штраф за холодный кэш: применяется только к аналитике на COL, если кэш не прогрет
        cache_penalty = (1.0 - self.col_cache_warmth) * \
            15.0 if used_action == COL and query.qtype == "analytic" else 0.0
        # штраф за лаг репликации (только для COL)
        lag_penalty = self.col_repl_lag * 10.0 if used_action == COL else 0.0

        # --- ИЗМЕНЕНИЕ (вариант 1: нелинейная M/M/1-подобная очередь) ---
        # Было (линейно): queue_wait = node.queue * base * 0.5
        # Теперь (опционально, по флагу queue_nonlinear_enabled):
        # эффективная ёмкость узла для расчёта коэффициента загрузки rho —
        # либо фактический предел очереди (если admission control включён
        # и max_queue конечен), либо условная референсная ёмкость
        # (queue_rho_reference) иначе.
        if self.queue_nonlinear_enabled:
            capacity = node.max_queue if node.max_queue < float("inf") else self.queue_rho_reference
            rho = min(node.queue / capacity, self.queue_rho_cap) if capacity > 0 else 0.0
            # Классическая M/M/1-форма среднего времени ожидания в очереди:
            # W_q ~ base * rho / (1 - rho). Вблизи насыщения (rho -> cap)
            # задержка растёт резко нелинейно, в отличие от прежней линейной
            # формы — это намеренно делает "почти заполненную" очередь
            # значительно опаснее, чем "наполовину заполненную", усиливая
            # ценность признаков row_q/col_q в state_vector для политик,
            # которые их используют.
            queue_wait = base * (rho / (1.0 - rho))
        else:
            # прежнее (линейное) поведение — используется по умолчанию
            queue_wait = node.queue * base * 0.5

        # Сетевая задержка: базовая величина + небольшой случайный джиттер
        network = node.base_rtt_ms + \
            abs(self.rng.normal(0, node.base_rtt_ms * 0.2 + 1e-6))
        # мультипликативный шум задержки (реалистичная вариативность)
        noise = self.rng.lognormal(mean=0.0, sigma=0.25)

        # Нормируем cost на типичную ("референсную") стоимость запроса ИМЕННО
        # этого типа — так SERVICE_BASE продолжает означать "типичное время
        # обслуживания" независимо от величины cost_scaling_exponent, а сам
        # экспонент калибрует только реакцию на ОТКЛОНЕНИЕ от типичного (хвост).
        cost_ratio = query.cost / self.cost_scaling_reference.get(
            query.qtype, self.COST_LOAD_REFERENCE)
        scale_processing = noise * (cost_ratio ** self.cost_scaling_exponent)
        # Сетевая компонента использует ОТДЕЛЬНЫЙ экспонент (по умолчанию 0.0,
        # т.е. cost-независима) — калибровка по Redset описывает только
        # серверное время обработки, а не сетевой RTT.
        scale_network = noise * \
            (cost_ratio ** self.network_cost_scaling_exponent)
        # Разделяем итоговую задержку на компоненту обработки узлом и сетевую компоненту (для диагностики)
        processing_ms = (base + cache_penalty + lag_penalty +
                          queue_wait + failover_penalty) * scale_processing
        network_ms = network * scale_network
        # суммарная наблюдаемая задержка запроса
        latency = processing_ms + network_ms

        # --- Обновление длины очередей узлов (пропускная способность vs новая нагрузка) ---
        # если вклад запроса в очередь масштабируется стоимостью
        if self.load_scales_with_cost:
            arrival_load = min(self.arrival_load * (query.cost / self.COST_LOAD_REFERENCE),
                                # пропорционально стоимости, но не выше предела
                                self.max_arrival_load)
        else:
            # иначе — фиксированная величина (старое поведение), также через self.arrival_load
            arrival_load = self.arrival_load
        # кластер, который реально обслужил запрос
        used_cluster = self.row_nodes if used_action == ROW else self.col_nodes
        # обновляем очередь каждого узла этого кластера
        for n in used_cluster:
            # обслужили + новая нагрузка
            n.queue = max(0.0, n.queue - n.service_rate) + \
                (arrival_load if n is node else 0.0)
            # очередь не может превысить предел ёмкости
            n.queue = min(n.queue, n.max_queue)
        # второй (не задействованный) кластер
        other_cluster = self.col_nodes if used_action == ROW else self.row_nodes
        # там нагрузка не добавляется, только обслуживание
        for n in other_cluster:
            n.queue = max(0.0, n.queue - n.service_rate)

        # Обновление прогретости кэша COL: разогревается при аналитике на COL, иначе остывает
        if used_action == COL and query.qtype == "analytic":
            self.col_cache_warmth = min(1.0, self.col_cache_warmth + 0.05)
        else:
            self.col_cache_warmth = max(0.0, self.col_cache_warmth - 0.02)
        # Обновление лага репликации: растёт при записи на ROW, иначе снижается
        if query.qtype in ("insert", "transactional") and used_action == ROW:
            self.col_repl_lag = min(1.0, self.col_repl_lag + 0.03)
        else:
            self.col_repl_lag = max(0.0, self.col_repl_lag - 0.05)

        # Дисбаланс — абсолютная разница между средней очередью ROW‐кластера и средней очередью COL‐кластера
        imbalance = abs(np.mean(
            [n.queue for n in self.row_nodes]) - np.mean([n.queue for n in self.col_nodes]))
        # штраф — текущая (обновлённая) очередь выбранного узла
        congestion_penalty = node.queue

        # Определение адаптивного порога для reward: либо общий (переданный извне) профиль, либо собственная история
        if self._external_thresholds is not None:                        # если задан общий профиль порога
            # индекс текущего шага в массиве профиля
            idx = min(self._t - 1, len(self._external_thresholds) - 1)
            # берём значение порога из этого профиля
            adaptive_threshold = float(self._external_thresholds[idx])
        # достаточно накопленной собственной истории
        elif len(self._lat_history) >= 20:
            adaptive_threshold = max(self.LAT_FLOOR_MS, float(
                np.percentile(self._lat_history, 95)))  # скользящий p95
        else:
            # начальный порог до накопления истории
            adaptive_threshold = self.INITIAL_THRESHOLD_MS
        # добавляем текущую задержку в историю
        self._lat_history.append(latency)

        # бинарный признак превышения адаптивного порога
        sla_violation = float(latency > adaptive_threshold)
        # локальная ссылка на веса reward (для краткости)
        w = self.reward_weights
        # Итоговая награда: нормированная задержка + взвешенная сумма штрафов за дисбаланс, перегрузку, SLA и failover
        reward = -(latency / adaptive_threshold) - w["imbalance"] * imbalance \
            - w["congestion"] * congestion_penalty - w["sla_violation"] * sla_violation \
            - w["failover"] * (1.0 if failover_penalty else 0.0)
        info = dict(node_id=node.node_id, network_ms=float(network_ms),
                    processing_ms=float(processing_ms),
                    # диагностическая информация
                    adaptive_threshold_ms=float(adaptive_threshold),
                    admission_rejected=False)
        # возвращаем всё, что нужно вызывающему коду
        return latency, reward, imbalance, info