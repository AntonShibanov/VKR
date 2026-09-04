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

ROW, COL = 0, 1                                       # числовые коды двух типов движков: 0=строчный, 1=колоночный
STATE_DIM = 11                                        # размерность вектора состояния, см. state_vector() ниже


@dataclass                                            # автогенерация __init__/__repr__ для простого контейнера данных
class Node:
    """Один физический узел (сервер) внутри кластера ROW или COL."""
    engine: int                                       # тип движка узла: ROW или COL
    service_rate: float                               # пропускная способность узла (сколько очереди обслуживает за шаг)
    base_rtt_ms: float                                # базовая сетевая задержка до этого узла, мс
    node_id: str = ""                                 # человекочитаемый идентификатор узла, например "row_0"
    health_factor: float = 1.0                        # коэффициент замедления: 1.0=норма, >1.0=деградация
    is_down: bool = False                             # True = узел полностью недоступен (отказ)
    queue: float = 0.0                                # текущая длина очереди узла (условные единицы нагрузки)
    max_queue: float = float("inf")                   # предельная ёмкость очереди (admission control); по умолчанию без ограничения


class HybridDBEnv:
    """Основной класс среды: инкапсулирует состояние обоих кластеров и логику одного шага симуляции."""

    # Базовое (немодифицированное) время обслуживания запроса (мс) в зависимости от типа запроса и движка.
    # Это ключевая "физика" модели — не должна подгоняться под желаемый результат эксперимента.
    SERVICE_BASE = {
        ("insert", ROW): 2.0, ("insert", COL): 6.0,           # вставка: быстрее на ROW
        ("analytic", ROW): 40.0, ("analytic", COL): 4.0,      # аналитика: значительно быстрее на COL
        ("transactional", ROW): 3.0, ("transactional", COL): 25.0,  # транзакция: быстрее на ROW
    }
    ARRIVAL_LOAD = 0.35                                # фиксированный вклад одного запроса в очередь узла (единиц)
    COST_LOAD_REFERENCE = 5.0                          # "типичная" стоимость запроса — точка нормировки нагрузки
    MAX_ARRIVAL_LOAD_MULT = 10.0                       # ограничение сверху вклада запроса в очередь (в разах от ARRIVAL_LOAD)
    TIMEOUT_MS = 5000.0                                # фиксированная "катастрофическая" задержка при полном отказе обоих кластеров
    LAT_WINDOW = 200                                   # размер скользящего окна истории задержек для адаптивного порога reward
    LAT_FLOOR_MS = 5.0                                 # нижняя граница адаптивного порога (не штрафовать шум при малой нагрузке)
    INITIAL_THRESHOLD_MS = 50.0                        # значение адаптивного порога до накопления достаточной истории

    # Веса компонентов reward по умолчанию (используются, если секция reward в конфиге не задана)
    DEFAULT_REWARD_WEIGHTS = dict(imbalance=0.3, congestion=0.0, sla_violation=0.5, failover=1.0)

    def __init__(self, seed=0, n_row_nodes=2, n_col_nodes=2,
                 row_service_rate=0.5, col_service_rate=0.35,
                 row_rtt_ms=0.3, col_rtt_ms=2.0, reward_weights=None,
                 load_scales_with_cost=False, cost_load_reference=None,
                 admission_control_enabled=False, max_queue_row=float("inf"),
                 max_queue_col=float("inf"), reject_timeout_ms=600.0,
                 admission_reroute_penalty_ms=8.0):
        self.rng = np.random.default_rng(seed)         # собственный генератор случайных чисел для воспроизводимости
        # Создаём список узлов ROW-кластера: каждому присваивается уникальный node_id вида "row_0", "row_1", ...
        self.row_nodes = [Node(ROW, row_service_rate, row_rtt_ms, node_id=f"row_{i}",
                                max_queue=max_queue_row) for i in range(n_row_nodes)]
        # Аналогично для COL-кластера
        self.col_nodes = [Node(COL, col_service_rate, col_rtt_ms, node_id=f"col_{i}",
                                max_queue=max_queue_col) for i in range(n_col_nodes)]
        self.admission_control_enabled = bool(admission_control_enabled)  # флаг: включена ли защита от перегрузки очереди
        self.reject_timeout_ms = reject_timeout_ms       # задержка при полном отказе (нет доступных узлов вообще)
        self.admission_reroute_penalty_ms = admission_reroute_penalty_ms  # штраф за принудительное перенаправление
        self.col_cache_warmth = 0.3                       # текущая "прогретость" кэша колоночного кластера (0..1)
        self.col_repl_lag = 0.0                             # текущий лаг репликации ROW -> COL (0..1)
        self._t = 0                                          # счётчик выполненных шагов (номер текущего запроса)
        self._degradation_events = []                         # список запланированных событий деградации узлов
        self._lat_history = deque(maxlen=self.LAT_WINDOW)      # скользящее окно последних задержек (для адаптивного порога)
        self._external_thresholds = None                        # общий (одинаковый для всех методов) профиль порога reward
        # Итоговые веса reward = дефолтные, перезаписанные тем, что явно передано (если что-то передано)
        self.reward_weights = {**self.DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        self.load_scales_with_cost = bool(load_scales_with_cost)  # включён ли масштабируемый вклад запроса в очередь
        self.cost_load_reference = cost_load_reference or self.COST_LOAD_REFERENCE  # точка нормировки стоимости
        self.max_arrival_load = self.ARRIVAL_LOAD * self.MAX_ARRIVAL_LOAD_MULT  # верхний предел вклада запроса в очередь

    def set_external_thresholds(self, thresholds):
        """Подключить общий профиль адаптивного порога, построенный нейтральной эвристикой (см. run_experiment.py),
        чтобы reward всех методов реагировал на дрейф нагрузки одинаково, а не на собственное качество маршрутизации."""
        self._external_thresholds = np.asarray(thresholds)   # сохраняем как numpy-массив для быстрой индексации

    def schedule_degradation(self, engine, node_idx, start_q, end_q, health_factor=1.0, is_down=False,
                              ramp_queries=0):
        """Зарегистрировать событие деградации конкретного узла в заданном диапазоне номеров запросов."""
        # Ограничиваем длину "рампы" половиной окна — иначе деградация растянется дольше заявленного интервала
        ramp_queries = min(ramp_queries, max(0, (end_q - start_q) // 2))
        self._degradation_events.append(dict(engine=engine, node_idx=node_idx, start=start_q,
                                              end=end_q, health_factor=health_factor, is_down=is_down,
                                              ramp_queries=ramp_queries))       # сохраняем параметры события в список

    def _apply_scheduled_degradation(self):
        """Применить все зарегистрированные события деградации к текущему моменту self._t."""
        for ev in self._degradation_events:                                   # проходим по всем событиям деградации
            node = (self.row_nodes if ev["engine"] == ROW else self.col_nodes)[ev["node_idx"]]  # находим целевой узел
            active = ev["start"] <= self._t < ev["end"]                        # активно ли событие СЕЙЧАС
            node.is_down = ev["is_down"] and active                            # полный отказ — мгновенный (без рампы)

            ramp = ev.get("ramp_queries", 0)                                   # длина рампы для health_factor
            if ramp <= 0:                                                      # если рампа не задана — старое поведение
                node.health_factor = ev["health_factor"] if active else 1.0    # мгновенный шаг health_factor
                continue                                                       # переходим к следующему событию

            if self._t < ev["start"]:                                          # до начала окна деградации
                node.health_factor = 1.0                                       # узел полностью здоров
            elif self._t < ev["start"] + ramp:                                 # фаза линейного подъёма деградации
                frac = (self._t - ev["start"]) / ramp                          # доля пройденного пути подъёма (0..1)
                node.health_factor = 1.0 + frac * (ev["health_factor"] - 1.0)  # линейная интерполяция от 1.0 до цели
            elif self._t < ev["end"] - ramp:                                    # фаза плато (полная деградация)
                node.health_factor = ev["health_factor"]                       # держим целевое значение
            elif self._t < ev["end"]:                                           # фаза линейного спада деградации
                frac = (ev["end"] - self._t) / ramp                            # доля оставшегося пути спада (0..1)
                node.health_factor = 1.0 + frac * (ev["health_factor"] - 1.0)  # линейная интерполяция обратно к 1.0
            else:                                                               # после окончания окна
                node.health_factor = 1.0                                       # узел снова полностью здоров

    def _select_node(self, healthy):
        """Выбрать наименее загруженный узел среди переданных (здоровых, с запасом ёмкости)."""
        min_q = min(n.queue for n in healthy)                                  # минимальная длина очереди среди кандидатов
        candidates = [n for n in healthy if n.queue - min_q < 1e-6]            # все узлы с (почти) минимальной очередью
        return candidates[self.rng.integers(len(candidates))]                  # случайный выбор среди равных — избегаем смещения

    def _nodes_with_capacity(self, nodes):
        """Отфильтровать здоровые узлы, у которых очередь не превысила предел ёмкости (admission control)."""
        healthy = [n for n in nodes if not n.is_down]                          # сначала отбрасываем полностью отказавшие узлы
        if not self.admission_control_enabled:                                 # если защита от перегрузки выключена
            return healthy                                                    # возвращаем всех здоровых без проверки очереди
        return [n for n in healthy if n.queue < n.max_queue]                   # иначе — только тех, у кого есть запас ёмкости

    def valid_actions(self, query):
        """Список допустимых действий (движков) для данного запроса."""
        # Запросы со строгой согласованностью (транзакции/биллинг) не могут уйти на COL — там нет строгой консистентности
        return [ROW] if query.strict_consistency else [ROW, COL]

    @staticmethod
    def _cluster_features(nodes):
        """Агрегированные признаки состояния одного кластера (для вектора состояния агента)."""
        healthy = [n for n in nodes if not n.is_down]                          # список живых узлов кластера
        healthy_frac = len(healthy) / len(nodes)                               # доля живых узлов от общего числа
        min_queue = min((n.queue for n in healthy), default=50.0)              # минимальная очередь; 50.0 — признак "все мертвы"
        # Средний коэффициент деградации по здоровым узлам — делает деградацию НАБЛЮДАЕМОЙ для всех политик
        avg_health = float(np.mean([n.health_factor for n in healthy])) if healthy else 1.0
        return min_queue, healthy_frac, avg_health                             # возвращаем три агрегированных признака

    def state_vector(self, query):
        """Построить 11-мерный вектор состояния — это то, что "видит" любая политика маршрутизации."""
        qcode = {"insert": 0, "analytic": 1, "transactional": 2}[query.qtype]  # числовой код типа запроса
        row_q, row_health, row_degradation = self._cluster_features(self.row_nodes)  # признаки ROW-кластера
        col_q, col_health, col_degradation = self._cluster_features(self.col_nodes)  # признаки COL-кластера
        return np.array([
            qcode,                              # тип текущего запроса (0/1/2)
            np.log1p(query.cost),               # логарифм стоимости запроса (сглаживает большой разброс значений)
            float(query.strict_consistency),    # требует ли запрос строгой согласованности (0/1)
            row_q, row_health,                  # минимальная очередь и доля здоровых узлов кластера ROW
            col_q, col_health,                  # то же для кластера COL
            self.col_cache_warmth,              # текущая прогретость колоночного кэша
            self.col_repl_lag,                  # текущий лаг репликации ROW -> COL
            row_degradation, col_degradation,   # средний health_factor каждого кластера (видимость деградации)
        ], dtype=np.float32)                    # приводим к float32 для совместимости с PyTorch

    def step(self, query, action):
        """Выполнить один шаг симуляции: обработать один запрос выбранным действием (движком)."""
        self._t += 1                                            # увеличиваем счётчик выполненных запросов
        self._apply_scheduled_degradation()                     # обновляем health_factor/is_down всех узлов по расписанию

        nodes = self.row_nodes if action == ROW else self.col_nodes   # список узлов выбранного кластера
        healthy = [n for n in nodes if not n.is_down]            # живые узлы этого кластера
        used_action, failover_penalty = action, 0.0              # реально использованное действие и штраф за перенаправление

        # --- Случай 1: весь запрошенный кластер физически недоступен (полный отказ) ---
        if not healthy:
            if query.strict_consistency:                         # для строгих транзакций нет запасного варианта
                info = dict(node_id="timeout", network_ms=0.0, processing_ms=self.TIMEOUT_MS)
                return self.TIMEOUT_MS, -10.0, 1.0, info          # возвращаем таймаут с большим штрафом
            alt = self.col_nodes if action == ROW else self.row_nodes  # пробуем альтернативный кластер
            healthy = [n for n in alt if not n.is_down]           # живые узлы альтернативного кластера
            if not healthy:                                       # если и он полностью недоступен
                info = dict(node_id="timeout", network_ms=0.0, processing_ms=self.TIMEOUT_MS)
                return self.TIMEOUT_MS, -10.0, 1.0, info          # крайний случай — оба кластера недоступны
            used_action, failover_penalty = (COL if action == ROW else ROW), 15.0  # переключаемся, штраф за failover
            available = self._nodes_with_capacity(self.row_nodes if used_action == ROW else self.col_nodes)
        else:
            available = self._nodes_with_capacity(nodes)          # узлы исходного кластера с запасом ёмкости

        # --- Случай 2: узел жив, но его очередь заполнена (admission control / backpressure) ---
        if not available:
            alt_nodes = self.col_nodes if used_action == ROW else self.row_nodes  # альтернативный кластер
            alt_available = ([] if query.strict_consistency        # строгим запросам нельзя уходить на COL
                              else self._nodes_with_capacity(alt_nodes))
            if alt_available:                                       # если в альтернативном кластере есть место
                used_action = COL if used_action == ROW else ROW    # меняем используемое действие
                available = alt_available                          # берём доступные узлы альтернативы
                failover_penalty = max(failover_penalty, self.admission_reroute_penalty_ms)  # фиксируем штраф
            else:                                                    # альтернативы тоже нет — отклоняем запрос
                reject_latency = self.reject_timeout_ms * self.rng.lognormal(0.0, 0.15)  # случайная задержка отклонения
                info = dict(node_id="admission_rejected", network_ms=0.0,
                            processing_ms=float(reject_latency), admission_rejected=True)
                reward = -(reject_latency / max(self.INITIAL_THRESHOLD_MS, 1.0)) \
                    - self.reward_weights["sla_violation"]           # награда штрафуется пропорционально задержке отклонения
                return reject_latency, reward, 0.0, info             # возвращаем результат отклонённого запроса

        node = self._select_node(available)                          # выбираем конкретный узел (наименее загруженный)
        base = self.SERVICE_BASE[(query.qtype, used_action)] * node.health_factor  # базовое время с учётом деградации
        # Штраф за холодный кэш: применяется только к аналитике на COL, если кэш не прогрет
        cache_penalty = (1.0 - self.col_cache_warmth) * 15.0 if used_action == COL and query.qtype == "analytic" else 0.0
        lag_penalty = self.col_repl_lag * 10.0 if used_action == COL else 0.0  # штраф за лаг репликации (только для COL)
        queue_wait = node.queue * base * 0.5                          # ожидание в очереди пропорционально её длине
        # Сетевая задержка: базовая величина + небольшой случайный джиттер
        network = node.base_rtt_ms + abs(self.rng.normal(0, node.base_rtt_ms * 0.2 + 1e-6))

        noise = self.rng.lognormal(mean=0.0, sigma=0.25)               # мультипликативный шум задержки (реалистичная вариативность)
        scale = noise * (query.cost ** 0.3)                            # общий масштабирующий коэффициент от шума и стоимости
        # Разделяем итоговую задержку на компоненту обработки узлом и сетевую компоненту (для диагностики)
        processing_ms = (base + cache_penalty + lag_penalty + queue_wait + failover_penalty) * scale
        network_ms = network * scale
        latency = processing_ms + network_ms                          # суммарная наблюдаемая задержка запроса

        # --- Обновление длины очередей узлов (пропускная способность vs новая нагрузка) ---
        if self.load_scales_with_cost:                                 # если вклад запроса в очередь масштабируется стоимостью
            arrival_load = min(self.ARRIVAL_LOAD * (query.cost / self.cost_load_reference),
                                self.max_arrival_load)                 # пропорционально стоимости, но не выше предела
        else:
            arrival_load = self.ARRIVAL_LOAD                           # иначе — фиксированная величина (старое поведение)
        used_cluster = self.row_nodes if used_action == ROW else self.col_nodes  # кластер, который реально обслужил запрос
        for n in used_cluster:                                          # обновляем очередь каждого узла этого кластера
            n.queue = max(0.0, n.queue - n.service_rate) + (arrival_load if n is node else 0.0)  # обслужили + новая нагрузка
            n.queue = min(n.queue, n.max_queue)                          # очередь не может превысить предел ёмкости
        other_cluster = self.col_nodes if used_action == ROW else self.row_nodes  # второй (не задействованный) кластер
        for n in other_cluster:                                          # там нагрузка не добавляется, только обслуживание
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

        # Дисбаланс — абсолютная разница между средней очередью ROW-кластера и средней очередью COL-кластера
        imbalance = abs(np.mean([n.queue for n in self.row_nodes]) - np.mean([n.queue for n in self.col_nodes]))

        congestion_penalty = node.queue                                 # штраф — текущая (обновлённая) очередь выбранного узла

        # Определение адаптивного порога для reward: либо общий (переданный извне) профиль, либо собственная история
        if self._external_thresholds is not None:                        # если задан общий профиль порога
            idx = min(self._t - 1, len(self._external_thresholds) - 1)   # индекс текущего шага в массиве профиля
            adaptive_threshold = float(self._external_thresholds[idx])   # берём значение порога из этого профиля
        elif len(self._lat_history) >= 20:                                # достаточно накопленной собственной истории
            adaptive_threshold = max(self.LAT_FLOOR_MS, float(np.percentile(self._lat_history, 95)))  # скользящий p95
        else:
            adaptive_threshold = self.INITIAL_THRESHOLD_MS                # начальный порог до накопления истории
        self._lat_history.append(latency)                                 # добавляем текущую задержку в историю

        sla_violation = float(latency > adaptive_threshold)                # бинарный признак превышения адаптивного порога
        w = self.reward_weights                                           # локальная ссылка на веса reward (для краткости)
        # Итоговая награда: нормированная задержка + взвешенная сумма штрафов за дисбаланс, перегрузку, SLA и failover
        reward = -(latency / adaptive_threshold) - w["imbalance"] * imbalance \
            - w["congestion"] * congestion_penalty - w["sla_violation"] * sla_violation \
            - w["failover"] * (1.0 if failover_penalty else 0.0)
        info = dict(node_id=node.node_id, network_ms=float(network_ms), processing_ms=float(processing_ms),
                    adaptive_threshold_ms=float(adaptive_threshold), admission_rejected=False)  # диагностическая информация
        return latency, reward, imbalance, info                           # возвращаем всё, что нужно вызывающему коду