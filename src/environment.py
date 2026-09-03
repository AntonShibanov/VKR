"""
Ключевые свойства, отвечающие на вопросы по инфраструктуре:
 - у каждого узла своя пропускная способность (service_rate) и очередь (queue);
 - у каждого узла своя сетевая задержка (base_rtt_ms);
 - узел может деградировать (health_factor > 1) или отказать (is_down) по расписанию,
   деградация нарастает и спадает ПОСТЕПЕННО (см. ramp_queries в schedule_degradation);
 - средний health_factor кластера НАБЛЮДАЕМ агентом через state_vector() — все политики,
   способные использовать признаки состояния (бандит, GBM, RL), могут в принципе научиться
   адаптивно реагировать на деградацию, а не просто пассивно её переживать;
 - "неправильный" тип движка не даёт ошибки, а просто увеличивает задержку;
   полный отказ кластера -> прозрачный failover с штрафом, отказ обоих -> таймаут;
 - штраф в reward привязан к АДАПТИВНОМУ порогу (скользящий p95 недавней истории
   задержек)
"""
import numpy as np
from collections import deque
from dataclasses import dataclass, field

ROW, COL = 0, 1
STATE_DIM = 11  # длина вектора состояния, см. state_vector()


@dataclass
class Node:
    engine: int
    service_rate: float     # пропускная способность узла (queue-units / шаг)
    base_rtt_ms: float      # сетевая задержка до узла
    node_id: str = ""       # уникальный id узла, напр. "row_0", "col_1"
    health_factor: float = 1.0   # >1 = замедление обслуживания (деградация)
    is_down: bool = False
    queue: float = 0.0
    max_queue: float = float("inf")  # конечная ёмкость очереди (admission control,
                                        # модель M/M/1/K). По умолчанию inf — без изменений
                                        # config.yaml поведение остаётся прежним.


class HybridDBEnv:
    # базовое время обслуживания (мс) по типу запроса и типу движка
    SERVICE_BASE = {
        ("insert", ROW): 2.0, ("insert", COL): 6.0,
        ("analytic", ROW): 40.0, ("analytic", COL): 4.0,
        ("transactional", ROW): 3.0, ("transactional", COL): 25.0,
    }
    ARRIVAL_LOAD = 0.35
    COST_LOAD_REFERENCE = 5.0        # "типичная" стоимость запроса — точка нормировки вклада в очередь
    MAX_ARRIVAL_LOAD_MULT = 10.0       # ограничение сверху (в разах от ARRIVAL_LOAD) — защита от
                                          # экстремальных значений в хвосте лог-нормального cost
    TIMEOUT_MS = 5000.0
    LAT_WINDOW = 200
    LAT_FLOOR_MS = 5.0
    INITIAL_THRESHOLD_MS = 50.0

    DEFAULT_REWARD_WEIGHTS = dict(imbalance=0.3, congestion=0.0, sla_violation=0.5, failover=1.0)

    def __init__(self, seed=0, n_row_nodes=2, n_col_nodes=2,
                 row_service_rate=0.5, col_service_rate=0.35,
                 row_rtt_ms=0.3, col_rtt_ms=2.0, reward_weights=None,
                 load_scales_with_cost=False, cost_load_reference=None,
                 admission_control_enabled=False, max_queue_row=float("inf"),
                 max_queue_col=float("inf"), reject_timeout_ms=600.0,
                 admission_reroute_penalty_ms=8.0):
        self.rng = np.random.default_rng(seed)
        self.row_nodes = [Node(ROW, row_service_rate, row_rtt_ms, node_id=f"row_{i}",
                                max_queue=max_queue_row) for i in range(n_row_nodes)]
        self.col_nodes = [Node(COL, col_service_rate, col_rtt_ms, node_id=f"col_{i}",
                                max_queue=max_queue_col) for i in range(n_col_nodes)]
        # Admission control (backpressure), см. паттерны Vitess Snake / OJP / Resilience4j:
        # узел, чья очередь достигла max_queue, перестаёт принимать НОВЫЕ запросы — они
        # либо перенаправляются на другой движок (аналог proxy-level load balancing), либо,
        # если альтернативы нет, ОТКЛОНЯЮТСЯ с ограниченной задержкой reject_timeout_ms
        # (клиентский timeout + retry), а НЕ ставятся в бесконечно растущую очередь.
        self.admission_control_enabled = bool(admission_control_enabled)
        self.reject_timeout_ms = reject_timeout_ms
        self.admission_reroute_penalty_ms = admission_reroute_penalty_ms
        self.col_cache_warmth = 0.3
        self.col_repl_lag = 0.0
        self._t = 0
        self._degradation_events = []
        self._lat_history = deque(maxlen=self.LAT_WINDOW)
        self._external_thresholds = None
        self.reward_weights = {**self.DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        # НЕ меняет физику ROW/COL (SERVICE_BASE/RTT/service_rate) — только устраняет
        # внутреннюю несогласованность: query.cost уже влияет на ВЫХОДНУЮ задержку, но
        # не влиял на то, сколько РЕСУРСОВ ОЧЕРЕДИ запрос занимает у узла. Реалистичная
        # поправка: тяжёлый запрос занимает узел дольше -> должен и нагружать очередь
        # пропорционально больше, а не фиксированной константой независимо от типа запроса.
        self.load_scales_with_cost = bool(load_scales_with_cost)
        self.cost_load_reference = cost_load_reference or self.COST_LOAD_REFERENCE
        self.max_arrival_load = self.ARRIVAL_LOAD * self.MAX_ARRIVAL_LOAD_MULT

    def set_external_thresholds(self, thresholds):
        """Подключить общий (одинаковый для всех методов) профиль адаптивного порога,
        построенный нейтральной эвристикой (см. run_experiment.build_reference_thresholds).
        Устраняет самоподкрепляющуюся петлю: метод, ухудшающий собственную латентность,
        больше не поднимает СЕБЕ порог и не ослабляет тем самым собственный штраф."""
        self._external_thresholds = np.asarray(thresholds)

    def schedule_degradation(self, engine, node_idx, start_q, end_q, health_factor=1.0, is_down=False,
                              ramp_queries=0):
        """
        ramp_queries — число запросов, за которое health_factor линейно нарастает от 1.0
        до целевого значения в начале окна и линейно спадает обратно к 1.0 в конце окна.
        При ramp_queries=0 (по умолчанию) мгновенный шаг.
        Полный отказ узла (is_down=True) остаётся мгновенным — реальный крах узла (OOM,
        аппаратный сбой) происходит скачкообразно; рампа применяется только
        к health_factor, имитирующему постепенное исчерпание ресурса (память, диск, троттлинг).
        """
        ramp_queries = min(ramp_queries, max(0, (end_q - start_q) // 2))
        self._degradation_events.append(dict(engine=engine, node_idx=node_idx, start=start_q,
                                              end=end_q, health_factor=health_factor, is_down=is_down,
                                              ramp_queries=ramp_queries))

    def _apply_scheduled_degradation(self):
        for ev in self._degradation_events:
            node = (self.row_nodes if ev["engine"] == ROW else self.col_nodes)[ev["node_idx"]]
            active = ev["start"] <= self._t < ev["end"]
            node.is_down = ev["is_down"] and active

            ramp = ev.get("ramp_queries", 0)
            if ramp <= 0:
                node.health_factor = ev["health_factor"] if active else 1.0
                continue

            if self._t < ev["start"]:
                node.health_factor = 1.0
            elif self._t < ev["start"] + ramp:                       # линейный подъём
                frac = (self._t - ev["start"]) / ramp
                node.health_factor = 1.0 + frac * (ev["health_factor"] - 1.0)
            elif self._t < ev["end"] - ramp:                          # плато на целевом уровне
                node.health_factor = ev["health_factor"]
            elif self._t < ev["end"]:                                   # линейный спад
                frac = (ev["end"] - self._t) / ramp
                node.health_factor = 1.0 + frac * (ev["health_factor"] - 1.0)
            else:
                node.health_factor = 1.0

    def _select_node(self, healthy):
        """
        Выбрать наименее загруженный узел среди здоровых. Тай-брейк — СЛУЧАЙНЫЙ, а не
        "первый по списку": иначе при равенстве очередей (что происходит почти каждый шаг,
        т.к. очередь быстро возвращается к 0) min() всегда отдаёт предпочтение узлу с
        наименьшим индексом, и остальные узлы кластера фактически никогда не используются
        """
        min_q = min(n.queue for n in healthy)
        candidates = [n for n in healthy if n.queue - min_q < 1e-6]
        return candidates[self.rng.integers(len(candidates))]

    def _nodes_with_capacity(self, nodes):
        """Здоровые узлы, у которых очередь НЕ достигла предела вместимости (admission
        control). При admission_control_enabled=False совпадает со списком здоровых узлов."""
        healthy = [n for n in nodes if not n.is_down]
        if not self.admission_control_enabled:
            return healthy
        return [n for n in healthy if n.queue < n.max_queue]

    def valid_actions(self, query):
        """Запросы со строгой согласованностью не допускают отказоустойчивого сброса на COL."""
        return [ROW] if query.strict_consistency else [ROW, COL]

    @staticmethod
    def _cluster_features(nodes):
        healthy = [n for n in nodes if not n.is_down]
        healthy_frac = len(healthy) / len(nodes)
        min_queue = min((n.queue for n in healthy), default=50.0)  # "50" - признак полностью нездорового кластера
        # avg_health_factor — НОВОЕ: средний коэффициент замедления обслуживания по здоровым
        # узлам кластера (1.0 = норма, >1.0 = деградация). Делает деградацию узла НАБЛЮДАЕМОЙ
        # для всех политик через state_vector() — раньше health_factor влиял только на итоговую
        # задержку ПОСЛЕ выбора узла и был полностью скрыт от агента, из-за чего ни одна
        # стратегия не могла ни обнаружить, ни предвидеть деградацию (см. пункт 1 предыдущего
        # ответа про синхронный скачок на графике у всех методов одновременно).
        # ПРИМЕЧАНИЕ ДЛЯ ПЕРЕНОСА НА РЕАЛЬНЫЕ ДАННЫЕ: health_factor — внутренний параметр
        # симулятора. В реальной системе его наблюдаемый аналог — скользящее отношение
        # текущего среднего/p95 времени обработки узла к историческому базовому уровню
        # (штатная метрика мониторинга), а не величина, "подсмотренная" из внутреннего
        # состояния среды.
        avg_health = float(np.mean([n.health_factor for n in healthy])) if healthy else 1.0
        return min_queue, healthy_frac, avg_health

    def state_vector(self, query):
        qcode = {"insert": 0, "analytic": 1, "transactional": 2}[query.qtype]
        row_q, row_health, row_degradation = self._cluster_features(self.row_nodes)
        col_q, col_health, col_degradation = self._cluster_features(self.col_nodes)
        return np.array([
            qcode, np.log1p(query.cost), float(query.strict_consistency),
            row_q, row_health, col_q, col_health,
            self.col_cache_warmth, self.col_repl_lag,
            row_degradation, col_degradation,   # НОВОЕ: средний health_factor кластера, см. выше
        ], dtype=np.float32)

    def step(self, query, action):
        self._t += 1
        self._apply_scheduled_degradation()

        nodes = self.row_nodes if action == ROW else self.col_nodes
        healthy = [n for n in nodes if not n.is_down]
        used_action, failover_penalty = action, 0.0

        # случай 1: узел физически недоступен — отказ, а не перегрузка
        if not healthy:
            if query.strict_consistency:
                info = dict(node_id="timeout", network_ms=0.0, processing_ms=self.TIMEOUT_MS)
                return self.TIMEOUT_MS, -10.0, 1.0, info   # нет отказоустойчивого сброса для строгих транзакций
            alt = self.col_nodes if action == ROW else self.row_nodes
            healthy = [n for n in alt if not n.is_down]
            if not healthy:
                info = dict(node_id="timeout", network_ms=0.0, processing_ms=self.TIMEOUT_MS)
                return self.TIMEOUT_MS, -10.0, 1.0, info    # крайний случай: оба кластера недоступны
            used_action, failover_penalty = (COL if action == ROW else ROW), 15.0
            available = self._nodes_with_capacity(self.row_nodes if used_action == ROW else self.col_nodes)
        else:
            available = self._nodes_with_capacity(nodes)

        # случай 2: узел ЖИВ, но его очередь заполнена (admission control / backpressure,
        # см. Vitess Snake, OJP, Resilience4j) — не отказ, а нормальное load-shedding событие.
        if not available:
            alt_nodes = self.col_nodes if used_action == ROW else self.row_nodes
            alt_available = ([] if query.strict_consistency
                              else self._nodes_with_capacity(alt_nodes))
            if alt_available:
                # Перенаправление на другой движок силами proxy-слоя (backpressure), а НЕ
                # решением маршрутизирующей политики — это происходит НИЖЕ уровня, на
                # котором работают RL/бандит/эвристики, как в реальных системах (OJP).
                used_action = COL if used_action == ROW else ROW
                available = alt_available
                failover_penalty = max(failover_penalty, self.admission_reroute_penalty_ms)
            else:
                # Нет доступной альтернативы -> ОГРАНИЧЕННЫЙ отказ (timeout + retry клиента),
                # а НЕ бесконечно растущая очередь. Это ключевое отличие от старой модели:
                # раньше latency могла расти пропорционально queue (который сам рос без
                # предела); теперь задержка ограничена сверху reject_timeout_ms независимо
                # от того, насколько сильно перегружена система.
                reject_latency = self.reject_timeout_ms * self.rng.lognormal(0.0, 0.15)
                info = dict(node_id="admission_rejected", network_ms=0.0,
                            processing_ms=float(reject_latency), admission_rejected=True)
                reward = -(reject_latency / max(self.INITIAL_THRESHOLD_MS, 1.0)) \
                    - self.reward_weights["sla_violation"]
                return reject_latency, reward, 0.0, info

        node = self._select_node(available)      # балансировка на наименее загруженный УЗЕЛ С ЗАПАСОМ ЁМКОСТИ
        base = self.SERVICE_BASE[(query.qtype, used_action)] * node.health_factor
        cache_penalty = (1.0 - self.col_cache_warmth) * 15.0 if used_action == COL and query.qtype == "analytic" else 0.0
        lag_penalty = self.col_repl_lag * 10.0 if used_action == COL else 0.0
        queue_wait = node.queue * base * 0.5
        network = node.base_rtt_ms + abs(self.rng.normal(0, node.base_rtt_ms * 0.2 + 1e-6))

        noise = self.rng.lognormal(mean=0.0, sigma=0.25)
        scale = noise * (query.cost ** 0.3)
        # разделяем итоговую задержку на компоненту обработки на узле и сетевую компоненту
        processing_ms = (base + cache_penalty + lag_penalty + queue_wait + failover_penalty) * scale
        network_ms = network * scale
        latency = processing_ms + network_ms

        # --- обновление нагрузки узлов (пропускная способность vs очередь) ---
        if self.load_scales_with_cost:
            # Вклад запроса в очередь пропорционален его стоимости (нормировано на
            # "типичную" стоимость cost_load_reference), с ограничением сверху от
            # экстремальных значений в хвосте лог-нормального распределения cost.
            # Именно это создаёт содержательную многошаговую динамику: несколько
            # тяжёлых аналитических запросов подряд теперь реально "раздувают" очередь
            # COL на много шагов вперёд, а не гасятся за 1-2 шага, как раньше.
            arrival_load = min(self.ARRIVAL_LOAD * (query.cost / self.cost_load_reference),
                                self.max_arrival_load)
        else:
            arrival_load = self.ARRIVAL_LOAD
        used_cluster = self.row_nodes if used_action == ROW else self.col_nodes
        for n in used_cluster:
            n.queue = max(0.0, n.queue - n.service_rate) + (arrival_load if n is node else 0.0)
            n.queue = min(n.queue, n.max_queue)     # НОВОЕ: очередь не может превысить ёмкость (M/M/1/K)
        other_cluster = self.col_nodes if used_action == ROW else self.row_nodes
        for n in other_cluster:
            n.queue = max(0.0, n.queue - n.service_rate)

        if used_action == COL and query.qtype == "analytic":
            self.col_cache_warmth = min(1.0, self.col_cache_warmth + 0.05)
        else:
            self.col_cache_warmth = max(0.0, self.col_cache_warmth - 0.02)
        if query.qtype in ("insert", "transactional") and used_action == ROW:
            self.col_repl_lag = min(1.0, self.col_repl_lag + 0.03)
        else:
            self.col_repl_lag = max(0.0, self.col_repl_lag - 0.05)

        imbalance = abs(np.mean([n.queue for n in self.row_nodes]) - np.mean([n.queue for n in self.col_nodes]))

        # Congestion-штраф: очередь ВЫБРАННОГО узла ПОСЛЕ обновления (node.queue уже
        # содержит новое значение). Явный немедленный сигнал
        # о нагрузке, оставленной будущим запросам на этом узле — то, что иначе агенту приходится
        # выводить только через дисконтированный бутстрэп по gamma, а очередь затухает так быстро
        # (service_rate), что этот сигнал почти не доходит до Q-сети.
        congestion_penalty = node.queue

        # Приоритет — ОБЩИЙ профиль порога (self._external_thresholds), одинаковый для
        # всех методов в рамках прогона: порог реагирует на реальный дрейф нагрузки, но
        # НЕ на то, ухудшил ли латентность именно ЭТОТ метод (см. set_external_thresholds).
        # Если общий профиль не передан (например, автономные тесты/офлайн-логи) —
        # используется старый режим: скользящий p95 собственной истории этой среды.
        if self._external_thresholds is not None:
            idx = min(self._t - 1, len(self._external_thresholds) - 1)
            adaptive_threshold = float(self._external_thresholds[idx])
        elif len(self._lat_history) >= 20:
            adaptive_threshold = max(self.LAT_FLOOR_MS, float(np.percentile(self._lat_history, 95)))
        else:
            adaptive_threshold = self.INITIAL_THRESHOLD_MS
        self._lat_history.append(latency)

        sla_violation = float(latency > adaptive_threshold)
        w = self.reward_weights
        reward = -(latency / adaptive_threshold) - w["imbalance"] * imbalance \
            - w["congestion"] * congestion_penalty - w["sla_violation"] * sla_violation \
            - w["failover"] * (1.0 if failover_penalty else 0.0)
        info = dict(node_id=node.node_id, network_ms=float(network_ms), processing_ms=float(processing_ms),
                    adaptive_threshold_ms=float(adaptive_threshold), admission_rejected=False)
        return latency, reward, imbalance, info