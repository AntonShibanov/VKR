"""
Построение всех PNG-графиков результатов эксперимента с поддержкой
фильтрации по подмножествам стратегий (plots.subsets в конфиге).
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
from src.workload import DEFAULT_PHASES

OUTPUT_DIR = "results/figures"                                    # папка, куда сохраняются все графики
DPI = 300                                                           # разрешение сохраняемых изображений

METHOD_LABELS = {                                                    # человекочитаемые названия методов для осей/легенд
    "round_robin": "Round-Robin", "hash": "Hash", "cost_threshold": "Cost-threshold",
    "gbm_static": "GBM (static)", "bandit_lints": "Bandit (LinTS)", "rl_dqn": "RL (DQN)",
}
_METHOD_ORDER = list(METHOD_LABELS)                                    # порядок ключей method (как в CSV)
STRATEGIES = [METHOD_LABELS[m] for m in _METHOD_ORDER]                  # список человекочитаемых имён в этом же порядке
COLORS = dict(zip(STRATEGIES, sns.color_palette("tab10", len(STRATEGIES))))  # фиксированный цвет на каждую стратегию

MIN_PHASE_LABEL_FRACTION = 0.02                                          # минимальная доля эпизода, чтобы подпись фазы рисовалась


def _resolve_methods(methods):
    """Проверить и нормализовать список ключей методов; None/[] означает "все методы"."""
    if not methods:
        return list(_METHOD_ORDER)
    resolved = [m for m in methods if m in METHOD_LABELS]                     # оставляем только известные методы
    unknown = [m for m in methods if m not in METHOD_LABELS]                    # находим неизвестные (опечатки в конфиге)
    if unknown:
        print(f"[make_plots] Предупреждение: неизвестные методы в plotting.method_sets: {unknown} — пропущены")
    return resolved or list(_METHOD_ORDER)                                        # если список пуст — возвращаем все методы


def _filename(base, suffix):
    """Сформировать имя файла графика с учётом суффикса подмножества."""
    return f"{base}.png" if not suffix or suffix == "all" else f"{base}__{suffix}.png"


def _build_phase_style(cfg):
    """Построить короткие коды, цвета и полные описания для всех фаз дрейфа нагрузки."""
    phases_cfg = cfg.get("workload", {}).get("phases") or DEFAULT_PHASES         # берём фазы из конфига либо дефолт
    palette = sns.color_palette("Set2", len(phases_cfg))                          # уникальный цвет на каждую фазу
    codes = {p["name"]: f"Тип {i + 1}" for i, p in enumerate(phases_cfg)}           # короткий код "Тип N" для отображения на графике
    descriptions = {p["name"]: p.get("label_ru", p["name"]) for p in phases_cfg}     # полное русское название для легенды
    colors = {p["name"]: palette[i] for i, p in enumerate(phases_cfg)}                # цвет каждой фазы
    return codes, colors, descriptions


def _annotate_phase_zones(ax, phases, phase_codes, phase_colors):
    """Закрасить фон графика по фазам и подписать их короткими кодами сверху."""
    change_points = np.where(phases[1:] != phases[:-1])[0] + 1                     # индексы, где фаза меняется
    boundaries = [0] + list(change_points) + [len(phases)]                            # границы всех зон (включая начало/конец)
    total_len = len(phases)                                                             # общая длина эпизода
    for start, end in zip(boundaries[:-1], boundaries[1:]):                                # для каждой непрерывной зоны фазы
        phase = phases[start]                                                                # имя фазы в этой зоне
        ax.axvspan(start, end, color=phase_colors.get(phase, "#EEEEEE"), alpha=0.18, zorder=0)  # закрашиваем фон
        if (end - start) / total_len >= MIN_PHASE_LABEL_FRACTION:                              # если зона достаточно длинная
            mid = (start + end) / 2                                                              # середина зоны для размещения подписи
            ax.text(mid, 1.02, phase_codes.get(phase, phase), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8)                                          # подпись коротким кодом сверху графика
    for cp in change_points:                                                                          # для каждой границы смены фазы
        ax.axvline(cp, color="grey", ls="--", lw=0.7, zorder=1)                                          # пунктирная вертикальная линия


def _add_phase_code_legend(ax, phase_codes, phase_descriptions, phase_colors):
    """Отдельная легенда, расшифровывающая короткие коды фаз, размещённая под графиком."""
    handles = [mpatches.Patch(facecolor=phase_colors[name], alpha=0.5,
                               label=f"{phase_codes[name]} — {phase_descriptions[name]}")
               for name in phase_codes]                                                                    # заплатка на каждую фазу
    ax.legend(handles=handles, title="Периоды нагрузки", loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=min(len(handles), 4), fontsize=8,
              title_fontsize=9, frameon=True)                                                                # рисуем легенду под графиком


def _with_strategy_column(df):
    """Добавить в датафрейм колонку strategy с человекочитаемым именем метода."""
    df = df.copy()
    df["strategy"] = df["method"].map(METHOD_LABELS)
    return df


def _ci95(x):
    """95%-й доверительный интервал для среднего (нормальная аппроксимация)."""
    return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))


def _bar_with_ci(summary, column, ylabel, title, filename, methods=None):
    """Общий шаблон столбчатого графика "среднее значение + 95% ДИ" по стратегиям."""
    method_keys = methods or _METHOD_ORDER                             # какие методы отображать
    strat_labels = [METHOD_LABELS[m] for m in method_keys]                # их человекочитаемые имена
    plt.figure(figsize=(10, 6))                                            # создаём новую фигуру
    x_pos = np.arange(len(strat_labels))                                     # позиции столбцов по оси X
    means, errs = [], []                                                        # накопители средних и погрешностей
    for key in method_keys:                                                       # для каждого метода
        vals = summary[summary.method == key][column].values                       # все значения метрики по прогонам
        means.append(vals.mean())                                                     # среднее значение
        errs.append(_ci95(vals))                                                        # доверительный интервал
    plt.bar(x_pos, means, yerr=errs, capsize=5, color=[COLORS[s] for s in strat_labels])  # рисуем столбцы с усами ошибок
    plt.xticks(x_pos, strat_labels, rotation=45)                                            # подписи оси X с наклоном
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()                                                                        # убираем лишние поля
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=DPI)                                    # сохраняем в файл
    plt.close()                                                                                   # закрываем фигуру (освобождаем память)


def plot_mean_latency(summary, methods=None, filename="mean_latency.png"):
    _bar_with_ci(summary, "mean_latency_ms", "Среднее время отклика (мс)",
                 "Сравнение стратегий по среднему времени отклика (95% ДИ)", filename, methods)


def plot_network_latency(summary, methods=None, filename="network_latency.png"):
    _bar_with_ci(summary, "mean_network_ms", "Средняя сетевая задержка (мс)",
                 "Сетевая задержка выбранного узла по стратегиям (95% ДИ)", filename, methods)


def plot_processing_time(summary, methods=None, filename="processing_time.png"):
    _bar_with_ci(summary, "mean_processing_ms", "Среднее время обработки на узле (мс)",
                 "Время обработки на узле без учёта сети по стратегиям (95% ДИ)", filename, methods)


def plot_sla_share(sla, methods=None, suffix="all"):
    """Доля запросов, обработанных в пределах SLA, по стратегиям."""
    methods = _resolve_methods(methods)                                     # проверяем и нормализуем список методов
    strategies = [METHOD_LABELS[m] for m in methods]
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(strategies))
    shares = [sla.loc[key, "k_under_sla"] / sla.loc[key, "n_total"] for key in methods]  # доля успешных запросов
    plt.bar(x_pos, shares, color=[COLORS[s] for s in strategies])
    plt.xticks(x_pos, strategies, rotation=45)
    plt.ylabel("Доля запросов в пределах 200 мс")
    plt.title("Соблюдение SLA по стратегиям")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, _filename("sla_share", suffix)), dpi=DPI)
    plt.close()


def plot_latency_boxplot(summary, methods=None, filename="latency_boxplot.png"):
    """Разброс среднего времени отклика по прогонам (boxplot) — сравнение стратегий."""
    method_keys = methods or _METHOD_ORDER
    strat_labels = [METHOD_LABELS[m] for m in method_keys]
    df = _with_strategy_column(summary)
    df = df[df["strategy"].isin(strat_labels)]                              # оставляем только выбранное подмножество
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="strategy", y="mean_latency_ms", order=strat_labels,
                hue="strategy", palette=COLORS, legend=False)                    # рисуем боксплот по каждой стратегии
    plt.title("Разброс среднего времени отклика по прогонам")
    plt.ylabel("Среднее время отклика за прогон (мс)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=DPI)
    plt.close()


def plot_node_usage(node_usage, methods=None, filename="node_usage.png"):
    """Частота использования каждого конкретного узла по стратегиям."""
    method_keys = methods or _METHOD_ORDER
    strat_labels = [METHOD_LABELS[m] for m in method_keys]
    df = _with_strategy_column(node_usage)
    df = df[df["strategy"].isin(strat_labels)]
    pivot = df.pivot(index="strategy", columns="node_id", values="count").fillna(0).reindex(strat_labels)  # таблица метод x узел
    shares = pivot.div(pivot.sum(axis=1), axis=0)                              # переводим счётчики в доли (по строкам)

    def _sort_key(c):
        """Ключ сортировки узлов: сначала по кластеру (row/col), затем по номеру."""
        parts = c.split("_")
        return (parts[0], int(parts[1])) if len(parts) > 1 and parts[1].isdigit() else ("zz", 0)

    node_cols = sorted(shares.columns, key=_sort_key)                            # сортированный список всех узлов
    row_cols = [c for c in node_cols if c.startswith("row_")]                       # только ROW-узлы
    col_cols = [c for c in node_cols if c.startswith("col_")]                        # только COL-узлы
    other_cols = [c for c in node_cols if c not in row_cols and c not in col_cols]     # прочие (например "timeout")

    row_palette = sns.color_palette("Blues", max(len(row_cols), 1))[::-1]                # синяя палитра для ROW
    col_palette = sns.color_palette("Oranges", max(len(col_cols), 1))[::-1]                # оранжевая палитра для COL
    other_palette = sns.color_palette("Greys", max(len(other_cols), 1))                      # серая палитра для прочих
    color_map = dict(list(zip(row_cols, row_palette)) + list(zip(col_cols, col_palette)) +
                      list(zip(other_cols, other_palette)))                                    # объединяем все палитры в один словарь

    x_pos = np.arange(len(strat_labels))                                                          # позиции групп столбцов
    width = 0.8 / max(len(node_cols), 1)                                                            # ширина одного столбца
    fig_width = max(12, 1.1 * len(node_cols) + 6)                                                     # автоматическая ширина фигуры
    plt.figure(figsize=(fig_width, 6))
    for j, node_id in enumerate(node_cols):                                                             # для каждого узла — свой столбец
        plt.bar(x_pos + j * width, shares[node_id].values, width=width,
                label=node_id, color=color_map[node_id])
    plt.xticks(x_pos + width * (len(node_cols) - 1) / 2, strat_labels, rotation=45)                        # подписи по центру групп
    plt.ylabel("Доля запросов, направленных на узел")
    plt.title("Частота использования узлов по стратегиям (ROW — синие, COL — оранжевые)")
    plt.legend(title="Узел", ncol=min(len(node_cols), 6), bbox_to_anchor=(1.02, 1), loc="upper left")          # легенда за пределами графика
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_drift_adaptation(cfg, window=500, methods=None, filename="drift_adaptation.png"):
    """Динамика задержки в течение эпизода (скользящее среднее) для указанных стратегий."""
    method_keys = methods or _METHOD_ORDER
    phases = np.load("results/trace_phases.npy", allow_pickle=True)               # загружаем трассу фаз (прогон #0)
    phase_codes, phase_colors, phase_descriptions = _build_phase_style(cfg)          # стили фаз
    fig, ax = plt.subplots(figsize=(14, 7))
    for key in method_keys:                                                            # рисуем кривую для каждой выбранной стратегии
        strat = METHOD_LABELS[key]
        lat = np.load(f"results/trace_latency_{key}.npy")                                # трасса задержки этого метода
        rolling_mean = pd.Series(lat).rolling(window, min_periods=1).mean()                 # скользящее среднее
        ax.plot(rolling_mean.index, rolling_mean, label=strat, linewidth=2, color=COLORS.get(strat), zorder=2)

    _annotate_phase_zones(ax, phases, phase_codes, phase_colors)                          # закрашиваем фон по фазам

    d = cfg["degradation"]                                                                   # секция деградации из конфига
    if d["enabled"]:                                                                            # если основное событие деградации включено
        ax.axvspan(d["start_query"], d["end_query"], color="red", alpha=0.15,
                   label=f"Деградация узла {d['engine']}#{d['node_idx']}", zorder=1)              # красная зона деградации

    ax.set_title(f"Динамика задержки при дрейфе нагрузки (окно {window} запросов)", pad=45)          # заголовок с отступом
    ax.set_xlabel("Номер запроса в эпизоде")
    ax.set_ylabel("Задержка, мс (скользящее среднее)")
    strat_legend = ax.legend(ncol=2, fontsize=9, loc="upper left")                                     # легенда стратегий
    ax.add_artist(strat_legend)                                                                            # закрепляем её (не удалять при следующем legend())
    _add_phase_code_legend(ax, phase_codes, phase_descriptions, phase_colors)                                # легенда расшифровки фаз
    ax.grid(True, zorder=0)                                                                                    # сетка на фоне
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_reward_dynamics(cfg, window=500, methods=None, filename="reward_dynamics.png"):
    """Динамика мгновенной награды в течение эпизода — аналог plot_drift_adaptation, но для reward."""
    method_keys = methods or _METHOD_ORDER
    phases = np.load("results/trace_phases.npy", allow_pickle=True)
    phase_codes, phase_colors, phase_descriptions = _build_phase_style(cfg)
    fig, ax = plt.subplots(figsize=(14, 7))
    for key in method_keys:
        strat = METHOD_LABELS[key]
        rew = np.load(f"results/trace_reward_{key}.npy")
        rolling_mean = pd.Series(rew).rolling(window, min_periods=1).mean()
        ax.plot(rolling_mean.index, rolling_mean, label=strat, linewidth=2, color=COLORS.get(strat), zorder=2)

    _annotate_phase_zones(ax, phases, phase_codes, phase_colors)

    d = cfg["degradation"]
    if d["enabled"]:
        ax.axvspan(d["start_query"], d["end_query"], color="red", alpha=0.15,
                   label=f"Деградация узла {d['engine']}#{d['node_idx']}", zorder=1)

    ax.set_title(f"Динамика мгновенной награды (окно {window} запросов)", pad=45)
    ax.set_xlabel("Номер запроса в эпизоде")
    ax.set_ylabel("Награда (скользящее среднее)")
    strat_legend = ax.legend(ncol=2, fontsize=9, loc="upper left")
    ax.add_artist(strat_legend)
    _add_phase_code_legend(ax, phase_codes, phase_descriptions, phase_colors)
    ax.grid(True, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_epsilon_curve(cfg):
    """Динамика эпсилон (интенсивности разведки) RL-агента в течение эпизода."""
    eps = np.load("results/trace_epsilon_rl_dqn.npy")                              # трасса эпсилон (только для rl_dqn)
    phases = np.load("results/trace_phases.npy", allow_pickle=True)
    phase_codes, phase_colors, phase_descriptions = _build_phase_style(cfg)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(eps, color=COLORS[METHOD_LABELS["rl_dqn"]], linewidth=1.5, label="epsilon (RL DQN)", zorder=2)

    _annotate_phase_zones(ax, phases, phase_codes, phase_colors)

    d = cfg["degradation"]
    if d["enabled"]:
        ax.axvspan(d["start_query"], d["end_query"], color="red", alpha=0.15,
                   label=f"Деградация узла {d['engine']}#{d['node_idx']}", zorder=1)

    ax.set_title("Динамика интенсивности разведки (epsilon) RL-агента", pad=45)
    ax.set_xlabel("Номер запроса в эпизоде")
    ax.set_ylabel("epsilon")
    strat_legend = ax.legend(loc="upper left")
    ax.add_artist(strat_legend)
    _add_phase_code_legend(ax, phase_codes, phase_descriptions, phase_colors)
    ax.grid(True, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "epsilon_curve.png"), dpi=DPI, bbox_inches="tight")
    plt.close()


def main():
    """Точка входа: строит все графики для всех подмножеств стратегий, заданных в конфиге."""
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))               # загружаем конфиг
    os.makedirs(OUTPUT_DIR, exist_ok=True)                                               # создаём папку для графиков
    summary = pd.read_csv("results/episode_summary.csv")                                  # читаем итоговые метрики
    sla = pd.read_csv("results/sla_counts.csv").set_index("method")                          # читаем счётчики SLA
    node_usage = pd.read_csv("results/node_usage.csv")                                          # читаем частоту использования узлов

    subsets = cfg.get("plots", {}).get("subsets") or {"all": list(_METHOD_ORDER)}                 # подмножества стратегий из конфига
    for subset_name, methods in subsets.items():                                                    # для каждого именованного подмножества
        methods = [m for m in methods if m in _METHOD_ORDER]                                            # защита от опечаток в конфиге
        if not methods:
            print(f"Подмножество '{subset_name}' пустое или содержит неизвестные методы — пропущено")
            continue
        suffix = "" if subset_name == "all" else f"_{subset_name}"                                        # суффикс имени файла

        plot_mean_latency(summary, methods=methods, filename=f"mean_latency{suffix}.png")                   # строим все графики
        plot_latency_boxplot(summary, methods=methods, filename=f"latency_boxplot{suffix}.png")               # для этого подмножества
        plot_network_latency(summary, methods=methods, filename=f"network_latency{suffix}.png")
        plot_processing_time(summary, methods=methods, filename=f"processing_time{suffix}.png")
        plot_node_usage(node_usage, methods=methods, filename=f"node_usage{suffix}.png")
        plot_drift_adaptation(cfg, methods=methods, filename=f"drift_adaptation{suffix}.png")
        plot_reward_dynamics(cfg, methods=methods, filename=f"reward_dynamics{suffix}.png")

    plot_sla_share(sla)                                                                                        # SLA-график всегда для всех методов
    plot_epsilon_curve(cfg)                                                                                       # epsilon-график только для rl_dqn
    print(f"Графики сохранены в {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()