"""
Визуализация результатов эксперимента: PNG-графики в results/figures/.
Стилистика графиков приведена в соответствие с принятой в проекте нотацией:
фиксированный словарь COLORS на стратегию, sns.boxplot для разброса метрик,
прямой plt.bar с x_pos для сравнений с ДИ, единый rotation=45 для подписей осей.
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
from src.workload import DEFAULT_PHASES

OUTPUT_DIR = "results/figures"
DPI = 300  # выше, чем в референсном примере (150), т.к. ранее было явное требование на повышенную чёткость

METHOD_LABELS = {
    "round_robin": "Round-Robin", "hash": "Hash", "cost_threshold": "Cost-threshold",
    "gbm_static": "GBM (static)", "bandit_lints": "Bandit (LinTS)", "rl_dqn": "RL (DQN)",
}
_METHOD_ORDER = list(METHOD_LABELS)                 # ключи method, как в CSV
STRATEGIES = [METHOD_LABELS[m] for m in _METHOD_ORDER] # человекочитаемые имена для осей/легенд
COLORS = dict(zip(STRATEGIES, sns.color_palette("tab10", len(STRATEGIES))))  # фиксированный цвет на стратегию

# Минимальная доля длины эпизода, которую должна занимать зона фазы, чтобы её код-подпись
# рисовалась на графике — доп. защита от наложения даже коротких кодов при очень частых фазах.
MIN_PHASE_LABEL_FRACTION = 0.02


def _resolve_methods(methods):
    """methods — список ключей method (как в CSV/METHOD_LABELS), либо None/[] (= все стратегии).
    Неизвестные ключи отбрасываются с предупреждением — типичная причина: опечатка в
    configs/config.yaml -> plotting.method_sets."""
    if not methods:
        return list(_METHOD_ORDER)
    resolved = [m for m in methods if m in METHOD_LABELS]
    unknown = [m for m in methods if m not in METHOD_LABELS]
    if unknown:
        print(f"[make_plots] Предупреждение: неизвестные методы в plotting.method_sets: {unknown} — пропущены")
    return resolved or list(_METHOD_ORDER)


def _filename(base, suffix):
    """base.png для набора 'all' (или если суффикс не задан); иначе base__suffix.png —
    чтобы разные отфильтрованные варианты графика не перезатирали друг друга на диске."""
    return f"{base}.png" if not suffix or suffix == "all" else f"{base}__{suffix}.png"


def _build_phase_style(cfg):
    """
    Строит стили периодов дрейфа из cfg['workload']['phases'] (или DEFAULT_PHASES).
    Возвращает три словаря {имя_фазы: ...}:
      - codes       — короткий код для отображения НА графике (Тип 1, Тип 2, ...);
      - colors      — цвет зоны;
      - descriptions — полное русское название для ОТДЕЛЬНОЙ легенды под графиком.

    Раньше на графике рисовались полные названия ("Насыщение аналитикой" и т.п.) прямо
    внутри области построения — при большом числе фаз или их малой продолжительности
    эти подписи накладывались друг на друга. Теперь на графике — только короткий код,
    а расшифровка кодов выносится в отдельную легенду (см. _add_phase_code_legend).
    """
    phases_cfg = cfg.get("workload", {}).get("phases") or DEFAULT_PHASES
    palette = sns.color_palette("Set2", len(phases_cfg))
    codes = {p["name"]: f"Тип {i + 1}" for i, p in enumerate(phases_cfg)}
    descriptions = {p["name"]: p.get("label_ru", p["name"]) for p in phases_cfg}
    colors = {p["name"]: palette[i] for i, p in enumerate(phases_cfg)}
    return codes, colors, descriptions


def _annotate_phase_zones(ax, phases, phase_codes, phase_colors):
    """Закрасить фон графика по стадиям нагрузки и подписать их КОРОТКИМИ кодами (Тип 1,
    Тип 2, ...) сверху; пунктирные вертикальные линии отмечают точные границы смены
    стадии. Расшифровка кодов даётся отдельной легендой под графиком, см.
    _add_phase_code_legend — это устраняет наложение длинных названий друг на друга.

    Подпись фазы рисуется только если её зона занимает не менее MIN_PHASE_LABEL_FRACTION
    от длины эпизода — иначе при большом числе коротких фаз подписи накладываются друг на
    друга по горизонтали."""
    change_points = np.where(phases[1:] != phases[:-1])[0] + 1
    boundaries = [0] + list(change_points) + [len(phases)]
    total_len = len(phases)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        phase = phases[start]
        ax.axvspan(start, end, color=phase_colors.get(phase, "#EEEEEE"), alpha=0.18, zorder=0)
        if (end - start) / total_len >= MIN_PHASE_LABEL_FRACTION:
            mid = (start + end) / 2
            ax.text(mid, 1.02, phase_codes.get(phase, phase), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8)
    for cp in change_points:
        ax.axvline(cp, color="grey", ls="--", lw=0.7, zorder=1)


def _add_phase_code_legend(ax, phase_codes, phase_descriptions, phase_colors):
    """
    Отдельная легенда «код -> полное название периода дрейфа», размещённая ПОД графиком
    (за пределами области построения), чтобы не перекрывать ни легенду стратегий, ни
    заголовок. Вызывается ПОСЛЕ того, как легенда стратегий уже создана и закреплена на
    графике через ax.add_artist(...) — иначе повторный вызов ax.legend() удалит первую
    легенду с осей.
    """
    handles = [mpatches.Patch(facecolor=phase_colors[name], alpha=0.5,
                               label=f"{phase_codes[name]} — {phase_descriptions[name]}")
               for name in phase_codes]
    ax.legend(handles=handles, title="Периоды нагрузки", loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=min(len(handles), 4), fontsize=8,
              title_fontsize=9, frameon=True)


def _with_strategy_column(df):
    """Добавить колонку strategy с человекочитаемым именем метода (для группировки и цвета)."""
    df = df.copy()
    df["strategy"] = df["method"].map(METHOD_LABELS)
    return df


def _ci95(x):
    return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))


def _bar_with_ci(summary, column, ylabel, title, filename, methods=None):
    """Общий шаблон для бар-графиков "среднее + 95% ДИ" (задержка, сеть, обработка на узле).
    methods — список внутренних ключей методов (из METHOD_LABELS) для отображения; None = все.
    Позволяет строить график только по подмножеству стратегий — удобно, когда какие-то
    из них сильно выбиваются по масштабу и "сжимают" остальные кривые/столбцы в незаметный диапазон."""
    method_keys = methods or _METHOD_ORDER
    strat_labels = [METHOD_LABELS[m] for m in method_keys]

    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(strat_labels))
    means, errs = [], []
    for key in method_keys:
        vals = summary[summary.method == key][column].values
        means.append(vals.mean())
        errs.append(_ci95(vals))
    plt.bar(x_pos, means, yerr=errs, capsize=5, color=[COLORS[s] for s in strat_labels])
    plt.xticks(x_pos, strat_labels, rotation=45)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=DPI)
    plt.close()


def plot_mean_latency(summary, methods=None, filename="mean_latency.png"):
    _bar_with_ci(summary, "mean_latency_ms", "Среднее время отклика (мс)",
                 "Сравнение стратегий по среднему времени отклика (95% ДИ)", filename, methods)


def plot_network_latency(summary, methods=None, filename="network_latency.png"):
    _bar_with_ci(summary, "mean_network_ms", "Средняя сетевая задержка (мс)",
                 "Сетевая задержка выбранного узла по стратегиям (95% ДИ)", filename, methods)


def plot_processing_time(summary, methods=None, filename="processing_time.png"):
    _bar_with_ci(summary, "mean_processing_ms", "Среднее время обработки на узле (мс)",
                 "Время обработки на узле без учёта сети по стратегиям (95% ДИ)", filename, methods)


def plot_sla_share(sla, methods=None, suffix="all"):  # не используется в main(), но поддерживает фильтр
    """Доля запросов, обработанных в пределах SLA (200 мс), по стратегиям."""
    methods = _resolve_methods(methods)
    strategies = [METHOD_LABELS[m] for m in methods]
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(strategies))
    shares = [sla.loc[key, "k_under_sla"] / sla.loc[key, "n_total"] for key in methods]
    plt.bar(x_pos, shares, color=[COLORS[s] for s in strategies])
    plt.xticks(x_pos, strategies, rotation=45)
    plt.ylabel("Доля запросов в пределах 200 мс")
    plt.title("Соблюдение SLA по стратегиям")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, _filename("sla_share", suffix)), dpi=DPI)
    plt.close()


def plot_latency_boxplot(summary, methods=None, filename="latency_boxplot.png"):
    """Разброс среднего времени отклика по прогонам, сравнение стратегий (или подмножества)."""
    method_keys = methods or _METHOD_ORDER
    strat_labels = [METHOD_LABELS[m] for m in method_keys]
    df = _with_strategy_column(summary)
    df = df[df["strategy"].isin(strat_labels)]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="strategy", y="mean_latency_ms", order=strat_labels,
                hue="strategy", palette=COLORS, legend=False)
    plt.title("Разброс среднего времени отклика по прогонам")
    plt.ylabel("Среднее время отклика за прогон (мс)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=DPI)
    plt.close()


def plot_node_usage(node_usage, methods=None, filename="node_usage.png"):
    """Частота использования узлов (row_0..row_N, col_0..col_M) по стратегиям (или подмножеству).
    Кластеры различаются цветовой гаммой (ROW — синие, COL — оранжевые оттенки),
    ширина фигуры и вынос легенды за пределы графика подстраиваются под число узлов,
    заданное в configs/config.yaml (nodes.n_row_nodes/n_col_nodes)."""
    method_keys = methods or _METHOD_ORDER
    strat_labels = [METHOD_LABELS[m] for m in method_keys]
    df = _with_strategy_column(node_usage)
    df = df[df["strategy"].isin(strat_labels)]
    pivot = df.pivot(index="strategy", columns="node_id", values="count").fillna(0).reindex(strat_labels)
    shares = pivot.div(pivot.sum(axis=1), axis=0)

    def _sort_key(c):  # "row_0"/"col_1" -> сортировка по кластеру и номеру; "timeout" уходит в конец
        parts = c.split("_")
        return (parts[0], int(parts[1])) if len(parts) > 1 and parts[1].isdigit() else ("zz", 0)

    node_cols = sorted(shares.columns, key=_sort_key)
    row_cols = [c for c in node_cols if c.startswith("row_")]
    col_cols = [c for c in node_cols if c.startswith("col_")]
    other_cols = [c for c in node_cols if c not in row_cols and c not in col_cols]

    row_palette = sns.color_palette("Blues", max(len(row_cols), 1))[::-1]
    col_palette = sns.color_palette("Oranges", max(len(col_cols), 1))[::-1]
    other_palette = sns.color_palette("Greys", max(len(other_cols), 1))
    color_map = dict(list(zip(row_cols, row_palette)) + list(zip(col_cols, col_palette)) +
                      list(zip(other_cols, other_palette)))

    x_pos = np.arange(len(strat_labels))
    width = 0.8 / max(len(node_cols), 1)
    fig_width = max(12, 1.1 * len(node_cols) + 6)      # авто-расширение под число узлов
    plt.figure(figsize=(fig_width, 6))
    for j, node_id in enumerate(node_cols):
        plt.bar(x_pos + j * width, shares[node_id].values, width=width,
                label=node_id, color=color_map[node_id])
    plt.xticks(x_pos + width * (len(node_cols) - 1) / 2, strat_labels, rotation=45)
    plt.ylabel("Доля запросов, направленных на узел")
    plt.title("Частота использования узлов по стратегиям (ROW — синие, COL — оранжевые)")
    plt.legend(title="Узел", ncol=min(len(node_cols), 6), bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_drift_adaptation(cfg, window=500, methods=None, filename="drift_adaptation.png"):
    """Динамика задержки в течение эпизода с дрейфом IoT-нагрузки и деградацией узла (прогон #0,
    указанное подмножество стратегий). Фон графика закрашен по стадиям нагрузки, зоны подписаны
    короткими кодами (Тип 1, Тип 2, ...), расшифровка — отдельной легендой под графиком."""
    method_keys = methods or _METHOD_ORDER
    phases = np.load("results/trace_phases.npy", allow_pickle=True)
    phase_codes, phase_colors, phase_descriptions = _build_phase_style(cfg)
    fig, ax = plt.subplots(figsize=(14, 7))
    for key in method_keys:
        strat = METHOD_LABELS[key]
        lat = np.load(f"results/trace_latency_{key}.npy")
        rolling_mean = pd.Series(lat).rolling(window, min_periods=1).mean()
        ax.plot(rolling_mean.index, rolling_mean, label=strat, linewidth=2, color=COLORS.get(strat), zorder=2)

    _annotate_phase_zones(ax, phases, phase_codes, phase_colors)

    d = cfg["degradation"]
    if d["enabled"]:
        ax.axvspan(d["start_query"], d["end_query"], color="red", alpha=0.15,
                   label=f"Деградация узла {d['engine']}#{d['node_idx']}", zorder=1)

    ax.set_title(f"Динамика задержки при дрейфе нагрузки (окно {window} запросов)", pad=45)
    ax.set_xlabel("Номер запроса в эпизоде")
    ax.set_ylabel("Задержка, мс (скользящее среднее)")
    strat_legend = ax.legend(ncol=2, fontsize=9, loc="upper left")
    ax.add_artist(strat_legend)   # закрепляем легенду стратегий, чтобы её не убрал следующий вызов legend()
    _add_phase_code_legend(ax, phase_codes, phase_descriptions, phase_colors)
    ax.grid(True, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_reward_dynamics(cfg, window=500, methods=None, filename="reward_dynamics.png"):
    """Динамика мгновенной награды указанного подмножества методов в течение эпизода (прогон #0)."""
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
    """Динамика epsilon (интенсивности разведки) RL-агента в течение эпизода (прогон #0):
    затухание eps_start -> eps_base за eps_decay_steps шагов и всплески при обнаружении
    дрейфа нагрузки (eps_drift). Определена только для rl_dqn (остальные методы либо не
    используют epsilon-разведку, либо используют Thompson sampling без явного epsilon)."""
    eps = np.load("results/trace_epsilon_rl_dqn.npy")
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
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = pd.read_csv("results/episode_summary.csv")
    sla = pd.read_csv("results/sla_counts.csv").set_index("method")
    node_usage = pd.read_csv("results/node_usage.csv")

    # Именованные подмножества стратегий из configs/config.yaml -> plots.subsets.
    # Позволяет строить один и тот же набор графиков для разных комбинаций стратегий —
    # удобно, когда какие-то методы сильно выбиваются по масштабу и "сжимают" остальные
    # кривые/столбцы. Подмножество "all" (или отсутствие секции plots в конфиге) сохраняет
    # старые имена файлов без суффикса; остальные подмножества получают суффикс _<имя>.
    subsets = cfg.get("plots", {}).get("subsets") or {"all": list(_METHOD_ORDER)}
    for subset_name, methods in subsets.items():
        methods = [m for m in methods if m in _METHOD_ORDER]   # защита от опечаток в конфиге
        if not methods:
            print(f"Подмножество '{subset_name}' пустое или содержит неизвестные методы — пропущено")
            continue
        suffix = "" if subset_name == "all" else f"_{subset_name}"

        plot_mean_latency(summary, methods=methods, filename=f"mean_latency{suffix}.png")
        plot_latency_boxplot(summary, methods=methods, filename=f"latency_boxplot{suffix}.png")
        plot_network_latency(summary, methods=methods, filename=f"network_latency{suffix}.png")
        plot_processing_time(summary, methods=methods, filename=f"processing_time{suffix}.png")
        plot_node_usage(node_usage, methods=methods, filename=f"node_usage{suffix}.png")
        plot_drift_adaptation(cfg, methods=methods, filename=f"drift_adaptation{suffix}.png")
        plot_reward_dynamics(cfg, methods=methods, filename=f"reward_dynamics{suffix}.png")

    plot_sla_share(sla)     # не используется в проверке гипотезы; подмножества не применяются
    plot_epsilon_curve(cfg)  # определён только для rl_dqn, подмножества методов не применимы
    print(f"Графики сохранены в {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()