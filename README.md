# Установка и зависимости
Для воспроизведения эксперимента установите зависимости:
```bash
pip install -r requirements.txt
```

## Инструкция по запуску эксперимента (занимает ~12-15 часов на ПК)
1. Оценка необходимого количества прогонов.

```bash
python -m experiments.power_analysis
```

2. Запуск основного эксперимента.

```bash
python -m experiments.run_experiment
```

3. Анализ полученных результатов

```bash
python -m experiments.analyze_results
```
Результаты сохраняются в папку results/ в виде CSV-файлов

4. Построение графиков

```bash
python -m experiments.make_plots
```
Графики сохраняются в папку results/figures

5. Построение сравнительных таблиц

```bash
python -m experiments.build_presentation_tables
```
Таблицы сохраняются в папку results/tables
