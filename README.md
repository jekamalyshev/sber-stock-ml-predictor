# SBER Stock ML Predictor

> Проверка принципиальной возможности получить предиктивные данные по акциям Сбербанка (SBER) с использованием методов машинного обучения.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## О проекте

Jupyter Notebook, в котором реализован полный пайплайн предсказания направления движения цены акций Сбербанка (MOEX: SBER) на основе 5-минутных тиковых данных. Данные загружены с [Finam.ru](https://www.finam.ru/profile/moex-akcii/sberbank/export/), период — январь 2021 — апрель 2022 (≈51 735 5-минутных свечей). Используется классификатор **Gradient Boosting** из `scikit-learn`.

## Структура проекта

```
.
├── SBER_probe.ipynb         # Основной ноутбук с анализом и моделью
├── SBER_210101_220424.csv   # Исходные данные (необходимо скачать отдельно)
└── README.md
```

## Зависимости

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ta pandas-ta
```

| Библиотека | Назначение |
|---|---|
| `pandas` | Работа с табличными данными |
| `numpy` | Математические операции |
| `matplotlib` / `seaborn` | Визуализация |
| `ta` | Технические индикаторы (bukosabino/ta) |
| `pandas_ta` | Расширенные технические индикаторы |
| `scikit-learn` | Gradient Boosting Classifier |

## Данные

Файл `SBER_210101_220424.csv` — тиковые данные SBER с MOEX в формате Finam (разделитель `;`), период 5 мин, 51 735 строк:

| Колонка | Описание |
|---|---|
| `<TICKER>` | Тикер (SBER) |
| `<PER>` | Период (5 мин) |
| `<DATE>` | Дата (YYYYMMDD) |
| `<TIME>` | Время (HHMMSS) |
| `<OPEN>` | Цена открытия |
| `<HIGH>` | Максимальная цена |
| `<LOW>` | Минимальная цена |
| `<CLOSE>` | Цена закрытия |
| `<VOL>` | Объём |

Скачать данные можно здесь: https://www.finam.ru/profile/moex-akcii/sberbank/export/

## Алгоритм работы ноутбука

### Шаг 1. Импорт библиотек
Загружаются все необходимые библиотеки: `pandas`, `numpy`, `sklearn`, `ta`, `pandas_ta`, `matplotlib`, `seaborn`.

### Шаг 2. Вспомогательные функции

**`series_to_supervised(data, n_in, n_out, dropnan)`** — преобразует временной ряд в формат задачи обучения с учителем (supervised learning). Создаёт лаговые признаки: для каждого момента времени `t` добавляются значения предыдущих `n_in` шагов как входные признаки `(t-n, ..., t-1)` и `n_out` будущих шагов как целевые метки `(t, t+1, ...)`.

**`ohlc_to_candlestick(row)`** — декомпозирует японскую свечу (OHLC) в 5 структурных признаков: тип свечи (бычья/медвежья), верхний фитиль, нижний фитиль, размер тела, общий размер свечи.

**`create_datetime_features(df, col)`** — создаёт временные признаки из datetime-столбца (год, месяц, день, час, сессия, день недели, квартал и пр.) + рассчитывает технические индикаторы через `pandas_ta` (RSI, RSX, Zscore, Bollinger Bands, Williams %R и др.).

**`get_session(hour)`** — кодирует час в биржевую сессию: утро (0), день (1), вечер (2), ночь (3).

### Шаг 3. Загрузка и первичный осмотр данных
Чтение CSV-файла, вывод `info()` и `head()`, удаление технических колонок `<TICKER>` и `<PER>`.

### Шаг 4. Ограничение выборки
Из всего датасета берутся первые **10 000 строк** для ускорения экспериментов (`fixed_df = frame[:10000]`).

### Шаг 5. Признаки свечного анализа
Применение `ohlc_to_candlestick` через `apply()` — добавление 5 новых колонок с характеристиками каждой японской свечи.

### Шаг 6. Технические индикаторы
Вычисление полного набора технических индикаторов из библиотеки `ta` (более 80 признаков):
- **Volume**: ADI, OBV, CMF, MFI, VWAP, NVI и др.
- **Volatility**: Bollinger Bands, Keltner Channels, ATR, Donchian Channels
- **Trend**: MACD, SMA/EMA, Ichimoku, ADX, CCI, PSAR, Aroon, Vortex, KST
- **Momentum**: RSI, Stochastic RSI, TSI, Williams %R, ROC, PPO, KAMA
- **Others**: Daily Return, Daily Log Return, Cumulative Return

### Шаг 7. Целевая переменная
Создаётся бинарный целевой признак `target`: `1` — если следующая свеча бычья (close > open), `0` — медвежья.

### Шаг 8. Преобразование в supervised learning
Применение `series_to_supervised` с параметрами `n_in=4, n_out=1` — создание матрицы признаков, где каждая строка содержит значения 4 предыдущих временных шагов (~490 признаков на 4 лага).

### Шаг 9. Обучение модели (Gradient Boosting)
Разбивка на train/test, обучение `GradientBoostingClassifier` из `sklearn`. Оценка качества через метрики accuracy и classification report.

## Быстрый старт

```bash
git clone https://github.com/jekamalyshev/sber-stock-ml-predictor.git
cd sber-stock-ml-predictor
pip install -r requirements.txt
jupyter notebook SBER_probe.ipynb
```

> **Важно:** поместите файл `SBER_210101_220424.csv` в корень проекта перед запуском.

## Лицензия

MIT License — см. [LICENSE](LICENSE).
