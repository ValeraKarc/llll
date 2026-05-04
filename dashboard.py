import streamlit as st, pandas as pd, numpy as np, plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import base64, time, gc, os, warnings
warnings.filterwarnings('ignore')

import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# ---------------------------- Праздники РФ (основные нерабочие дни) ----------------------------
RUSSIAN_HOLIDAYS = {
    # фиксированные даты для 2020-2026 гг.
    '2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-01-05','2020-01-06','2020-01-07','2020-01-08',
    '2020-02-23','2020-03-08','2020-05-01','2020-05-09','2020-06-12','2020-11-04',
    '2021-01-01','2021-01-02','2021-01-03','2021-01-04','2021-01-05','2021-01-06','2021-01-07','2021-01-08',
    '2021-02-23','2021-03-08','2021-05-01','2021-05-09','2021-06-12','2021-11-04',
    '2022-01-01','2022-01-02','2022-01-03','2022-01-04','2022-01-05','2022-01-06','2022-01-07','2022-01-08',
    '2022-02-23','2022-03-08','2022-05-01','2022-05-09','2022-06-12','2022-11-04',
    '2023-01-01','2023-01-02','2023-01-03','2023-01-04','2023-01-05','2023-01-06','2023-01-07','2023-01-08',
    '2023-02-23','2023-03-08','2023-05-01','2023-05-09','2023-06-12','2023-11-04',
    '2024-01-01','2024-01-02','2024-01-03','2024-01-04','2024-01-05','2024-01-06','2024-01-07','2024-01-08',
    '2024-02-23','2024-03-08','2024-05-01','2024-05-09','2024-06-12','2024-11-04',
    '2025-01-01','2025-01-02','2025-01-03','2025-01-04','2025-01-05','2025-01-06','2025-01-07','2025-01-08',
    '2025-02-23','2025-03-08','2025-05-01','2025-05-09','2025-06-12','2025-11-04',
    '2026-01-01','2026-01-02','2026-01-03','2026-01-04','2026-01-05','2026-01-06','2026-01-07','2026-01-08',
    '2026-02-23','2026-03-08','2026-05-01','2026-05-09','2026-06-12','2026-11-04',
}

def is_holiday(dt):
    return dt.strftime('%Y-%m-%d') in RUSSIAN_HOLIDAYS

# ---------------------------- Вспомогательные функции ----------------------------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)
    mask = y_true != 0
    if np.sum(mask) == 0: return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def train_rf_model(train, test_index, lags, freq, holiday_series=None):
    """Обучение Random Forest с лагами и опциональным признаком праздников."""
    X = pd.DataFrame(index=train.index)
    for lag in range(1, lags+1):
        X[f'lag_{lag}'] = train.shift(lag)
    if holiday_series is not None:
        X['holiday'] = holiday_series
    y = train.copy()
    # Удаление NaN, возникших от лагов
    valid = ~X.isna().any(axis=1)
    X, y = X.loc[valid], y.loc[valid]
    if len(X) < 5:
        return None, None
    rf = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    # Рекурсивный прогноз на тестовый период
    test_pred = []
    hist = y.iloc[-lags:].tolist()
    for i in range(len(test_index)):
        feat = {}
        for j in range(lags):
            feat[f'lag_{j+1}'] = hist[-j-1] if len(hist) > j else np.nan
        if holiday_series is not None:
            feat['holiday'] = 1 if is_holiday(test_index[i]) else 0
        X_row = pd.DataFrame([feat])
        pred = rf.predict(X_row)[0]
        test_pred.append(pred)
        hist.append(pred)
    return np.array(test_pred), rf

# ---------------------------- Интерфейс ----------------------------
st.set_page_config(page_title="Интеллектуальная модель прогнозирования продаж", layout="wide")
st.title("📈 Интеллектуальная модель прогнозирования продаж")
st.markdown("Загрузите CSV-файл с продажами и получите прогноз с автоматическим выбором лучшей модели.")

# Боковая панель с краткой справкой
with st.sidebar:
    st.info("**Обязательные столбцы:** date, time, category, product, quantity, price, total\n\n"
            "**Формат даты:** любой (дд.мм.гггг, гггг-мм-дд и т.д.)\n\n"
            "**Кодировка:** автоопределение или выберите вручную")

uploaded = st.file_uploader("📂 Загрузите CSV-файл (до 150 МБ)", type="csv")
if uploaded:
    if uploaded.size > 150 * 1024 * 1024:
        st.error("❌ Размер файла превышает 150 МБ. Пожалуйста, загрузите файл меньшего размера.")
        st.stop()

    # Выбор кодировки
    enc_choice = st.selectbox("📝 Кодировка файла", ['auto', 'utf-8', 'cp1251', 'latin1', 'iso-8859-1'])
    if enc_choice == 'auto':
        raw = uploaded.read()
        try:
            import chardet
            enc = chardet.detect(raw)['encoding'] or 'utf-8'
        except ImportError:
            enc = 'utf-8'
        uploaded.seek(0)
    else:
        enc = enc_choice

    # Чтение CSV только с нужными столбцами и лёгкими типами
    try:
        df = pd.read_csv(
            uploaded,
            encoding=enc,
            usecols=['date', 'time', 'category', 'product', 'quantity', 'price', 'total'],
            dtype={'date': str, 'time': str, 'category': str, 'product': str,
                   'quantity': np.float32, 'price': np.float32, 'total': np.float32},
            on_bad_lines='skip'
        )
    except Exception as e:
        st.error(f"❌ Ошибка чтения файла: {e}")
        st.stop()

    # Проверка обязательных столбцов
    required = ['date', 'time', 'category', 'product', 'quantity', 'price', 'total']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"❌ Отсутствуют обязательные столбцы: {', '.join(missing)}")
        st.stop()

    # Проверка на инъекции
    for col in df.select_dtypes(object).columns:
        if df[col].astype(str).str.startswith(('=', '+', '-', '@')).any():
            st.error("⚠️ Обнаружены ячейки, начинающиеся с '=', '+', '-', '@'. Загрузка остановлена.")
            st.stop()

    # Очистка данных
    df['date'] = df['date'].astype(str).str.strip()
    df['time'] = df['time'].astype(str).str.strip()
    time_is_empty = df['time'].str.replace(r'[\s\.]', '', regex=True).eq('').all()
    if not time_is_empty:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    else:
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['datetime', 'quantity', 'price', 'total'], inplace=True)
    df = df[df['total'] > 0].drop_duplicates()
    df.sort_values('datetime', inplace=True)
    if df.empty:
        st.error("❌ После очистки не осталось данных. Проверьте файл.")
        st.stop()

    st.success(f"✅ Данные загружены: {len(df)} записей")
    with st.expander("🔍 Первые 10 строк загруженных данных"):
        st.dataframe(df.head(10))

    # Настройки прогноза
    col1, col2, col3 = st.columns(3)
    freq_map = {'час': 'h', 'день': 'D', 'неделя': 'W-MON', 'месяц': 'MS'}
    freq_label = col1.selectbox("📅 Периодичность", list(freq_map.keys()), index=3)  # по умолчанию месяц
    freq = freq_map[freq_label]
    cats = ['Все'] + sorted(df['category'].unique())
    selected_cat = col2.selectbox("🏷️ Категория", cats)
    if selected_cat != 'Все':
        prods = ['Все'] + sorted(df[df['category'] == selected_cat]['product'].unique())
    else:
        prods = ['Все']
    selected_prod = col3.selectbox("📦 Товар", prods) if prods else None
    horizon = st.slider("🔮 Горизонт прогноза (периодов)", min_value=1, max_value=52, value=5)

    # Режимы
    use_ml = st.checkbox("🌲 Расширенный режим (Random Forest + аналитика)", value=False,
                         help="Добавляет Random Forest и сравнение моделей. Может быть медленнее.")
    show_advanced = st.checkbox("📊 Расширенная аналитика (декомпозиция, остатки, сравнение)",
                                value=False, help="Дополнительные графики для оценки качества модели.")

    # Фильтрация данных
    df_filtered = df.copy()
    if selected_cat != 'Все':
        df_filtered = df_filtered[df_filtered['category'] == selected_cat]
    if selected_prod and selected_prod != 'Все':
        df_filtered = df_filtered[df_filtered['product'] == selected_prod]
    if df_filtered.empty:
        st.warning("⚠️ Для выбранной комбинации категории/товара нет данных.")
        st.stop()

    # Кнопка прогноза
    if st.button("🚀 Построить прогноз"):
        start_time = time.time()
        progress = st.progress(0)
        status = st.empty()

        try:
            # ---------- Агрегация ----------
            status.text("⏳ Агрегация данных...")
            progress.progress(10)
            ts = df_filtered.set_index('datetime')['total'].astype(np.float64).resample(freq).sum()
            del df_filtered; gc.collect()
            ts = ts.asfreq(freq).interpolate().bfill().ffill().dropna()
            if len(ts) < horizon + 5:
                st.error(f"❌ Недостаточно данных для прогноза (всего {len(ts)} точек). Уменьшите горизонт или измените периодичность.")
                st.stop()

            train, test = ts.iloc[:-horizon], ts.iloc[-horizon:]

            # Параметры сезонности
            sp = {'h': 24, 'D': 7, 'W-MON': 52, 'MS': 12}[freq]
            if sp >= len(train):
                sp = max(2, len(train) // 2)
            lags = min(6, len(train) // 2)

            # Признак праздников (только для дневной и недельной частоты)
            holiday_series = None
            if freq in ('D', 'W-MON'):
                holiday_series = pd.Series(
                    [1 if is_holiday(d) else 0 for d in train.index],
                    index=train.index, dtype=np.int8
                )

            models = {}

            # ---------- Holt-Winters ----------
            status.text("🧠 Обучение Holt-Winters...")
            progress.progress(25)
            try:
                hw_model = ExponentialSmoothing(
                    train, trend='add', seasonal='add',
                    seasonal_periods=sp, initialization_method='estimated'
                ).fit()
                pred_hw = hw_model.forecast(horizon)
                rmse_hw = np.sqrt(mean_squared_error(test, pred_hw))
                mape_hw = mape(test, pred_hw) * 100
                models['Holt-Winters'] = {
                    'rmse': rmse_hw, 'mape': mape_hw,
                    'pred_test': pred_hw, 'model': hw_model
                }
            except Exception as e:
                st.warning(f"Holt-Winters не обучилась: {e}")

            # ---------- Random Forest (только в расширенном режиме) ----------
            if use_ml:
                status.text("🌲 Обучение Random Forest...")
                progress.progress(50)
                try:
                    pred_rf, rf_model = train_rf_model(train, test.index, lags, freq, holiday_series)
                    if pred_rf is not None:
                        rmse_rf = np.sqrt(mean_squared_error(test, pred_rf))
                        mape_rf = mape(test, pred_rf) * 100
                        models['Random Forest'] = {
                            'rmse': rmse_rf, 'mape': mape_rf,
                            'pred_test': pred_rf, 'model': rf_model
                        }
                except Exception as e:
                    st.warning(f"Random Forest не обучилась: {e}")

            if not models:
                st.error("❌ Ни одна модель не обучилась.")
                st.stop()

            # Выбор лучшей модели по RMSE
            best_name = min(models, key=lambda k: models[k]['rmse'])
            best = models[best_name]

            status.text(f"🏆 Лучшая модель: {best_name} | Построение прогноза...")
            progress.progress(70)

            # ---------- Финальный прогноз на полном ряду ----------
            full_ts = pd.concat([train, test])
            if best_name == 'Holt-Winters':
                full_model = ExponentialSmoothing(
                    full_ts, trend='add', seasonal='add',
                    seasonal_periods=sp, initialization_method='estimated'
                ).fit()
                forecast = full_model.forecast(horizon)
            else:  # Random Forest
                # Переобучаем на полном ряду
                X_full = pd.DataFrame(index=full_ts.index)
                for lag in range(1, lags+1):
                    X_full[f'lag_{lag}'] = full_ts.shift(lag)
                if holiday_series is not None:
                    full_hol = pd.Series(
                        [1 if is_holiday(d) else 0 for d in full_ts.index],
                        index=full_ts.index, dtype=np.int8
                    )
                    X_full['holiday'] = full_hol
                y_full = full_ts.copy()
                valid = ~X_full.isna().any(axis=1)
                X_full, y_full = X_full.loc[valid], y_full.loc[valid]
                rf_full = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
                rf_full.fit(X_full, y_full)
                # Рекурсивный прогноз будущего
                future_holiday = None
                if freq in ('D', 'W-MON'):
                    future_holiday = [1 if is_holiday(d) else 0 for d in pd.date_range(
                        start=full_ts.index[-1] + pd.tseries.frequencies.to_offset(freq),
                        periods=horizon, freq=freq)]
                hist = y_full.iloc[-lags:].tolist()
                forecast = []
                for i in range(horizon):
                    feat = {}
                    for j in range(lags):
                        feat[f'lag_{j+1}'] = hist[-j-1] if len(hist) > j else np.nan
                    if future_holiday is not None:
                        feat['holiday'] = future_holiday[i]
                    X_row = pd.DataFrame([feat])
                    pred = rf_full.predict(X_row)[0]
                    forecast.append(pred)
                    hist.append(pred)
                forecast = np.array(forecast)

            # Будущие даты (безопасный способ)
            from pandas.tseries.frequencies import to_offset
            start_future = full_ts.index[-1] + to_offset(freq)
            future = pd.date_range(start=start_future, periods=horizon, freq=freq)

            # 90% доверительный интервал
            std_res = np.std(np.array(test) - np.array(best['pred_test']))
            lower = forecast - 1.645 * std_res
            upper = forecast + 1.645 * std_res

            progress.progress(90)
            status.empty()
            progress.empty()

            # ---------- Результаты ----------
            st.subheader(f"🏆 Результаты прогнозирования (модель: **{best_name}**)")
            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{best['rmse']:,.2f}")
            col2.metric("MAPE", f"{best['mape']:.2f}%")

            # Основной график
            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(x=train.index, y=train.values,
                                          name='Обучающие', line=dict(color='#1f77b4')))
            fig_main.add_trace(go.Scatter(x=test.index, y=test.values,
                                          name='Тестовые', line=dict(color='#ff7f0e')))
            fig_main.add_trace(go.Scatter(x=future, y=forecast,
                                          name='Прогноз', line=dict(color='#2ca02c')))
            fig_main.add_trace(go.Scatter(
                x=np.concatenate([future, future[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself', fillcolor='rgba(44,160,44,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='90% дов. интервал'))
            # Вертикальная линия (используем add_shape для надёжности)
            split_date = test.index[0]
            fig_main.add_shape(type='line', x0=split_date, x1=split_date,
                               y0=0, y1=1, yref='paper',
                               line=dict(color='red', dash='dash'))
            fig_main.add_annotation(x=split_date, y=1, yref='paper',
                                    text='Начало прогноза', showarrow=False,
                                    xanchor='left', textangle=-90)
            fig_main.update_layout(
                title=f'Прогноз продаж ({best_name})',
                xaxis_title='Дата', yaxis_title='Сумма продаж',
                hovermode='x unified', template='plotly_white'
            )
            st.plotly_chart(fig_main, use_container_width=True,
                            config={'scrollZoom': True, 'displayModeBar': True})

            # Таблица прогноза (даты в формате ДД-ММ-ГГГГ)
            st.subheader("📋 Прогнозные значения")
            st.dataframe(pd.DataFrame({
                'Дата': future.strftime('%d-%m-%Y'),
                'Прогноз': forecast.round(2),
                'Нижняя граница (90%)': lower.round(2),
                'Верхняя граница (90%)': upper.round(2)
            }), use_container_width=True)

            # ---------- Дополнительная аналитика ----------
            if show_advanced:
                st.subheader("📊 Расширенная аналитика")
                # Сравнение моделей, если их несколько
                if len(models) > 1:
                    comp_df = pd.DataFrame([
                        {'Модель': name, 'RMSE': d['rmse'], 'MAPE': d['mape']}
                        for name, d in models.items()
                    ]).sort_values('RMSE')
                    fig_comp = make_subplots(rows=1, cols=2, subplot_titles=("RMSE", "MAPE"))
                    fig_comp.add_trace(go.Bar(x=comp_df['Модель'], y=comp_df['RMSE'], name='RMSE'), 1, 1)
                    fig_comp.add_trace(go.Bar(x=comp_df['Модель'], y=comp_df['MAPE'], name='MAPE'), 1, 2)
                    fig_comp.update_layout(showlegend=False)
                    st.plotly_chart(fig_comp, use_container_width=True)

                # Сезонная декомпозиция (если данных достаточно)
                if len(train) >= 2 * sp + 10:
                    try:
                        decomp = seasonal_decompose(train, model='additive', period=sp)
                        fig_dec = make_subplots(
                            rows=4, cols=1,
                            subplot_titles=("Наблюдения", "Тренд", "Сезонность", "Остатки")
                        )
                        fig_dec.add_trace(go.Scatter(x=train.index, y=decomp.observed), row=1, col=1)
                        fig_dec.add_trace(go.Scatter(x=train.index, y=decomp.trend), row=2, col=1)
                        fig_dec.add_trace(go.Scatter(x=train.index, y=decomp.seasonal), row=3, col=1)
                        fig_dec.add_trace(go.Scatter(x=train.index, y=decomp.resid), row=4, col=1)
                        fig_dec.update_layout(height=800, showlegend=False,
                                              title="Сезонная декомпозиция обучающих данных")
                        st.plotly_chart(fig_dec, use_container_width=True)
                    except Exception:
                        st.info("Не удалось построить сезонную декомпозицию.")

                # Остатки на тестовом периоде
                resid_test = test.values - best['pred_test']
                fig_resid = go.Figure()
                fig_resid.add_trace(go.Scatter(x=test.index, y=resid_test, name='Остатки'))
                fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                fig_resid.update_layout(title="Остатки модели на тестовом периоде",
                                        xaxis_title="Дата", yaxis_title="Ошибка")
                st.plotly_chart(fig_resid, use_container_width=True)

            elapsed = time.time() - start_time
            st.caption(f"⏱️ Прогноз построен за {elapsed:.1f} секунд.")

        except Exception as e:
            st.error(f"❌ Во время выполнения прогноза произошла ошибка: {e}")
        finally:
            del train, test, ts
            gc.collect()
