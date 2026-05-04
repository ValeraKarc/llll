import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import base64
import os, warnings, time
warnings.filterwarnings('ignore')

import matplotlib
if 'DISPLAY' not in os.environ and 'MPLBACKEND' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError: HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError: HAS_LGB = False
try:
    import pmdarima as pm
    HAS_ARIMA = True
except ImportError: HAS_ARIMA = False

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fpdf import FPDF

# ---------------------------- Оптимизации памяти ----------------------------
pd.options.mode.copy_on_write = True   # предотвращает лишнее копирование DataFrame

# ---------------------------- Праздники РФ ---------------------------------
RUSSIAN_HOLIDAYS = {
    '2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-01-05',
    '2020-01-06','2020-01-07','2020-01-08','2020-01-09','2020-01-10',
    '2020-01-11','2020-01-12','2020-02-23','2020-02-24','2020-03-08',
    '2020-05-01','2020-05-09','2020-06-12','2020-11-04',
    '2021-01-01','2021-01-02','2021-01-03','2021-01-04','2021-01-05',
    '2021-01-06','2021-01-07','2021-01-08','2021-01-09','2021-01-10',
    '2021-01-11','2021-01-12','2021-02-23','2021-03-08',
    '2021-05-01','2021-05-09','2021-06-12','2021-11-04',
    '2022-01-01','2022-01-02','2022-01-03','2022-01-04','2022-01-05',
    '2022-01-06','2022-01-07','2022-01-08','2022-01-09','2022-01-10',
    '2022-01-11','2022-01-12','2022-02-23','2022-03-08',
    '2022-05-01','2022-05-09','2022-06-12','2022-11-04',
    '2023-01-01','2023-01-02','2023-01-03','2023-01-04','2023-01-05',
    '2023-01-06','2023-01-07','2023-01-08','2023-01-09','2023-01-10',
    '2023-01-11','2023-01-12','2023-02-23','2023-03-08',
    '2023-05-01','2023-05-09','2023-06-12','2023-11-04',
    '2024-01-01','2024-01-02','2024-01-03','2024-01-04','2024-01-05',
    '2024-01-06','2024-01-07','2024-01-08','2024-01-09','2024-01-10',
    '2024-01-11','2024-01-12','2024-02-23','2024-03-08',
    '2024-05-01','2024-05-09','2024-06-12','2024-11-04',
    '2025-01-01','2025-01-02','2025-01-03','2025-01-04','2025-01-05',
    '2025-01-06','2025-01-07','2025-01-08','2025-01-09','2025-01-10',
    '2025-01-11','2025-01-12','2025-02-23','2025-03-08',
    '2025-05-01','2025-05-09','2025-06-12','2025-11-04',
    '2026-01-01','2026-01-02','2026-01-03','2026-01-04','2026-01-05',
    '2026-01-06','2026-01-07','2026-01-08','2026-01-09','2026-01-10',
    '2026-01-11','2026-01-12','2026-02-23','2026-03-08',
    '2026-05-01','2026-05-09','2026-06-12','2026-11-04',
}

def is_holiday(dt):
    return dt.strftime('%Y-%m-%d') in RUSSIAN_HOLIDAYS

# ---------------------------- Функции метрик ----------------------------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

# ---------------------------- Создание признаков (облегчённое) ----------------------------
def create_lag_features(series, lags, freq_str, holiday_series=None):
    df_feat = pd.DataFrame(index=series.index)
    # Лаги
    for lag in range(1, lags+1):
        df_feat[f'lag_{lag}'] = series.shift(lag)
    # Только два скользящих окна
    for w in [3, 7]:
        if w < len(series):
            df_feat[f'roll_mean_{w}'] = series.rolling(w).mean()
            df_feat[f'roll_std_{w}'] = series.rolling(w).std()
    # Временные метки
    dt_index = series.index
    if freq_str == 'h':
        df_feat['hour'] = dt_index.hour
        df_feat['dayofweek'] = dt_index.dayofweek
        df_feat['month'] = dt_index.month
    elif freq_str == 'D':
        df_feat['dayofweek'] = dt_index.dayofweek
        df_feat['month'] = dt_index.month
        df_feat['quarter'] = dt_index.quarter
        df_feat['year'] = dt_index.year
    elif freq_str == 'W-MON':
        df_feat['weekofyear'] = dt_index.isocalendar().week.astype(int)
        df_feat['month'] = dt_index.month
        df_feat['quarter'] = dt_index.quarter
        df_feat['year'] = dt_index.year
    elif freq_str == 'MS':
        df_feat['month'] = dt_index.month
        df_feat['quarter'] = dt_index.quarter
        df_feat['year'] = dt_index.year
    if holiday_series is not None:
        df_feat['holiday'] = holiday_series.loc[df_feat.index].fillna(0).astype(np.int8)
    df_feat.dropna(inplace=True)
    return df_feat, series[df_feat.index]

def recursive_forecast(model, history_series, forecast_dates, lags, freq_str):
    hist = history_series.copy()
    preds = []
    for dt in forecast_dates:
        last_vals = hist.iloc[-lags:]
        feat = {}
        for lag in range(1, lags+1):
            feat[f'lag_{lag}'] = last_vals.iloc[-lag] if len(last_vals) >= lag else np.nan
        for w in [3, 7]:
            if len(hist) >= w:
                feat[f'roll_mean_{w}'] = hist.iloc[-w:].mean()
                feat[f'roll_std_{w}'] = hist.iloc[-w:].std()
            else:
                feat[f'roll_mean_{w}'] = np.mean(hist)
                feat[f'roll_std_{w}'] = np.std(hist)
        if freq_str == 'h':
            feat['hour'] = dt.hour; feat['dayofweek'] = dt.dayofweek; feat['month'] = dt.month
        elif freq_str == 'D':
            feat['dayofweek'] = dt.dayofweek; feat['month'] = dt.month
            feat['quarter'] = dt.quarter; feat['year'] = dt.year
        elif freq_str == 'W-MON':
            feat['weekofyear'] = dt.isocalendar().week; feat['month'] = dt.month
            feat['quarter'] = dt.quarter; feat['year'] = dt.year
        elif freq_str == 'MS':
            feat['month'] = dt.month; feat['quarter'] = dt.quarter; feat['year'] = dt.year
        # Праздник для будущей даты
        feat['holiday'] = 1 if is_holiday(dt) else 0
        X = pd.DataFrame([feat])
        pred = model.predict(X)[0]
        preds.append(pred)
        # Эффективное добавление новой точки без перестроения всего объекта
        new_point = pd.Series({dt: pred})
        hist = pd.concat([hist, new_point])
    return np.array(preds)

def train_ml_model(model, train_series, test_index, lags, freq_str, holiday_series=None):
    X_train, y_train = create_lag_features(train_series, lags, freq_str, holiday_series)
    if len(X_train) == 0:
        return None, None
    model.fit(X_train, y_train)
    preds = recursive_forecast(model, train_series, test_index, lags, freq_str)
    return preds

# ---------------------------- Интерфейс Streamlit ----------------------------
st.set_page_config(layout="wide")
st.title("📈 Прогнозирование продаж (конфиденциальная версия)")

# Загрузка файла
uploaded = st.file_uploader("Загрузите CSV-файл (до 150 МБ)", type="csv")
if uploaded is not None:
    if uploaded.size > 150 * 1024 * 1024:
        st.error("❌ Размер файла превышает 150 МБ. Загрузите файл меньшего размера.")
        st.stop()

    # Выбор кодировки
    enc_choice = st.selectbox("Кодировка файла", ['auto','utf-8','cp1251','latin1','iso-8859-1','cp1252'])
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

    # Быстрая загрузка CSV только с нужными столбцами и типами
    try:
        dtype_dict = {
            'date': str, 'time': str, 'category': str, 'product': str,
            'quantity': np.float64, 'price': np.float64, 'total': np.float64
        }
        df = pd.read_csv(
            uploaded,
            encoding=enc,
            usecols=['date','time','category','product','quantity','price','total'],
            dtype=dtype_dict,
            on_bad_lines='skip'
        )
    except Exception as e:
        st.error(f"❌ Ошибка чтения файла: {e}")
        st.stop()

    # Проверка наличия всех колонок
    required = ['date','time','category','product','quantity','price','total']
    if not all(col in df.columns for col in required):
        st.error(f"❌ Отсутствуют обязательные столбцы: {', '.join(set(required)-set(df.columns))}")
        st.stop()

    # Проверка на инъекции (начало ячеек с '=', '+', '-', '@')
    def has_injection(val):
        s = str(val).strip()
        return s.startswith(('=', '+', '-', '@'))
    injection_found = False
    for col in df.columns:
        if df[col].dtype == object and df[col].apply(has_injection).any():
            injection_found = True
            break
    if injection_found:
        st.error("⚠️ Обнаружены ячейки, начинающиеся с '=', '+', '-', '@'. Загрузка остановлена.")
        st.stop()

    # Очистка данных
    df['date'] = df['date'].astype(str).str.strip()
    df['time'] = df['time'].astype(str).str.strip()
    time_empty = df['time'].str.replace(r'[\s\.]','',regex=True).eq('').all()
    if not time_empty:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    else:
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    # Числовые колонки уже в нужном типе, но на всякий случай
    for c in ['quantity','price','total']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['quantity','price','total'], inplace=True)
    df.drop_duplicates(inplace=True)
    df = df[df['total'] > 0]
    df.sort_values('datetime', inplace=True)

    if df.empty:
        st.error("❌ После очистки не осталось данных. Проверьте файл.")
        st.stop()

    st.success(f"✅ Данные загружены. {len(df)} записей после очистки.")
    st.subheader("Первые 10 строк")
    st.dataframe(df.head(10))

    # Выбор периодичности
    freq_map = {'час': 'h', 'день': 'D', 'неделя': 'W-MON', 'месяц': 'MS'}
    freq_label = st.selectbox("Периодичность агрегации", list(freq_map.keys()))
    freq = freq_map[freq_label]

    # Выбор категории и товара
    categories = ['Все'] + sorted(df['category'].unique().tolist())
    selected_category = st.selectbox("Категория товаров", categories)
    if selected_category != 'Все':
        products = ['Все'] + sorted(df[df['category'] == selected_category]['product'].unique().tolist())
    else:
        products = ['Все']
    selected_product = st.selectbox("Конкретный товар", products)

    # Горизонт прогноза
    horizon = st.slider("Горизонт прогноза (периодов)", 1, 52, 8)

    # Фильтрация данных
    if selected_category == 'Все':
        df_filtered = df.copy()
    else:
        df_filtered = df[df['category'] == selected_category]
        if selected_product != 'Все':
            df_filtered = df_filtered[df_filtered['product'] == selected_product]
    if df_filtered.empty:
        st.warning("⚠️ Нет данных для выбранной комбинации.")
        st.stop()

    # Кнопка запуска прогноза
    if st.button("🚀 Построить прогноз"):
        start_time = time.time()
        with st.spinner("Идёт агрегация и обучение моделей... Это может занять до минуты."):
            # Агрегация временного ряда
            ts = df_filtered.set_index('datetime').resample(freq)['total'].sum()
            del df_filtered  # освобождаем память
            ts = ts.asfreq(freq)
            ts.interpolate(method='linear', inplace=True)
            ts.bfill(inplace=True)
            ts.ffill(inplace=True)
            ts.dropna(inplace=True)

            if len(ts) < horizon + 5:
                st.error(f"❌ Недостаточно данных для прогноза (доступно {len(ts)} точек).")
                st.stop()

            train = ts.iloc[:-horizon]
            test = ts.iloc[-horizon:]

            # Параметры сезонности и лагов
            if freq == 'h':
                sp, lags = 24, min(24, len(train)//2)
            elif freq == 'D':
                sp, lags = 7, min(14, len(train)//2)
            elif freq == 'W-MON':
                sp, lags = 52, min(12, len(train)//2)
            else:  # MS
                sp, lags = 12, min(6, len(train)//2)

            # Признак праздника для ML
            holiday_series = pd.Series(
                [1 if is_holiday(d) else 0 for d in train.index],
                index=train.index, dtype=np.int8
            )

            results = {}

            # Holt-Winters (быстрая базовая модель)
            try:
                hw = ExponentialSmoothing(train, trend='add', seasonal='add',
                                          seasonal_periods=sp,
                                          initialization_method='estimated').fit()
                pred = hw.forecast(horizon)
                results['Holt-Winters'] = {
                    'rmse': np.sqrt(mean_squared_error(test, pred)),
                    'mape': mape(test, pred)*100,
                    'pred_test': pred,
                    'model': hw
                }
            except Exception as e:
                st.warning(f"Holt-Winters не обучена: {e}")

            # ARIMA с жесткими ограничениями по времени
            if HAS_ARIMA:
                try:
                    arima = pm.auto_arima(
                        train, seasonal=True, m=sp,
                        suppress_warnings=True, error_action='ignore',
                        stepwise=True, trace=False,
                        maxiter=10, time_limit=20
                    )
                    pred = arima.predict(n_periods=horizon)
                    results['ARIMA'] = {
                        'rmse': np.sqrt(mean_squared_error(test, pred)),
                        'mape': mape(test, pred)*100,
                        'pred_test': pred,
                        'model': arima
                    }
                except Exception as e:
                    st.warning(f"ARIMA не обучена: {e}")

            # Random Forest (оптимизированные параметры)
            rf = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42, n_jobs=-1)
            pred = train_ml_model(rf, train, test.index, lags, freq, holiday_series)
            if pred is not None:
                results['Random Forest'] = {
                    'rmse': np.sqrt(mean_squared_error(test, pred)),
                    'mape': mape(test, pred)*100,
                    'pred_test': pred,
                    'model': rf
                }

            # XGBoost
            if HAS_XGB:
                xgb = XGBRegressor(n_estimators=80, max_depth=6, learning_rate=0.1,
                                   random_state=42, verbosity=0, n_jobs=-1)
                pred = train_ml_model(xgb, train, test.index, lags, freq, holiday_series)
                if pred is not None:
                    results['XGBoost'] = {
                        'rmse': np.sqrt(mean_squared_error(test, pred)),
                        'mape': mape(test, pred)*100,
                        'pred_test': pred,
                        'model': xgb
                    }

            # LightGBM
            if HAS_LGB:
                lgbm = LGBMRegressor(n_estimators=80, max_depth=6, learning_rate=0.1,
                                     random_state=42, verbose=-1, n_jobs=-1)
                pred = train_ml_model(lgbm, train, test.index, lags, freq, holiday_series)
                if pred is not None:
                    results['LightGBM'] = {
                        'rmse': np.sqrt(mean_squared_error(test, pred)),
                        'mape': mape(test, pred)*100,
                        'pred_test': pred,
                        'model': lgbm
                    }

            if not results:
                st.error("❌ Ни одна модель не смогла обучиться. Проверьте данные.")
                st.stop()

            best_name = min(results, key=lambda k: results[k]['rmse'])
            best = results[best_name]

            st.subheader(f"🏆 Лучшая модель: {best_name}")
            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{best['rmse']:.2f}")
            col2.metric("MAPE", f"{best['mape']:.2f}%")

            # Обучение лучшей модели на полном ряде
            full_ts = pd.concat([train, test])
            if best_name in ['Holt-Winters', 'ARIMA']:
                if best_name == 'Holt-Winters':
                    full_model = ExponentialSmoothing(
                        full_ts, trend='add', seasonal='add',
                        seasonal_periods=sp, initialization_method='estimated'
                    ).fit()
                else:
                    full_model = pm.auto_arima(
                        full_ts, seasonal=True, m=sp,
                        suppress_warnings=True, error_action='ignore',
                        stepwise=True, trace=False, maxiter=10, time_limit=20
                    )
            else:
                X_full, y_full = create_lag_features(full_ts, lags, freq, holiday_series)
                if best_name == 'Random Forest':
                    full_model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42, n_jobs=-1)
                elif best_name == 'XGBoost':
                    full_model = XGBRegressor(n_estimators=80, max_depth=6, learning_rate=0.1,
                                              random_state=42, verbosity=0, n_jobs=-1)
                else:
                    full_model = LGBMRegressor(n_estimators=80, max_depth=6, learning_rate=0.1,
                                               random_state=42, verbose=-1, n_jobs=-1)
                full_model.fit(X_full, y_full)

            # Будущие даты
            if freq == 'MS':
                start = full_ts.index[-1] + pd.DateOffset(months=1)
            elif freq == 'W-MON':
                start = full_ts.index[-1] + pd.DateOffset(weeks=1)
            elif freq == 'h':
                start = full_ts.index[-1] + pd.DateOffset(hours=1)
            elif freq == 'D':
                start = full_ts.index[-1] + pd.DateOffset(days=1)
            else:
                start = full_ts.index[-1] + pd.Timedelta(1, unit=freq)
            future = pd.date_range(start=start, periods=horizon, freq=freq)

            if best_name in ['Holt-Winters']:
                forecast = full_model.forecast(horizon)
            elif best_name == 'ARIMA':
                forecast = full_model.predict(n_periods=horizon)
            else:
                forecast = recursive_forecast(full_model, full_ts, future, lags, freq)

            # 90% доверительный интервал
            std_res = np.std(np.array(test) - np.array(best['pred_test']))
            lower = forecast - 1.645 * std_res
            upper = forecast + 1.645 * std_res

            elapsed = time.time() - start_time
            st.caption(f"Прогноз построен за {elapsed:.1f} сек.")

            # --------------------------- Графики ---------------------------
            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(x=train.index, y=train.values,
                                          name='Train', line=dict(color='blue')))
            fig_main.add_trace(go.Scatter(x=test.index, y=test.values,
                                          name='Test', line=dict(color='orange')))
            fig_main.add_trace(go.Scatter(x=future, y=forecast,
                                          name='Forecast', line=dict(color='green')))
            fig_main.add_trace(go.Scatter(
                x=np.concatenate([future, future[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself', fillcolor='rgba(0,100,80,0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                name='90% CI'))
            split_date = test.index[0]
            fig_main.add_shape(type='line', x0=split_date, x1=split_date,
                               y0=0, y1=1, yref='paper',
                               line=dict(color='red', dash='dash'))
            fig_main.add_annotation(x=split_date, y=1, yref='paper',
                                    text='Прогноз', showarrow=False,
                                    xanchor='left', textangle=-90)
            fig_main.update_layout(title=f"Прогноз ({best_name})",
                                   xaxis_title='Дата', yaxis_title='Сумма (total)',
                                   hovermode='x unified')
            st.plotly_chart(fig_main, use_container_width=True,
                            config={'scrollZoom': True, 'displayModeBar': True})

            # Сравнение моделей
            st.subheader("Сравнение моделей")
            models_stats = pd.DataFrame([
                {'Модель': m, 'RMSE': d['rmse'], 'MAPE': d['mape']}
                for m, d in results.items()
            ]).sort_values('RMSE')
            fig_comp = make_subplots(rows=1, cols=2, subplot_titles=("RMSE", "MAPE"))
            fig_comp.add_trace(go.Bar(x=models_stats['Модель'], y=models_stats['RMSE'], name='RMSE'), 1, 1)
            fig_comp.add_trace(go.Bar(x=models_stats['Модель'], y=models_stats['MAPE'], name='MAPE'), 1, 2)
            fig_comp.update_layout(showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Таблица прогнозных значений
            st.subheader("📋 Прогнозные значения")
            forecast_table = pd.DataFrame({
                'Дата': future,
                'Прогноз': forecast,
                'Нижняя граница (90%)': lower,
                'Верхняя граница (90%)': upper
            })
            st.dataframe(forecast_table, use_container_width=True)

            # --------------------------- PDF-отчёт ---------------------------
            if st.button("📄 Скачать PDF-отчёт"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, "Отчёт о прогнозировании", ln=1, align='C')
                pdf.ln(10)
                pdf.set_font("Arial", size=10)
                pdf.cell(200, 10, f"Модель: {best_name}", ln=1)
                pdf.cell(200, 10, f"Категория: {selected_category}, Товар: {selected_product}", ln=1)
                pdf.cell(200, 10, f"Периодичность: {freq_label}", ln=1)
                pdf.cell(200, 10, f"Горизонт: {horizon} периодов", ln=1)
                pdf.cell(200, 10, f"RMSE: {best['rmse']:.2f}", ln=1)
                pdf.cell(200, 10, f"MAPE: {best['mape']:.2f}%", ln=1)
                pdf.ln(5)
                pdf.set_font("Arial", 'B', 9)
                pdf.cell(50, 8, "Дата", 1)
                pdf.cell(40, 8, "Прогноз", 1)
                pdf.cell(40, 8, "Нижняя", 1)
                pdf.cell(40, 8, "Верхняя", 1)
                pdf.ln()
                pdf.set_font("Arial", size=9)
                for i, dt in enumerate(future):
                    pdf.cell(50, 8, dt.strftime("%Y-%m-%d"), 1)
                    pdf.cell(40, 8, f"{forecast[i]:.2f}", 1)
                    pdf.cell(40, 8, f"{lower[i]:.2f}", 1)
                    pdf.cell(40, 8, f"{upper[i]:.2f}", 1)
                    pdf.ln()

                fig_mpl, ax = plt.subplots(figsize=(8,4))
                ax.plot(train.index, train.values, label='Train')
                ax.plot(test.index, test.values, label='Test')
                ax.plot(future, forecast, label='Forecast')
                ax.fill_between(future, lower, upper, alpha=0.2)
                ax.axvline(split_date, color='red', linestyle='--')
                ax.legend()
                buf = BytesIO()
                fig_mpl.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                plt.close(fig_mpl)
                pdf.image(buf, x=10, w=190)
                buf.close()

                pdf_bytes = pdf.output(dest='S').encode('latin-1')
                b64 = base64.b64encode(pdf_bytes).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="forecast_report.pdf">Скачать PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

        # Очистка временных переменных (дополнительная гарантия)
        del ts, train, test, holiday_series, results
        gc.collect()
