import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import base64
import os
import warnings
warnings.filterwarnings('ignore')

# Matplotlib
import matplotlib
if 'DISPLAY' not in os.environ and 'MPLBACKEND' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Модели
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
try:
    import pmdarima as pm
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False

# PDF
from fpdf import FPDF

# ---------------------------- Праздники РФ ---------------------------------
# Официальные нерабочие дни РФ (примерные даты на 2020-2026 гг.)
RUSSIAN_HOLIDAYS = {
    '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05',
    '2020-01-06', '2020-01-07', '2020-01-08', '2020-01-09', '2020-01-10',
    '2020-01-11', '2020-01-12', '2020-02-23', '2020-02-24', '2020-03-08',
    '2020-05-01', '2020-05-09', '2020-06-12', '2020-11-04',
    # 2021
    '2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05',
    '2021-01-06', '2021-01-07', '2021-01-08', '2021-01-09', '2021-01-10',
    '2021-01-11', '2021-01-12', '2021-02-23', '2021-03-08',
    '2021-05-01', '2021-05-09', '2021-06-12', '2021-11-04',
    # 2022
    '2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05',
    '2022-01-06', '2022-01-07', '2022-01-08', '2022-01-09', '2022-01-10',
    '2022-01-11', '2022-01-12', '2022-02-23', '2022-03-08',
    '2022-05-01', '2022-05-09', '2022-06-12', '2022-11-04',
    # 2023
    '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
    '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10',
    '2023-01-11', '2023-01-12', '2023-02-23', '2023-03-08',
    '2023-05-01', '2023-05-09', '2023-06-12', '2023-11-04',
    # 2024
    '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
    '2024-01-06', '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10',
    '2024-01-11', '2024-01-12', '2024-02-23', '2024-03-08',
    '2024-05-01', '2024-05-09', '2024-06-12', '2024-11-04',
    # 2025
    '2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05',
    '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10',
    '2025-01-11', '2025-01-12', '2025-02-23', '2025-03-08',
    '2025-05-01', '2025-05-09', '2025-06-12', '2025-11-04',
    # 2026
    '2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04', '2026-01-05',
    '2026-01-06', '2026-01-07', '2026-01-08', '2026-01-09', '2026-01-10',
    '2026-01-11', '2026-01-12', '2026-02-23', '2026-03-08',
    '2026-05-01', '2026-05-09', '2026-06-12', '2026-11-04',
}

def is_holiday(dt):
    """Проверяет, является ли дата праздничным днём в РФ."""
    date_str = dt.strftime('%Y-%m-%d')
    return date_str in RUSSIAN_HOLIDAYS

# --------------------------------------------------------------------------
# Функции метрик и приведения типов
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def create_lag_features(series, lags, freq_str, holiday_series=None):
    """
    Создаёт DataFrame с признаками: лаги, скользящие статистики, временные метки, праздники.
    holiday_series: pd.Series с таким же индексом, где 1, если дата выходной/праздник, иначе 0.
    """
    df_feat = pd.DataFrame(index=series.index)
    # Лаги
    for lag in range(1, lags+1):
        df_feat[f'lag_{lag}'] = series.shift(lag)
    # Скользящие статистики
    for w in [2, 3, 5, 7, 14]:
        if len(series) >= w:
            df_feat[f'rolling_mean_{w}'] = series.rolling(window=w).mean()
            df_feat[f'rolling_std_{w}'] = series.rolling(window=w).std()
    # Временные признаки
    if freq_str == 'h':
        df_feat['hour'] = series.index.hour
        df_feat['dayofweek'] = series.index.dayofweek
        df_feat['month'] = series.index.month
        df_feat['quarter'] = series.index.quarter
    elif freq_str == 'D':
        df_feat['dayofweek'] = series.index.dayofweek
        df_feat['month'] = series.index.month
        df_feat['quarter'] = series.index.quarter
        df_feat['year'] = series.index.year
    elif freq_str == 'W-MON':
        df_feat['weekofyear'] = series.index.isocalendar().week.astype(int)
        df_feat['month'] = series.index.month
        df_feat['quarter'] = series.index.quarter
        df_feat['year'] = series.index.year
    elif freq_str == 'MS':
        df_feat['month'] = series.index.month
        df_feat['quarter'] = series.index.quarter
        df_feat['year'] = series.index.year
    # Праздники
    if holiday_series is not None:
        df_feat['holiday'] = holiday_series.loc[df_feat.index].fillna(0).astype(int)
    df_feat.dropna(inplace=True)
    return df_feat, series[df_feat.index]

def recursive_forecast(model, history_series, forecast_dates, lags, freq_str, holiday_series=None):
    """Рекурсивный прогноз на несколько шагов."""
    hist = history_series.copy()
    preds = []
    for i, dt in enumerate(forecast_dates):
        last_vals = hist.iloc[-lags:]
        feat = {}
        for lag in range(1, lags+1):
            feat[f'lag_{lag}'] = last_vals.iloc[-lag] if len(last_vals) >= lag else np.nan
        for w in [2, 3, 5, 7, 14]:
            if len(hist) >= w:
                feat[f'rolling_mean_{w}'] = hist.iloc[-w:].mean()
                feat[f'rolling_std_{w}'] = hist.iloc[-w:].std()
            else:
                feat[f'rolling_mean_{w}'] = np.mean(hist)
                feat[f'rolling_std_{w}'] = np.std(hist)
        # Временные признаки
        if freq_str == 'h':
            feat['hour'] = dt.hour
            feat['dayofweek'] = dt.dayofweek
            feat['month'] = dt.month
            feat['quarter'] = dt.quarter
        elif freq_str == 'D':
            feat['dayofweek'] = dt.dayofweek
            feat['month'] = dt.month
            feat['quarter'] = dt.quarter
            feat['year'] = dt.year
        elif freq_str == 'W-MON':
            feat['weekofyear'] = dt.isocalendar().week
            feat['month'] = dt.month
            feat['quarter'] = dt.quarter
            feat['year'] = dt.year
        elif freq_str == 'MS':
            feat['month'] = dt.month
            feat['quarter'] = dt.quarter
            feat['year'] = dt.year
        # Признак праздника для будущей даты
        if holiday_series is not None:
            feat['holiday'] = 1 if is_holiday(dt) else 0
        X = pd.DataFrame([feat])
        pred = model.predict(X)[0]
        preds.append(pred)
        hist = pd.concat([hist, pd.Series({dt: pred})])
    return np.array(preds)

def train_ml_model(model, train_series, test_index, lags, freq_str, holiday_series=None):
    X_train, y_train = create_lag_features(train_series, lags, freq_str, holiday_series)
    if len(X_train) == 0:
        return None, None
    model.fit(X_train, y_train)
    preds = recursive_forecast(model, train_series, test_index, lags, freq_str, holiday_series)
    return preds

# ---------------------------- Интерфейс Streamlit ----------------------------
st.set_page_config(layout="wide")
st.title("📈 Прогнозирование продаж (расширенная система)")

# 1. Загрузка файла
uploaded = st.file_uploader("Загрузите CSV-файл (до 150 МБ)", type="csv")
if uploaded:
    # Ограничение размера (проверка вручную, st.file_uploader лимита не имеет)
    if uploaded.size > 150 * 1024 * 1024:
        st.error("Размер файла превышает 150 МБ. Загрузите файл меньшего размера.")
        st.stop()

    # Кодировка
    enc_choice = st.selectbox("Кодировка", ['auto','utf-8','cp1251','latin1','iso-8859-1','cp1252'])
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

    try:
        df = pd.read_csv(uploaded, encoding=enc)
    except Exception as e:
        st.error(f"Ошибка чтения: {e}")
        st.stop()

    # 2. Проверка обязательных столбцов
    required = ['date','time','category','product','quantity','price','total']
    if not all(col in df.columns for col in required):
        st.error(f"❌ Отсутствуют столбцы: {', '.join(set(required)-set(df.columns))}")
        st.stop()

    # 3. Проверка на опасные конструкции в ячейках
    danger_patterns = ['=', '@', '+', '-', 'DROP', 'DELETE', 'INSERT', 'UPDATE', '<script>']
    for col in df.columns:
        # Проверяем только строковые колонки
        if df[col].dtype == object:
            for pat in danger_patterns:
                if df[col].astype(str).str.contains(pat, case=False).any():
                    st.error(f"⚠️ Обнаружена потенциально опасная конструкция в столбце '{col}'. Загрузка остановлена.")
                    st.stop()

    # 4. Очистка данных
    df['date'] = df['date'].astype(str).str.strip()
    df['time'] = df['time'].astype(str).str.strip()
    time_empty = df['time'].str.replace(r'[\s\.]','',regex=True).eq('').all()
    if not time_empty:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    else:
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    for c in ['quantity','price','total']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['quantity','price','total'], inplace=True)
    # Удаление дубликатов и нулевых total
    df.drop_duplicates(inplace=True)
    df = df[df['total'] > 0]
    df.sort_values('datetime', inplace=True)

    if df.empty:
        st.error("Нет данных после очистки.")
        st.stop()

    # 5. Отображение первых 10 строк
    st.subheader("Первые 10 строк загруженных данных")
    st.dataframe(df.head(10))

    # 6. Выбор периодичности
    freq_map = {'час': 'h', 'день': 'D', 'неделя': 'W-MON', 'месяц': 'MS'}
    freq_label = st.selectbox("Периодичность агрегации", list(freq_map.keys()))
    freq = freq_map[freq_label]

    # 7. Выбор категории и продукта
    categories = ['Все'] + sorted(df['category'].unique().tolist())
    selected_category = st.selectbox("Категория товаров", categories)
    if selected_category != 'Все':
        products = ['Все'] + sorted(df[df['category'] == selected_category]['product'].unique().tolist())
    else:
        products = ['Все']
    selected_product = st.selectbox("Товар (если выбрана категория)", products)

    # 8. Горизонт прогноза
    horizon = st.slider("Горизонт прогноза (периодов)", 1, 52, 8)

    # 9. Фильтрация данных
    if selected_category == 'Все':
        df_filtered = df.copy()
    else:
        df_filtered = df[df['category'] == selected_category]
        if selected_product != 'Все':
            df_filtered = df_filtered[df_filtered['product'] == selected_product]

    if df_filtered.empty:
        st.warning("Нет данных для выбранной комбинации категории/товара.")
        st.stop()

    # 10. Запуск прогноза
    if st.button("🚀 Построить прогноз"):
        with st.spinner("Обучение моделей... Пожалуйста, подождите."):
            # Агрегация
            ts = df_filtered.set_index('datetime').resample(freq)['total'].sum()
            # Обеспечиваем полный ряд и интерполяцию
            ts = ts.asfreq(freq)
            ts.interpolate(method='linear', inplace=True)
            ts.bfill(inplace=True)
            ts.ffill(inplace=True)
            ts.dropna(inplace=True)

            if len(ts) < horizon + 5:
                st.error(f"Недостаточно данных. Требуется минимум {horizon+5} точек, доступно {len(ts)}.")
                st.stop()

            # Разбивка train/test
            train = ts.iloc[:-horizon]
            test = ts.iloc[-horizon:]

            # Параметры сезонности и лагов
            if freq == 'h':
                sp = 24
                lags = min(48, len(train)//2)
            elif freq == 'D':
                sp = 7
                lags = min(30, len(train)//2)
            elif freq == 'W-MON':
                sp = 52
                lags = min(26, len(train)//2)
            else: # MS
                sp = 12
                lags = min(12, len(train)//2)

            # Создаём признак праздника для ML-моделей (только если даты присутствуют)
            holiday_series = pd.Series(
                [1 if is_holiday(d) else 0 for d in train.index],
                index=train.index
            )

            results = {}

            # 1. Holt-Winters
            try:
                model_hw = ExponentialSmoothing(
                    train, trend='add', seasonal='add',
                    seasonal_periods=sp,
                    initialization_method='estimated'
                ).fit()
                pred_hw = model_hw.forecast(horizon)
                rmse = np.sqrt(mean_squared_error(test, pred_hw))
                mape_val = mape(test, pred_hw) * 100
                results['Holt-Winters'] = {'rmse': rmse, 'mape': mape_val, 'pred_test': pred_hw, 'model': model_hw}
            except Exception as e:
                st.warning(f"Holt-Winters не обучена: {e}")

            # 2. ARIMA
            if HAS_ARIMA:
                try:
                    model_arima = pm.auto_arima(
                        train, seasonal=True, m=sp, suppress_warnings=True,
                        error_action='ignore', stepwise=True, trace=False, maxiter=30
                    )
                    pred_arima = model_arima.predict(n_periods=horizon)
                    rmse = np.sqrt(mean_squared_error(test, pred_arima))
                    mape_val = mape(test, pred_arima) * 100
                    results['ARIMA'] = {'rmse': rmse, 'mape': mape_val, 'pred_test': pred_arima, 'model': model_arima}
                except Exception as e:
                    st.warning(f"ARIMA не обучена: {e}")

            # 3. Random Forest (с признаками)
            rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
            pred_rf = train_ml_model(rf, train, test.index, lags, freq, holiday_series)
            if pred_rf is not None:
                rmse = np.sqrt(mean_squared_error(test, pred_rf))
                mape_val = mape(test, pred_rf) * 100
                results['Random Forest'] = {'rmse': rmse, 'mape': mape_val, 'pred_test': pred_rf, 'model': rf}

            # 4. XGBoost
            if HAS_XGB:
                xgb = XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
                pred_xgb = train_ml_model(xgb, train, test.index, lags, freq, holiday_series)
                if pred_xgb is not None:
                    rmse = np.sqrt(mean_squared_error(test, pred_xgb))
                    mape_val = mape(test, pred_xgb) * 100
                    results['XGBoost'] = {'rmse': rmse, 'mape': mape_val, 'pred_test': pred_xgb, 'model': xgb}

            # 5. LightGBM
            if HAS_LGB:
                lgbm = LGBMRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)
                pred_lgbm = train_ml_model(lgbm, train, test.index, lags, freq, holiday_series)
                if pred_lgbm is not None:
                    rmse = np.sqrt(mean_squared_error(test, pred_lgbm))
                    mape_val = mape(test, pred_lgbm) * 100
                    results['LightGBM'] = {'rmse': rmse, 'mape': mape_val, 'pred_test': pred_lgbm, 'model': lgbm}

            if not results:
                st.error("Ни одна модель не обучилась.")
                st.stop()

            # Выбор лучшей модели
            best_name = min(results, key=lambda k: results[k]['rmse'])
            best = results[best_name]
            st.subheader(f"🏆 Лучшая модель: {best_name}")
            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{best['rmse']:.2f}")
            col2.metric("MAPE", f"{best['mape']:.2f}%")

            # Финальное обучение на всех данных для прогноза будущего
            full_ts = pd.concat([train, test])
            if best_name in ['Holt-Winters', 'ARIMA']:
                # Используем модель обученную на полном ряду (переобучаем)
                if best_name == 'Holt-Winters':
                    full_model = ExponentialSmoothing(full_ts, trend='add', seasonal='add',
                                                      seasonal_periods=sp,
                                                      initialization_method='estimated').fit()
                else:
                    full_model = pm.auto_arima(full_ts, seasonal=True, m=sp, suppress_warnings=True,
                                               error_action='ignore', stepwise=True, trace=False, maxiter=30)
            else:
                # ML: создаём признаки на полном ряду и переобучаем ту же архитектуру
                X_full, y_full = create_lag_features(full_ts, lags, freq, holiday_series)
                if best_name == 'Random Forest':
                    full_model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
                elif best_name == 'XGBoost':
                    full_model = XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)
                else:
                    full_model = LGBMRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)
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

            # Прогноз
            if best_name in ['Holt-Winters']:
                forecast = full_model.forecast(horizon)
            elif best_name == 'ARIMA':
                forecast = full_model.predict(n_periods=horizon)
            else:
                forecast = recursive_forecast(full_model, full_ts, future, lags, freq, holiday_series)

            # 90% доверительный интервал (z=1.645)
            std_res = np.std(np.array(test) - np.array(best['pred_test']))
            lower = forecast - 1.645 * std_res
            upper = forecast + 1.645 * std_res

            # ---------- Графики ----------
            # Основной график
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

            # Дополнительные графики: сравнение моделей
            st.subheader("Сравнение моделей")
            models_stats = pd.DataFrame([
                {'Модель': m, 'RMSE': d['rmse'], 'MAPE': d['mape']}
                for m, d in results.items()
            ]).sort_values('RMSE')
            fig_comp = make_subplots(rows=1, cols=2, subplot_titles=("RMSE", "MAPE"))
            fig_comp.add_trace(go.Bar(x=models_stats['Модель'], y=models_stats['RMSE'], name='RMSE'), row=1, col=1)
            fig_comp.add_trace(go.Bar(x=models_stats['Модель'], y=models_stats['MAPE'], name='MAPE'), row=1, col=2)
            fig_comp.update_layout(showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Сезонная декомпозиция (если ряд достаточно длинный)
            if len(train) >= 2*sp+10:
                try:
                    decomp = seasonal_decompose(train, model='additive', period=sp)
                    fig_seas = make_subplots(rows=4, cols=1,
                                             subplot_titles=("Наблюдения", "Тренд", "Сезонность", "Остатки"))
                    fig_seas.add_trace(go.Scatter(x=train.index, y=decomp.observed), row=1, col=1)
                    fig_seas.add_trace(go.Scatter(x=train.index, y=decomp.trend), row=2, col=1)
                    fig_seas.add_trace(go.Scatter(x=train.index, y=decomp.seasonal), row=3, col=1)
                    fig_seas.add_trace(go.Scatter(x=train.index, y=decomp.resid), row=4, col=1)
                    fig_seas.update_layout(height=800, showlegend=False)
                    st.plotly_chart(fig_seas, use_container_width=True)
                except Exception as e:
                    st.info("Сезонная декомпозиция не построена: недостаточно данных или ошибка.")

            # Остатки модели на тесте
            resid_test = test.values - best['pred_test']
            fig_resid = go.Figure()
            fig_resid.add_trace(go.Scatter(x=test.index, y=resid_test, name='Остатки'))
            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
            fig_resid.update_layout(title="Остатки на тестовом периоде")
            st.plotly_chart(fig_resid, use_container_width=True)

            # Таблица прогнозных значений
            st.subheader("Прогнозные значения")
            forecast_table = pd.DataFrame({
                'Дата': future,
                'Прогноз': forecast,
                'Нижняя граница (90%)': lower,
                'Верхняя граница (90%)': upper
            })
            st.dataframe(forecast_table, use_container_width=True)

            # ---------- PDF-отчёт ----------
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
                # Таблица
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

                # Основной график в PDF
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
