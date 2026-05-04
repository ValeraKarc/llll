import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
if 'DISPLAY' not in os.environ and 'MPLBACKEND' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

st.set_page_config(layout="wide")
st.title("Прогнозирование (Holt-Winters) с очисткой данных")

enc_choice = st.selectbox("Кодировка", ['auto','utf-8','cp1251','latin1','iso-8859-1','cp1252'], index=0)
uploaded = st.file_uploader("CSV-файл", type="csv")

if uploaded:
    # ---------- Кодировка ----------
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

    # ---------- Проверка столбцов ----------
    needed = ['date','time','category','product','quantity','price','total']
    if not all(col in df.columns for col in needed):
        st.error(f"Нет обязательных столбцов: {', '.join(needed)}")
        st.stop()

    # ---------- Преобразование даты и времени ----------
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

    # ---------- Очистка данных (выбросы, дубликаты) ----------
    df.drop_duplicates(inplace=True)

    Q1 = df['total'].quantile(0.25)
    Q3 = df['total'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['total'] >= lower_bound) & (df['total'] <= upper_bound)]

    df = df[df['total'] > 0]   # убираем нулевые/отрицательные

    df.sort_values('datetime', inplace=True)

    if df.empty:
        st.error("Нет данных после очистки")
        st.stop()

    st.success(f"Данных после очистки: {len(df)} строк")

    # ---------- Настройки прогноза ----------
    freq_map = {'час': 'h', 'день': 'D', 'неделя': 'W-MON', 'месяц': 'MS'}
    freq_label = st.selectbox("Периодичность", list(freq_map.keys()))
    freq = freq_map[freq_label]
    horizon = st.slider("Горизонт прогноза", 1, 52, 8)

    if st.button("🚀 Создать прогноз"):
        with st.spinner("Обработка и прогноз..."):
            # ---------- Агрегация и заполнение пропусков ----------
            ts = df.set_index('datetime').resample(freq)['total'].sum()
            ts = ts.asfreq(freq)                        # полный календарь
            ts.interpolate(method='linear', inplace=True)  # линейная интерполяция пропусков
            ts.bfill(inplace=True)                      # заполняем начало
            ts.ffill(inplace=True)                      # заполняем конец
            ts.dropna(inplace=True)

            if len(ts) < horizon + 5:
                st.error(f"Недостаточно данных после агрегации (осталось {len(ts)} точек).")
                st.stop()

            train = ts.iloc[:-horizon]
            test = ts.iloc[-horizon:]

            # ---------- Параметры сезонности ----------
            if freq == 'h':
                sp = 24
            elif freq == 'D':
                sp = 7
            elif freq == 'W-MON':
                sp = 52
            else:
                sp = 12
            if sp >= len(train):
                sp = max(2, len(train)//2)

            # ---------- Обучение Holt-Winters ----------
            try:
                model = ExponentialSmoothing(
                    train,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=sp,
                    initialization_method='estimated'
                ).fit()
                test_pred = model.forecast(horizon)
                rmse_val = np.sqrt(mean_squared_error(test, test_pred))
                mape_val = mape(test, test_pred) * 100
            except Exception as e:
                st.error(f"Ошибка обучения Holt-Winters: {e}")
                st.stop()

            st.subheader("Метрики модели")
            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{rmse_val:.2f}")
            col2.metric("MAPE", f"{mape_val:.1f}%")

            # ---------- Прогноз на будущее ----------
            full = pd.concat([train, test])
            if freq == 'MS':
                start = full.index[-1] + pd.DateOffset(months=1)
            elif freq == 'W-MON':
                start = full.index[-1] + pd.DateOffset(weeks=1)
            elif freq == 'h':
                start = full.index[-1] + pd.DateOffset(hours=1)
            elif freq == 'D':
                start = full.index[-1] + pd.DateOffset(days=1)
            else:
                start = full.index[-1] + pd.Timedelta(1, unit=freq)
            future = pd.date_range(start=start, periods=horizon, freq=freq)

            full_model = ExponentialSmoothing(
                full,
                trend='add',
                seasonal='add',
                seasonal_periods=sp,
                initialization_method='estimated'
            ).fit()
            forecast = full_model.forecast(horizon)

            std_res = np.std(train - test_pred)
            lower = forecast - 1.96 * std_res
            upper = forecast + 1.96 * std_res

            # ---------- График ----------
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train.values,
                                     name='Train', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test.index, y=test.values,
                                     name='Test', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=future, y=forecast,
                                     name='Forecast', line=dict(color='green')))
            fig.add_trace(go.Scatter(
                x=np.concatenate([future, future[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself', fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI'))

            split_date = test.index[0]
            fig.add_shape(type='line', x0=split_date, x1=split_date,
                          y0=0, y1=1, yref='paper',
                          line=dict(color='red', dash='dash'))
            fig.add_annotation(x=split_date, y=1, yref='paper',
                               text='Прогноз', showarrow=False,
                               xanchor='left', textangle=-90)

            fig.update_layout(title='Прогноз (Holt-Winters с очисткой)',
                              xaxis_title='Дата', yaxis_title='Сумма (total)',
                              hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True,
                            config={'scrollZoom': True, 'displayModeBar': True})
