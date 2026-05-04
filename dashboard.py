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

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ---------------------------- Праздники РФ ----------------------------
RUSSIAN_HOLIDAYS = {
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

# ---------------------------- Метрика ----------------------------
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=np.float64), np.array(y_pred, dtype=np.float64)
    mask = y_true != 0
    if np.sum(mask) == 0: return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

# ---------------------------- Обучение ML (RF и XGBoost) ----------------------------
def train_ml_model(model, train_series, test_index, lags, freq, holiday_series=None):
    X = pd.DataFrame(index=train_series.index)
    for lag in range(1, lags+1):
        X[f'lag_{lag}'] = train_series.shift(lag)
    if holiday_series is not None:
        X['holiday'] = holiday_series
    y = train_series.copy()
    valid = ~X.isna().any(axis=1)
    X, y = X.loc[valid], y.loc[valid]
    if len(X) < 5:
        return None, None
    model.fit(X, y)
    # Рекурсивный прогноз на тестовый период
    test_pred = []
    hist = y.iloc[-lags:].tolist()
    for i, dt in enumerate(test_index):
        feat = {}
        for j in range(lags):
            feat[f'lag_{j+1}'] = hist[-j-1] if len(hist) > j else np.nan
        if holiday_series is not None:
            feat['holiday'] = 1 if is_holiday(dt) else 0
        X_row = pd.DataFrame([feat])
        pred = model.predict(X_row)[0]
        test_pred.append(pred)
        hist.append(pred)
    return np.array(test_pred), model

# ---------------------------- Интерфейс ----------------------------
st.set_page_config(page_title="Интеллектуальная модель прогнозирования продаж", layout="wide")
st.title("📈 Интеллектуальная модель прогнозирования продаж")
st.markdown("Загрузите CSV-файл с продажами и получите прогноз с автоматическим выбором лучшей модели.")

with st.sidebar:
    st.info("**Обязательные столбцы:** date, time, category, product, quantity, price, total\n\n"
            "**Формат даты:** любой\n"
            "**Кодировка:** авто или вручную")

uploaded = st.file_uploader("📂 Загрузите CSV-файл (до 150 МБ)", type="csv")
if uploaded:
    if uploaded.size > 150*1024*1024:
        st.error("❌ Файл > 150 МБ"); st.stop()

    enc = st.selectbox("Кодировка", ['auto','utf-8','cp1251'])
    if enc == 'auto':
        raw = uploaded.read()
        try:
            import chardet
            enc = chardet.detect(raw)['encoding'] or 'utf-8'
        except: enc = 'utf-8'
        uploaded.seek(0)

    try:
        df = pd.read_csv(uploaded, encoding=enc,
                         usecols=['date','time','category','product','quantity','price','total'],
                         dtype={'date':str,'time':str,'category':str,'product':str,
                                'quantity':np.float32,'price':np.float32,'total':np.float32})
    except Exception as e:
        st.error(f"Ошибка чтения: {e}"); st.stop()

    required = ['date','time','category','product','quantity','price','total']
    if missing := [c for c in required if c not in df.columns]:
        st.error(f"❌ Отсутствуют столбцы: {', '.join(missing)}"); st.stop()

    for col in df.select_dtypes(object).columns:
        if df[col].astype(str).str.startswith(('=', '+', '-', '@')).any():
            st.error("⚠️ Обнаружены ячейки с инъекциями"); st.stop()

    df['date'] = df['date'].astype(str).str.strip()
    df['time'] = df['time'].astype(str).str.strip()
    time_empty = df['time'].str.replace(r'[\s\.]','',regex=True).eq('').all()
    df['datetime'] = pd.to_datetime(df['date'] + (' ' + df['time'] if not time_empty else ''), errors='coerce')
    df.dropna(subset=['datetime','quantity','price','total'], inplace=True)
    df = df[df['total'] > 0].drop_duplicates()
    df.sort_values('datetime', inplace=True)
    if df.empty:
        st.error("Нет данных после очистки"); st.stop()

    st.success(f"✅ {len(df)} записей загружено")
    with st.expander("🔍 Первые 10 строк"):
        st.dataframe(df.head(10))

    col1, col2, col3 = st.columns(3)
    freq_map = {'час':'h','день':'D','неделя':'W-MON','месяц':'MS'}
    freq_label = col1.selectbox("Периодичность", list(freq_map.keys()), index=3)
    freq = freq_map[freq_label]
    cats = ['Все'] + sorted(df['category'].unique())
    selected_cat = col2.selectbox("Категория", cats)
    prods = ['Все'] + (sorted(df[df['category']==selected_cat]['product'].unique()) if selected_cat!='Все' else [])
    selected_prod = col3.selectbox("Товар", prods) if prods else None
    horizon = st.slider("Горизонт (периодов)", 1, 52, 5)
    show_advanced = st.checkbox("📊 Расширенная аналитика (декомпозиция, остатки, сравнение)")

    # Фильтрация
    df_f = df.copy()
    if selected_cat != 'Все': df_f = df_f[df_f['category']==selected_cat]
    if selected_prod and selected_prod != 'Все': df_f = df_f[df_f['product']==selected_prod]
    if df_f.empty:
        st.warning("Нет данных для выбранной комбинации"); st.stop()

    if st.button("🚀 Построить прогноз"):
        start = time.time()
        progress = st.progress(0)
        status = st.empty()

        try:
            status.text("Агрегация..."); progress.progress(10)
            ts = df_f.set_index('datetime')['total'].astype(np.float64).resample(freq).sum()
            del df_f; gc.collect()
            ts = ts.asfreq(freq).interpolate().bfill().ffill().dropna()
            if len(ts) < horizon + 5:
                st.error(f"Мало данных ({len(ts)} точек)"); st.stop()

            train, test = ts.iloc[:-horizon], ts.iloc[-horizon:]

            # Параметры
            sp = {'h':24,'D':7,'W-MON':52,'MS':12}[freq]
            if sp >= len(train): sp = max(2, len(train)//2)
            lags = 12 if freq == 'W-MON' else min(6, len(train)//2)

            holiday_series = None
            if freq in ('D','W-MON'):
                holiday_series = pd.Series(
                    [1 if is_holiday(d) else 0 for d in train.index],
                    index=train.index, dtype=np.int8
                )

            models = {}

            # 1. Holt-Winters
            status.text("Holt-Winters..."); progress.progress(20)
            try:
                hw = ExponentialSmoothing(train, trend='add', seasonal='add',
                                          seasonal_periods=sp,
                                          initialization_method='estimated').fit()
                pred = hw.forecast(horizon)
                models['Holt-Winters'] = {
                    'rmse': np.sqrt(mean_squared_error(test, pred)),
                    'mape': mape(test, pred)*100,
                    'pred_test': pred,
                    'model': hw
                }
            except Exception as e:
                st.warning(f"Holt-Winters: {e}")

            # 2. Random Forest
            status.text("Random Forest..."); progress.progress(40)
            rf = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
            pred_rf, rf_model = train_ml_model(rf, train, test.index, lags, freq, holiday_series)
            if pred_rf is not None:
                models['Random Forest'] = {
                    'rmse': np.sqrt(mean_squared_error(test, pred_rf)),
                    'mape': mape(test, pred_rf)*100,
                    'pred_test': pred_rf,
                    'model': rf_model
                }

            # 3. XGBoost (если доступен)
            if HAS_XGB:
                status.text("XGBoost..."); progress.progress(60)
                xgb = XGBRegressor(n_estimators=30, max_depth=5, learning_rate=0.1,
                                   random_state=42, verbosity=0, n_jobs=-1)
                pred_xgb, xgb_model = train_ml_model(xgb, train, test.index, lags, freq, holiday_series)
                if pred_xgb is not None:
                    models['XGBoost'] = {
                        'rmse': np.sqrt(mean_squared_error(test, pred_xgb)),
                        'mape': mape(test, pred_xgb)*100,
                        'pred_test': pred_xgb,
                        'model': xgb_model
                    }

            if not models:
                st.error("Модели не обучились"); st.stop()

            # Выбор лучшей модели по MAPE (меньше – лучше)
            best_name = min(models, key=lambda k: models[k]['mape'])
            best = models[best_name]

            status.text(f"Лучшая модель: {best_name} (MAPE={best['mape']:.2f}%)"); progress.progress(75)

            # Финальное обучение на всех данных
            full_ts = pd.concat([train, test])
            from pandas.tseries.frequencies import to_offset
            start_future = full_ts.index[-1] + to_offset(freq)
            future = pd.date_range(start=start_future, periods=horizon, freq=freq)

            if best_name == 'Holt-Winters':
                full_model = ExponentialSmoothing(full_ts, trend='add', seasonal='add',
                                                  seasonal_periods=sp,
                                                  initialization_method='estimated').fit()
                forecast = full_model.forecast(horizon)
            else:
                # Обучение ML на полном ряду
                X_full = pd.DataFrame(index=full_ts.index)
                for lag in range(1, lags+1):
                    X_full[f'lag_{lag}'] = full_ts.shift(lag)
                if holiday_series is not None:
                    full_hol = pd.Series([1 if is_holiday(d) else 0 for d in full_ts.index],
                                         index=full_ts.index, dtype=np.int8)
                    X_full['holiday'] = full_hol
                y_full = full_ts.copy()
                valid = ~X_full.isna().any(axis=1)
                X_full, y_full = X_full.loc[valid], y_full.loc[valid]

                # Создаём новую модель такого же типа
                if best_name == 'Random Forest':
                    full_model = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
                else:  # XGBoost
                    full_model = XGBRegressor(n_estimators=30, max_depth=5, learning_rate=0.1,
                                              random_state=42, verbosity=0, n_jobs=-1)
                full_model.fit(X_full, y_full)

                future_hol = None
                if freq in ('D','W-MON'):
                    future_hol = [1 if is_holiday(d) else 0 for d in future]

                hist = y_full.iloc[-lags:].tolist()
                forecast = []
                for i in range(horizon):
                    feat = {}
                    for j in range(lags):
                        feat[f'lag_{j+1}'] = hist[-j-1] if len(hist) > j else np.nan
                    if future_hol is not None:
                        feat['holiday'] = future_hol[i]
                    X_row = pd.DataFrame([feat])
                    pred = full_model.predict(X_row)[0]
                    forecast.append(pred)
                    hist.append(pred)
                forecast = np.array(forecast)

            # 90% доверительный интервал
            std_res = np.std(np.array(test) - np.array(best['pred_test']))
            lower = forecast - 1.645*std_res
            upper = forecast + 1.645*std_res

            progress.progress(95)
            status.empty(); progress.empty()

            # ---------- Результаты ----------
            st.subheader(f"🏆 Результаты (модель: {best_name})")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{best['rmse']:,.2f}")
            col2.metric("MAPE", f"{best['mape']:.2f}%")
            # Также покажем лучшую другую модель для сравнения
            other_models = [m for m in models if m != best_name]
            if other_models:
                best_other = min(other_models, key=lambda x: models[x]['mape'])
                col3.metric(f"Альтернатива: {best_other}", f"MAPE {models[best_other]['mape']:.2f}%")

            # График
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train.values, name='Обучающие', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test.index, y=test.values, name='Тестовые', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=future, y=forecast, name='Прогноз', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=np.concatenate([future, future[::-1]]),
                                     y=np.concatenate([upper, lower[::-1]]),
                                     fill='toself', fillcolor='rgba(44,160,44,0.2)',
                                     line=dict(color='rgba(255,255,255,0)'), name='90% CI'))
            split = test.index[0]
            fig.add_shape(type='line', x0=split, x1=split, y0=0, y1=1, yref='paper',
                          line=dict(color='red', dash='dash'))
            fig.add_annotation(x=split, y=1, yref='paper', text='Прогноз', showarrow=False,
                               xanchor='left', textangle=-90)
            fig.update_layout(title=f'Прогноз ({best_name})', xaxis_title='Дата', yaxis_title='Сумма')
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

            # Таблица прогноза
            st.subheader("📋 Таблица прогнозных значений")
            st.dataframe(pd.DataFrame({
                'Дата': future.strftime('%d-%m-%Y'),
                'Прогноз': forecast.round(2),
                'Нижняя граница (90%)': lower.round(2),
                'Верхняя граница (90%)': upper.round(2)
            }), use_container_width=True)

            # Расширенная аналитика
            if show_advanced:
                st.subheader("📊 Расширенная аналитика")
                # Сравнение моделей
                comp = pd.DataFrame([
                    {'Модель':n, 'RMSE':d['rmse'], 'MAPE':d['mape']} for n,d in models.items()
                ]).sort_values('MAPE')
                st.dataframe(comp, use_container_width=True)

                fig_c = make_subplots(rows=1, cols=2, subplot_titles=('RMSE','MAPE'))
                fig_c.add_trace(go.Bar(x=comp['Модель'], y=comp['RMSE'], name='RMSE'), 1, 1)
                fig_c.add_trace(go.Bar(x=comp['Модель'], y=comp['MAPE'], name='MAPE'), 1, 2)
                fig_c.update_layout(showlegend=False)
                st.plotly_chart(fig_c, use_container_width=True)

                # Декомпозиция
                if len(train) >= 2*sp+10:
                    try:
                        dec = seasonal_decompose(train, model='additive', period=sp)
                        fd = make_subplots(rows=4, cols=1, subplot_titles=('Наблюдения','Тренд','Сезонность','Остатки'))
                        fd.add_trace(go.Scatter(x=train.index, y=dec.observed), 1, 1)
                        fd.add_trace(go.Scatter(x=train.index, y=dec.trend), 2, 1)
                        fd.add_trace(go.Scatter(x=train.index, y=dec.seasonal), 3, 1)
                        fd.add_trace(go.Scatter(x=train.index, y=dec.resid), 4, 1)
                        fd.update_layout(height=800, showlegend=False)
                        st.plotly_chart(fd, use_container_width=True)
                    except: pass

                # Остатки на тесте
                resid = test.values - best['pred_test']
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=test.index, y=resid, name='Остатки'))
                fig_r.add_hline(y=0, line_dash='dash', line_color='red')
                fig_r.update_layout(title='Остатки на тестовом периоде')
                st.plotly_chart(fig_r, use_container_width=True)

            st.caption(f"⏱️ Прогноз построен за {time.time()-start:.1f} сек.")

        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
        finally:
            del train, test, ts
            gc.collect()
